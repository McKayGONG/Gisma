#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <Application.h>
#include <Utility.h>
#include <Graph.h>
#include <thread>
#include <fstream>
#include <random>
#include <chrono>
#include <memory>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>

// GED-gen exact-search iteration cap (AppForComputation's exact_max_iter), overridable via
// env GED_GEN_ITER (default 1000000 = true exact). Training labels must be exact GED, so we
// use AppForComputation (searches to the optimum); pairs that do not converge within the cap
// return -1 and are skipped. Lower the cap only for timing tests, not for producing labels.
static int g_ged_gen_iter = []{ const char* e = std::getenv("GED_GEN_ITER"); return (e && atoi(e) > 0) ? atoi(e) : 1000000; }();
// Method switch: default exact (AppForComputation). Set GED_GEN_APP=<iter> to use the fast
// iteration-capped App() (upper-bound labels). Needed only for the SYN grid, where exact GED
// is intractable on i.i.d. random graphs above ~16 nodes (all pairs high-GED, no similar pairs
// to compute). App(iter) with UB-exit gives usable labels there; real datasets stay exact.
static int g_app_iter = []{ const char* e = std::getenv("GED_GEN_APP"); return (e && atoi(e) > 0) ? atoi(e) : 0; }();
#include <iomanip>

using namespace std;
using namespace chrono;

// Global mutex: protect file write operations
static std::mutex g_write_mutex;

// Global mutex: protect log file operations
static std::mutex g_log_mutex;

// Atomic: completed task count for progress bar  
static std::atomic<long long> tasksDone(0);

// Atomic: statistics counters
static std::atomic<long long> totalValidPairs(0);
static std::atomic<long long> totalProcessedPairs(0);

// Chunk size for periodic writes
constexpr int CHUNK_SIZE = 100;

// Batch parameter structure
struct BatchParam {
    long long N;
    int max_ged;
    int num_threads;
    string description;
};

// Mixed task structure for load balancing
struct MixedTask {
    long long pair_idx;  // Random pair index
    int max_ged;         // GED threshold for this task
    int param_set_id;    // Which parameter set this task belongs to
};

// Streaming task generator to avoid memory overflow
class StreamingTaskGenerator {
private:
    vector<BatchParam> params;
    vector<long long> cumulative_counts;
    long long total_tasks;
    long long total_possible;
    mutable mt19937 rng;
    mutable uniform_int_distribution<long long> dis;
    
public:
    StreamingTaskGenerator(const vector<BatchParam>& batch_params, size_t db_size) 
        : params(batch_params), total_tasks(0), total_possible((long long)db_size * db_size), 
          rng(random_device{}()), dis(0, total_possible - 1) {
        
        cumulative_counts.reserve(params.size());
        for (const auto& param : params) {
            total_tasks += param.N;
            cumulative_counts.push_back(total_tasks);
        }
    }
    
    long long getTotalTasks() const { return total_tasks; }
    
    // Get task by global index (thread-safe)
    MixedTask getTask(long long global_idx) const {
        if (global_idx >= total_tasks) {
            throw runtime_error("Task index out of range");
        }
        
        // Binary search to find which parameter set this task belongs to
        int param_set_id = lower_bound(cumulative_counts.begin(), cumulative_counts.end(), 
                                     global_idx + 1) - cumulative_counts.begin();
        
        // Generate random pair_idx for this specific task
        // Use global_idx as seed modifier to ensure reproducibility while maintaining randomness
        mt19937 task_rng(rng() ^ global_idx);  
        uniform_int_distribution<long long> task_dis(0, total_possible - 1);
        
        MixedTask task;
        task.pair_idx = task_dis(task_rng);
        task.max_ged = params[param_set_id].max_ged;
        task.param_set_id = param_set_id;
        
        return task;
    }
};

// Logging functions
void writeLog(const string& logFile, const string& message) {
    lock_guard<mutex> lock(g_log_mutex);
    ofstream log(logFile, ios::app);
    if (log) {
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        log << "[" << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
            << "." << setfill('0') << setw(3) << ms.count() << "] " 
            << message << endl;
    }
}

string generateLogFileName(const string& outputFile) {
    // Extract directory and base name from output file
    size_t lastSlash = outputFile.find_last_of("/\\");
    string directory = "";
    string fileName = outputFile;
    
    if (lastSlash != string::npos) {
        directory = outputFile.substr(0, lastSlash + 1);
        fileName = outputFile.substr(lastSlash + 1);
    }
    
    // Generate training data generation log filename
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&time_t));
    
    return directory + "training_data_generation_" + string(timestamp) + ".log";
}

/**
 * Fixed version: correctly decode random index to valid graph pair indices
 * Ensures both i and j are within [0, n) range
 */
static inline void decodePair(long long x, int n, int &i, int &j) {
    // Use simple modular arithmetic to ensure both indices are in valid range
    i = (int)(x % n);
    j = (int)((x / n) % n);
    
    // Safety check: ensure j is within bounds (should not happen with correct logic)
    if (j >= n) {
        j = j % n;  // Force into valid range as failsafe
    }
}

// Streaming mixed task worker function: processes tasks with different max_ged values
void compute_streaming_ged_worker(
    const vector<Graph*>& db,
    const StreamingTaskGenerator& taskGenerator,
    const char* output_filename,
    long long total_count,
    atomic<long long>& nextIdx
)
{
    // Local cache
    vector<tuple<int,int,int>> local_cache;
    local_cache.reserve(CHUNK_SIZE);

    int n = (int)db.size();

    while(true) {
        long long idx = nextIdx.fetch_add(1, memory_order_relaxed);
        if(idx >= total_count) {
            break; // No more tasks
        }

        // Get mixed task dynamically
        MixedTask task = taskGenerator.getTask(idx);
        int i, j;
        decodePair(task.pair_idx, n, i, j);

        // Allow i == j, generate GED=0 training data
        Graph* db_graph    = db[i];
        Graph* query_graph = db[j];

        // GED calculation with task-specific max_ged. Default: EXACT (AppForComputation, no UB
        // early-exit; non-converged pairs return -1 and are skipped). If GED_GEN_APP is set, use
        // the fast UB-exit App(iter) instead (upper-bound labels; only for the SYN grid).
        Application app(task.max_ged, "BMao", (g_app_iter > 0 ? g_app_iter : 2300), g_ged_gen_iter);
        app.init(db_graph, query_graph);
        int ged_res = g_app_iter > 0 ? (int)app.App(nullptr, nullptr)
                                     : (int)app.AppForComputation(nullptr, nullptr);

        // Update statistics
        totalProcessedPairs.fetch_add(1, memory_order_relaxed);
        
        // Skip if not exact GED (reached computation limit)
        if(ged_res == -1) {
            // Skip this sample - not exact GED
            tasksDone.fetch_add(1, memory_order_relaxed);
            continue;
        }
        
        // If qualified, store in local_cache
        if(ged_res <= task.max_ged) {
            local_cache.emplace_back(i, j, ged_res);
            totalValidPairs.fetch_add(1, memory_order_relaxed);
        }

        // Completed tasks +1
        tasksDone.fetch_add(1, memory_order_relaxed);

        // If accumulated CHUNK_SIZE, write to file
        if((int)local_cache.size() >= CHUNK_SIZE) {
            lock_guard<mutex> lock(g_write_mutex);
            ofstream ofs(output_filename, ios::app);
            if(!ofs) {
                cerr << "Error opening file for write: " << output_filename << endl;
            } else {
                for(const auto &tp : local_cache) {
                    int gg1, gg2, dist;
                    tie(gg1, gg2, dist) = tp;
                    ofs << gg1 << "," << gg2 << "," << dist << "\n";
                }
            }
            local_cache.clear();
        }
    }

    // Cleanup: if there are unwritten items, write once more
    if(!local_cache.empty()) {
        lock_guard<mutex> lock(g_write_mutex);
        ofstream ofs(output_filename, ios::app);
        if(!ofs) {
            cerr << "Error opening file for write: " << output_filename << endl;
        } else {
            for(const auto &tp : local_cache) {
                int g1, g2, dist;
                tie(g1, g2, dist) = tp;
                ofs << g1 << "," << g2 << "," << dist << "\n";
            }
        }
    }
}

// Original thread function: dynamic scheduling + write every CHUNK_SIZE accumulations
void compute_ged_worker(
    const vector<Graph*>& db,
    const vector<long long>& randIndices,
    int max_ged,
    const char* output_filename,
    int total_count,      // Number of indices to process
    atomic<int>& nextIdx
)
{
    // Local cache
    vector<tuple<int,int,int>> local_cache;
    local_cache.reserve(CHUNK_SIZE);

    int n = (int)db.size();

    while(true) {
        int idx = nextIdx.fetch_add(1, memory_order_relaxed);
        if(idx >= total_count) {
            break; // No more tasks
        }

        // Get random index and decode to (i, j)
        long long x = randIndices[idx];
        int i, j;
        decodePair(x, n, i, j);

        // Allow i == j, generate GED=0 training data, very important for model learning
        Graph* db_graph    = db[i];
        Graph* query_graph = db[j];

        // GED calculation. Default exact (AppForComputation); GED_GEN_APP switches to fast
        // UB-exit App(iter) with upper-bound labels (SYN grid only, where exact is intractable).
        Application app(max_ged, "BMao", (g_app_iter > 0 ? g_app_iter : 2300), g_ged_gen_iter);
        app.init(db_graph, query_graph);
        int ged_res = g_app_iter > 0 ? (int)app.App(nullptr, nullptr)
                                     : (int)app.AppForComputation(nullptr, nullptr);

        // Skip if not exact GED (reached computation limit)
        if(ged_res == -1) {
            // Skip this sample - not exact GED
            tasksDone.fetch_add(1, memory_order_relaxed);
            continue;
        }

        // If qualified, store in local_cache
        if(ged_res <= max_ged) {
            local_cache.emplace_back(i, j, ged_res);
        }

        // Completed tasks +1
        tasksDone.fetch_add(1, memory_order_relaxed);

        // If accumulated CHUNK_SIZE, write to file
        if((int)local_cache.size() >= CHUNK_SIZE) {
            lock_guard<mutex> lock(g_write_mutex);
            ofstream ofs(output_filename, ios::app);
            if(!ofs) {
                cerr << "Error opening file for write: " << output_filename << endl;
            } else {
                for(const auto &tp : local_cache) {
                    int gg1, gg2, dist;
                    tie(gg1, gg2, dist) = tp;
                    ofs << gg1 << "," << gg2 << "," << dist << "\n";
                }
            }
            local_cache.clear();
        }
    }

    // Cleanup: if there are unwritten items, write once more
    if(!local_cache.empty()) {
        lock_guard<mutex> lock(g_write_mutex);
        ofstream ofs(output_filename, ios::app);
        if(!ofs) {
            cerr << "Error opening file for write: " << output_filename << endl;
        } else {
            for(const auto &tp : local_cache) {
                int g1, g2, dist;
                tie(g1, g2, dist) = tp;
                ofs << g1 << "," << g2 << "," << dist << "\n";
            }
        }
    }
}

// Process single parameter set with pre-loaded database
int process_single_param(const vector<Graph*>& db, const BatchParam& param, const char* output_filename) {
    int n = (int)db.size();
    if(n < 2) {
        cerr << "Need at least 2 graphs to form pairs.\n";
        return 1;
    }

    long long N = param.N;
    int max_ged = param.max_ged;
    int num_threads = param.num_threads;

    cout << "  Processing: N=" << N << ", max_ged=" << max_ged << ", threads=" << num_threads << " - " << param.description << endl;

    // Generate random indices
    long long total_possible = (long long)n * n;
    if(N > total_possible) {
        N = total_possible;
    }

    long long threshold = total_possible / 100;
    if(threshold < 1) {
        threshold = 1;
    }

    vector<long long> randIndices;
    randIndices.reserve((size_t)N);

    if(N < threshold) {
        // Method A: one-by-one random selection (not considering duplicates)
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<long long> dis(0, total_possible - 1);

        // No deduplication here: same x may appear multiple times
        while((long long)randIndices.size() < N) {
            long long x = dis(gen);
            randIndices.push_back(x);
        }
    } else {
        // Method B: shuffle then take first N
        randIndices.resize((size_t)total_possible);
        for(long long i = 0; i < total_possible; i++) {
            randIndices[(size_t)i] = i;
        }
        {
            random_device rd;
            mt19937 gen(rd());
            shuffle(randIndices.begin(), randIndices.end(), gen);
        }
        randIndices.resize((size_t)N);
    }

    // Multi-threaded processing
    tasksDone.store(0, memory_order_relaxed);
    auto t0 = high_resolution_clock::now();

    atomic<int> nextIdx(0);
    vector<thread> workers;
    workers.reserve(num_threads);

    for(int t = 0; t < num_threads; ++t) {
        workers.emplace_back(
            compute_ged_worker,
            cref(db),
            cref(randIndices),
            max_ged,
            output_filename,
            (int)N,
            ref(nextIdx)
        );
    }

    for(auto &th : workers) {
        th.join();
    }

    auto t1 = high_resolution_clock::now();
    auto dur = duration_cast<milliseconds>(t1 - t0);
    
    long long done = tasksDone.load(memory_order_relaxed);
    cout << "    Processed " << done << "/" << N << " pairs, Time: " << dur.count() << " ms" << endl;

    return 0;
}

// Parse batch configuration file
vector<BatchParam> parse_batch_config(const char* config_file) {
    vector<BatchParam> params;
    ifstream ifs(config_file);
    if (!ifs) {
        cerr << "Error opening config file: " << config_file << endl;
        return params;
    }

    string line;
    while (getline(ifs, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse CSV: N,max_ged,threads,description
        stringstream ss(line);
        string item;
        vector<string> tokens;
        
        while (getline(ss, item, ',')) {
            tokens.push_back(item);
        }

        if (tokens.size() >= 3) {
            BatchParam param;
            param.N = stoll(tokens[0]);
            param.max_ged = stoi(tokens[1]);
            param.num_threads = stoi(tokens[2]);
            param.description = tokens.size() > 3 ? tokens[3] : "";
            params.push_back(param);
        }
    }

    return params;
}

// Batch processing mode
int batch_mode(const char* config_file, const char* db_file, const char* output_file) {
    cout << "=== GREED Batch Training Data Generation ===" << endl;
    cout << "Config: " << config_file << endl;
    cout << "Database: " << db_file << endl;
    cout << "Output: " << output_file << endl;
    cout << endl;
    
    // Initialize logging
    string logFile = generateLogFileName(string(output_file));
    
    // Clear log file at start
    {
        ofstream log(logFile, ios::trunc);
    }
    
    writeLog(logFile, "=== GREED Batch Training Data Generation Started ===");
    writeLog(logFile, "Config: " + string(config_file));
    writeLog(logFile, "Database: " + string(db_file));
    writeLog(logFile, "Output: " + string(output_file));
    writeLog(logFile, "Log: " + logFile);

    // Parse configuration
    vector<BatchParam> params = parse_batch_config(config_file);
    if (params.empty()) {
        string error = "No valid parameters found in config file.";
        cerr << error << endl;
        writeLog(logFile, "ERROR: " + error);
        return 1;
    }

    cout << "Found " << params.size() << " parameter sets:" << endl;
    writeLog(logFile, "Found " + to_string(params.size()) + " parameter sets:");
    for (size_t i = 0; i < params.size(); ++i) {
        string paramInfo = "[" + to_string(i+1) + "] N=" + to_string(params[i].N) +
                          ", max_ged=" + to_string(params[i].max_ged) +
                          ", threads=" + to_string(params[i].num_threads) +
                          " - " + params[i].description;
        cout << "  " << paramInfo << endl;
        writeLog(logFile, "  " + paramInfo);
    }
    cout << endl;

    // Load database ONCE
    vector<Graph*> db;
    map<string, ui> vM, eM;
    cout << "Loading database from " << db_file << "..." << endl;
    writeLog(logFile, "Loading database from " + string(db_file));
    
    auto load_start = high_resolution_clock::now();
    ui max_db_n = Utility::load_db(db_file, db, vM, eM);
    auto load_end = high_resolution_clock::now();
    auto load_dur = duration_cast<milliseconds>(load_end - load_start);
    
    cout << "Loaded " << db.size() << " graphs." << endl;
    cout << endl;
    
    string loadInfo = "Loaded " + to_string(db.size()) + " graphs in " + 
                     to_string(load_dur.count()) + " ms";
    writeLog(logFile, loadInfo);
    writeLog(logFile, "Max vertex labels: " + to_string(vM.size()) + 
                     ", Max edge labels: " + to_string(eM.size()));

    // Clear output file
    {
        ofstream ofs(output_file, ios::trunc);
        if (!ofs) {
            cerr << "Error creating output file: " << output_file << endl;
            return 1;
        }
    }

    // Create mixed task pool for better load balancing
    auto batch_start = high_resolution_clock::now();
    
    cout << "Creating mixed task pool from " << params.size() << " parameter sets..." << endl;
    
    // Create streaming task generator (avoids memory overflow)
    StreamingTaskGenerator taskGenerator(params, db.size());
    long long total_tasks = taskGenerator.getTotalTasks();
    
    cout << "Created " << total_tasks << " mixed tasks, starting processing" << endl;
    writeLog(logFile, "Created " + to_string(total_tasks) + " mixed tasks, starting processing");
    
    // Process mixed tasks with optimal thread count
    int optimal_threads = params.empty() ? 4 : params[0].num_threads;
    for (const auto& param : params) {
        if (param.num_threads > optimal_threads) {
            optimal_threads = param.num_threads;
        }
    }
    
    writeLog(logFile, "Using " + to_string(optimal_threads) + " threads for optimal performance");
    
    tasksDone.store(0, memory_order_relaxed);
    atomic<long long> nextIdx(0);
    vector<thread> workers;
    workers.reserve(optimal_threads);
    
    // Progress tracking thread with time estimation
    bool keepRunning = true;
    auto progress_start_time = high_resolution_clock::now();
    
    thread progressThread([&](){
        while(keepRunning) {
            this_thread::sleep_for(chrono::seconds(2));
            long long done = tasksDone.load(memory_order_relaxed);
            long long total = total_tasks;
            double progress = total > 0 ? (double)done * 100.0 / (double)total : 0.0;
            
            // Calculate elapsed time
            auto current_time = high_resolution_clock::now();
            auto elapsed = duration_cast<chrono::seconds>(current_time - progress_start_time);
            
            // Format elapsed time
            int elapsed_seconds = (int)elapsed.count();
            int elapsed_hours = elapsed_seconds / 3600;
            int elapsed_minutes = (elapsed_seconds % 3600) / 60;
            int elapsed_secs = elapsed_seconds % 60;
            
            string elapsed_str;
            if (elapsed_hours > 0) {
                elapsed_str = to_string(elapsed_hours) + "h" + 
                             to_string(elapsed_minutes) + "m" + 
                             to_string(elapsed_secs) + "s";
            } else if (elapsed_minutes > 0) {
                elapsed_str = to_string(elapsed_minutes) + "m" + to_string(elapsed_secs) + "s";
            } else {
                elapsed_str = to_string(elapsed_secs) + "s";
            }
            
            // Calculate ETA (Estimated Time of Arrival)
            string eta_str = "N/A";
            if (done > 0 && progress > 0) {
                double rate = (double)done / elapsed_seconds;
                if (rate > 0) {
                    long long remaining = total - done;
                    int eta_seconds = (int)(remaining / rate);
                    int eta_hours = eta_seconds / 3600;
                    int eta_minutes = (eta_seconds % 3600) / 60;
                    int eta_secs = eta_seconds % 60;
                    
                    if (eta_hours > 0) {
                        eta_str = to_string(eta_hours) + "h" + 
                                 to_string(eta_minutes) + "m" + 
                                 to_string(eta_secs) + "s";
                    } else if (eta_minutes > 0) {
                        eta_str = to_string(eta_minutes) + "m" + to_string(eta_secs) + "s";
                    } else {
                        eta_str = to_string(eta_secs) + "s";
                    }
                }
            }
            
            // Enhanced progress bar with time information
            int barWidth = 40; // Slightly smaller to fit more info
            int pos = (int)(progress * barWidth / 100.0);
            cout << "\rProgress: [";
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) cout << "=";
                else if (i == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << (int)progress << "% (" << done << "/" << total << ") "
                 << "Elapsed: " << elapsed_str << " ETA: " << eta_str << flush;
        }
    });
    
    for(int t = 0; t < optimal_threads; ++t) {
        workers.emplace_back(
            compute_streaming_ged_worker,
            cref(db),
            cref(taskGenerator),
            output_file,
            total_tasks,
            ref(nextIdx)
        );
    }
    
    for(auto &th : workers) {
        th.join();
    }
    
    // Stop progress thread and show final result
    keepRunning = false;
    progressThread.join();
    
    long long final_done = tasksDone.load(memory_order_relaxed);
    long long final_valid = totalValidPairs.load(memory_order_relaxed);
    long long final_processed = totalProcessedPairs.load(memory_order_relaxed);
    
    // Calculate final elapsed time
    auto final_time = high_resolution_clock::now();
    auto final_elapsed = duration_cast<chrono::seconds>(final_time - progress_start_time);
    int final_elapsed_seconds = (int)final_elapsed.count();
    int final_hours = final_elapsed_seconds / 3600;
    int final_minutes = (final_elapsed_seconds % 3600) / 60;
    int final_secs = final_elapsed_seconds % 60;
    
    string final_elapsed_str;
    if (final_hours > 0) {
        final_elapsed_str = to_string(final_hours) + "h" + 
                           to_string(final_minutes) + "m" + 
                           to_string(final_secs) + "s";
    } else if (final_minutes > 0) {
        final_elapsed_str = to_string(final_minutes) + "m" + to_string(final_secs) + "s";
    } else {
        final_elapsed_str = to_string(final_secs) + "s";
    }
    
    cout << "\rProgress: [";
    for (int i = 0; i < 40; ++i) cout << "=";
    cout << "] 100% (" << final_done << "/" << total_tasks << ") "
         << "Total: " << final_elapsed_str << " - Complete!" << endl;

    auto batch_end = high_resolution_clock::now();
    auto total_dur = duration_cast<milliseconds>(batch_end - batch_start);

    cout << endl;
    cout << "=== Batch Processing Completed ===" << endl;
    cout << "Total time: " << total_dur.count() << " ms" << endl;
    cout << "Results saved to: " << output_file << endl;

    // Detailed statistics logging
    double efficiency = final_processed > 0 ? (double)final_valid * 100.0 / (double)final_processed : 0.0;
    double throughput = total_dur.count() > 0 ? (double)final_processed * 1000.0 / (double)total_dur.count() : 0.0;
    
    writeLog(logFile, "=== Batch Processing Completed ===");
    writeLog(logFile, "Total execution time: " + to_string(total_dur.count()) + " ms");
    writeLog(logFile, "Total tasks processed: " + to_string(final_done) + "/" + to_string(total_tasks));
    writeLog(logFile, "Total graph pairs computed: " + to_string(final_processed));
    writeLog(logFile, "Valid training pairs generated: " + to_string(final_valid));
    writeLog(logFile, "Filter efficiency: " + to_string(efficiency) + "% (valid/computed)");
    writeLog(logFile, "Processing throughput: " + to_string(throughput) + " pairs/second");
    writeLog(logFile, "Results saved to: " + string(output_file));
    writeLog(logFile, "=== Session completed successfully ===");
    
    cout << "Log file saved to: " << logFile << endl;

    // Cleanup
    for(auto* g : db) {
        delete g;
    }

    return 0;
}

// Single processing mode (original)
int single_mode(const char* db_name, const char* output_filename, long long N, int max_ged, int num_threads) {
    // Original single processing logic
    vector<Graph*> db;
    map<string, ui> vM, eM;
    cout << "Loading DB from " << db_name << "...\n";
    ui max_db_n = Utility::load_db(db_name, db, vM, eM);
    cout << "max_vlabel: " << vM.size() << " max_elabel: " << eM.size() << endl;
    cout << "Loaded " << db.size() << " graphs.\n";

    int n = (int)db.size();
    if(n < 2) {
        cerr << "Need at least 2 graphs to form i != j.\n";
        return 1;
    }

    long long total_possible = (long long)n * n;
    if(N > total_possible) {
        N = total_possible;
    }

    long long threshold = total_possible / 100;
    if(threshold < 1) {
        threshold = 1;
    }

    vector<long long> randIndices; 
    randIndices.reserve((size_t)N);

    if(N < threshold) {
        cout << "[INFO] Using one-by-one random approach (N < total_possible/100).\n";
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<long long> dis(0, total_possible - 1);

        while((long long)randIndices.size() < N) {
            long long x = dis(gen);
            randIndices.push_back(x);
        }

    } else {
        cout << "[INFO] Using shuffle approach (N >= total_possible/100).\n";
        randIndices.resize((size_t)total_possible);
        for(long long i = 0; i < total_possible; i++) {
            randIndices[(size_t)i] = i;
        }
        {
            random_device rd;
            mt19937 gen(rd());
            shuffle(randIndices.begin(), randIndices.end(), gen);
        }
        randIndices.resize((size_t)N);
    }

    // Open output file: append mode
    {
        ofstream ofs(output_filename, ios::app);
        if(!ofs) {
            cerr << "Error opening output file: " << output_filename << endl;
            return 1;
        }
    }

    // Multi-threaded processing
    tasksDone.store(0, memory_order_relaxed);
    bool keepRunning = true;
    thread progressThread([&](){
        while(keepRunning) {
            this_thread::sleep_for(1s);
            long long done = tasksDone.load(memory_order_relaxed);
            double ratio = (N > 0) ? (double)done * 100.0 / (double)N : 0.0;
            cerr << "\rProcessed " << done << "/" << N
                 << " pairs (" << (int)ratio << "%)..." << flush;
        }
    });

    auto t0 = high_resolution_clock::now();

    atomic<int> nextIdx(0);
    vector<thread> workers;
    workers.reserve(num_threads);

    for(int t = 0; t < num_threads; ++t) {
        workers.emplace_back(
            compute_ged_worker,
            cref(db),
            cref(randIndices),
            max_ged,
            output_filename,
            (int)N,
            ref(nextIdx)
        );
    }

    for(auto &th : workers) {
        th.join();
    }

    keepRunning = false;
    progressThread.join();

    {
        long long done = tasksDone.load(memory_order_relaxed);
        cerr << "\rProcessed " << done << "/" << N << " pairs (100%)\n";
    }

    auto t1 = high_resolution_clock::now();
    auto dur = duration_cast<milliseconds>(t1 - t0);
    cout << "Results appended to " << output_filename << endl;
    cout << "Total time: " << dur.count() << " ms\n";

    // Cleanup
    for(auto*g : db) {
        delete g;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    // Check for batch mode
    if (argc >= 2 && string(argv[1]) == "--batch") {
        // Batch mode: ./program --batch config.txt db.txt output.txt
        if (argc < 5) {
            cerr << "Batch mode usage: " << argv[0] 
                 << " --batch <config_file> <db_file> <output_file>\n";
            return 1;
        }
        return batch_mode(argv[2], argv[3], argv[4]);
    }

    // Original single mode
    if(argc < 6) {
        cerr << "Single mode usage: " << argv[0]
             << " <db_file> <output_file> <N> <max_ged> <num_threads>\n";
        cerr << "Batch mode usage: " << argv[0]
             << " --batch <config_file> <db_file> <output_file>\n";
        return 1;
    }

    const char* db_name         = argv[1];
    const char* output_filename = argv[2];
    long long N  = stoll(argv[3]);
    int max_ged  = stoi(argv[4]);
    int num_threads = stoi(argv[5]);

    return single_mode(db_name, output_filename, N, max_ged, num_threads);
}