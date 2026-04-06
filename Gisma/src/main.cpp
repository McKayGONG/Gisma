#include "GismaIndexBuilder.h"
#include "GismaSearchEngine.h"
#include "Graph.h"
#include "Node.h"
#include "Anchor.h"
#include "Utility.h"
#include "Application.h"
#include "EditPathTree.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <cstring>  // for argument parsing
#include <chrono>
#include <cstdlib>  // for exit() function
#include <thread>
#include <popl.hpp>
#include "Config.h"
#include <unordered_set>
#include <atomic>
#include <future>
#include <mutex>
#include <iomanip>
#include <random>
#include <algorithm>
#include <sstream>
#include <set>
#include <numeric>


namespace fs = std::filesystem;
// Define pair_hash for unordered_map with pair<int, int> key
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};





// Function declarations
void print_epf_info(const std::string& index_name, int total_anchors = -1);
void select_alpha_mode(Config& config);

// Compute GED between two graphs
void compute_ged_mode(Config& config) {
    std::cout << "Compute GED Mode" << std::endl;

    std::string query_file = config.query_file;
    std::string target_file = config.target_file;
    std::string ged_algorithm = config.ged_algorithm;

    std::cout << "Query file: " << query_file << std::endl;
    std::cout << "Target file: " << target_file << std::endl;
    std::cout << "GED algorithm: " << ged_algorithm << std::endl;

    // Load graphs
    std::map<std::string, ui> vM, eM;

    // Load query graph
    std::vector<Graph*> query_graphs;
    Utility::load_db(query_file.c_str(), query_graphs, vM, eM);
    if (query_graphs.empty()) {
        std::cerr << "Error: Failed to load query graph from " << query_file << std::endl;
        return;
    }
    Graph* query_graph = query_graphs[0];
    std::cout << "Loaded query graph: " << query_graph->id
              << " (nodes=" << query_graph->n << ", edges=" << query_graph->m << ")" << std::endl;

    // Load target graph
    std::vector<Graph*> target_graphs;
    Utility::load_db(target_file.c_str(), target_graphs, vM, eM);
    if (target_graphs.empty()) {
        std::cerr << "Error: Failed to load target graph from " << target_file << std::endl;
        for (auto g : query_graphs) delete g;
        return;
    }
    Graph* target_graph = target_graphs[0];
    std::cout << "Loaded target graph: " << target_graph->id
              << " (nodes=" << target_graph->n << ", edges=" << target_graph->m << ")" << std::endl;

    // Set feature dimension
    Graph::FEATURE_DIM = vM.size();

    // Compute GED
    double ged_result = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use tau_search as upper_bound if provided (> 0), otherwise use INF for exact computation
    ui upper_bound = (config.tau_search > 0) ? static_cast<ui>(config.tau_search) : INF;
    if (upper_bound != INF) {
        std::cout << "Using upper_bound (tau): " << upper_bound << std::endl;
    }

    if (ged_algorithm == "AppForComputation") {
        // Use AppForComputation (exact)
        std::cout << "Using AppForComputation..." << std::endl;
        Application* app = new Application(upper_bound, "BMao");
        app->init(query_graph, target_graph);
        ged_result = app->AppForComputation(nullptr, nullptr);
        delete app;
    } else if (ged_algorithm == "App") {
        // Use App with app_max_iter (approximate)
        std::cout << "Using App (app_max_iter=" << config.app_max_iter << ")..." << std::endl;
        Application* app = new Application(upper_bound, "BMao", config.app_max_iter);
        app->init(query_graph, target_graph);
        ged_result = app->App(nullptr, nullptr);
        delete app;
    } else {
        // Default: use Application::compute_ged
        std::cout << "Using default compute_ged (A* based)..." << std::endl;
        ged_result = Application::compute_ged(query_graph, target_graph, upper_bound);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Output result
    std::cout << "\n=== GED Computation Result ===" << std::endl;
    std::cout << "Query Graph: " << query_graph->id << std::endl;
    std::cout << "Target Graph: " << target_graph->id << std::endl;
    std::cout << "GED: " << ged_result << std::endl;
    std::cout << "Computation Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Algorithm: " << ged_algorithm << std::endl;

    // Clean up
    for (auto g : query_graphs) delete g;
    for (auto g : target_graphs) delete g;
}

// Batch GED mode for GHash integration
// Input: query_id, candidate_ids_file (one ID per line), tau_search
// Output: GED values (one per line), then CPU time in seconds
void batch_ged_mode(Config& config) {
    // Load all graphs from dataset
    std::map<std::string, ui> vM, eM;

    // Load database graphs
    std::vector<Graph*> db_graphs;
    Utility::load_db(config.db_name.c_str(), db_graphs, vM, eM);
    if (db_graphs.empty()) {
        std::cerr << "Error: Failed to load database from " << config.db_name << std::endl;
        return;
    }

    // Load query graphs
    std::vector<Graph*> query_graphs;
    Utility::load_db(config.query_name.c_str(), query_graphs, vM, eM);
    if (query_graphs.empty()) {
        std::cerr << "Error: Failed to load queries from " << config.query_name << std::endl;
        for (auto g : db_graphs) delete g;
        return;
    }

    // Set feature dimension
    Graph::FEATURE_DIM = vM.size();

    // Build ID to graph index map for db (ID is string, convert to int)
    std::map<int, int> db_id_to_idx;
    for (size_t i = 0; i < db_graphs.size(); i++) {
        int graph_id = std::stoi(db_graphs[i]->id);
        db_id_to_idx[graph_id] = i;
    }

    // Find query graph by ID
    Graph* query_graph = nullptr;
    std::string query_id_str = std::to_string(config.query_id);
    for (auto g : query_graphs) {
        if (g->id == query_id_str) {
            query_graph = g;
            break;
        }
    }
    if (query_graph == nullptr) {
        std::cerr << "Error: Query graph ID " << config.query_id << " not found" << std::endl;
        for (auto g : db_graphs) delete g;
        for (auto g : query_graphs) delete g;
        return;
    }

    // Read candidate IDs from file
    std::vector<int> candidate_ids;
    std::ifstream cand_file(config.candidate_ids_file);
    if (!cand_file.is_open()) {
        std::cerr << "Error: Cannot open candidate IDs file: " << config.candidate_ids_file << std::endl;
        for (auto g : db_graphs) delete g;
        for (auto g : query_graphs) delete g;
        return;
    }

    int cand_id;
    while (cand_file >> cand_id) {
        candidate_ids.push_back(cand_id);
    }
    cand_file.close();

    // Upper bound for GED computation
    ui upper_bound = (config.tau_search > 0) ? static_cast<ui>(config.tau_search) + 1 : INF;

    // Compute GED for each candidate
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int cid : candidate_ids) {
        auto it = db_id_to_idx.find(cid);
        if (it == db_id_to_idx.end()) {
            // Candidate not found, output upper_bound
            std::cout << upper_bound << std::endl;
            continue;
        }

        Graph* target_graph = db_graphs[it->second];

        // Use AppForComputation
        Application* app = new Application(upper_bound, "BMao");
        app->init(query_graph, target_graph);
        double ged = app->AppForComputation(nullptr, nullptr);
        delete app;

        // Output GED value
        std::cout << static_cast<int>(ged) << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double cpu_time = duration.count() / 1000000.0;  // Convert to seconds

    // Output CPU time as last line
    std::cout << cpu_time << std::endl;

    // Clean up
    for (auto g : db_graphs) delete g;
    for (auto g : query_graphs) delete g;
}

// NetDag-only construction mode
void construct_ND_mode(Config& config) {
    std::cout << "Construct NetDag (ND) Mode" << std::endl;

    double alpha = config.alpha;
    double tau_index = config.tau_index;
    double error_tolerance_index = config.error_tolerance_index;
    std::string db_name = config.db_name;
    bool has_ged_matrix = config.has_ged_matrix;
    double max_exact_ged_for_EPT = config.max_exact_ged_for_EPT;

    // If using default value -1, set to alpha + 4
    if (max_exact_ged_for_EPT < 0) {
        max_exact_ged_for_EPT = alpha + 4.0;
        std::cout << "[INFO] Setting max_exact_ged_for_EPT to alpha + 4 = " << max_exact_ged_for_EPT << std::endl;
    }

    std::cout << "alpha: " << alpha << ", Tau Index: " << tau_index
              << ", Error Tolerance Index: " << error_tolerance_index
              << ", DB Name: " << db_name << ", Has GED Matrix: " << has_ged_matrix << std::endl;

    std::string database = db_name;
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(database.c_str(), db, vM, eM);
    std::cout << "Loaded " << db.size() << " graphs with max node count: " << max_db_n << std::endl;
    Graph::FEATURE_DIM = vM.size();

    std::vector<std::shared_ptr<Graph>> db_graphs;
    std::vector<std::string> nameList;
    for (const auto& graph_ptr : db) {
        db_graphs.emplace_back(std::shared_ptr<Graph>(graph_ptr));
        nameList.push_back(graph_ptr->id);
    }

    std::vector<double> ged_matrix;
    int N = 0;

    // Decide whether to load GED matrix based on has_ged_matrix
    if (has_ged_matrix) {
        std::string binary_file = "../GED_matrix/ensembled_ged_matrix_good_upper.bin";
        std::ifstream infile(binary_file, std::ios::binary);

        if (!infile) {
            std::cerr << "Cannot open file " << binary_file << std::endl;
            return;
        }

        infile.read(reinterpret_cast<char*>(&N), sizeof(int));
        if (!infile) {
            std::cerr << "Failed to read matrix size." << std::endl;
            return;
        }

        size_t num_elements = static_cast<size_t>(N) * (N + 1) / 2;
        ged_matrix.resize(num_elements);
        infile.read(reinterpret_cast<char*>(ged_matrix.data()), num_elements * sizeof(double));
        infile.close();

        std::cout << "GED matrix loaded successfully. Matrix size: " << num_elements << std::endl;
    }

    // Load embedding vectors for NetDag construction
    std::vector<std::vector<float>> embeddings;
    std::cout << "[construct_ND] Loading embedding vectors..." << std::endl;
    std::string embedding_file = "./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin";
    if (Utility::load_embeddings(embedding_file, embeddings)) {
        std::cout << "[construct_ND] Embedding vectors loaded successfully from " << embedding_file << std::endl;
        std::cout << "[construct_ND] Loaded " << embeddings.size() << " embeddings for NetDag construction" << std::endl;
    } else {
        std::cout << "[construct_ND] Warning: Failed to load embeddings from " << embedding_file << std::endl;
        std::cout << "[construct_ND] NetDag construction will proceed without ML assistance" << std::endl;
    }

    // Use Utility to get index name
    std::string index_name = Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db_graphs.size());

    std::shared_ptr<NetDag> netdag_ptr = nullptr;
    std::string dataset_dir = "./NetDags/" + config.dataset + "/";
    std::string configs_dir = dataset_dir + "configs/";
    std::string reassigned_dir = dataset_dir + "reassigned/";

    std::filesystem::create_directories(configs_dir);
    std::filesystem::create_directories(reassigned_dir);

    std::string file_path = configs_dir + index_name + ".dat";
    std::string reassigned_file_path = reassigned_dir + index_name + ".dat";

    std::cout << "NetDag file path will be: " << file_path << std::endl;

    // Create GismaIndexBuilder object
    GismaIndexBuilder netDagConst(
        index_name, db_graphs, nameList, config.root_ind, alpha, tau_index, error_tolerance_index,
        netdag_ptr, false, has_ged_matrix, ged_matrix, N, max_exact_ged_for_EPT,
        db_name, vM, eM, max_db_n, embeddings, config.enable_friends_reassign
    );
    netDagConst.dataset = config.dataset;  // fix: dataset name may contain underscore (e.g. SYN_10K)

    // Force rebuild NetDag regardless of whether it already exists
    std::cout << "[construct_ND] Building new NetDag (will overwrite existing)..." << std::endl;
    netDagConst.overall_NetDag_const();
    std::cout << "[construct_ND] NetDag construction completed." << std::endl;

    // Print NetDag information
    if (netDagConst.net_dag) {
        std::cout << "\n====== Constructed NetDag Information ======\n";
        netDagConst.net_dag->print_netdag_info();
        std::cout << "============================================\n\n";
    }

    // Save to configs/ directory
    std::string save_path = configs_dir + index_name + ".dat";
    std::cout << "[construct_ND] Saving NetDag to: " << save_path << std::endl;
    netDagConst.net_dag->save_to_file(save_path);
    std::cout << "[construct_ND] NetDag saved successfully!" << std::endl;

    std::cout << "[construct_ND] NetDag construction completed!" << std::endl;
    std::cout << "[construct_ND] Next step: run 'compute_paths' to compute exact GED and node matching." << std::endl;
}

// Mode for computing exact GED and node matching
void compute_paths_mode(Config& config) {
    std::cout << "Compute Paths Mode (GED + Node Matching)" << std::endl;

    double alpha = config.alpha;
    double tau_index = config.tau_index;
    double error_tolerance_index = config.error_tolerance_index;
    std::string db_name = config.db_name;
    bool has_ged_matrix = config.has_ged_matrix;
    double max_exact_ged_for_EPT = config.max_exact_ged_for_EPT;

    // If using default value -1, set to alpha + 4
    if (max_exact_ged_for_EPT < 0) {
        max_exact_ged_for_EPT = alpha + 4.0;
        std::cout << "[INFO] Setting max_exact_ged_for_EPT to alpha + 4 = " << max_exact_ged_for_EPT << std::endl;
    }

    std::cout << "alpha: " << alpha << ", Tau Index: " << tau_index
              << ", Error Tolerance Index: " << error_tolerance_index
              << ", DB Name: " << db_name << std::endl;

    // Load data
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(db_name.c_str(), db, vM, eM);
    std::cout << "Loaded " << db.size() << " graphs with max node count: " << max_db_n << std::endl;
    Graph::FEATURE_DIM = vM.size();

    std::vector<std::shared_ptr<Graph>> db_graphs;
    std::vector<std::string> nameList;
    for (const auto& graph_ptr : db) {
        db_graphs.emplace_back(std::shared_ptr<Graph>(graph_ptr));
        nameList.push_back(graph_ptr->id);
    }

    std::vector<double> ged_matrix;
    int N = 0;

    // Decide whether to load GED matrix based on has_ged_matrix
    if (has_ged_matrix) {
        std::string binary_file = "../GED_matrix/ensembled_ged_matrix_good_upper.bin";
        std::ifstream infile(binary_file, std::ios::binary);
        if (infile) {
            infile.read(reinterpret_cast<char*>(&N), sizeof(int));
            size_t num_elements = static_cast<size_t>(N) * (N + 1) / 2;
            ged_matrix.resize(num_elements);
            infile.read(reinterpret_cast<char*>(ged_matrix.data()), num_elements * sizeof(double));
            infile.close();
            std::cout << "GED matrix loaded successfully." << std::endl;
        }
    }

    // compute_paths does not need embeddings, pass empty vector
    std::vector<std::vector<float>> embeddings;

    // Get index name and path
    std::string index_name = Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db_graphs.size());
    std::string dataset_dir = "./NetDags/" + config.dataset + "/";
    std::string configs_dir = dataset_dir + "configs/";
    std::string reassigned_dir = dataset_dir + "reassigned/";
    std::string ged_results_dir = dataset_dir + "ged_results/";

    std::filesystem::create_directories(reassigned_dir);
    std::filesystem::create_directories(ged_results_dir);

    std::string configs_file_path = configs_dir + index_name + ".dat";

    // Check if NetDag exists
    if (!fs::exists(configs_file_path)) {
        std::cerr << "[ERROR] NetDag not found at: " << configs_file_path << std::endl;
        std::cerr << "Please run 'construct_ND' mode first to build the NetDag." << std::endl;
        return;
    }

    // Create GismaIndexBuilder object
    std::shared_ptr<NetDag> netdag_ptr = nullptr;
    GismaIndexBuilder netDagConst(
        index_name, db_graphs, nameList, config.root_ind, alpha, tau_index, error_tolerance_index,
        netdag_ptr, false, has_ged_matrix, ged_matrix, N, max_exact_ged_for_EPT,
        db_name, vM, eM, max_db_n, embeddings, config.enable_friends_reassign
    );
    netDagConst.dataset = config.dataset;  // fix: dataset name may contain underscore (e.g. SYN_10K)

    // Load NetDag
    std::cout << "[compute_paths] Loading NetDag from: " << configs_file_path << std::endl;
    netdag_ptr = std::make_shared<NetDag>();
    netdag_ptr->load_from_file(*netdag_ptr, configs_file_path);

    std::cout << "\n====== Loaded NetDag Information ======\n";
    netdag_ptr->print_netdag_info();
    std::cout << "======================================\n\n";

    netDagConst.setNetDag(netdag_ptr);
    netDagConst.setAnchors(netdag_ptr->anchors);
    netDagConst.setNodes(netdag_ptr->nodes);

    std::cout << "Number of anchors: " << netDagConst.get_anchors().size() << std::endl;
    std::cout << "Number of nodes: " << netDagConst.get_nodes().size() << std::endl;

    // Compute exact GED and node matching, save to csv (without modifying NetDag)
    std::cout << "[compute_paths] Computing exact GED and node matching..." << std::endl;
    netDagConst.compute_ged_to_csv();  // Only compute GED, save to csv, without modifying NetDag
    std::cout << "[compute_paths] GED computation completed." << std::endl;

    std::cout << "[compute_paths] Results saved to: " << ged_results_dir << std::endl;
    std::cout << "[compute_paths] Paths computation completed successfully!" << std::endl;
    std::cout << "[compute_paths] Next step: run 'reassign' to reassign nodes to clusters." << std::endl;
}

// Reassign mode: load NetDag from configs/, reassign using csv, save to reassigned/
void reassign_mode(Config& config) {
    std::cout << "Reassign Mode" << std::endl;

    double alpha = config.alpha;
    double tau_index = config.tau_index;
    double error_tolerance_index = config.error_tolerance_index;
    std::string db_name = config.db_name;
    bool has_ged_matrix = config.has_ged_matrix;
    double max_exact_ged_for_EPT = config.max_exact_ged_for_EPT;

    // If using default value -1, set to alpha + 4
    if (max_exact_ged_for_EPT < 0) {
        max_exact_ged_for_EPT = alpha + 4.0;
        std::cout << "[INFO] Setting max_exact_ged_for_EPT to alpha + 4 = " << max_exact_ged_for_EPT << std::endl;
    }

    // Use Utility to get index name
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(db_name.c_str(), db, vM, eM);
    std::vector<std::shared_ptr<Graph>> db_graphs;
    for (const auto& graph_ptr : db) {
        db_graphs.emplace_back(std::shared_ptr<Graph>(graph_ptr));
    }

    std::string index_name = Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db_graphs.size());
    std::string dataset_dir = "./NetDags/" + config.dataset + "/";
    std::string configs_dir = dataset_dir + "configs/";
    std::string reassigned_dir = dataset_dir + "reassigned/";

    std::filesystem::create_directories(reassigned_dir);

    std::string configs_file_path = configs_dir + index_name + ".dat";
    std::string reassigned_file_path = reassigned_dir + index_name + ".dat";
    std::string ged_results_dir = dataset_dir + "ged_results/";
    std::string ged_csv_path = ged_results_dir + index_name + "_exact_ged_results.csv";

    // Check if configs NetDag exists
    if (!fs::exists(configs_file_path)) {
        std::cerr << "[ERROR] NetDag not found at: " << configs_file_path << std::endl;
        std::cerr << "Please run 'construct_ND' mode first to build the NetDag." << std::endl;
        return;
    }

    // Check if GED csv exists
    if (!fs::exists(ged_csv_path)) {
        std::cerr << "[ERROR] GED results not found at: " << ged_csv_path << std::endl;
        std::cerr << "Please run 'compute_paths' mode first to compute GED and node matching." << std::endl;
        return;
    }

    std::cout << "[reassign] NetDag and GED results found, proceeding..." << std::endl;

    // Load data and initialize
    std::cout << "Loaded " << db.size() << " graphs with max node count: " << max_db_n << std::endl;
    Graph::FEATURE_DIM = vM.size();

    std::vector<std::string> nameList;
    for (const auto& graph_ptr : db) {
        nameList.push_back(graph_ptr->id);
    }

    std::vector<double> ged_matrix;
    int N = 0;

    // Decide whether to load GED matrix based on has_ged_matrix
    if (has_ged_matrix) {
        std::string binary_file = "../GED_matrix/ensembled_ged_matrix_good_upper.bin";
        std::ifstream infile(binary_file, std::ios::binary);

        if (!infile) {
            std::cerr << "Cannot open file " << binary_file << std::endl;
            return;
        }

        infile.read(reinterpret_cast<char*>(&N), sizeof(int));
        if (!infile) {
            std::cerr << "Failed to read matrix size." << std::endl;
            return;
        }

        size_t num_elements = static_cast<size_t>(N) * (N + 1) / 2;
        ged_matrix.resize(num_elements);
        infile.read(reinterpret_cast<char*>(ged_matrix.data()), num_elements * sizeof(double));
        infile.close();

        std::cout << "GED matrix loaded successfully. Matrix size: " << num_elements << std::endl;
    }

    // Load embedding vectors
    std::vector<std::vector<float>> embeddings;
    std::cout << "[reassign] Loading embedding vectors..." << std::endl;
    std::string embedding_file = "./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin";
    if (Utility::load_embeddings(embedding_file, embeddings)) {
        std::cout << "[reassign] Embedding vectors loaded successfully from " << embedding_file << std::endl;
        std::cout << "[reassign] Loaded " << embeddings.size() << " embeddings" << std::endl;
    } else {
        std::cout << "[reassign] Warning: Failed to load embeddings from " << embedding_file << std::endl;
    }

    // Create GismaIndexBuilder object
    std::shared_ptr<NetDag> netdag_ptr = nullptr;
    GismaIndexBuilder netDagConst(
        index_name, db_graphs, nameList, config.root_ind, alpha, tau_index, error_tolerance_index,
        netdag_ptr, false, has_ged_matrix, ged_matrix, N, max_exact_ged_for_EPT,
        db_name, vM, eM, max_db_n, embeddings, config.enable_friends_reassign
    );
    netDagConst.dataset = config.dataset;  // fix: dataset name may contain underscore (e.g. SYN_10K)

    // Load configs NetDag
    std::cout << "[reassign] Loading NetDag from configs..." << std::endl;
    netdag_ptr = std::make_shared<NetDag>();
    netdag_ptr->load_from_file(*netdag_ptr, configs_file_path);
    std::cout << "[reassign] NetDag loaded." << std::endl;

    netDagConst.setNetDag(netdag_ptr);
    netDagConst.setAnchors(netdag_ptr->anchors);
    netDagConst.setNodes(netdag_ptr->nodes);

    // Read GED results from csv and reassign
    std::cout << "[reassign] Reassigning nodes based on GED results from csv..." << std::endl;
    netDagConst.reassign_nodes_in_cluster_with_csv();
    std::cout << "[reassign] Reassignment completed." << std::endl;

    // Save to reassigned directory
    std::cout << "[reassign] Saving reassigned NetDag to: " << reassigned_file_path << std::endl;
    netDagConst.net_dag->save_to_file(reassigned_file_path);

    // Print NetDag information
    std::cout << "\n====== Reassigned NetDag Information ======\n";
    netdag_ptr->print_netdag_info();
    std::cout << "==========================================\n\n";

    // Collect cluster statistics
    int num_nodes_in_cluster = 0;
    int num_nodes_in_exact_cluster = 0;
    for (auto anchor : netDagConst.get_anchors()) {
        num_nodes_in_cluster += anchor->nodes_in_cluster.size();
        num_nodes_in_exact_cluster += anchor->nodes_in_exact_cluster.size();
    }
    std::cout << "Number of nodes in cluster: " << num_nodes_in_cluster << std::endl;
    std::cout << "Number of nodes in exact cluster: " << num_nodes_in_exact_cluster << std::endl;
    std::cout << "Number of anchors: " << netDagConst.get_anchors().size() << std::endl;

    std::cout << "[reassign] Reassign mode completed successfully!" << std::endl;
    std::cout << "[reassign] Next step: run 'construct_EPF' to build EPF." << std::endl;
}

// EPF-only construction mode: load NetDag from reassigned/, build EPF
void construct_EPF_mode(Config& config) {
    std::cout << "Construct EPF Mode" << std::endl;

    double alpha = config.alpha;
    double tau_index = config.tau_index;
    double error_tolerance_index = config.error_tolerance_index;
    std::string db_name = config.db_name;
    bool has_ged_matrix = config.has_ged_matrix;
    double max_exact_ged_for_EPT = config.max_exact_ged_for_EPT;

    // If using default value -1, set to alpha + 4
    if (max_exact_ged_for_EPT < 0) {
        max_exact_ged_for_EPT = alpha + 4.0;
        std::cout << "[INFO] Setting max_exact_ged_for_EPT to alpha + 4 = " << max_exact_ged_for_EPT << std::endl;
    }

    // Use Utility to get index name
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(db_name.c_str(), db, vM, eM);
    std::vector<std::shared_ptr<Graph>> db_graphs;
    for (const auto& graph_ptr : db) {
        db_graphs.emplace_back(std::shared_ptr<Graph>(graph_ptr));
    }

    std::string index_name = Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db_graphs.size());
    std::string dataset_dir = "./NetDags/" + config.dataset + "/";
    std::string reassigned_dir = dataset_dir + "reassigned/";
    std::string reassigned_file_path = reassigned_dir + index_name + ".dat";

    // Check if reassigned NetDag exists
    if (!fs::exists(reassigned_file_path)) {
        std::cerr << "[ERROR] Reassigned NetDag not found at: " << reassigned_file_path << std::endl;
        std::cerr << "Please run 'reassign' mode first." << std::endl;
        return;
    }

    std::cout << "[construct_EPF] Reassigned NetDag found, proceeding..." << std::endl;

    // Load data and initialize
    std::cout << "Loaded " << db.size() << " graphs with max node count: " << max_db_n << std::endl;
    Graph::FEATURE_DIM = vM.size();

    std::vector<std::string> nameList;
    for (const auto& graph_ptr : db) {
        nameList.push_back(graph_ptr->id);
    }

    std::vector<double> ged_matrix;
    int N = 0;

    // Decide whether to load GED matrix based on has_ged_matrix
    if (has_ged_matrix) {
        std::string binary_file = "../GED_matrix/ensembled_ged_matrix_good_upper.bin";
        std::ifstream infile(binary_file, std::ios::binary);

        if (!infile) {
            std::cerr << "Cannot open file " << binary_file << std::endl;
            return;
        }

        infile.read(reinterpret_cast<char*>(&N), sizeof(int));
        if (!infile) {
            std::cerr << "Failed to read matrix size." << std::endl;
            return;
        }

        size_t num_elements = static_cast<size_t>(N) * (N + 1) / 2;
        ged_matrix.resize(num_elements);
        infile.read(reinterpret_cast<char*>(ged_matrix.data()), num_elements * sizeof(double));
        infile.close();

        std::cout << "GED matrix loaded successfully. Matrix size: " << num_elements << std::endl;
    }

    // Create GismaIndexBuilder object (EPF construction doesn't need embeddings)
    std::vector<std::vector<float>> embeddings;  // empty, not used in EPF
    std::shared_ptr<NetDag> netdag_ptr = nullptr;
    GismaIndexBuilder netDagConst(
        index_name, db_graphs, nameList, config.root_ind, alpha, tau_index, error_tolerance_index,
        netdag_ptr, false, has_ged_matrix, ged_matrix, N, max_exact_ged_for_EPT,
        db_name, vM, eM, max_db_n, embeddings, config.enable_friends_reassign
    );
    netDagConst.dataset = config.dataset;  // fix: dataset name may contain underscore (e.g. SYN_10K)

    // Load reassigned NetDag (minimal mode: only read anchor ID + exact_cluster, skip nodes/graph/children)
    std::cout << "[construct_EPF] Loading anchors from reassigned (flat mode)..." << std::endl;
    netdag_ptr = std::make_shared<NetDag>();
    netdag_ptr->load_anchors_only(*netdag_ptr, reassigned_file_path);
    std::cout << "[construct_EPF] Anchors loaded." << std::endl;

    // Set each anchor's graph from db (load_anchors_only skipped graph data)
    std::unordered_map<int, std::shared_ptr<Graph>> db_graph_map;
    int null_graphs = 0;
    for (size_t i = 0; i < db_graphs.size(); i++) {
        if (db_graphs[i]) {
            int gid = 0;
            try { gid = std::stoi(db_graphs[i]->id); } catch (...) { gid = (int)i; }
            db_graph_map[gid] = db_graphs[i];
        } else {
            null_graphs++;
        }
    }
    std::cout << "[construct_EPF] db_graph_map size: " << db_graph_map.size()
              << ", null graphs: " << null_graphs
              << ", db_graphs total: " << db_graphs.size() << std::endl;

    // Debug: show first 5 db graph IDs
    std::cout << "[construct_EPF] First 5 db graph IDs: ";
    for (size_t i = 0; i < std::min((size_t)5, db_graphs.size()); i++) {
        if (db_graphs[i]) std::cout << db_graphs[i]->id << " ";
    }
    std::cout << std::endl;

    int graph_set_count = 0;
    int missing_count = 0;
    for (auto& anchor : netdag_ptr->anchors) {
        auto it = db_graph_map.find(anchor->node_id);
        if (it != db_graph_map.end()) {
            anchor->graph = it->second;
            graph_set_count++;
        } else {
            if (missing_count < 10) {
                std::cout << "[construct_EPF] MISS: anchor_id=" << anchor->anchor_id
                          << " node_id=" << anchor->node_id << std::endl;
            }
            missing_count++;
        }
    }
    std::cout << "[construct_EPF] Set graph for " << graph_set_count << "/" << netdag_ptr->anchors.size()
              << " anchors from db. Missing: " << missing_count << std::endl;

    netDagConst.setNetDag(netdag_ptr);
    netDagConst.setAnchors(netdag_ptr->anchors);
    // nodes is empty (flat mode does not load nodes), but construct_EPT does not need nodes
    // netDagConst.setNodes(netdag_ptr->nodes);

    // Print anchor/cluster statistics (Do not call print_netdag_info since hierarchy was not loaded)
    std::cout << "\n====== Loaded Anchor Information (flat mode) ======\n";
    std::cout << "Number of anchors: " << netdag_ptr->anchors.size() << std::endl;
    std::cout << "====================================================\n\n";

    // Collect cluster statistics
    int num_nodes_in_cluster = 0;
    int num_nodes_in_exact_cluster = 0;
    for (auto anchor : netDagConst.get_anchors()) {
        num_nodes_in_cluster += anchor->nodes_in_cluster.size();
        num_nodes_in_exact_cluster += anchor->nodes_in_exact_cluster.size();
    }
    std::cout << "Number of nodes in cluster: " << num_nodes_in_cluster << std::endl;
    std::cout << "Number of nodes in exact cluster: " << num_nodes_in_exact_cluster << std::endl;
    std::cout << "Number of anchors: " << netDagConst.get_anchors().size() << std::endl;

    // Build EPF (force rebuild, overwrite existing)
    std::cout << "[construct_EPF] Building EPF (will overwrite existing)..." << std::endl;
    netDagConst.construct_EPT_parallel();
    std::cout << "[construct_EPF] EPF construction completed successfully!" << std::endl;

    // Print EPF information after construction
    std::cout << "\n====== EPF Construction Summary ======\n";
    print_epf_info(index_name, netDagConst.get_anchors().size());
    std::cout << "=====================================\n";
}

void construct_mode(Config& config) {
    std::cout << "========================================" << std::endl;
    std::cout << "Construct Mode (Full Pipeline)" << std::endl;
    std::cout << "  Step 1: construct_ND  (NetDag structure)" << std::endl;
    std::cout << "  Step 2: compute_paths (GED + node matching)" << std::endl;
    std::cout << "  Step 3: reassign      (Reassign nodes to clusters)" << std::endl;
    std::cout << "  Step 4: construct_EPF (EPF filters)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Step 1: Construct NetDag
    std::cout << "\n[Step 1/4] Constructing NetDag..." << std::endl;
    construct_ND_mode(config);

    // Step 2: Compute paths (GED + node matching)
    std::cout << "\n[Step 2/4] Computing paths (GED + node matching)..." << std::endl;
    compute_paths_mode(config);

    // Step 3: Reassign nodes
    std::cout << "\n[Step 3/4] Reassigning nodes to clusters..." << std::endl;
    reassign_mode(config);

    // Step 4: Construct EPF
    std::cout << "\n[Step 4/4] Constructing EPF..." << std::endl;
    construct_EPF_mode(config);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Construct Mode completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
}







// EPF info printing function
void print_epf_info(const std::string& index_name, int total_anchors) {
    // Check if EPF directory exists
    std::string epf_dir_path = "./EPFs/" + index_name;

    try {
        if (std::filesystem::exists(epf_dir_path) && std::filesystem::is_directory(epf_dir_path)) {
            printf("EPF directory: %s\n", epf_dir_path.c_str());

            // Count EPT files
            int ept_count = 0;
            uint64_t total_epf_size = 0;
            std::filesystem::file_time_type latest_epf_time = std::filesystem::file_time_type::min();

            for (const auto& entry : std::filesystem::directory_iterator(epf_dir_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                    std::string filename = entry.path().filename().string();
                    if (filename.find("EPT_anchor_") == 0) {
                        ept_count++;
                        total_epf_size += entry.file_size();
                        auto file_time = entry.last_write_time();
                        if (file_time > latest_epf_time) {
                            latest_epf_time = file_time;
                        }
                    }
                }
            }

            printf("Number of EPT files: %d\n", ept_count);
            printf("Total EPF size: %.2f MB\n", total_epf_size / (1024.0 * 1024.0));

            if (ept_count > 0) {
                auto epf_sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    latest_epf_time - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                std::time_t epf_cftime = std::chrono::system_clock::to_time_t(epf_sctp);
                printf("Latest EPF modified: %s", std::ctime(&epf_cftime));

                // Calculate coverage if total_anchors is provided
                if (total_anchors > 0) {
                    double coverage = (double)ept_count / total_anchors * 100.0;
                    printf("EPF coverage: %d/%d anchors (%.1f%%)\n", ept_count, total_anchors, coverage);
                } else {
                    printf("EPT file count: %d files\n", ept_count);
                }

                // Detailed EPT statistics
                printf("\n--- EPT Detailed Statistics ---\n");

                // Statistics collectors
                int total_nodes = 0;
                int max_nodes = 0;
                int min_nodes = std::numeric_limits<int>::max();
                int total_depth = 0;
                int max_depth = 0;
                int min_depth = std::numeric_limits<int>::max();
                int valid_ept_count = 0;

                // New statistics for path nodes vs data graphs ratio
                int total_completed_graphs = 0;
                int total_path_nodes = 0;
                double total_expansion_ratio = 0.0;

                // Sample a subset of EPT files for statistics (to avoid long processing time)
                int sample_size = std::min(100, ept_count);
                int step = std::max(1, ept_count / sample_size);

                int file_index = 0;
                for (const auto& entry : std::filesystem::directory_iterator(epf_dir_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                        std::string filename = entry.path().filename().string();
                        if (filename.find("EPT_anchor_") == 0) {
                            if (file_index % step == 0) {
                                try {
                                    EditPathTree ept;
                                    ept.load_from_file(entry.path().string());

                                    int node_count = ept.tree_nodes.size();

                                    // Calculate depth manually by traversing the tree
                                    int depth = 0;
                                    if (!ept.tree_nodes.empty()) {
                                        std::function<int(size_t, int)> calculate_depth = [&](size_t node_idx, int current_depth) -> int {
                                            int max_child_depth = current_depth;
                                            for (size_t child_idx : ept.tree_nodes[node_idx].children_indices) {
                                                max_child_depth = std::max(max_child_depth, calculate_depth(child_idx, current_depth + 1));
                                            }
                                            return max_child_depth;
                                        };
                                        depth = calculate_depth(ept.root_index, 0);
                                    }

                                    // Calculate path nodes vs data graphs ratio
                                    std::unordered_set<int> unique_completed_graphs;
                                    for (const auto& node : ept.tree_nodes) {
                                        for (int graph_id : node.completed_db_graph_ids) {
                                            unique_completed_graphs.insert(graph_id);
                                        }
                                    }
                                    int completed_graph_count = unique_completed_graphs.size();

                                    // Each EPT represents at least the anchor graph itself
                                    // If no completed_db_graph_ids found, it means only the anchor graph is represented
                                    if (completed_graph_count == 0) {
                                        completed_graph_count = 1; // Count the anchor graph itself
                                    }

                                    // Update statistics
                                    total_completed_graphs += completed_graph_count;
                                    total_path_nodes += node_count;

                                    double expansion_ratio = (double)node_count / completed_graph_count;
                                    total_expansion_ratio += expansion_ratio;


                                    total_nodes += node_count;
                                    max_nodes = std::max(max_nodes, node_count);
                                    min_nodes = std::min(min_nodes, node_count);

                                    total_depth += depth;
                                    max_depth = std::max(max_depth, depth);
                                    min_depth = std::min(min_depth, depth);

                                    valid_ept_count++;
                                } catch (const std::exception& e) {
                                    // Skip corrupted files silently
                                }
                            }
                            file_index++;
                        }
                    }
                }

                if (valid_ept_count > 0) {
                    double avg_nodes = (double)total_nodes / valid_ept_count;
                    double avg_depth = (double)total_depth / valid_ept_count;

                    printf("Node statistics (sampled %d/%d EPTs):\n", valid_ept_count, ept_count);
                    printf("  Average nodes per EPT: %.1f\n", avg_nodes);
                    printf("  Maximum nodes per EPT: %d\n", max_nodes);
                    printf("  Minimum nodes per EPT: %d\n", min_nodes == std::numeric_limits<int>::max() ? 0 : min_nodes);

                    printf("Depth statistics:\n");
                    printf("  Average depth: %.1f\n", avg_depth);
                    printf("  Maximum depth: %d\n", max_depth);
                    printf("  Minimum depth: %d\n", min_depth == std::numeric_limits<int>::max() ? 0 : min_depth);

                    // Path nodes vs data graphs ratio analysis
                    printf("Path expansion statistics:\n");
                    printf("  Total path nodes (sampled): %d\n", total_path_nodes);
                    printf("  Total completed graphs (sampled): %d\n", total_completed_graphs);
                    if (total_completed_graphs > 0) {
                        double overall_expansion_ratio = (double)total_path_nodes / total_completed_graphs;
                        double avg_expansion_ratio = total_expansion_ratio / valid_ept_count;
                        printf("  Overall expansion ratio: %.2fx (path nodes / data graphs)\n", overall_expansion_ratio);
                        printf("  Average expansion ratio per EPT: %.2fx\n", avg_expansion_ratio);
                    } else {
                        printf("  No completed graphs found in samples\n");
                    }
                } else {
                    printf("Unable to load EPT statistics (files may be corrupted)\n");
                }
            } else {
                printf("EPF Status: Directory exists but no EPT files found\n");
            }

        } else {
            printf("EPF Status: Not found (directory does not exist)\n");
            printf("Expected path: %s\n", epf_dir_path.c_str());
            printf("Run 'construct_EPF' mode to build EPF\n");
        }
    } catch (const std::exception& e) {
        printf("EPF Status: Error checking EPF directory: %s\n", e.what());
    }
}

// Include experiment mode implementation
#include "experiment_mode_impl.cpp"

void search_mode(const Config& config) {
    printf("[search_mode] Starting search mode...\n");
    
    // Extract configuration parameters
    double alpha = config.alpha;
    double tau_index = config.tau_index;
    double error_tolerance_index = config.error_tolerance_index;
    std::string db_name = config.db_name;
    std::string query_name = config.query_name;
    std::string ground_truth_path = config.ground_truth_path;
    std::string search_method = "Gisma";  // Search mode always uses Gisma
    double tau_search = config.tau_search;
    double error_tolerance_search = config.error_tolerance_search;
    int q_start = config.q_start;
    int q_end = config.q_end;
    bool has_ged_matrix = config.has_ged_matrix;
    

    // Load database graphs
    printf("[search_mode] Step 1: Loading database graphs...\n");
    std::cout << "Loading database graphs..." << std::endl;
    std::vector<std::shared_ptr<Node>> db_node_list;
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    
    auto load_start = std::chrono::high_resolution_clock::now();
    ui max_db_n = Utility::load_db(db_name.c_str(), db, vM, eM);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
    
    printf("[search_mode] Database graphs loaded: %zu graphs in %ld ms\n", db.size(), load_duration.count());
    std::cout << "Loaded " << db.size() << " graphs with max node count: " << max_db_n << std::endl;

    Graph::FEATURE_DIM = vM.size();
    printf("[search_mode] Feature dimension set to: %zu\n", vM.size());
    
    // Pre-allocate space
    db_node_list.resize(db.size());
    
    // Convert database graphs in parallel
    printf("[search_mode] Step 2: Converting database graphs to Node format (parallel)...\n");
    auto convert_start = std::chrono::high_resolution_clock::now();
    
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 
                                    static_cast<int>(db.size()));
    
    if (num_threads > 1 && db.size() > 1000) {
        printf("[search_mode] Using %d threads for parallel conversion\n", num_threads);
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 100)
        for (size_t i = 0; i < db.size(); ++i) {
            auto graph_ptr = db[i];
            auto node = std::make_shared<Node>();
            node->node_id = i;
            node->graph = std::make_shared<Graph>(*graph_ptr);
            node->file_name = graph_ptr->id;
            node->is_anchor = false;
            node->nearest_anchor = -1;
            node->nearest_anchor_dist = std::numeric_limits<double>::infinity();

            db_node_list[i] = node;
        }
    } else {
        // Serial processing (for small data)
        for (size_t i = 0; i < db.size(); ++i) {
            // if (i % 100000 == 0 && i > 0) {
            //     printf("[search_mode] Processing DB node %zu/%zu\n", i, db.size());
            // }
            
            auto graph_ptr = db[i];
            auto node = std::make_shared<Node>();
            node->node_id = i;
            node->graph = std::make_shared<Graph>(*graph_ptr);
            node->file_name = graph_ptr->id;
            node->is_anchor = false;
            node->nearest_anchor = -1;
            node->nearest_anchor_dist = std::numeric_limits<double>::infinity();

            // node->graph->initialize_vectors_from_arrays();
            // node->ml_graph = node->graph->to_ML_graph();

            db_node_list[i] = node;
        }
    }
    
    auto convert_end = std::chrono::high_resolution_clock::now();
    auto convert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(convert_end - convert_start);
    printf("[search_mode] Database nodes conversion completed in %ld ms\n", convert_duration.count());

    // Load GED matrix if needed
    std::vector<double> ged_matrix;
    int N = 0;
    if (has_ged_matrix) {
        printf("[search_mode] Step 3: Loading GED matrix...\n");
        std::string binary_file = "../GED_matrix/ensembled_ged_matrix_good_upper.bin";
        std::ifstream infile(binary_file, std::ios::binary);

        if (!infile) {
            std::cerr << "Cannot open file " << binary_file << std::endl;
            return;
        }

        infile.read(reinterpret_cast<char*>(&N), sizeof(int));
        if (!infile) {
            std::cerr << "Failed to read matrix size." << std::endl;
            return;
        }

        size_t num_elements = static_cast<size_t>(N) * (N + 1) / 2;
        ged_matrix.resize(num_elements);
        infile.read(reinterpret_cast<char*>(ged_matrix.data()), num_elements * sizeof(double));
        infile.close();

        printf("[search_mode] GED matrix loaded: N=%d, elements=%zu\n", N, num_elements);
        std::cout << "GED matrix loaded successfully. Matrix size: " << num_elements << std::endl;
    } else {
        printf("[search_mode] Step 3: Skipping GED matrix loading (has_ged_matrix=false)\n");
    }

    // Get index name
    std::string index_name = Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db.size());

    // Load NetDag object (needed for Gisma, Base+GS, Base+SS, Base_All_EPT)
    std::shared_ptr<NetDag> netdag_ptr = std::make_shared<NetDag>();
    if (config.search_method == "Gisma" || config.search_method == "Gisma-no-reuse" || config.search_method == "Base+GS" || config.search_method == "Base+SS" || config.search_method == "Base_All_EPT") {
        printf("[search_mode] Step 4: Loading NetDag index...\n");
        printf("[search_mode] Index name: %s\n", index_name.c_str());

        std::string dataset_dir = "./NetDags/" + config.dataset + "/";
        std::string configs_dir = dataset_dir + "configs/";
        std::string reassigned_dir = dataset_dir + "reassigned/";
        std::filesystem::create_directories(configs_dir);
        std::filesystem::create_directories(reassigned_dir);
        std::string reassigned_file_path = reassigned_dir + index_name + ".dat";
        printf("[search_mode] Loading NetDag from: %s\n", reassigned_file_path.c_str());

        netdag_ptr->load_from_file(*netdag_ptr, reassigned_file_path);
        printf("[search_mode] NetDag loaded successfully\n");
    } else {
        printf("[search_mode] Step 4: Skipping NetDag loading (not needed for %s)\n", config.search_method.c_str());
    }

    // Prepare query node list and query graphs
    printf("[search_mode] Step 5: Loading query graphs...\n");
    std::vector<std::shared_ptr<Node>> query_node_list;
    std::vector<Graph*> query_db;
    ui max_query_n = Utility::load_db(query_name.c_str(), query_db, vM, eM);
    printf("[search_mode] Query graphs loaded: %zu graphs\n", query_db.size());
    std::cout << "Loaded " << query_db.size() << " query graphs with max node count: " << max_query_n << std::endl;

    if (q_start == -1) {
        q_start = 0;
    }
    if (q_end == -1) {
        q_end = query_db.size() - 1;
    }
    printf("[search_mode] Processing queries from %d to %d\n", q_start, q_end);
    
    // Convert query graphs (usually small count, but can be parallelized)
    printf("[search_mode] Step 6: Converting query graphs to Node format...\n");
    int num_queries = q_end - q_start + 1;
    query_node_list.resize(num_queries);
    
    if (num_queries > 10 && num_threads > 1) {
        // Process query graphs in parallel
        #pragma omp parallel for num_threads(std::min(num_threads, num_queries))
        for (int idx = 0; idx < num_queries; ++idx) {
            int i = q_start + idx;
            auto graph_ptr = query_db[i];
            auto node = std::make_shared<Node>();
            node->node_id = i;
            node->graph = std::make_shared<Graph>(*graph_ptr);
            node->file_name = graph_ptr->id;
            node->is_anchor = false;
            node->nearest_anchor = -1;
            node->nearest_anchor_dist = std::numeric_limits<double>::infinity();

            // node->graph->initialize_vectors_from_arrays();
            // node->ml_graph = node->graph->to_ML_graph();

            query_node_list[idx] = node;
        }
    } else {
        // Serial processing
        for (int idx = 0; idx < num_queries; ++idx) {
            int i = q_start + idx;
            auto graph_ptr = query_db[i];
            auto node = std::make_shared<Node>();
            node->node_id = i;
            node->graph = std::make_shared<Graph>(*graph_ptr);
            node->file_name = graph_ptr->id;
            node->is_anchor = false;
            node->nearest_anchor = -1;
            node->nearest_anchor_dist = std::numeric_limits<double>::infinity();

            // node->graph->initialize_vectors_from_arrays();
            // node->ml_graph = node->graph->to_ML_graph();

            query_node_list[idx] = node;
        }
    }
    printf("[search_mode] Query nodes conversion completed\n");

    if (query_node_list.empty()) {
        std::cerr << "Error: query_node_list is empty after loading queries." << std::endl;
        return;
    } else {
        printf("[search_mode] Loaded %zu query nodes\n", query_node_list.size());
        std::cout << "Loaded " << query_node_list.size() << " query nodes." << std::endl;
    }

    // Initialize auxiliary variables
    ui max_n = std::max(max_db_n, max_query_n);
    printf("[search_mode] Max node count: %u\n", max_n);

    // Load exact ground truth
    printf("[search_mode] Step 7: Loading ground truth...\n");
    std::map<int, std::map<double, std::vector<int>>> ground_truth;
    if (!Utility::load_exact_ground_truth(ground_truth_path, ground_truth)) {
        std::cerr << "Error: Failed to load ground truth from " << ground_truth_path << std::endl;
        return;
    } else {
        printf("[search_mode] Ground truth loaded: %zu queries\n", ground_truth.size());
        std::cout << "Ground truth loaded successfully. Number of queries: " << ground_truth.size() << std::endl;
    }

    // Load EPTs (needed for Gisma, Base+GS, Base+SS, Base_All_EPT)
    EditPathTreeManager ept_manager;
    if (config.search_method == "Gisma" || config.search_method == "Gisma-no-reuse" || config.search_method == "Base+GS" || config.search_method == "Base+SS" || config.search_method == "Base_All_EPT") {
        printf("[search_mode] Step 8: Loading EPTs...\n");
        std::string ept_directory_path = "./EPFs/" + index_name;
        std::cout << "Loading EPTs from directory: " << ept_directory_path << "..." << std::endl;

        printf("[search_mode] Using parallel EPT loading for %s method\n", config.search_method.c_str());
        ept_manager.load_all_epts_from_directory_parallel(ept_directory_path);
        printf("[search_mode] EPT loading completed: %zu EPTs loaded\n", ept_manager.get_ept_count());
        std::cout << "Loaded " << ept_manager.get_ept_count() << " EPTs." << std::endl;

        // Fill global coverage sets for GismaSearchEngine statistics
        g_ept_coverage.clear();
        g_extra_coverage.clear();
        for (const auto& anchor : netdag_ptr->anchors) {
            // EPT coverage: anchor itself + all tree_node completed_db_graph_ids
            g_ept_coverage.insert(anchor->node_id);
            EditPathTree* ept = ept_manager.get_ept(anchor->node_id);
            if (ept) {
                for (const auto& tree_node : ept->tree_nodes) {
                    for (int id : tree_node.completed_db_graph_ids) {
                        g_ept_coverage.insert(id);
                    }
                }
            }
            // Extra coverage: nodes_in_cluster
            auto cluster_copy = anchor->nodes_in_cluster;
            while (!cluster_copy.empty()) {
                g_extra_coverage.insert(cluster_copy.top().second);
                cluster_copy.pop();
            }
        }
        std::unordered_set<int> total_union = g_ept_coverage;
        total_union.insert(g_extra_coverage.begin(), g_extra_coverage.end());
        printf("[search_mode] Database Coverage: EPT=%zu graphs, Extra=%zu graphs, Union=%zu graphs (total DB=%zu)\n",
               g_ept_coverage.size(), g_extra_coverage.size(), total_union.size(), db.size());
    } else {
        printf("[search_mode] Step 8: Skipping EPT loading (not needed for %s)\n", config.search_method.c_str());
    }

    // Fill anchor vectors in parallel (needed for Gisma, Base+GS, Base+SS, Base_All_EPT)
    auto fill_start = std::chrono::high_resolution_clock::now();
    if (config.search_method == "Gisma" || config.search_method == "Gisma-no-reuse" || config.search_method == "Base+GS" || config.search_method == "Base+SS" || config.search_method == "Base_All_EPT") {
        printf("[search_mode] Step 9: Filling anchor vectors (parallel)...\n");

        if (netdag_ptr->anchors.size() > 100 && num_threads > 1) {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 10)
            for (size_t i = 0; i < netdag_ptr->anchors.size(); ++i) {
                netdag_ptr->anchors[i]->fill_vectors_from_queues();
            }
        } else {
            for (auto & anchorPtr : netdag_ptr->anchors) {
                anchorPtr->fill_vectors_from_queues();
            }
        }

        auto fill_end = std::chrono::high_resolution_clock::now();
        auto fill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fill_end - fill_start);
        printf("[search_mode] Anchor vectors filled in %ld ms\n", fill_duration.count());
    } else {
        printf("[search_mode] Step 9: Skipping anchor vector filling (not needed for %s)\n", config.search_method.c_str());
    }
    auto fill_end = std::chrono::high_resolution_clock::now();
    auto fill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fill_end - fill_start);
    
    // Create GismaSearchEngine instance (scoped)
    printf("[search_mode] Step 10: Creating GismaSearchEngine instance...\n");

    // Use scope to limit search engine lifetime
    {
        GismaSearchEngine searcher(
            netdag_ptr,
            tau_index,
            // tau_search,  // removed: now a function parameter
            error_tolerance_search,
            q_start,
            q_end,
            has_ged_matrix,
            ged_matrix,
            search_method,
            config.dataset,  // dataset_name
            db_node_list,
            query_node_list,
            N,
            db,
            ground_truth,
            vM,
            eM,
            max_n,
            &ept_manager,
            config.nd_mode,          // pass NetDag mode
            config.dfs_mode,         // pass DFS traversal mode
            !config.disable_ept_filters,  // pass EPT filter switch (enabled by default, --disable_ept_filters to disable)
            config.only_compute_db_graph,   // only compute db graph vertices
            config.app_max_iter,  // A* max iteration count
            !config.disable_fast_down,  // fast-down strategy switch (inverted: disable_fast_down=false means enabled)
            config.exact_max_iter,  // exact computation iteration limit
            config.nd_filter_ratio,  // NetDag filter tightening ratio
            config.disable_lsa_pruning,  // Disable LSa pruning
            config.disable_reuse_lsa,    // disable LSa recomputation in reuse
            config.verify_reuse_baseline,  // verify reuse effectiveness
            config.chain_reuse  // chain reuse
        );
        printf("[search_mode] GismaSearchEngine instance created\n");

        // Set whether to save logs
        searcher.save_logs = config.save_logs;

        // Execute search
        printf("[search_mode] Step 12: Starting search...\n");
        if (config.use_parallel) {
            std::cout << "Using parallel search." << std::endl;
            printf("[search_mode] Executing parallel search\n");
            searcher.perform_search_parallel(tau_search);
        } else {
            std::cout << "Using sequential search." << std::endl;
            printf("[search_mode] Executing sequential search\n");
            searcher.perform_search(tau_search);
        }
        printf("[search_mode] Search completed\n");

        printf("[search_mode] Step 12.1: Destroying search engine...\n");
    } // searcher destroyed here
    printf("[search_mode] Search engine destroyed\n");

    // Ensure all parallel operations complete
    printf("[search_mode] Step 12.2: Waiting for all threads to complete...\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    printf("[search_mode] All threads completed\n");

    printf("[search_mode] search_mode function completed successfully\n");

    // Output performance summary
    printf("\n[Performance Summary]\n");
    printf("- Database loading: %ld ms\n", load_duration.count());
    printf("- DB conversion: %ld ms\n", convert_duration.count());
    printf("- Anchor filling: %ld ms\n", fill_duration.count());

    // Clean up memory - following original version
    printf("[search_mode] Step 13: Cleaning up memory...\n");

    printf("[search_mode] Cleaning up database graphs...\n");
    for (auto* graph : db) {
        delete graph;
    }
    db.clear();
    printf("[search_mode] Database graphs cleaned\n");

    printf("[search_mode] Cleaning up query graphs...\n");
    for (auto* graph : query_db) {
        delete graph;
    }
    query_db.clear();
    printf("[search_mode] Query graphs cleaned\n");
    if (num_threads > 1 && db.size() > 1000) {
        printf("- Used %d threads for parallel processing\n", num_threads);
    }
}

void info_mode(Config& config) {
    printf("[info_mode] Starting NetDag info mode...\n");
    fflush(stdout);

    // Construct file path directly for SYN dataset (avoid directory traversal issues)
    std::string netdag_file_path;
    if (config.dataset == "SYN") {
        // SYN dataset uses known filename directly (skip filesystem check to avoid Windows bug)
        netdag_file_path = "./NetDags/SYN/reassigned/SYN_999900_6.0_4.0_0.0.dat";
        printf("[info_mode] Using predefined NetDag file for SYN: %s\n", netdag_file_path.c_str());
        fflush(stdout);
    } else {
        // Other datasets use directory scan
        printf("[info_mode] Scanning for NetDag files for dataset: %s\n", config.dataset.c_str());
        fflush(stdout);
        std::vector<std::string> available_netdags;

        try {
            std::string reassigned_dir = "./NetDags/" + config.dataset + "/reassigned/";
            if (std::filesystem::exists(reassigned_dir)) {
                for (const auto& entry : std::filesystem::directory_iterator(reassigned_dir)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                        std::string filename = entry.path().filename().string();
                        if (filename.find(config.dataset + "_") == 0) {
                            available_netdags.push_back(filename);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            printf("[info_mode] Error scanning NetDags directory: %s\n", e.what());
            return;
        }

        if (available_netdags.empty()) {
            printf("[info_mode] No NetDag files found for dataset %s\n", config.dataset.c_str());
            printf("[info_mode] Please run construct mode first\n");
            return;
        }

        printf("[info_mode] Found %zu NetDag file(s) for %s:\n", available_netdags.size(), config.dataset.c_str());
        for (size_t i = 0; i < available_netdags.size(); i++) {
            printf("[info_mode]   %zu: %s\n", i + 1, available_netdags[i].c_str());
        }

        // Use the first available NetDag file
        netdag_file_path = "./NetDags/" + config.dataset + "/reassigned/" + available_netdags[0];
        printf("[info_mode] Using NetDag file: %s\n", netdag_file_path.c_str());
    }

    // Load NetDag
    printf("[info_mode] Loading NetDag index...\n");
    fflush(stdout);
    NetDag netdag;
    auto load_netdag_start = std::chrono::high_resolution_clock::now();

    printf("[info_mode] About to call NetDag::load_from_file...\n");
    fflush(stdout);
    try {
        NetDag::load_from_file(netdag, netdag_file_path);
        printf("[info_mode] NetDag::load_from_file returned\n");
        fflush(stdout);
        auto load_netdag_end = std::chrono::high_resolution_clock::now();
        auto load_netdag_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_netdag_end - load_netdag_start);

        printf("[info_mode] NetDag loaded successfully in %lld ms\n", load_netdag_duration.count());

        // Display NetDag information
        printf("\n[info_mode] ====== NetDag Information ======\n");
        netdag.print_netdag_info();

        // Display file information
        printf("\n[info_mode] ====== File Information ======\n");
        std::filesystem::path netdag_path(netdag_file_path);
        auto file_size = std::filesystem::file_size(netdag_path);
        printf("NetDag file size: %.2f MB\n", file_size / (1024.0 * 1024.0));

        auto ftime = std::filesystem::last_write_time(netdag_path);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);
        printf("Last modified: %s", std::ctime(&cftime));

        // Display EPF information
        printf("\n[info_mode] ====== EPF Information ======\n");

        // Extract index name from the NetDag file path (remove .dat extension)
        std::string netdag_filename = std::filesystem::path(netdag_file_path).filename().string();
        std::string index_name = netdag_filename;
        if (index_name.length() >= 4 && index_name.substr(index_name.length() - 4) == ".dat") {
            index_name = index_name.substr(0, index_name.length() - 4);
        }

        // Calculate total anchors for coverage calculation
        int total_anchors = 0;
        for (const auto& anchor : netdag.anchors) {
            total_anchors++;
        }

        // Use the dedicated EPF info printing function
        print_epf_info(index_name, total_anchors);

        printf("\n[info_mode] Info mode completed successfully\n");

        // ========== Coverage analysis ==========
        printf("\n[info_mode] ====== Coverage Analysis ======\n");

        // First count priority_queue size
        size_t total_pq_cluster = 0, total_pq_exact = 0;
        for (const auto& anchor : netdag.anchors) {
            total_pq_cluster += anchor->nodes_in_cluster.size();
            total_pq_exact += anchor->nodes_in_exact_cluster.size();
        }
        printf("Before fill_vectors: pq_cluster=%zu, pq_exact=%zu\n", total_pq_cluster, total_pq_exact);

        // First fill vectors (from priority queues)
        printf("Filling anchor vectors from queues...\n");
        for (const auto& anchor : netdag.anchors) {
            anchor->fill_vectors_from_queues();
        }

        // Count vector size after filling
        size_t total_vec_cluster = 0, total_vec_exact = 0;
        for (const auto& anchor : netdag.anchors) {
            total_vec_cluster += anchor->nodes_in_cluster_vec.size();
            total_vec_exact += anchor->nodes_in_exact_cluster_vec.size();
        }
        printf("After fill_vectors: vec_cluster=%zu, vec_exact=%zu\n", total_vec_cluster, total_vec_exact);

        std::unordered_set<int> all_anchors_set;
        std::unordered_set<int> all_nodes_in_cluster;
        std::unordered_set<int> all_nodes_in_exact_cluster;

        for (const auto& anchor : netdag.anchors) {
            // anchor itself
            all_anchors_set.insert(anchor->node_id);

            // nodes_in_cluster_vec
            for (const auto& p : anchor->nodes_in_cluster_vec) {
                all_nodes_in_cluster.insert(p.second);
            }

            // nodes_in_exact_cluster_vec
            for (const auto& p : anchor->nodes_in_exact_cluster_vec) {
                all_nodes_in_exact_cluster.insert(p.second);
            }
        }

        // Compute union
        std::unordered_set<int> netdag_coverage;
        netdag_coverage.insert(all_anchors_set.begin(), all_anchors_set.end());
        netdag_coverage.insert(all_nodes_in_cluster.begin(), all_nodes_in_cluster.end());
        netdag_coverage.insert(all_nodes_in_exact_cluster.begin(), all_nodes_in_exact_cluster.end());

        int total_db_size = static_cast<int>(netdag.nodes.size());

        printf("Database size: %d graphs\n", total_db_size);
        printf("Anchors: %zu\n", all_anchors_set.size());
        printf("nodes_in_cluster (Extra): %zu unique graphs\n", all_nodes_in_cluster.size());
        printf("nodes_in_exact_cluster (EPT source): %zu unique graphs\n", all_nodes_in_exact_cluster.size());
        printf("NetDag coverage (union): %zu graphs\n", netdag_coverage.size());
        printf("Uncovered graphs: %d\n", total_db_size - static_cast<int>(netdag_coverage.size()));

        // Find uncovered graph IDs and analyze
        if (netdag_coverage.size() < static_cast<size_t>(total_db_size)) {
            printf("\nUncovered graph IDs (first 20):\n");
            std::vector<int> uncovered_ids;
            for (int i = 0; i < total_db_size; i++) {
                if (netdag_coverage.find(i) == netdag_coverage.end()) {
                    uncovered_ids.push_back(i);
                }
            }

            for (size_t i = 0; i < std::min(uncovered_ids.size(), size_t(20)); i++) {
                printf("  %d", uncovered_ids[i]);
            }
            printf("\n");

            // Analyze characteristics of uncovered graphs
            printf("\nAnalyzing uncovered graphs (first 10):\n");
            for (size_t i = 0; i < std::min(uncovered_ids.size(), size_t(10)); i++) {
                int node_id = uncovered_ids[i];
                auto node = netdag.nodes[node_id];
                if (node) {
                    printf("  Graph %d: is_anchor=%d, nearest_anchor=%d, nearest_dist=%.2f\n",
                           node_id, node->is_anchor ? 1 : 0,
                           node->nearest_anchor, node->nearest_anchor_dist);
                } else {
                    printf("  Graph %d: node is null!\n", node_id);
                }
            }

            // Count nearest_anchor distribution of uncovered graphs
            printf("\nNearest anchor distribution for uncovered graphs:\n");
            std::map<int, int> nearest_anchor_count;
            for (int id : uncovered_ids) {
                auto node = netdag.nodes[id];
                if (node) {
                    nearest_anchor_count[node->nearest_anchor]++;
                }
            }
            printf("  Total uncovered: %zu\n", uncovered_ids.size());
            printf("  Unique nearest_anchors: %zu\n", nearest_anchor_count.size());
            // Show top 5
            std::vector<std::pair<int, int>> sorted_counts(nearest_anchor_count.begin(), nearest_anchor_count.end());
            std::sort(sorted_counts.begin(), sorted_counts.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            printf("  Top 5 nearest_anchors:\n");
            for (size_t i = 0; i < std::min(sorted_counts.size(), size_t(5)); i++) {
                printf("    anchor %d: %d uncovered graphs\n", sorted_counts[i].first, sorted_counts[i].second);
            }

            // Check what root node is
            int root_id = -1;
            if (netdag.root) {
                root_id = netdag.root->node_id;
                printf("\nRoot node ID: %d\n", root_id);
            }

            // Count nearest_anchor pointing to root
            int points_to_root = 0;
            for (int id : uncovered_ids) {
                auto node = netdag.nodes[id];
                if (node && node->nearest_anchor == root_id) {
                    points_to_root++;
                }
            }
            printf("Uncovered graphs pointing to root: %d / %zu\n", points_to_root, uncovered_ids.size());

            // Check detailed nearest_anchor info of some uncovered graphs
            printf("\nChecking nearest_anchor details for first 5 uncovered graphs:\n");
            for (size_t i = 0; i < std::min(uncovered_ids.size(), size_t(5)); i++) {
                int node_id = uncovered_ids[i];
                auto node = netdag.nodes[node_id];
                if (!node) continue;

                int na = node->nearest_anchor;
                printf("  Graph %d -> nearest_anchor %d:\n", node_id, na);

                // Check if nearest_anchor is an anchor
                bool found_anchor = false;
                for (const auto& anchor : netdag.anchors) {
                    if (anchor->node_id == na) {
                        found_anchor = true;
                        printf("    Anchor exists, cluster_size=%zu, exact_cluster_size=%zu\n",
                               anchor->nodes_in_cluster_vec.size(),
                               anchor->nodes_in_exact_cluster_vec.size());

                        // Check if node_id is in cluster
                        bool in_cluster = false, in_exact = false;
                        for (const auto& p : anchor->nodes_in_cluster_vec) {
                            if (p.second == node_id) { in_cluster = true; break; }
                        }
                        for (const auto& p : anchor->nodes_in_exact_cluster_vec) {
                            if (p.second == node_id) { in_exact = true; break; }
                        }
                        printf("    Graph %d in cluster: %s, in exact_cluster: %s\n",
                               node_id, in_cluster ? "YES" : "NO", in_exact ? "YES" : "NO");
                        break;
                    }
                }
                if (!found_anchor) {
                    printf("    WARNING: nearest_anchor %d is NOT an anchor!\n", na);
                }
            }
        }

    } catch (const std::exception& e) {
        printf("[info_mode] Error loading NetDag: %s\n", e.what());
    }
}

























// Helper structures for statistics collection
struct ClusterSnapshot {
    int r_integer;              // r_k corresponding integer value
    double r_k_actual;          // actual r_k value
    int phase;                  // current phase
    int num_anchors;            // number of anchors
    std::vector<int> cluster_sizes;  // size of each cluster
    std::vector<int> friends_counts;  // number of friends within 3r_k range for each anchor (doubling constant)

    // Computed statistics
    double avg_cluster_size = 0.0;
    double max_cluster_size = 0.0;
    double total_nodes_in_clusters = 0.0;
    double avg_friends_count = 0.0;
    double max_friends_count = 0.0;

    // Net-Tree statistics (cover count based on hierarchical splitting)
    double avg_net_tree_cover = 0.0;
    double max_net_tree_cover = 0.0;
};

struct AlphaStatistics {
    double alpha;
    double tau_index;
    int phi;  // number of NetDag layers
    std::vector<ClusterSnapshot> snapshots;  // snapshot for each integer r
};

// select-alpha mode: build NetDag once (using alpha_min), collect per-layer statistics
// Since greedy permutation is independent of alpha, upper-layer data is identical across different alpha values,
// so building once to the minimum alpha suffices; all larger alpha data are subsets.
void select_alpha_mode(Config& config) {
    double alpha = config.alpha_min;  // build once with minimum alpha
    std::cout << "[select-alpha] Starting select-alpha mode (single build, alpha=" << alpha << ")..." << std::endl;

    // 1. Load database
    std::cout << "[select-alpha] Step 1: Loading database..." << std::endl;
    std::string database = config.db_name;
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(database.c_str(), db, vM, eM);
    std::cout << "[select-alpha] Loaded " << db.size() << " graphs with max node count: " << max_db_n << std::endl;
    Graph::FEATURE_DIM = vM.size();

    std::vector<std::shared_ptr<Graph>> db_graphs;
    std::vector<std::string> nameList;
    for (const auto& graph_ptr : db) {
        db_graphs.emplace_back(std::shared_ptr<Graph>(graph_ptr));
        nameList.push_back(graph_ptr->id);
    }

    // Prepare output directory
    std::string output_dir = "./alpha_selection";
    std::filesystem::create_directories(output_dir);

    // 2. Build NetDag once
    std::vector<AlphaStatistics> all_alpha_stats;

    {
        AlphaStatistics alpha_stats;
        alpha_stats.alpha = alpha;
        alpha_stats.tau_index = config.tau_index;

        // 3. Build NetDag for each alpha and collect statistics
        std::string index_name = Utility::get_index_name(
            config.dataset, alpha, config.tau_index, config.error_tolerance_index, db_graphs.size()
        );

        // Load embeddings (if available)
        std::vector<std::vector<float>> embeddings;
        std::string embedding_file = "./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin";
        if (std::filesystem::exists(embedding_file)) {
            std::cout << "[select-alpha]   Loading embeddings from " << embedding_file << std::endl;
            if (Utility::load_embeddings(embedding_file, embeddings)) {
                std::cout << "[select-alpha]   Loaded " << embeddings.size() << " embeddings" << std::endl;
            } else {
                std::cout << "[select-alpha]   Warning: Failed to load embeddings" << std::endl;
            }
        }

        // Create GismaIndexBuilder
        std::shared_ptr<NetDag> netdag_ptr = nullptr;
        std::vector<double> ged_matrix;  // empty matrix

        GismaIndexBuilder builder(
            index_name, db_graphs, nameList, config.root_ind,
            alpha, config.tau_index, config.error_tolerance_index,
            netdag_ptr, false, false, ged_matrix, 0,
            config.max_exact_ged_for_EPT < 0 ? alpha + 4.0 : config.max_exact_ged_for_EPT,
            database, vM, eM, max_db_n, embeddings, config.enable_friends_reassign
        );

        // Net-Tree statistics: track anchor creation history
        struct AnchorCreationInfo {
            int anchor_node_id;
            double r_k_created;
            int parent_anchor_node_id;  // -1 indicates initial anchor
        };
        std::vector<AnchorCreationInfo> anchor_creation_history;
        std::set<int> known_anchor_ids;  // known anchor ids

        // Set callback to collect statistics
        builder.set_cluster_stats_callback(
            [&alpha_stats, &anchor_creation_history, &known_anchor_ids, &builder](
                double r_k, const std::vector<std::shared_ptr<Anchor>>& anchors, int phase) {
                ClusterSnapshot snapshot;
                snapshot.r_k_actual = r_k;
                snapshot.r_integer = static_cast<int>(std::floor(r_k));
                snapshot.phase = phase;
                snapshot.num_anchors = anchors.size();

                // Collect size and friends count for each cluster
                double total_nodes = 0;
                double max_size = 0;
                double total_friends = 0;
                double max_friends = 0;

                for (const auto& anchor : anchors) {
                    // Cluster size
                    int cluster_size = anchor->nodes_in_cluster.size();
                    snapshot.cluster_sizes.push_back(cluster_size);
                    total_nodes += cluster_size;
                    max_size = std::max(max_size, static_cast<double>(cluster_size));

                    // Friends count (within 3*r_k range friends)
                    // Note: friends list stores (distance, friend_id) pairs
                    int friends_in_3r = 0;
                    double threshold_3r = 3.0 * r_k;
                    for (const auto& friend_pair : anchor->friends) {
                        if (friend_pair.first <= threshold_3r) {
                            friends_in_3r++;
                        }
                    }
                    snapshot.friends_counts.push_back(friends_in_3r);
                    total_friends += friends_in_3r;
                    max_friends = std::max(max_friends, static_cast<double>(friends_in_3r));
                }

                snapshot.total_nodes_in_clusters = total_nodes;
                snapshot.avg_cluster_size = anchors.empty() ? 0.0 : total_nodes / anchors.size();
                snapshot.max_cluster_size = max_size;
                snapshot.avg_friends_count = anchors.empty() ? 0.0 : total_friends / anchors.size();
                snapshot.max_friends_count = max_friends;

                // Net-Tree statistics: detect new anchors and record creation info
                for (const auto& anchor : anchors) {
                    if (known_anchor_ids.find(anchor->node_id) == known_anchor_ids.end()) {
                        // This is a new anchor
                        known_anchor_ids.insert(anchor->node_id);

                        // Non-intrusive approach: use anchor->nearest_anchor as net-tree parent
                        // This field records the anchor ID of the cluster where the node resided before becoming an anchor
                        int parent_id = anchor->nearest_anchor;

                        // Record creation info
                        anchor_creation_history.push_back({
                            anchor->node_id,
                            r_k,
                            parent_id
                        });
                    }
                }

                // Net-Tree statistics: compute Net-Tree-Cover-Count for current integer r_k
                // New approach: determine parent-child relationship based on distance
                int current_r_int = snapshot.r_integer;
                double r_k_min = static_cast<double>(current_r_int);
                double r_k_max = 2.0 * current_r_int;

                // 1. Collect anchors created in [r_k, 2*r_k) interval
                std::vector<int> new_anchor_ids;
                for (const auto& info : anchor_creation_history) {
                    if (info.r_k_created >= r_k_min && info.r_k_created < r_k_max) {
                        new_anchor_ids.push_back(info.anchor_node_id);
                    }
                }

                // 2. Collect old anchors created at 2*r_k or above
                std::vector<int> old_anchor_ids;
                for (const auto& info : anchor_creation_history) {
                    if (info.r_k_created >= r_k_max) {
                        old_anchor_ids.push_back(info.anchor_node_id);
                    }
                }

                // 3. For each new anchor, find the nearest old anchor
                std::map<int, int> parent_to_child_count;  // parent_id -> number of child anchors produced
                int total_new_anchors = new_anchor_ids.size();

                for (int new_anchor_id : new_anchor_ids) {
                    if (old_anchor_ids.empty()) {
                        // no old anchors, skip
                        continue;
                    }

                    // Get node of new anchor
                    auto new_anchor = builder.nodes[new_anchor_id];
                    if (!new_anchor || new_anchor->embedding.empty()) {
                        continue;
                    }

                    // Find nearest old anchor
                    int nearest_old_anchor_id = -1;
                    double min_distance = std::numeric_limits<double>::infinity();

                    for (int old_anchor_id : old_anchor_ids) {
                        auto old_anchor = builder.nodes[old_anchor_id];
                        if (!old_anchor || old_anchor->embedding.empty()) {
                            continue;
                        }

                        // Compute embedding space distance (Euclidean distance)
                        double dist = 0.0;
                        for (size_t i = 0; i < new_anchor->embedding.size(); ++i) {
                            double diff = new_anchor->embedding[i] - old_anchor->embedding[i];
                            dist += diff * diff;
                        }
                        dist = std::sqrt(dist);

                        if (dist < min_distance) {
                            min_distance = dist;
                            nearest_old_anchor_id = old_anchor_id;
                        }
                    }

                    if (nearest_old_anchor_id != -1) {
                        parent_to_child_count[nearest_old_anchor_id]++;
                    }
                }

                // Compute statistics
                double avg_net_tree_cover = 0.0;
                double max_net_tree_cover = 0.0;

                int total_children = 0;
                for (const auto& pair : parent_to_child_count) {
                    max_net_tree_cover = std::max(max_net_tree_cover, static_cast<double>(pair.second));
                    total_children += pair.second;
                }

                // Average over all old anchors (including those with no children)
                if (!old_anchor_ids.empty()) {
                    avg_net_tree_cover = static_cast<double>(total_children) / old_anchor_ids.size();
                }

                // Debug output for specific r values
                if (current_r_int == 20 || current_r_int == 30 || current_r_int == 40) {
                    std::cout << "\n[DEBUG] Net-Tree Cover for r=" << current_r_int << " (distance-based method, averaged over all old anchors):\n";
                    std::cout << "  new anchor interval: [" << r_k_min << ", " << r_k_max << ")\n";
                    std::cout << "  old anchor interval: [" << r_k_max << ", +∞)\n";
                    std::cout << "  number of new anchors: " << new_anchor_ids.size() << "\n";
                    std::cout << "  number of old anchors (denominator): " << old_anchor_ids.size() << "\n";
                    std::cout << "  number of old anchors with children: " << parent_to_child_count.size() << "\n";
                    std::cout << "  number of old anchors without children: " << (old_anchor_ids.size() - parent_to_child_count.size()) << "\n";
                    std::cout << "  total children count (numerator): " << total_children << "\n";
                    std::cout << "  average children per old anchor: " << avg_net_tree_cover << "\n";
                    std::cout << "  old anchor with most children: " << max_net_tree_cover << "\n";

                    // Show distribution
                    std::map<int, int> child_count_distribution;
                    for (const auto& pair : parent_to_child_count) {
                        child_count_distribution[pair.second]++;
                    }
                    std::cout << "  Children count distribution:\n";
                    for (const auto& dist_pair : child_count_distribution) {
                        std::cout << "    " << dist_pair.second << " parents have " << dist_pair.first << " children\n";
                    }
                    std::cout << std::endl;
                }

                snapshot.avg_net_tree_cover = avg_net_tree_cover;
                snapshot.max_net_tree_cover = max_net_tree_cover;

                alpha_stats.snapshots.push_back(snapshot);

                std::cout << "[select-alpha]   r_k crossed integer " << snapshot.r_integer
                          << " (r_k=" << r_k << ", phase=" << phase
                          << ", anchors=" << anchors.size()
                          << ", avg_cluster=" << snapshot.avg_cluster_size << ")" << std::endl;
            }
        );

        // Build NetDag (using net_tree_const instead of overall_NetDag_const)
        std::cout << "[select-alpha] Building NetDag..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        builder.net_tree_const();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Compute phi (number of layers) - find the maximum r_k value from snapshots
        if (!alpha_stats.snapshots.empty()) {
            // find the maximum r_k_actual (excluding INF)
            double max_r_k = 0;
            for (const auto& snap : alpha_stats.snapshots) {
                if (snap.r_k_actual < 1e6 && snap.r_k_actual > max_r_k) {  // exclude INF (1e7)
                    max_r_k = snap.r_k_actual;
                }
            }
            if (max_r_k > alpha) {
                alpha_stats.phi = static_cast<int>(std::log2(max_r_k / alpha)) + 1;
            } else {
                alpha_stats.phi = 0;
            }
            std::cout << "[select-alpha] NetDag built successfully. Layers (phi): " << alpha_stats.phi
                      << " (max_r_k: " << max_r_k << ")" << std::endl;
        } else {
            alpha_stats.phi = 0;
            std::cout << "[select-alpha] NetDag built but no snapshots collected." << std::endl;
        }

        std::cout << "[select-alpha] Construction time: " << duration.count() << " ms" << std::endl;
        std::cout << "[select-alpha] Collected " << alpha_stats.snapshots.size() << " snapshots" << std::endl;

        all_alpha_stats.push_back(alpha_stats);
    }

    // 3. Save to CSV
    std::cout << "\n[select-alpha] Step 3: Saving statistics to CSV..." << std::endl;

    // 4.1 Save detailed CSV (including size of each cluster)
    int alpha_int = static_cast<int>(alpha);
    std::string detailed_csv_path = output_dir + "/alpha_statistics_detailed_alpha" + std::to_string(alpha_int) + ".csv";
    std::ofstream detailed_csv(detailed_csv_path);

    if (!detailed_csv.is_open()) {
        std::cerr << "[select-alpha] Error: Cannot create CSV file: " << detailed_csv_path << std::endl;
        return;
    }

    // Detailed CSV header
    detailed_csv << "alpha,tau_index,phi,r_integer,r_k_actual,phase,num_anchors,cluster_sizes,friends_counts\n";

    for (const auto& alpha_stat : all_alpha_stats) {
        for (const auto& snapshot : alpha_stat.snapshots) {
            detailed_csv << alpha_stat.alpha << ","
                        << alpha_stat.tau_index << ","
                        << alpha_stat.phi << ","
                        << snapshot.r_integer << ","
                        << snapshot.r_k_actual << ","
                        << snapshot.phase << ","
                        << snapshot.num_anchors << ",\"";

            // Output all cluster sizes, separated by semicolons
            for (size_t i = 0; i < snapshot.cluster_sizes.size(); i++) {
                detailed_csv << snapshot.cluster_sizes[i];
                if (i < snapshot.cluster_sizes.size() - 1) {
                    detailed_csv << ";";
                }
            }
            detailed_csv << "\",\"";

            // Output all friends counts, separated by semicolons
            for (size_t i = 0; i < snapshot.friends_counts.size(); i++) {
                detailed_csv << snapshot.friends_counts[i];
                if (i < snapshot.friends_counts.size() - 1) {
                    detailed_csv << ";";
                }
            }
            detailed_csv << "\"\n";
        }
    }

    detailed_csv.close();
    std::cout << "[select-alpha] Detailed statistics saved to: " << detailed_csv_path << std::endl;

    // 4.2 Save summary CSV (statistics only)
    std::string summary_csv_path = output_dir + "/alpha_statistics_summary_alpha" + std::to_string(alpha_int) + ".csv";
    std::ofstream summary_csv(summary_csv_path);

    if (!summary_csv.is_open()) {
        std::cerr << "[select-alpha] Error: Cannot create summary CSV file: " << summary_csv_path << std::endl;
        return;
    }

    // Summary CSV header
    summary_csv << "alpha,tau_index,phi,r_integer,r_k_actual,phase,num_anchors,avg_cluster_size,max_cluster_size,total_nodes,avg_friends_count,max_friends_count,avg_net_tree_cover,max_net_tree_cover\n";

    for (const auto& alpha_stat : all_alpha_stats) {
        for (const auto& snapshot : alpha_stat.snapshots) {
            summary_csv << alpha_stat.alpha << ","
                       << alpha_stat.tau_index << ","
                       << alpha_stat.phi << ","
                       << snapshot.r_integer << ","
                       << snapshot.r_k_actual << ","
                       << snapshot.phase << ","
                       << snapshot.num_anchors << ","
                       << snapshot.avg_cluster_size << ","
                       << snapshot.max_cluster_size << ","
                       << snapshot.total_nodes_in_clusters << ","
                       << snapshot.avg_friends_count << ","
                       << snapshot.max_friends_count << ","
                       << snapshot.avg_net_tree_cover << ","
                       << snapshot.max_net_tree_cover << "\n";
        }
    }

    summary_csv.close();
    std::cout << "[select-alpha] Summary statistics saved to: " << summary_csv_path << std::endl;

    // Raw pointers in db are now managed by shared_ptr in db_graphs, no manual delete needed

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "[select-alpha] Completed successfully!" << std::endl;
    std::cout << "[select-alpha] Output files:" << std::endl;
    std::cout << "[select-alpha]   - Detailed CSV (with cluster sizes): " << detailed_csv_path << std::endl;
    std::cout << "[select-alpha]   - Summary CSV (statistics only): " << summary_csv_path << std::endl;
    std::cout << "[select-alpha] Next step: Run visualization script:" << std::endl;
    std::cout << "[select-alpha]   python scripts/analysis/visualize_alpha_selection.py \\" << std::endl;
    std::cout << "[select-alpha]     --csv " << summary_csv_path << " \\" << std::endl;
    std::cout << "[select-alpha]     --output_dir " << output_dir << "/plots" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void export_candidates_mode(const Config& config, const std::string& candidates_output_file) {
    printf("[export-candidates] Starting candidate export mode...\n");
    
    // Auto-generate output path and filename
    std::string output_dir = "../GED-via-Optimal-Transport/candidates/" + config.dataset;
    std::string auto_output_file = output_dir + "/candidates_" + config.dataset + "_tau" + std::to_string((int)config.tau_search) + ".csv";
    
    // Check if using default candidates_output or user specified one
    std::string default_path = "../GED-via-Optimal-Transport/candidates/candidates.txt";
    bool use_auto_path = candidates_output_file.empty() || candidates_output_file == default_path;
    std::string actual_output_file = use_auto_path ? auto_output_file : candidates_output_file;
    
    printf("[export-candidates] Output file: %s\n", actual_output_file.c_str());
    
    // Initialize base data
    std::map<std::string, ui> vM, eM;
    std::vector<Graph*> db;
    std::vector<Graph*> query_db;
    
    // Get parameters
    int q_start = config.q_start;
    int q_end = config.q_end;
    int tau_search = config.tau_search;
    
    // Build dataset path
    std::string db_name = "datasets/" + config.dataset + "/db.txt";
    std::string query_name = "datasets/" + config.dataset + "/queries.txt";
    
    printf("[export-candidates] Dataset: %s, tau_search: %d\n", config.dataset.c_str(), tau_search);
    
    // Load database
    ui max_db_n = Utility::load_db(db_name.c_str(), db, vM, eM);
    printf("[export-candidates] Database loaded: %zu graphs\n", db.size());
    
    // Load query graphs
    ui max_query_n = Utility::load_db(query_name.c_str(), query_db, vM, eM);
    printf("[export-candidates] Query graphs loaded: %zu graphs\n", query_db.size());
    
    // Set query range
    if (q_start == -1) q_start = 0;
    if (q_end == -1) q_end = query_db.size() - 1;
    printf("[export-candidates] Processing queries from %d to %d\n", q_start, q_end);
    
    // Initialize auxiliary variables
    ui max_n = std::max(max_db_n, max_query_n);
    
    // Ensure output directory exists
    std::string actual_output_dir = actual_output_file.substr(0, actual_output_file.find_last_of("/\\"));
    if (!actual_output_dir.empty()) {
        // Cross-platform mkdir command
#ifdef _WIN32
        std::string mkdir_command = "mkdir \"" + actual_output_dir + "\" 2>nul";
        // Replace forward slashes with backslashes for Windows
        for (size_t i = 0; i < mkdir_command.length(); i++) {
            if (mkdir_command[i] == '/') mkdir_command[i] = '\\';
        }
#else
        std::string mkdir_command = "mkdir -p \"" + actual_output_dir + "\"";
#endif
        system(mkdir_command.c_str());
        printf("[export-candidates] Created output directory: %s\n", actual_output_dir.c_str());
    }
    
    // Clear output file and write CSV header
    std::ofstream outfile(actual_output_file);
    if (!outfile.is_open()) {
        printf("[ERROR] Cannot create output file: %s\n", actual_output_file.c_str());
        return;
    }
    outfile << "query_id,tau_search,candidate_ids\n";
    outfile.close();
    
    printf("[export-candidates] Starting candidate export to: %s\n", actual_output_file.c_str());
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Export candidates for each query
    for (int i = q_start; i <= q_end; ++i) {
        Graph* query_graph = query_db[i];
        std::vector<int> candidates;
        
        auto filter_start = std::chrono::high_resolution_clock::now();
        
        // BMao lower bound filtering
        for (size_t node_id = 0; node_id < db.size(); ++node_id) {
            Graph* db_graph = db[node_id];
            
            ui lb = query_graph->ged_lower_bound_filter(
                db_graph, static_cast<ui>(tau_search), vM.size(), eM.size(), max_n);
            
            if (lb <= tau_search) {
                candidates.push_back(static_cast<int>(node_id));
            }
        }
        
        auto filter_end = std::chrono::high_resolution_clock::now();
        double filter_time = std::chrono::duration<double, std::milli>(filter_end - filter_start).count();
        
        printf("[TIMING] Query %d BMao_Filtering: %.2fms, Candidates: %d/%d (%.2f%% reduction)\n", 
               i, filter_time, (int)candidates.size(), (int)db.size(), 
               (1.0 - (double)candidates.size() / db.size()) * 100.0);
        
        // Write to file (including tau_search)
        std::ofstream outfile(actual_output_file, std::ios::app);
        outfile << i << "," << tau_search << ",";
        for (size_t j = 0; j < candidates.size(); ++j) {
            outfile << candidates[j];
            if (j < candidates.size() - 1) outfile << ";";
        }
        outfile << "\n";
        outfile.close();
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    printf("[export-candidates] Export completed!\n");
    printf("[TIMING] Total export time: %.2fms for %d queries\n", total_time, q_end - q_start + 1);
    printf("[export-candidates] Results saved to: %s\n", actual_output_file.c_str());
    
    // Clean up memory
    for (auto* graph : db) {
        delete graph;
    }
    for (auto* graph : query_db) {
        delete graph;
    }
}

// ======================== construct_from mode ========================
// Continue construction from existing (old_alpha, tau) NetDag to (new_alpha, tau)
// Use full-scan mode (no friends list), suitable when memory is insufficient with small old_alpha
// Example: continue building alpha=2 from alpha=4 NetDag
void construct_from_mode(Config& config) {
    std::cout << "=== Construct From Mode ===" << std::endl;

    double new_alpha = config.alpha;
    double old_alpha = config.old_alpha;
    double tau = config.tau_index;
    double old_tau = (config.old_tau_index > 0) ? config.old_tau_index : tau;  // old NetDag's tau, defaults to same as new tau
    double error_tolerance = config.error_tolerance_index;

    if (old_alpha <= 0) {
        std::cerr << "Error: --old_alpha must be specified and > 0" << std::endl;
        return;
    }
    if (new_alpha >= old_alpha) {
        std::cerr << "Error: --alpha (" << new_alpha << ") must be < --old_alpha (" << old_alpha << ")" << std::endl;
        return;
    }

    std::cout << "old_alpha=" << old_alpha << ", old_tau=" << old_tau
              << ", new_alpha=" << new_alpha << ", new_tau=" << tau
              << ", error=" << error_tolerance << std::endl;

    // 1. Load database graphs
    std::string database = config.db_name;
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(database.c_str(), db, vM, eM);
    std::cout << "Loaded " << db.size() << " graphs" << std::endl;

    std::string old_index_name = Utility::get_index_name(config.dataset, old_alpha, old_tau, error_tolerance, db.size());
    std::string new_index_name = Utility::get_index_name(config.dataset, new_alpha, tau, error_tolerance, db.size());
    std::string dataset_dir = "./NetDags/" + config.dataset + "/";
    std::string configs_dir = dataset_dir + "configs/";
    std::string reassigned_dir = dataset_dir + "reassigned/";
    std::string old_file = reassigned_dir + old_index_name + ".dat";
    // Save to configs/ directory using standard index_name (no _from suffix)
    // So subsequent compute_paths -> reassign -> construct_EPF pipeline can proceed directly
    std::string new_file = configs_dir + new_index_name + ".dat";

    std::filesystem::create_directories(configs_dir);

    if (std::filesystem::exists(new_file)) {
        std::cerr << "Warning: " << new_file << " already exists. Overwriting." << std::endl;
    }

    std::cout << "Old NetDag: " << old_file << std::endl;
    std::cout << "New NetDag (configs): " << new_file << std::endl;

    // 2. Load old NetDag
    auto netdag_ptr = std::make_shared<NetDag>();
    std::cout << "Loading old NetDag..." << std::endl;
    NetDag::load_from_file(*netdag_ptr, old_file);
    std::cout << "Loaded NetDag: " << netdag_ptr->anchors.size() << " anchors, "
              << netdag_ptr->nodes.size() << " nodes" << std::endl;

    // 3. Load embeddings
    std::string embedding_file = "./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin";
    std::vector<std::vector<float>> embeddings;
    if (!Utility::load_embeddings(embedding_file, embeddings)) {
        std::cerr << "Error: Failed to load embeddings from " << embedding_file << std::endl;
        for (auto* g : db) delete g;
        return;
    }
    std::cout << "Loaded " << embeddings.size() << " embeddings" << std::endl;

    // Assign embeddings to nodes
    for (size_t i = 0; i < embeddings.size() && i < netdag_ptr->nodes.size(); ++i) {
        if (netdag_ptr->nodes[i]) {
            netdag_ptr->nodes[i]->embedding = embeddings[i];
        }
    }
    std::cout << "Embeddings assigned to nodes" << std::endl;

    // 3.5 Reclaim exact_cluster nodes into nodes_in_cluster for the new construction loop
    // Note: exact_cluster stores GED distance, nodes_in_cluster uses embedding distance
    // so embedding distance needs to be recomputed
    std::cout << "Moving exact_cluster nodes back to nodes_in_cluster..." << std::endl;
    int exact_moved = 0;
    for (const auto& anchor : netdag_ptr->anchors) {
        while (!anchor->nodes_in_exact_cluster.empty()) {
            auto ecn = anchor->nodes_in_exact_cluster.top();
            anchor->nodes_in_exact_cluster.pop();
            // Recompute embedding distance
            double emb_dist = Utility::euclidean_distance(
                anchor->embedding, netdag_ptr->nodes[ecn.node_id]->embedding);
            anchor->nodes_in_cluster.push(std::make_pair(emb_dist, ecn.node_id));
            exact_moved++;
        }
    }
    std::cout << "Moved " << exact_moved << " exact_cluster nodes to nodes_in_cluster" << std::endl;

    // 4. Build new phaseList
    // Find current max_dist (maximum among all nodes_in_cluster heap tops)
    double max_dist = -1.0;
    std::shared_ptr<Node> max_node = nullptr;
    int max_anchor_node_id = -1;

    for (const auto& anchor : netdag_ptr->anchors) {
        if (!anchor->nodes_in_cluster.empty()) {
            auto top = anchor->nodes_in_cluster.top();
            if (top.first > max_dist) {
                max_dist = top.first;
                max_node = netdag_ptr->nodes[top.second];
                max_anchor_node_id = anchor->node_id;
            }
        }
    }

    std::cout << "Initial max_dist = " << max_dist << std::endl;
    if (max_dist <= new_alpha) {
        std::cout << "max_dist <= new_alpha, no new anchors needed. Saving copy with updated alpha." << std::endl;
        netdag_ptr->alpha = new_alpha;
        netdag_ptr->file_signature = new_index_name;
        netdag_ptr->save_to_file(new_file);
        for (auto* g : db) delete g;
        return;
    }

    // Generate phaseList (descending): [..., 4*new_alpha, 2*new_alpha, new_alpha]
    std::vector<double> phaseList;
    {
        double state = new_alpha;
        phaseList.push_back(state);
        while (state < max_dist) {
            state *= 2;
            phaseList.push_back(state);
        }
        std::reverse(phaseList.begin(), phaseList.end());
    }
    std::cout << "phaseList: ";
    for (double p : phaseList) std::cout << p << " ";
    std::cout << std::endl;

    // Update phase
    double r_k = max_dist;
    int phase = 0;
    auto update_phase_fn = [&]() {
        auto it = std::upper_bound(phaseList.begin(), phaseList.end(), max_dist,
            [](double value, double element) { return value > element; });
        if (it == phaseList.begin()) {
            phase = static_cast<int>(phaseList.front());
        } else {
            phase = static_cast<int>(*(it - 1));
        }
    };
    update_phase_fn();
    std::cout << "Initial phase = " << phase << std::endl;

    // 5. Main construction loop (full-scan mode)
    bool use_parallel = config.use_parallel;
    unsigned int num_threads_cfg = 0;
    if (use_parallel) {
        unsigned int max_cores = std::thread::hardware_concurrency();
        if (max_cores == 0) max_cores = 1;
        num_threads_cfg = config.num_workers > 0
            ? std::min(static_cast<unsigned int>(config.num_workers), max_cores)
            : max_cores;
        std::cout << "Parallel mode: " << num_threads_cfg << " threads" << std::endl;
    }

    int new_anchors_count = 0;
    double timing_find = 0, timing_promote = 0, timing_steal = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (max_dist > new_alpha) {
        int phase_before = phase;

        // find_max_dist: scan all anchor nodes_in_cluster heap tops (serial, fast)
        auto t0 = std::chrono::high_resolution_clock::now();
        max_dist = -1.0;
        max_node = nullptr;
        max_anchor_node_id = -1;
        for (const auto& anchor : netdag_ptr->anchors) {
            if (!anchor->nodes_in_cluster.empty()) {
                auto top = anchor->nodes_in_cluster.top();
                if (top.first > max_dist) {
                    max_dist = top.first;
                    max_node = netdag_ptr->nodes[top.second];
                    max_anchor_node_id = anchor->node_id;
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        timing_find += std::chrono::duration<double>(t1 - t0).count();

        if (max_dist <= new_alpha) break;

        // update_phase
        r_k = max_dist;
        update_phase_fn();

        if (phase < phase_before) {
            std::cout << "Phase decreased from " << phase_before << " to " << phase << std::endl;

            netdag_ptr->parent_by_phase_dict[phase_before] = std::vector<int>();
            for (const auto& a : netdag_ptr->anchors) {
                netdag_ptr->parent_by_phase_dict[phase_before].push_back(a->node_id);
            }
            netdag_ptr->parent_by_phase_dict[phase] = std::vector<int>();
            for (const auto& a : netdag_ptr->anchors) {
                netdag_ptr->parent_by_phase_dict[phase].push_back(a->node_id);
            }
        }

        // update_one_anchor (full-scan version)
        if (!max_node) {
            std::cerr << "Error: max_node is nullptr!" << std::endl;
            break;
        }

        // Promote max_node to new Anchor
        auto existing_anchor = std::dynamic_pointer_cast<Anchor>(max_node);
        if (!existing_anchor) {
            auto new_anchor = std::make_shared<Anchor>(
                max_node->node_id,
                max_node->graph,
                max_node->file_name,
                max_node->nearest_anchor,
                max_node->nearest_anchor_dist,
                max_node->embedding,
                static_cast<int>(netdag_ptr->anchors.size())
            );
            netdag_ptr->nodes[max_node->node_id] = new_anchor;
            netdag_ptr->anchors.push_back(new_anchor);
            existing_anchor = new_anchor;
        }

        // Remove from old anchor's cluster
        auto old_anchor = std::dynamic_pointer_cast<Anchor>(netdag_ptr->nodes[max_anchor_node_id]);
        if (old_anchor && old_anchor->node_id != existing_anchor->node_id) {
            std::priority_queue<std::pair<double, int>> new_cluster;
            while (!old_anchor->nodes_in_cluster.empty()) {
                auto p = old_anchor->nodes_in_cluster.top();
                old_anchor->nodes_in_cluster.pop();
                if (p.second != existing_anchor->node_id) {
                    new_cluster.push(p);
                }
            }
            old_anchor->nodes_in_cluster = std::move(new_cluster);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        timing_promote += std::chrono::duration<double>(t2 - t1).count();

        // Full scan: steal nodes
        double reassign_threshold = 2 * r_k + 3 * error_tolerance;
        int new_anchor_id = existing_anchor->node_id;
        const auto& new_anchor_emb = existing_anchor->embedding;
        int n_anchors = static_cast<int>(netdag_ptr->anchors.size());

        if (use_parallel && n_anchors > 1000) {
            // Parallel version: std::async dynamic task queue
            unsigned int n_threads = num_threads_cfg;
            std::vector<std::vector<std::pair<double, int>>> thread_stolen(n_threads);
            std::atomic<int> next_idx(0);

            std::vector<std::future<void>> futures;
            futures.reserve(n_threads);
            for (unsigned int t = 0; t < n_threads; ++t) {
                futures.emplace_back(std::async(std::launch::async,
                    [&, t]() {
                        auto& local_stolen = thread_stolen[t];
                        while (true) {
                            int i = next_idx.fetch_add(1, std::memory_order_relaxed);
                            if (i >= n_anchors) break;

                            auto& other_anchor = netdag_ptr->anchors[i];
                            if (other_anchor->node_id == new_anchor_id) continue;
                            if (other_anchor->embedding.empty()) continue;

                            double dist = Utility::euclidean_distance(new_anchor_emb, other_anchor->embedding);
                            if (dist > reassign_threshold) continue;

                            std::vector<std::pair<double, int>> waste_nodes;
                            while (!other_anchor->nodes_in_cluster.empty()) {
                                auto node_tuple = other_anchor->nodes_in_cluster.top();
                                double node_dist = node_tuple.first;
                                int node_id = node_tuple.second;

                                if (dist > 2 * node_dist + 3 * error_tolerance) break;

                                double dist_to_new = Utility::euclidean_distance(
                                    new_anchor_emb, netdag_ptr->nodes[node_id]->embedding);

                                other_anchor->nodes_in_cluster.pop();

                                if (dist_to_new <= node_dist) {
                                    netdag_ptr->nodes[node_id]->nearest_anchor = new_anchor_id;
                                    netdag_ptr->nodes[node_id]->nearest_anchor_dist = dist_to_new;
                                    local_stolen.emplace_back(dist_to_new, node_id);
                                } else {
                                    waste_nodes.push_back(node_tuple);
                                }
                            }
                            for (const auto& w : waste_nodes) {
                                other_anchor->nodes_in_cluster.push(w);
                            }
                        }
                    }
                ));
            }
            for (auto& f : futures) f.get();

            // Merge stolen nodes from all threads into new anchor
            for (auto& local : thread_stolen) {
                for (auto& p : local) {
                    existing_anchor->nodes_in_cluster.push(p);
                }
            }
        } else {
            // Serial version
            for (const auto& other_anchor : netdag_ptr->anchors) {
                if (other_anchor->node_id == new_anchor_id) continue;
                if (other_anchor->embedding.empty()) continue;

                double dist = Utility::euclidean_distance(new_anchor_emb, other_anchor->embedding);
                if (dist > reassign_threshold) continue;

                std::vector<std::pair<double, int>> waste_nodes;
                while (!other_anchor->nodes_in_cluster.empty()) {
                    auto node_tuple = other_anchor->nodes_in_cluster.top();
                    double node_dist = node_tuple.first;
                    int node_id = node_tuple.second;

                    if (dist > 2 * node_dist + 3 * error_tolerance) break;

                    double dist_to_new = Utility::euclidean_distance(
                        new_anchor_emb, netdag_ptr->nodes[node_id]->embedding);

                    other_anchor->nodes_in_cluster.pop();

                    if (dist_to_new <= node_dist) {
                        netdag_ptr->nodes[node_id]->nearest_anchor = new_anchor_id;
                        netdag_ptr->nodes[node_id]->nearest_anchor_dist = dist_to_new;
                        existing_anchor->nodes_in_cluster.push(std::make_pair(dist_to_new, node_id));
                    } else {
                        waste_nodes.push_back(node_tuple);
                    }
                }
                for (const auto& w : waste_nodes) {
                    other_anchor->nodes_in_cluster.push(w);
                }
            }
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        timing_steal += std::chrono::duration<double>(t3 - t2).count();

        new_anchors_count++;
        if (new_anchors_count % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            std::cout << "New anchors: " << new_anchors_count
                      << ", total anchors: " << netdag_ptr->anchors.size()
                      << ", max_dist: " << max_dist
                      << ", r_k: " << r_k
                      << ", elapsed: " << elapsed << "s"
                      << ", t_find=" << timing_find << "s"
                      << ", t_promote=" << timing_promote << "s"
                      << ", t_steal=" << timing_steal << "s" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double loop_seconds = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "\nConstruction loop completed in " << loop_seconds << "s" << std::endl;
    std::cout << "Added " << new_anchors_count << " new anchors" << std::endl;
    std::cout << "Total anchors: " << netdag_ptr->anchors.size() << std::endl;

    // 6. Update parent_by_phase_dict final state / clear hierarchy
    if (config.skip_hierarchy) {
        // --skip_hierarchy: clear all children and parent_by_phase_dict
        // Only keep anchor + cluster data for Base+SS search (GS not needed)
        std::cout << "\n--skip_hierarchy: Clearing all children and parent_by_phase_dict..." << std::endl;
        for (auto& anchor : netdag_ptr->anchors) {
            anchor->children.clear();
            anchor->friends.clear();
        }
        netdag_ptr->parent_by_phase_dict.clear();
        std::cout << "Hierarchy cleared. NetDag will only contain anchors + clusters." << std::endl;
    } else {
        // Ensure minimum phase contains all anchors
        int min_phase = static_cast<int>(new_alpha);
        if (netdag_ptr->parent_by_phase_dict.find(min_phase) == netdag_ptr->parent_by_phase_dict.end()) {
            netdag_ptr->parent_by_phase_dict[min_phase] = std::vector<int>();
        }
        netdag_ptr->parent_by_phase_dict[min_phase].clear();
        for (const auto& a : netdag_ptr->anchors) {
            netdag_ptr->parent_by_phase_dict[min_phase].push_back(a->node_id);
        }

        // 7. Rebuild all children edges (full scan)
        // Recompute children for all phases, ensuring new anchors are correctly linked
        std::cout << "\nRebuilding all children edges (full scan)..." << std::endl;
        int total_new_edges = 0;

        std::vector<int> phases_sorted;
        for (const auto& [ph, ids] : netdag_ptr->parent_by_phase_dict) {
            phases_sorted.push_back(ph);
        }
        std::sort(phases_sorted.begin(), phases_sorted.end());

        // Build anchor node_id -> anchor mapping
        std::unordered_map<int, std::shared_ptr<Anchor>> anchor_map;
        for (const auto& a : netdag_ptr->anchors) {
            anchor_map[a->node_id] = a;
        }

        for (int child_phase : phases_sorted) {
            int parent_phase = child_phase * 2;
            auto it = netdag_ptr->parent_by_phase_dict.find(parent_phase);
            if (it == netdag_ptr->parent_by_phase_dict.end()) continue;

            const auto& parent_ids = it->second;

            // r_k estimate: use child_phase + 1 (conservative estimate)
            double r_k_est = child_phase + 1.0;
            double child_range = 3.0 * r_k_est + 2.0 * tau + 4.0 * error_tolerance;

            int phase_new_edges = 0;

            for (int parent_id : parent_ids) {
                auto parent_it = anchor_map.find(parent_id);
                if (parent_it == anchor_map.end()) continue;
                auto& parent_anchor = parent_it->second;
                if (parent_anchor->embedding.empty()) continue;

                // Collect existing children
                std::unordered_set<int> existing_children;
                auto children_it = parent_anchor->children.find(child_phase);
                if (children_it != parent_anchor->children.end()) {
                    for (const auto& [cid, cdist] : children_it->second) {
                        existing_children.insert(cid);
                    }
                }

                // Full scan all anchors as candidate children
                for (const auto& candidate : netdag_ptr->anchors) {
                    if (existing_children.count(candidate->node_id)) continue;
                    if (candidate->embedding.empty()) continue;

                    double dist = Utility::euclidean_distance(parent_anchor->embedding, candidate->embedding);
                    if (dist <= child_range) {
                        parent_anchor->children[child_phase].emplace_back(candidate->node_id, dist);
                        phase_new_edges++;
                    }
                }
            }

            total_new_edges += phase_new_edges;
            if (phase_new_edges > 0) {
                std::cout << "Phase " << child_phase << " (parent_phase=" << parent_phase
                          << "): parents=" << parent_ids.size()
                          << ", child_range=" << child_range
                          << ", new_edges=" << phase_new_edges << std::endl;
            }
        }
        std::cout << "Total new children edges: " << total_new_edges << std::endl;
    }

    // 8. Update metadata and save
    netdag_ptr->alpha = new_alpha;
    netdag_ptr->tau = tau;
    netdag_ptr->file_signature = new_index_name;

    std::cout << "Saving new NetDag to " << new_file << "..." << std::endl;
    std::filesystem::create_directories(configs_dir);
    netdag_ptr->save_to_file(new_file);
    std::cout << "Saved successfully" << std::endl;

    // 9. Post-pipeline instructions
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "  Step 2: ./build/GismaProject -m compute_paths -s " << config.dataset
              << " --alpha " << new_alpha << " --tau_index " << tau << std::endl;
    std::cout << "  Step 3: ./build/GismaProject -m reassign -s " << config.dataset
              << " --alpha " << new_alpha << " --tau_index " << tau << std::endl;
    std::cout << "  Step 4: ./build/GismaProject -m construct_EPF -s " << config.dataset
              << " --alpha " << new_alpha << " --tau_index " << tau << std::endl;

    std::cout << "\n=== Construct From completed ===" << std::endl;

    for (auto* g : db) delete g;
}

// ======================== upgrade_tau mode ========================
// Upgrade existing (alpha, old_tau) NetDag to (alpha, new_tau)
// Only modify children edges (expand child_range), without changing anchor/cluster/EPT
void upgrade_tau_mode(Config& config) {
    std::cout << "=== Upgrade Tau Mode ===" << std::endl;

    double alpha = config.alpha;
    double new_tau = config.tau_index;
    double old_tau = config.old_tau_index;
    double error_tolerance_index = config.error_tolerance_index;

    if (old_tau <= 0) {
        std::cerr << "Error: --old_tau_index must be specified and > 0" << std::endl;
        return;
    }
    if (new_tau <= old_tau) {
        std::cerr << "Error: --tau_index (" << new_tau << ") must be > --old_tau_index (" << old_tau << ")" << std::endl;
        return;
    }

    std::cout << "alpha=" << alpha << ", old_tau=" << old_tau << ", new_tau=" << new_tau << std::endl;

    // 1. Load database graphs (only need db.size() to build index_name)
    std::string database = config.db_name;
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;
    ui max_db_n = Utility::load_db(database.c_str(), db, vM, eM);
    std::cout << "Loaded " << db.size() << " graphs" << std::endl;

    std::string old_index_name = Utility::get_index_name(config.dataset, alpha, old_tau, error_tolerance_index, db.size());
    std::string new_index_name = Utility::get_index_name(config.dataset, alpha, new_tau, error_tolerance_index, db.size());

    std::string dataset_dir = "./NetDags/" + config.dataset + "/";
    std::string reassigned_dir = dataset_dir + "reassigned/";
    std::string old_file = reassigned_dir + old_index_name + ".dat";
    std::string new_file = reassigned_dir + new_index_name + ".dat";

    // Check if new file already exists
    if (std::filesystem::exists(new_file)) {
        std::cerr << "Warning: " << new_file << " already exists. Overwriting." << std::endl;
    }

    std::cout << "Old NetDag: " << old_file << std::endl;
    std::cout << "New NetDag: " << new_file << std::endl;

    // 2. Load old NetDag
    auto netdag_ptr = std::make_shared<NetDag>();
    std::cout << "Loading old NetDag..." << std::endl;
    NetDag::load_from_file(*netdag_ptr, old_file);
    std::cout << "Loaded NetDag: " << netdag_ptr->anchors.size() << " anchors, "
              << netdag_ptr->nodes.size() << " nodes" << std::endl;

    // 3. Load embeddings
    std::string embedding_file = "./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin";
    std::vector<std::vector<float>> embeddings;
    if (!Utility::load_embeddings(embedding_file, embeddings)) {
        std::cerr << "Error: Failed to load embeddings from " << embedding_file << std::endl;
        for (auto* g : db) delete g;
        return;
    }
    std::cout << "Loaded " << embeddings.size() << " embeddings" << std::endl;

    // Assign embeddings to nodes
    for (size_t i = 0; i < embeddings.size() && i < netdag_ptr->nodes.size(); ++i) {
        if (netdag_ptr->nodes[i]) {
            netdag_ptr->nodes[i]->embedding = embeddings[i];
        }
    }
    std::cout << "Embeddings assigned to nodes" << std::endl;

    // 4. Iterate each phase, expand children
    int total_new_edges = 0;
    std::vector<int> phases_sorted;
    for (const auto& [phase, anchor_ids] : netdag_ptr->parent_by_phase_dict) {
        phases_sorted.push_back(phase);
    }
    std::sort(phases_sorted.begin(), phases_sorted.end());

    std::cout << "\nPhases in parent_by_phase_dict: ";
    for (int p : phases_sorted) std::cout << p << " ";
    std::cout << std::endl;

    // Build anchor node_id -> anchor index mapping
    std::unordered_map<int, int> anchor_id_to_idx;
    for (size_t i = 0; i < netdag_ptr->anchors.size(); ++i) {
        anchor_id_to_idx[netdag_ptr->anchors[i]->node_id] = i;
    }

    for (int child_phase : phases_sorted) {
        int parent_phase = child_phase * 2;

        // Find parent anchors for the given parent_phase
        auto it = netdag_ptr->parent_by_phase_dict.find(parent_phase);
        if (it == netdag_ptr->parent_by_phase_dict.end()) {
            continue;
        }

        const auto& parent_anchor_ids = it->second;

        // r_k estimate: r_k corresponding to phase ≈ child_phase + 1（conservative estimate）
        double r_k_est = child_phase + 1.0;
        double new_child_range = 3.0 * r_k_est + 2.0 * new_tau + 4.0 * error_tolerance_index;
        double old_child_range = 3.0 * r_k_est + 2.0 * old_tau + 4.0 * error_tolerance_index;

        int phase_new_edges = 0;

        for (int parent_id : parent_anchor_ids) {
            auto parent_node = netdag_ptr->nodes[parent_id];
            auto parent_anchor = std::dynamic_pointer_cast<Anchor>(parent_node);
            if (!parent_anchor || parent_anchor->embedding.empty()) continue;

            // Get existing child IDs from current children[child_phase]
            std::unordered_set<int> existing_child_ids;
            auto children_it = parent_anchor->children.find(child_phase);
            if (children_it != parent_anchor->children.end()) {
                for (const auto& [cid, cdist] : children_it->second) {
                    existing_child_ids.insert(cid);
                }
            }

            // Iterate all anchors as candidate new children
            for (const auto& candidate_anchor : netdag_ptr->anchors) {
                int cand_id = candidate_anchor->node_id;
                if (existing_child_ids.count(cand_id)) continue;
                if (candidate_anchor->embedding.empty()) continue;

                double dist = Utility::euclidean_distance(parent_anchor->embedding, candidate_anchor->embedding);
                if (dist <= new_child_range) {
                    parent_anchor->children[child_phase].emplace_back(cand_id, dist);
                    phase_new_edges++;
                }
            }
        }

        total_new_edges += phase_new_edges;
        if (phase_new_edges > 0) {
            std::cout << "Phase " << child_phase << " (parent_phase=" << parent_phase
                      << "): parents=" << parent_anchor_ids.size()
                      << ", old_range=" << old_child_range
                      << ", new_range=" << new_child_range
                      << ", new_edges=" << phase_new_edges << std::endl;
        }
    }

    std::cout << "\nTotal new edges added: " << total_new_edges << std::endl;

    // 5. Update tau
    netdag_ptr->tau = new_tau;

    // 6. Save new NetDag
    std::cout << "Saving new NetDag to " << new_file << "..." << std::endl;
    netdag_ptr->save_to_file(new_file);
    std::cout << "Saved successfully" << std::endl;

    // 7. Create EPF symlink
    std::string old_epf_dir = "./EPFs/" + old_index_name + "/";
    std::string new_epf_dir = "./EPFs/" + new_index_name + "/";
    if (!std::filesystem::exists(new_epf_dir)) {
        std::error_code ec;
        std::filesystem::create_symlink(
            std::filesystem::absolute(old_epf_dir),
            new_epf_dir, ec);
        if (ec) {
            std::cout << "Note: Could not create symlink (" << ec.message()
                      << "). Copy or link EPFs manually:" << std::endl;
            std::cout << "  ln -s " << old_epf_dir << " " << new_epf_dir << std::endl;
        } else {
            std::cout << "EPF symlink created: " << new_epf_dir << " -> " << old_epf_dir << std::endl;
        }
    } else {
        std::cout << "EPF directory already exists: " << new_epf_dir << std::endl;
    }

    std::cout << "\n=== Upgrade completed ===" << std::endl;
    std::cout << "To test: ./build/GismaProject -m experiment -s " << config.dataset
              << " --alpha " << alpha << " --tau_index " << new_tau
              << " --tau_values \"8\" --methods \"Gisma\" --use_parallel" << std::endl;

    // Cleanup
    for (auto* g : db) delete g;
}

int main(int argc, char* argv[]) {
    srand(time(0));
    Config config;

    // Use POPL for command line parsing
    popl::OptionParser op("Allowed options");

    // Common options
    auto help_option = op.add<popl::Switch>("h", "help", "Produce help message");
    auto mode_option = op.add<popl::Value<std::string>>("m", "mode", "Mode (construct, construct_ND, construct_EPF, search, export-candidates, info, etc.)", config.mode);
    auto dataset_option = op.add<popl::Value<std::string>>("s", "dataset", "Dataset name (AIDS, Chemical1M, etc.)", config.dataset);
    auto db_name_option = op.add<popl::Value<std::string>>("d", "db_name", "Database file name", config.db_name);
    auto ground_truth_path_option = op.add<popl::Value<std::string>>("", "ground_truth_path", "Path to ground truth data", config.ground_truth_path);
    auto candidates_output_option = op.add<popl::Value<std::string>>("", "candidates_output", "Output file for candidate export", "../GED-via-Optimal-Transport/candidates/candidates.txt");

    // Additional search_method option
    // search_method removed from CLI (always Gisma in search mode)
    // Compute GED mode options
    auto query_file_option = op.add<popl::Value<std::string>>("", "query_file", "Query graph file for compute_ged mode", config.query_file);
    auto target_file_option = op.add<popl::Value<std::string>>("", "target_file", "Target graph file for compute_ged mode", config.target_file);
    auto ged_algorithm_option = op.add<popl::Value<std::string>>("", "ged_algorithm", "GED algorithm (App/AppForComputation, AStar/AStarForComputation, default)", config.ged_algorithm);

    // Construction mode options
    auto alpha_option = op.add<popl::Value<double>>("", "alpha", "alpha value for index", config.alpha);
    auto tau_index_option = op.add<popl::Value<double>>("", "tau_index", "Tau index value for index", config.tau_index);
    auto error_tolerance_index_option = op.add<popl::Value<double>>("", "error_tolerance_index", "Error tolerance index for index", config.error_tolerance_index);
    auto old_tau_index_option = op.add<popl::Value<double>>("", "old_tau_index", "Old tau_index for upgrade_tau mode", config.old_tau_index);
    auto old_alpha_option = op.add<popl::Value<double>>("", "old_alpha", "Old alpha for construct_from mode", config.old_alpha);

    // Search mode options
    auto tau_search_option = op.add<popl::Value<double>>("", "tau_search", "Tau value for search", config.tau_search);
    auto error_tolerance_search_option = op.add<popl::Value<double>>("", "error_tolerance_search", "Error tolerance value for search", config.error_tolerance_search);
    // Additional options (for compute_edit_path_tree function)
    auto include_compute_dist_option = op.add<popl::Value<double>>("", "include_compute_dist", "Include compute distance range", config.include_compute_dist);
    auto max_exact_ged_for_EPT_option = op.add<popl::Value<double>>("", "max_exact_ged_for_EPT", "Compute lower bound (used as upper bound in GED computation)", config.max_exact_ged_for_EPT);
    auto root_ind_option = op.add<popl::Value<int>>("r", "root_ind", "Root node index", config.root_ind);

    // Server options
    auto server_id_option = op.add<popl::Value<int>>("", "server_id", "Server ID (0-based index)", config.server_id);
    auto total_servers_option = op.add<popl::Value<int>>("", "total_servers", "Total number of servers", config.total_servers);

    auto use_parallel = op.add<popl::Switch>("", "use_parallel", "Use parallel processing");
    auto num_workers_option = op.add<popl::Value<int>>("", "num_workers", "Number of parallel workers (0 = auto-detect CPU cores)", config.num_workers);
    auto enable_friends_reassign_option = op.add<popl::Switch>("", "enable_friends_reassign", "Enable friends reassignment mechanism (default: disabled)");
    auto q_start_option = op.add<popl::Value<int>>("", "q_start", "Query start index", config.q_start);
    auto q_end_option = op.add<popl::Value<int>>("", "q_end", "Query end index", config.q_end);
    auto save_logs_option = op.add<popl::Switch>("", "save_logs", "Save detailed experiment logs to files");
    auto nd_mode_option = op.add<popl::Value<std::string>>("", "nd_mode", "NetDag mode: filters (only LB filters) / astar (only AStar) / filters_astar (filters then AStar if lb<=tau)", config.nd_mode);
    auto dfs_mode_option = op.add<popl::Value<std::string>>("", "dfs_mode", "DFS traversal mode: default (all optimizations), or no_reuse / no_SP / no_LP / only_dfs for ablation", config.dfs_mode);
    auto disable_ept_filters_option = op.add<popl::Switch>("", "disable_ept_filters", "Disable lower bound filters for EPT (default: enabled)");
    auto only_compute_db_graph_option = op.add<popl::Switch>("", "only_compute_db_graph", "Only compute GED for EPT nodes that correspond to actual DB graphs (default: false)");
    auto disable_fast_down_option = op.add<popl::Switch>("", "disable_fast_down", "Disable fast down strategy (default: enabled, select first valid child instead of minimum lb)");
    auto app_max_iter_option = op.add<popl::Value<int>>("", "app_max_iter", "Maximum App iterations. Auto-set by dataset if not specified: AIDS=3000, PubChem=5000, Chemical1M=3000, SYN=350", config.app_max_iter);
    auto exact_max_iter_option = op.add<popl::Value<int>>("", "exact_max_iter", "Maximum iterations for exact computation (training data generation), default: 1000000", config.exact_max_iter);
    auto nd_filter_ratio_option = op.add<popl::Value<double>>("", "nd_filter_ratio", "NetDag filter ratio: condition becomes lb <= (alpha + tau) * ratio (default: 1.0, no tightening)", config.nd_filter_ratio);
    auto disable_all_lsa_option = op.add<popl::Switch>("", "disable_all_lsa", "Disable all LSA (automatically enables disable_lsa_pruning and disable_reuse_lsa)");
    auto disable_lsa_pruning_option = op.add<popl::Switch>("", "disable_lsa_pruning", "Disable LSa pruning in AStar (still save lsa_lb for reuse, default: false)");
    auto disable_reuse_lsa_option = op.add<popl::Switch>("", "disable_reuse_lsa", "Disable LSa recomputation in reuse (use simple ged_gap subtraction, default: false)");
    auto verify_reuse_baseline_option = op.add<popl::Switch>("", "verify_reuse_baseline", "Verify reuse effectiveness: compute AppForComputation baseline for each reuse (EXP-5)");
    auto chain_reuse_option = op.add<popl::Switch>("", "chain_reuse", "Enable chain reuse: nodes using reuse also save snapshot for subsequent nodes (default: false)");
    auto skip_hierarchy_option = op.add<popl::Switch>("", "skip_hierarchy", "Skip children/parent_by_phase_dict in construct_from (only keep anchors+clusters for Base+SS)");

    // experiment mode options
    auto tau_values_option = op.add<popl::Value<std::string>>("", "tau_values", "Comma-separated tau values for experiment mode (e.g., \"2,4,6,8\")", config.tau_values);
    auto exp_option = op.add<popl::Value<std::string>>("", "exp", "Experiment type: overall, ablation_gisma, ablation_epf", "overall");
    auto save_option = op.add<popl::Switch>("", "save", "Save results to files (default: not save, only display to console)");

    // select-alpha mode options
    auto alpha_min_option = op.add<popl::Value<double>>("", "alpha_min", "Minimum alpha value for select-alpha mode", config.alpha_min);
    auto alpha_max_option = op.add<popl::Value<double>>("", "alpha_max", "Maximum alpha value for select-alpha mode", config.alpha_max);
    auto alpha_step_option = op.add<popl::Value<double>>("", "alpha_step", "Alpha step for select-alpha mode", config.alpha_step);

    // batch_ged mode options (for GHash integration)
    auto candidate_ids_file_option = op.add<popl::Value<std::string>>("", "candidate_ids_file", "File containing candidate graph IDs, one per line", config.candidate_ids_file);
    auto query_id_option = op.add<popl::Value<int>>("", "query_id", "Query graph ID for batch_ged mode", config.query_id);

    // Parse command linearguments
    op.parse(argc, argv);

    if (help_option->is_set()) {
        std::cout << op << std::endl;
        return 0;
    }

    // Update configuration
    config.mode = mode_option->value();
    config.dataset = dataset_option->value();
    config.db_name = db_name_option->value();
    config.ground_truth_path = ground_truth_path_option->value();
    // config.search_method always "Gisma" in search mode
    config.query_file = query_file_option->value();
    config.target_file = target_file_option->value();
    config.ged_algorithm = ged_algorithm_option->value();
    config.alpha = alpha_option->value();
    config.tau_index = tau_index_option->value();
    config.error_tolerance_index = error_tolerance_index_option->value();
    config.old_tau_index = old_tau_index_option->value();
    config.old_alpha = old_alpha_option->value();
    config.tau_search = tau_search_option->value();
    config.error_tolerance_search = error_tolerance_search_option->value();

    // Update additional parameters
    config.include_compute_dist = include_compute_dist_option->value();
    config.max_exact_ged_for_EPT = max_exact_ged_for_EPT_option->value();
    config.q_start = q_start_option->value();
    config.q_end = q_end_option->value();
    config.server_id = server_id_option->value();
    config.total_servers = total_servers_option->value();
    config.use_parallel = use_parallel->is_set();
    config.num_workers = num_workers_option->value();
    config.root_ind = root_ind_option->value();
    config.save_logs = save_logs_option->is_set();
    config.nd_mode = nd_mode_option->value();
    config.dfs_mode = dfs_mode_option->value();
    config.disable_ept_filters = disable_ept_filters_option->is_set();
    config.only_compute_db_graph = only_compute_db_graph_option->is_set();
    config.disable_fast_down = disable_fast_down_option->is_set();
    // If user did not specify app_max_iter, auto-set default based on dataset
    if (app_max_iter_option->is_set()) {
        config.app_max_iter = app_max_iter_option->value();
    } else {
        // Set default based on dataset
        if (config.dataset == "AIDS") {
            config.app_max_iter = 3000;
        } else if (config.dataset == "PubChem") {
            config.app_max_iter = 5000;
        } else if (config.dataset == "Chemical1M") {
            config.app_max_iter = 3000;
        } else if (config.dataset == "SYN" || config.dataset.rfind("SYN_", 0) == 0) {
            config.app_max_iter = 350;
        } else {
            config.app_max_iter = 2300;  // default for other datasets 2300
        }
    }
    config.exact_max_iter = exact_max_iter_option->value();
    config.nd_filter_ratio = nd_filter_ratio_option->value();
    config.disable_all_lsa = disable_all_lsa_option->is_set();
    config.disable_lsa_pruning = disable_lsa_pruning_option->is_set();
    config.disable_reuse_lsa = disable_reuse_lsa_option->is_set();

    // If disable_all_lsa is enabled, automatically enable disable_lsa_pruning and disable_reuse_lsa
    if (config.disable_all_lsa) {
        config.disable_lsa_pruning = true;
        config.disable_reuse_lsa = true;
    }
    config.verify_reuse_baseline = verify_reuse_baseline_option->is_set();
    config.chain_reuse = chain_reuse_option->is_set();
    config.skip_hierarchy = skip_hierarchy_option->is_set();

    // Update experiment mode parameters
    config.tau_values = tau_values_option->value();
    // Map --exp to methods
    std::string exp_type = exp_option->value();
    if (exp_type == "overall") {
        config.methods = "App-BMao,AStar-BMao,Gisma";
    } else if (exp_type == "ablation_gisma") {
        config.methods = "App-BMao,Base+GS,Base+SS,Gisma";
    } else if (exp_type == "ablation_epf") {
        config.methods = "Gisma-no-reuse,Gisma-no-SP,Gisma-no-LP,Gisma";
    } else {
        std::cerr << "Unknown experiment type: " << exp_type << std::endl;
        std::cerr << "Available types: overall, ablation_gisma, ablation_epf" << std::endl;
        return 1;
    }
    config.save = save_option->is_set();

    // Update select-alpha mode parameters
    config.alpha_min = alpha_min_option->value();
    config.alpha_max = alpha_max_option->value();
    config.alpha_step = alpha_step_option->value();

    // Update batch_ged mode parameters
    config.candidate_ids_file = candidate_ids_file_option->value();
    config.query_id = query_id_option->value();

    // Handle friends adoption feature switch
    if (enable_friends_reassign_option->is_set()) {
        config.enable_friends_reassign = true;
    }
    // If not set, use default value (false, defined in Config.h)

    config.setDataset(config.dataset);

    // Restore user-specified paths (setDataset overwrites with defaults)
    if (db_name_option->is_set()) {
        config.db_name = db_name_option->value();
    }
    if (ground_truth_path_option->is_set()) {
        config.ground_truth_path = ground_truth_path_option->value();
    }

    // Call corresponding function based on mode
    if (config.mode == "construct") {
        construct_mode(config);
    } else if (config.mode == "construct_ND") {
        construct_ND_mode(config);
    } else if (config.mode == "compute_paths") {
        compute_paths_mode(config);
    } else if (config.mode == "reassign") {
        reassign_mode(config);
    } else if (config.mode == "construct_EPF") {
        construct_EPF_mode(config);
    } else if (config.mode == "search") {
        search_mode(config);
    } else if (config.mode == "export-candidates") {
        export_candidates_mode(config, candidates_output_option->value());
    } else if (config.mode == "info") {
        info_mode(config);
    } else if (config.mode == "compute_ged") {
        compute_ged_mode(config);
    } else if (config.mode == "select-alpha") {
        select_alpha_mode(config);
    } else if (config.mode == "experiment") {
        experiment_mode(config);
    } else if (config.mode == "batch_ged") {
        batch_ged_mode(config);
    } else if (config.mode == "upgrade_tau") {
        upgrade_tau_mode(config);
    } else if (config.mode == "construct_from") {
        construct_from_mode(config);
    } else {
        std::cerr << "Invalid mode: " << config.mode << std::endl;
        std::cerr << "Usage: " << argv[0] << " --mode [construct|construct_ND|compute_paths|reassign|construct_EPF|search|export-candidates|info|compute_ged|select-alpha|experiment|batch_ged|upgrade_tau|construct_from]" << std::endl;
        std::cerr << "\nConstruction modes:" << std::endl;
        std::cerr << "  construct     - Full pipeline: ND -> paths -> reassign -> EPF" << std::endl;
        std::cerr << "  construct_ND  - Build NetDag structure only (output: configs/)" << std::endl;
        std::cerr << "  compute_paths - Compute exact GED and node matching (output: ged_results/)" << std::endl;
        std::cerr << "  reassign      - Reassign nodes to clusters (input: configs/ + csv, output: reassigned/)" << std::endl;
        std::cerr << "  construct_EPF - Build EPF filters only (input: reassigned/)" << std::endl;
        std::cerr << "  upgrade_tau   - Upgrade tau_index of existing NetDag (--old_tau_index)" << std::endl;
        std::cerr << "  construct_from - Continue construction from existing NetDag with smaller alpha (--old_alpha)" << std::endl;
        return 1;
    }

    return 0;
}


