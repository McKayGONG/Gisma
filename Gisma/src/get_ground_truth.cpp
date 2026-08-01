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
#include <set>
#include <chrono>
#include <memory>
#include <mutex>
#include <atomic>

using namespace std;
using namespace chrono;

// ========================
// 全局容器
// ========================

// 1) 存放“通过下界筛选”的 (query_idx, db_idx) 对
static vector<pair<int,int>> filtered_pairs;
// A* 精确计算的迭代预算(第 6 个命令行参数, 默认 1e6 = 原行为)
static int gt_exact_max_iter = 1000000;

// 2) aggregator: 
//    aggregator[query_id][ged_val] = [db_ids ...]
//    用于最终一次性写到文件
static map<int, map<int, vector<int>>> aggregator;

// 3) 互斥锁，保护 aggregator
static mutex aggregator_mutex;

// 4) 原子计数，记录"已完成任务数"，用于进度条
static atomic<int> tasksDone(0);

// 5) 统计信息
static atomic<int> timeout_count(0);     // 达到100万限制的对数
static atomic<int> successful_count(0);  // 成功计算GED的对数
static atomic<int> filtered_out_count(0); // 在最终检查中被过滤的对数


// ------------------------------------------------------------
// 阶段一：下界筛选 (可多线程或单线程，这里展示多线程均分 queries)
// ------------------------------------------------------------
void compute_lower_bounds(
    int start_idx,
    int end_idx,
    const vector<Graph*>& db,
    const vector<Graph*>& queries,
    int max_ged,
    const map<string, ui>& vM,
    const map<string, ui>& eM,
    ui max_n,
    const vector<int>& query_ids,
    vector<pair<int,int>>& local_pass // [输出] 保存通过的 (q_idx, db_idx)
)
{
    for (int i = start_idx; i < end_idx; ++i) {
        int qIndex = i;         // queries[i]
        int qid    = query_ids[i];
        auto query_graph_ptr = queries[qIndex];

        // 遍历 DB
        int query_passed_count = 0;
        for (int db_id = 0; db_id < (int)db.size(); ++db_id) {
            auto db_graph_ptr = db[db_id];
            ui lb = query_graph_ptr->ged_lower_bound_filter(
                db_graph_ptr, (ui)max_ged, vM.size(), eM.size(), max_n
            );
            if (lb <= (ui)max_ged) {
                // 通过下界筛选
                local_pass.emplace_back(qIndex, db_id);
                query_passed_count++;
            }
        }
        // 每1000个query打印一次通过统计
        if (qIndex % 1000 == 0) {
            cout << "Query " << qIndex << " passed " << query_passed_count << "/" << db.size() << " pairs" << endl;
        }
    }
}


// ------------------------------------------------------------
// 阶段二：worker 函数 (A*)
// 使用动态任务分配: 依赖 atomic<int> nextIdx;
// tasksDone 用于更新进度
// ------------------------------------------------------------
void astar_worker(
    const vector<Graph*>& db,
    const vector<Graph*>& queries,
    const vector<int>& query_ids,
    int max_ged,
    atomic<int>& nextIdx,               // 动态分配索引
    const vector<pair<int,int>>& tasks, // 即 filtered_pairs
    int total_tasks                     // tasks.size()
)
{
    while (true) {
        // 1) 取下一个任务索引
        int i = nextIdx.fetch_add(1, memory_order_relaxed);
        if (i >= total_tasks) {
            break; // 没有任务了
        }

        // 2) A* 计算
        int qIndex = tasks[i].first;
        int dbIndex= tasks[i].second;

        int qid = query_ids[qIndex];
        auto query_graph_ptr = queries[qIndex];
        auto db_graph_ptr    = db[dbIndex];

        Application app(max_ged, "BMao", 2300, gt_exact_max_iter);
        app.init(db_graph_ptr, query_graph_ptr);
        int ged_val = app.AppForComputation(nullptr, nullptr);

        // 检查是否因为计数限制提前终止
        // 原判据 ged_val == -1 恒不成立(见文件头注释), 改用 hit_iter_limit() 记号。
        if (ged_val == -1 || app.hit_iter_limit()) {
            // AppForComputation 达到循环限制，未找到精确解
            int current_count = timeout_count.fetch_add(1);
            
            // 更频繁地报告前期情况，后期每1000次报告一次
            bool should_report = false;
            if (current_count < 100) {
                should_report = (current_count % 10 == 0);  // 前100次每10次报告
            } else if (current_count < 1000) {
                should_report = (current_count % 100 == 0); // 前1000次每100次报告
            } else {
                should_report = (current_count % 1000 == 0); // 之后每1000次报告
            }
            
            if (should_report) {
                cout << "[TIMEOUT] " << current_count + 1 << " pairs hit 1M computation limit (query " 
                     << qid << " vs db " << dbIndex << ")" << endl;
            }
        } else if (ged_val <= max_ged) {
            // 3) 若 ged_val <= max_ged, 写入 aggregator
            successful_count.fetch_add(1);
            lock_guard<mutex> lk(aggregator_mutex);
            aggregator[qid][ged_val].push_back(dbIndex);
        } else {
            // GED计算成功但超过max_ged阈值
            filtered_out_count.fetch_add(1);
        }

        // 4) 进度更新: 已完成任务数 +1
        tasksDone.fetch_add(1, memory_order_relaxed);
    }
}


// ------------------------------------------------------------
// 主函数
// 用法: ./this_program db.txt queries.txt max_ged output.txt num_threads
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0]
             << " <db_file> <queries_file> <max_ged> <output_file> <num_threads> [exact_max_iter]"
             << endl;
        return 1;
    }

    // 解析命令行
    const char* db_name         = argv[1];
    const char* queries_name    = argv[2];
    int max_ged                 = stoi(argv[3]);
    const char* output_filename = argv[4];
    int num_threads             = stoi(argv[5]);
    if (argc > 6) gt_exact_max_iter = stoi(argv[6]);
    cout << "[GT] exact_max_iter = " << gt_exact_max_iter << endl;

    // 1) 加载 DB
    vector<Graph*> db;
    map<string, ui> vM, eM;
    cout << "Loading DB from " << db_name << " ..." << endl;
    ui max_db_n = Utility::load_db(db_name, db, vM, eM);
    cout << "Loaded " << db.size() << " db graphs. max node = " << max_db_n << endl;

    // 2) 加载 queries
    vector<Graph*> queries;
    {
        cout << "Loading Queries from " << queries_name << " ..." << endl;
        map<string, ui> vM2, eM2;
        ui max_query_n = Utility::load_db(queries_name, queries, vM2, eM2);
        cout << "Loaded " << queries.size() << " query graphs. max node = "
             << max_query_n << endl;
    }

    // 3) 构造 query_ids
    vector<int> query_ids(queries.size());
    for (int i = 0; i < (int)queries.size(); ++i) {
        query_ids[i] = i;
    }

    // 清空输出文件(若想续写就把 ios::app)
    {
        ofstream of(output_filename);
        if (!of) {
            cerr << "Error opening file: " << output_filename << endl;
            return 1;
        }
    }

    auto t0 = high_resolution_clock::now();

    //=====================================================
    // 阶段一: 下界筛选 (多线程均分 queries)
    //=====================================================
    vector<pair<int,int>> all_passed; // 结果

    {
        cout << "[Phase 1] Lower-bound filtering..." << endl;

        int total_queries = (int)queries.size();
        int queries_per_thread = (total_queries + num_threads - 1) / num_threads;

        // 每个线程的输出放 local, 最后合并
        vector<vector<pair<int,int>>> local_passes(num_threads);

        vector<thread> ths;
        for (int t = 0; t < num_threads; ++t) {
            int st = t * queries_per_thread;
            if (st >= total_queries) break;
            int ed = min(st + queries_per_thread, total_queries);

            ths.push_back(thread(
                [&](int tid, int startQ, int endQ) {
                    compute_lower_bounds(
                        startQ, endQ,
                        cref(db),
                        cref(queries),
                        max_ged,
                        cref(vM),
                        cref(eM),
                        max_db_n,
                        cref(query_ids),
                        local_passes[tid]
                    );
                },
                t, st, ed
            ));
        }
        for (auto &th : ths) {
            th.join();
        }

        // 合并
        for (auto &lp : local_passes) {
            all_passed.insert(all_passed.end(), lp.begin(), lp.end());
        }
    }

    // swap 到 filtered_pairs
    filtered_pairs.swap(all_passed);
    auto t1 = high_resolution_clock::now();

    cout << "[Phase 1 done], filtered_pairs.size() = " << filtered_pairs.size() << endl;
    cout << "Total possible pairs: " << queries.size() << " * " << db.size() << " = " << (queries.size() * db.size()) << endl;
    cout << "Filtering efficiency: " << (100.0 * filtered_pairs.size() / (queries.size() * db.size())) << "%" << endl;

    //=====================================================
    // 阶段二: 动态任务分配 => A* + 进度条
    //=====================================================
    int total_pairs = (int)filtered_pairs.size();
    // 进度计数清零
    tasksDone.store(0, memory_order_relaxed);

    if (total_pairs == 0) {
        cerr << "[Phase 2] No pairs to process! (All filtered out)" << endl;
    } else {
        cout << "[Phase 2] A* for " << total_pairs << " filtered pairs (dynamic scheduling)..." << endl;
    }

    // 1) 启动进度线程: 每秒输出一次
    bool keepRunning = true;
    thread progressThread([&](){
        while(keepRunning) {
            this_thread::sleep_for(chrono::seconds(1));
            int done = tasksDone.load(memory_order_relaxed);
            if (total_pairs > 0) {
                double ratio = 100.0 * (double)done / total_pairs;
                cerr << "\rProgress: " << done << "/" << total_pairs
                     << " (" << (int)ratio << "%)   " << flush;
            }
        }
    });

    // 2) worker 线程 => 动态分配
    atomic<int> nextIdx(0);
    vector<thread> workers;
    workers.reserve(num_threads);

    for (int t=0; t<num_threads; ++t) {
        workers.emplace_back(
            [&](int thread_id){
                astar_worker(
                    cref(db),
                    cref(queries),
                    cref(query_ids),
                    max_ged,
                    ref(nextIdx),
                    cref(filtered_pairs),
                    total_pairs
                );
            },
            t
        );
    }

    for (auto &w : workers) w.join();

    // 3) 让进度线程退出
    keepRunning = false;
    progressThread.join();
    // 打印 100%
    if(total_pairs > 0){
        int done = tasksDone.load(memory_order_relaxed);
        cerr << "\rProgress: " << done << "/" << total_pairs
             << " (100%)" << endl;
    }

    auto t2 = high_resolution_clock::now();

    //=====================================================
    // 写 aggregator => output_filename
    //=====================================================
    {
        ofstream out_file(output_filename, ios::app);
        if (!out_file) {
            cerr << "Error opening file: " << output_filename << endl;
            return 1;
        }

        // aggregator[qid][ged_val] => db_ids
        for (auto &qpair : aggregator) {
            int qid = qpair.first;
            auto & ged_map = qpair.second;
            for (auto & kv : ged_map) {
                int ged_val = kv.first;
                auto & db_list = kv.second;

                // 输出: query_id ged [db1, db2, ...]
                out_file << qid << " " << ged_val << ".0 [";
                for (size_t i = 0; i < db_list.size(); i++) {
                    out_file << db_list[i];
                    if(i+1 < db_list.size()) out_file << ", ";
                }
                out_file << "]" << endl;
            }
        }
    }

    auto t3 = high_resolution_clock::now();

    //=====================================================
    // 释放
    //=====================================================
    for (auto &g : db) delete g;
    for (auto &q : queries) delete q;

    // 打印时间
    auto dur1      = duration_cast<milliseconds>(t1 - t0).count(); // LB阶段
    auto dur2      = duration_cast<milliseconds>(t2 - t1).count(); // 动态A*
    auto durWrite  = duration_cast<milliseconds>(t3 - t2).count(); // 写聚合
    auto durTotal  = duration_cast<milliseconds>(t3 - t0).count();

    cout << "\n[Phase 1: LB filter] = " << dur1 << " ms" << endl;
    cout << "[Phase 2: A* dynamic] = " << dur2 << " ms" << endl;
    cout << "[Phase 2: aggregator write] = " << durWrite << " ms" << endl;
    cout << "[TOTAL time] = " << durTotal << " ms" << endl;
    
    // 统计总结
    int total_computed = total_pairs;
    int timeouts = timeout_count.load();
    int successful = successful_count.load();
    int filtered = filtered_out_count.load();
    
    cout << "\n========== COMPUTATION STATISTICS ==========" << endl;
    cout << "Total query-db pairs processed: " << total_computed << endl;
    cout << "Hit 1M computation limit (timeout): " << timeouts 
         << " (" << (100.0 * timeouts / total_computed) << "%)" << endl;
    cout << "Successfully computed GED <= " << max_ged << ": " << successful 
         << " (" << (100.0 * successful / total_computed) << "%)" << endl;
    cout << "Computed GED > " << max_ged << " (filtered): " << filtered
         << " (" << (100.0 * filtered / total_computed) << "%)" << endl;
    cout << "Verification: " << (timeouts + successful + filtered) 
         << " / " << total_computed << " = " 
         << (100.0 * (timeouts + successful + filtered) / total_computed) << "%" << endl;
    cout << "===========================================" << endl;
    
    cout << "Results saved to " << output_filename << endl;

    return 0;
}
