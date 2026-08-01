#include "EditPathForestConstructor.h"
#include "GismaIndexBuilder.h"
#include "Application.h"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <atomic>
#include <future>
#include <thread>

// Hash function for pair keys (outside class scope for template compatibility)
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};
#include <mutex>
#include <iomanip>
#include <random>
#include <algorithm>
#include <sstream>
#include <set>
#include <vector>
#include <utility>

namespace fs = std::filesystem;

EditPathForestConstructor::EditPathForestConstructor(GismaIndexBuilder* parent) : parent_(parent) {}

void EditPathForestConstructor::construct_EPT() {
    double total_anchor_mapping_time = 0.0;
    double total_graph_lookup_time = 0.0;
    double total_compute_mapping_cost_time = 0.0;
    double total_build_edit_path_tree_time = 0.0;
    double total_save_ept_time = 0.0;

    auto total_start_time = std::chrono::high_resolution_clock::now();


    // 1) 检查 parent_->net_dag 指针是否有效
    if (!parent_->net_dag) {
        std::cerr << "[ERROR] net_dag is nullptr in construct_EPT()!\n";
        return;
    }

    // 2) 获取 anchors
    const auto& anchors = parent_->net_dag->anchors;
    if (anchors.empty()) {
        std::cerr << "[WARN] No anchors found in NetDag => return.\n";
        return;
    }


    // 4) 构建 graph_id_map
    std::unordered_map<ui, Graph*> graph_id_map;
    for (size_t i = 0; i < parent_->graphs.size(); i++) {
        auto &g_ptr = parent_->graphs[i];
        if (!g_ptr) {
            std::cerr << "[WARN] parent_->graphs[" << i << "] is nullptr!\n";
            continue;
        }
        // 转成无符号 long
        unsigned long graph_key = 0;
        try {
            graph_key = std::stoul(g_ptr->id);
        } catch (...) {
            std::cerr << "[ERROR] std::stoul failed for g_ptr->id='" << g_ptr->id << "' => skipping.\n";
            continue;
        }
        graph_id_map[graph_key] = g_ptr.get();
    }

    // 5) 创建输出目录
    std::string epf_directory = "./EPFs/" + parent_->index_name;
    fs::create_directories(epf_directory);

    // 6) 准备处理 anchors
    size_t total_anchors = anchors.size();
    size_t processed_anchors = 0;
    std::cout << "Processing " << total_anchors << " anchors for EPF construction..." << std::endl;

    for (size_t idx = 0; idx < total_anchors; ++idx) {
        auto& anchor = anchors[idx];
        if (!anchor) {
            std::cerr << "[ERROR] anchor at index " << idx << " is nullptr!\n";
            continue;
        }

        ui anchor_id = static_cast<ui>(anchor->node_id);

        // 7) 检查 anchor->graph
        if (!anchor->graph) {
            std::cerr << "[ERROR] anchor->graph is nullptr => cannot build EPT.\n";
            continue;
        }

        Graph* anchor_graph = anchor->graph.get();

        // 8) 复制 nodes_in_exact_cluster
        auto anchor_mapping_start = std::chrono::high_resolution_clock::now();
        std::vector<Anchor::ExactClusterNode> anchor_mappings;
        auto nodes_in_exact_cluster_pq = anchor->nodes_in_exact_cluster;


        while (!nodes_in_exact_cluster_pq.empty()) {
            anchor_mappings.push_back(nodes_in_exact_cluster_pq.top());
            nodes_in_exact_cluster_pq.pop();
        }
        auto anchor_mapping_end = std::chrono::high_resolution_clock::now();
        total_anchor_mapping_time += std::chrono::duration<double>(anchor_mapping_end - anchor_mapping_start).count();

        // Debug output for anchor 12735
        if (anchor_id == 12735) {
            std::cout << "\n[DEBUG 12735] Anchor 12735 exact_cluster has " << anchor_mappings.size() << " members:" << std::endl;
            for (const auto& cn : anchor_mappings) {
                std::cout << "  node_id=" << cn.node_id << ", dist=" << cn.dist << std::endl;
            }
        }


        // 8.1) 统计 other_id 频次
        {
            std::unordered_map<ui,int> freq_map;
            for (auto &cn : anchor_mappings) {
                freq_map[cn.node_id]++;
            }
        }

        // 如果 anchor_mappings 为空，构建空 EPT 并保存
        if (anchor_mappings.empty()) {
            EditPathTree ept(anchor_id);
            TreeNode root_node;
            root_node.anchor_id = anchor_id;
            root_node.ept_node_id = 0;
            root_node.completed_db_graph_ids.push_back((int)anchor_id);
            root_node.tree_node_graph_id = (int)anchor_id;
            root_node.level = 0;

            root_node.pseudo_graph = PseudoGraph(*anchor_graph);
            root_node.graph_ptr = std::make_shared<Graph>(*anchor_graph);
            root_node.graph_ptr->id = std::to_string(root_node.tree_node_graph_id);

            ept.tree_nodes.push_back(root_node);
            ept.root_index = 0;

            auto save_ept_start = std::chrono::high_resolution_clock::now();
            ept.save_to_file(epf_directory + "/EPT_anchor_" + std::to_string(anchor_id) + ".dat");
            auto save_ept_end = std::chrono::high_resolution_clock::now();
            total_save_ept_time += std::chrono::duration<double>(save_ept_end - save_ept_start).count();

            processed_anchors++;

            // 显示进度
            double progress = (double)processed_anchors / total_anchors * 100.0;
            if (processed_anchors % 50 == 0 || processed_anchors == total_anchors) {
                std::cout << "Progress: " << processed_anchors << "/" << total_anchors
                          << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
            }
            continue;
        }

        // 9) 对 anchor_mappings 做编辑操作
        std::vector<std::vector<EditOperation>> edit_operations_list;
        std::vector<ui> target_graph_ids;
        std::vector<std::vector<std::pair<ui, ui>>> all_mappings;
        std::vector<ui> identical_graph_ids;  // GED=0的图（与anchor完全相同）

        for (size_t c_idx = 0; c_idx < anchor_mappings.size(); ++c_idx) {
            auto &cluster_node = anchor_mappings[c_idx];
            ui other_id = cluster_node.node_id;


            auto graph_lookup_start = std::chrono::high_resolution_clock::now();
            Graph* other_graph = nullptr;
            if (auto it = graph_id_map.find(other_id); it != graph_id_map.end()) {
                other_graph = it->second;
            } else {
                std::cerr << "[WARN] other_id=" << other_id
                          << " not found in graph_id_map => skip.\n";
                continue;
            }
            auto graph_lookup_end = std::chrono::high_resolution_clock::now();
            total_graph_lookup_time += std::chrono::duration<double>(graph_lookup_end - graph_lookup_start).count();

            if (!other_graph) {
                std::cerr << "[WARN] other_graph is nullptr => skip.\n";
                continue;
            }

            auto cmc_start = std::chrono::high_resolution_clock::now();
            std::vector<EditOperation> edit_operations;
            anchor_graph->compute_mapping_cost(*other_graph, cluster_node.mapping, edit_operations);
            auto cmc_end = std::chrono::high_resolution_clock::now();
            total_compute_mapping_cost_time += std::chrono::duration<double>(cmc_end - cmc_start).count();

            if (!edit_operations.empty()) {
                // 有编辑操作，正常加入EPT
                edit_operations_list.push_back(std::move(edit_operations));
                target_graph_ids.push_back(other_id);
                all_mappings.push_back(cluster_node.mapping);
            } else {
                // GED=0: 完全相同的图，直接记录ID，稍后加到root节点
                identical_graph_ids.push_back(other_id);
            }
        }

        // 10) build EPT
        auto bept_start = std::chrono::high_resolution_clock::now();
        EditPathTree ept(anchor_id);
        build_edit_path_tree(ept, edit_operations_list, target_graph_ids, anchor_id, *anchor_graph);
        auto bept_end = std::chrono::high_resolution_clock::now();
        total_build_edit_path_tree_time += std::chrono::duration<double>(bept_end - bept_start).count();

        // 10.1) 将GED=0的图加入root节点的completed_db_graph_ids
        if (!identical_graph_ids.empty() && !ept.tree_nodes.empty()) {
            for (ui id : identical_graph_ids) {
                ept.tree_nodes[0].completed_db_graph_ids.push_back((int)id);
            }
            if (anchor_id == 12735) {
                std::cout << "[DEBUG 12735] Added " << identical_graph_ids.size()
                          << " GED=0 graphs to root node: ";
                for (ui id : identical_graph_ids) {
                    std::cout << id << " ";
                }
                std::cout << std::endl;
            }
        }

        // 11) save
        auto save_ept_start = std::chrono::high_resolution_clock::now();
        std::string ept_filename = epf_directory + "/EPT_anchor_" + std::to_string(anchor_id) + ".dat";
        ept.save_to_file(ept_filename);
        auto save_ept_end = std::chrono::high_resolution_clock::now();
        total_save_ept_time += std::chrono::duration<double>(save_ept_end - save_ept_start).count();

        processed_anchors++;

        // 显示进度
        double progress = (double)processed_anchors / total_anchors * 100.0;
        if (processed_anchors % 50 == 0 || processed_anchors == total_anchors) {
            std::cout << "Progress: " << processed_anchors << "/" << total_anchors
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
        }
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end_time - total_start_time).count();
    std::cout << "EPF construction completed!" << std::endl;
    std::cout << "Total anchors processed: " << processed_anchors << "/" << anchors.size() << std::endl;
    std::cout << "Total elapsed time: " << std::fixed << std::setprecision(2) << total_elapsed << " seconds" << std::endl;
    // E1 construction timing (serial): wall == work (single thread).
    {
        // E1: work = 纯 compute（anchor_mapping + build_ept）。save 是磁盘 I/O，并行写盘相加会虚高，不计入 work，单列在 breakdown。
        double epf_work = total_anchor_mapping_time + total_build_edit_path_tree_time;
        std::cout << "[E1_EPF] threads=1 wall_seconds=" << total_elapsed
                  << " work_seconds=" << epf_work
                  << " anchors=" << processed_anchors << std::endl;
        std::cout << "[E1_EPF] breakdown anchor_mapping=" << total_anchor_mapping_time
                  << " (graph_lookup=" << total_graph_lookup_time
                  << " compute_mapping_cost=" << total_compute_mapping_cost_time << ")"
                  << " build_ept=" << total_build_edit_path_tree_time
                  << " save=" << total_save_ept_time << std::endl;
        Utility::append_e1_timing("index_name=" + parent_->index_name + "\tdataset=" + parent_->dataset
            + "\tstage=construct_EPF\tthreads=1\twall_seconds=" + std::to_string(total_elapsed)
            + "\twork_seconds=" + std::to_string(epf_work)
            + "\tanchors=" + std::to_string(processed_anchors)
            + "\tsave_seconds=" + std::to_string(total_save_ept_time));
    }
}

void EditPathForestConstructor::construct_EPT_parallel() {
    // Timing variables
    double total_anchor_mapping_time = 0.0;
    double total_graph_lookup_time = 0.0;
    double total_compute_mapping_cost_time = 0.0;
    double total_build_edit_path_tree_time = 0.0;
    double total_save_ept_time = 0.0;

    auto total_start_time = std::chrono::high_resolution_clock::now();

    // 1) Check if net_dag pointer is valid
    if (!parent_->net_dag) {
        std::cerr << "[ERROR] net_dag is nullptr in construct_EPT_parallel()!\n";
        return;
    }

    // 2) Get anchors
    const auto& anchors = parent_->net_dag->anchors;
    if (anchors.empty()) {
        std::cerr << "[WARN] No anchors found in NetDag => return.\n";
        return;
    }

    // 3) Build graph_id_map
    std::unordered_map<ui, Graph*> graph_id_map;
    for (size_t i = 0; i < parent_->graphs.size(); i++) {
        auto &g_ptr = parent_->graphs[i];
        if (!g_ptr) continue;
        // Convert to unsigned long
        unsigned long graph_key = std::stoul(g_ptr->id);
        graph_id_map[graph_key] = g_ptr.get();
    }

    // 5) Create EPF directory for this NetDag
    std::string epf_directory = "./EPFs/" + parent_->index_name;
    if (!std::filesystem::exists(epf_directory)) {
        std::filesystem::create_directories(epf_directory);
    }

    // 6) Set up parallel processing
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = std::min(num_threads, 200u); // Limit maximum threads (unified to 200, matching GED stage)

    std::atomic<size_t> processed_anchors(0);
    std::mutex io_mutex, stats_mutex;
    std::vector<std::thread> threads;

    // 7) Start threads
    for (unsigned int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            double thread_anchor_mapping_time = 0.0;
            double thread_graph_lookup_time = 0.0;
            double thread_compute_mapping_cost_time = 0.0;
            double thread_build_edit_path_tree_time = 0.0;
            double thread_save_ept_time = 0.0;

            for (size_t i = t; i < anchors.size(); i += num_threads) {
                const auto& anchor = anchors[i];
                int anchor_id = anchor->node_id;

                // 8) Get anchor graph
                Graph* anchor_graph = anchor->graph.get();
                if (!anchor_graph) {
                    // Skip anchors without graphs, but count them as processed
                    processed_anchors++;
                    double progress = (double)processed_anchors / anchors.size() * 100.0;
                    if (processed_anchors % 50 == 0 || processed_anchors == anchors.size()) {
                        std::cout << "Progress: " << processed_anchors << "/" << anchors.size()
                                  << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
                    }
                    continue;
                }

                // 9) Get nodes in exact cluster
                auto am_start = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<EditOperation>> edit_operations_list;
                std::vector<ui> target_graph_ids;
                std::vector<std::map<int, int>> all_mappings;
                std::vector<ui> identical_graph_ids;  // GED=0的图（与anchor完全相同）

                auto nodes_in_exact_cluster = anchor->nodes_in_exact_cluster;
                while (!nodes_in_exact_cluster.empty()) {
                    auto cluster_node = nodes_in_exact_cluster.top();
                    nodes_in_exact_cluster.pop();

                    ui other_id = cluster_node.node_id;

                    auto graph_lookup_start = std::chrono::high_resolution_clock::now();
                    Graph* other_graph = nullptr;
                    auto it = graph_id_map.find(other_id);
                    if (it != graph_id_map.end()) {
                        other_graph = it->second;
                    }

                    if (!other_graph) continue;
                    auto graph_lookup_end = std::chrono::high_resolution_clock::now();
                    thread_graph_lookup_time += std::chrono::duration<double>(graph_lookup_end - graph_lookup_start).count();

                    auto cmc_start = std::chrono::high_resolution_clock::now();
                    std::vector<EditOperation> edit_operations;
                    anchor_graph->compute_mapping_cost(*other_graph, cluster_node.mapping, edit_operations);
                    auto cmc_end = std::chrono::high_resolution_clock::now();
                    thread_compute_mapping_cost_time += std::chrono::duration<double>(cmc_end - cmc_start).count();

                    if (!edit_operations.empty()) {
                        // 有编辑操作，正常加入EPT
                        edit_operations_list.push_back(std::move(edit_operations));
                        target_graph_ids.push_back(other_id);
                        // Convert vector<pair> to map<int,int>
                        std::map<int, int> mapping_map;
                        for (const auto& pair : cluster_node.mapping) {
                            mapping_map[pair.first] = pair.second;
                        }
                        all_mappings.push_back(mapping_map);
                    } else {
                        // GED=0: 完全相同的图，直接记录ID，稍后加到root节点
                        identical_graph_ids.push_back(other_id);
                    }
                }
                auto am_end = std::chrono::high_resolution_clock::now();
                thread_anchor_mapping_time += std::chrono::duration<double>(am_end - am_start).count();

                // 10) Build EPT
                auto bept_start = std::chrono::high_resolution_clock::now();
                EditPathTree ept(anchor_id);
                build_edit_path_tree(ept, edit_operations_list, target_graph_ids, anchor_id, *anchor_graph);
                auto bept_end = std::chrono::high_resolution_clock::now();
                thread_build_edit_path_tree_time += std::chrono::duration<double>(bept_end - bept_start).count();

                // 10.1) 将GED=0的图加入root节点的completed_db_graph_ids
                if (!identical_graph_ids.empty() && !ept.tree_nodes.empty()) {
                    for (ui id : identical_graph_ids) {
                        ept.tree_nodes[0].completed_db_graph_ids.push_back((int)id);
                    }
                }

                // 11) Save
                auto save_ept_start = std::chrono::high_resolution_clock::now();
                std::string ept_filename = epf_directory + "/EPT_anchor_" + std::to_string(anchor_id) + ".dat";
                ept.save_to_file(ept_filename);
                auto save_ept_end = std::chrono::high_resolution_clock::now();
                thread_save_ept_time += std::chrono::duration<double>(save_ept_end - save_ept_start).count();

                processed_anchors++;

                // 显示进度
                double progress = (double)processed_anchors / anchors.size() * 100.0;
                if (processed_anchors % 50 == 0 || processed_anchors == anchors.size()) {
                    std::cout << "Progress: " << processed_anchors << "/" << anchors.size()
                              << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
                }
            }

            // Merge thread statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                total_anchor_mapping_time += thread_anchor_mapping_time;
                total_graph_lookup_time += thread_graph_lookup_time;
                total_compute_mapping_cost_time += thread_compute_mapping_cost_time;
                total_build_edit_path_tree_time += thread_build_edit_path_tree_time;
                total_save_ept_time += thread_save_ept_time;
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end_time - total_start_time).count();
    std::cout << "EPF parallel construction completed!" << std::endl;
    std::cout << "Total anchors processed: " << processed_anchors << "/" << anchors.size() << std::endl;
    std::cout << "Total elapsed time: " << std::fixed << std::setprecision(2) << total_elapsed << " seconds" << std::endl;
    // E1 construction timing (parallel): wall = real elapsed; work = sum of per-thread per-task
    // compute time (= single-thread-equivalent CPU work, undiluted by parallelism).
    {
        // E1: work = 纯 compute（anchor_mapping + build_ept）。save 是磁盘 I/O，并行写盘相加会虚高，不计入 work，单列在 breakdown。
        double epf_work = total_anchor_mapping_time + total_build_edit_path_tree_time;
        std::cout << "[E1_EPF] threads=" << num_threads << " wall_seconds=" << total_elapsed
                  << " work_seconds=" << epf_work
                  << " anchors=" << processed_anchors.load() << std::endl;
        std::cout << "[E1_EPF] breakdown anchor_mapping=" << total_anchor_mapping_time
                  << " (graph_lookup=" << total_graph_lookup_time
                  << " compute_mapping_cost=" << total_compute_mapping_cost_time << ")"
                  << " build_ept=" << total_build_edit_path_tree_time
                  << " save=" << total_save_ept_time << std::endl;
        Utility::append_e1_timing("index_name=" + parent_->index_name + "\tdataset=" + parent_->dataset
            + "\tstage=construct_EPF\tthreads=" + std::to_string(num_threads)
            + "\twall_seconds=" + std::to_string(total_elapsed)
            + "\twork_seconds=" + std::to_string(epf_work)
            + "\tanchors=" + std::to_string(processed_anchors.load())
            + "\tsave_seconds=" + std::to_string(total_save_ept_time));
    }
}

void EditPathForestConstructor::compute_edit_path_and_save_to_csv() {
    namespace fs = std::filesystem;
    std::string csv_filename = "./NetDags/gisma_index_exact_ged_results_.csv";

    // Create CSV file if it doesn't exist
    if(!fs::exists("./NetDags/")) {
        fs::create_directories("./NetDags/");
    }

    if(!fs::exists(csv_filename)) {
        std::ofstream create_empty(csv_filename);
        if(!create_empty.is_open()) {
            std::cerr << "[ERROR] Unable to create: " << csv_filename << std::endl;
            return;
        } else {
            std::cout << "[INFO] Created empty CSV file: " << csv_filename << std::endl;
        }
    }

    // Load existing results to avoid recomputation
    // pair_hash defined outside local scope

    std::unordered_map<std::pair<int,int>, std::tuple<int,std::string>, pair_hash> computed_results;
    {
        std::ifstream ifs(csv_filename);
        if(ifs.is_open()) {
            std::string line;
            while(std::getline(ifs,line)) {
                std::stringstream ss(line);
                std::string id1_str, id2_str, ged_str, mapping_str;
                if(!std::getline(ss,id1_str,',')) continue;
                if(!std::getline(ss,id2_str,',')) continue;
                if(!std::getline(ss,ged_str,',')) continue;
                if(!std::getline(ss,mapping_str))  continue;

                try {
                    int id1 = std::stoi(id1_str);
                    int id2 = std::stoi(id2_str);
                    int ged = std::stoi(ged_str);
                    int min_id = std::min(id1,id2);
                    int max_id = std::max(id1,id2);
                    computed_results[ {min_id,max_id} ] = {ged,mapping_str};
                }
                catch(...) {
                    continue;
                }
            }
            ifs.close();
            std::cout << "[INFO] Loaded existing GED results from " << csv_filename << std::endl;
        }
    }

    // Find pairs that need computation
    std::vector<std::pair<int,int>> pairs_to_compute;
    for(auto &anchor : parent_->net_dag->anchors) {
        int anchor_id = anchor->node_id;
        auto temp_pq = anchor->nodes_in_cluster;
        while(!temp_pq.empty()) {
            auto [dist, node_id] = temp_pq.top();
            temp_pq.pop();

            if(anchor_id == node_id) {
                continue; // Skip self
            }
            int min_id = std::min(anchor_id,node_id);
            int max_id = std::max(anchor_id,node_id);
            auto it = computed_results.find({min_id,max_id});
            if(it == computed_results.end()) {
                pairs_to_compute.emplace_back(anchor_id,node_id);
            }
        }
    }

    size_t total_tasks = pairs_to_compute.size();
    if(total_tasks == 0) {
        std::cout << "[INFO] No new pairs to compute. All done.\n";
        return;
    }

    std::cout << "[INFO] Computing GED for " << total_tasks << " pairs..." << std::endl;

    // Atomic counters for statistics
    std::atomic<size_t> done_count(0);
    std::atomic<size_t> wrote_count(0);
    std::atomic<size_t> infplus_count(0);
    std::atomic<size_t> exceed_count(0);

    // Open file for appending results
    std::ofstream ofs(csv_filename, std::ios::app);
    if(!ofs.is_open()) {
        std::cerr << "[ERROR] Cannot open for append: " << csv_filename << std::endl;
        return;
    }

    // Threading setup
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads > 200) num_threads = 200;
    if(num_threads == 0) num_threads = 4;

    std::atomic<size_t> nextIdx(0);
    static std::mutex g_write_mutex;

    // Progress monitoring thread
    bool keepRunning = true;
    std::thread progressThread([&](){
        while(keepRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            size_t d = done_count.load(std::memory_order_relaxed);
            double ratio = 100.0 * (double)d / (double)total_tasks;
            size_t w = wrote_count.load(std::memory_order_relaxed);
            size_t i = infplus_count.load(std::memory_order_relaxed);
            size_t e = exceed_count.load(std::memory_order_relaxed);

            std::cerr << "\rProcessed " << d << "/" << total_tasks
                      << " (" << (int)ratio << "%). wrote=" << w
                      << ", INF+1=" << i << ", exceed=" << e << "..." << std::flush;
        }
    });

    auto t0 = std::chrono::steady_clock::now();

    // Worker thread function
    auto worker = [&](int thread_id){
        const int CHUNK_SIZE = 100;
        std::vector<std::tuple<int,int,int,std::string>> local_cache;
        local_cache.reserve(CHUNK_SIZE);

        while(true) {
            size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
            if(idx >= total_tasks) break;

            int anchor_id = pairs_to_compute[idx].first;
            int node_id = pairs_to_compute[idx].second;

            auto node_anchor = parent_->net_dag->nodes[anchor_id];
            auto node_other = parent_->net_dag->nodes[node_id];
            auto graph1 = node_anchor->graph.get();
            auto graph2 = node_other->graph.get();

            // Compute GED with mapping using AppForComputation
            Application app(8u, "BMao");
            app.set_all_edge_labels_same(parent_->all_edge_labels_same);
            app.init(graph1, graph2);
            int ged_res = app.AppForComputation();

            std::vector<std::pair<ui,ui>> mapping;
            app.get_mapping(mapping);

            // Categorize results
            if(ged_res == INF+1) {
                infplus_count.fetch_add(1);
            } else if(ged_res > 8) {
                exceed_count.fetch_add(1);
            } else {
                wrote_count.fetch_add(1);

                // Serialize mapping
                std::ostringstream oss;
                oss << "\"[";
                for(size_t k = 0; k < mapping.size(); k++){
                    oss << "(" << mapping[k].first << "," << mapping[k].second << ")";
                    if(k + 1 < mapping.size()) oss << ",";
                }
                oss << "]\"";

                local_cache.emplace_back(anchor_id, node_id, ged_res, oss.str());
            }

            done_count.fetch_add(1);

            // Batch write to file
            if((int)local_cache.size() >= CHUNK_SIZE) {
                std::lock_guard<std::mutex> lk(g_write_mutex);
                for(auto &tup: local_cache) {
                    int g1, g2, dist;
                    std::string maps;
                    std::tie(g1, g2, dist, maps) = tup;
                    ofs << g1 << "," << g2 << "," << dist << "," << maps << "\n";
                }
                local_cache.clear();
            }
        }

        // Write remaining cached results
        if(!local_cache.empty()) {
            std::lock_guard<std::mutex> lk(g_write_mutex);
            for(auto &tup: local_cache) {
                int g1, g2, dist;
                std::string maps;
                std::tie(g1, g2, dist, maps) = tup;
                ofs << g1 << "," << g2 << "," << dist << "," << maps << "\n";
            }
        }
    };

    // Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t i = 0; i < num_threads; i++){
        threads.emplace_back(worker, i);
    }

    // Wait for all threads to complete
    for(auto &t : threads){
        t.join();
    }

    // Stop progress monitoring
    keepRunning = false;
    progressThread.join();

    ofs.close();

    auto t1 = std::chrono::steady_clock::now();
    double dur = std::chrono::duration<double>(t1 - t0).count();

    // Final statistics
    size_t d = done_count.load();
    size_t w = wrote_count.load();
    size_t i = infplus_count.load();
    size_t e = exceed_count.load();
    std::cout << "\n[INFO] All done. total=" << total_tasks
              << ", wrote=" << w << ", INF+1=" << i
              << ", exceed=" << e << std::endl;
    std::cout << "[INFO] Time elapsed: " << dur << " s\n";
    std::cout << "[INFO] Results appended to: " << csv_filename << std::endl;
}

void EditPathForestConstructor::find_within_tau_and_save_csv(const std::string &csv_filename) {
    int root = 0; // Use default root index
    double T = parent_->net_dag->tau;

    if(root < 0 || root >= (int)parent_->graphs.size()){
        std::cerr << "[ERROR] root=" << root << " out_of_range\n";
        return;
    }

    auto root_g_ptr = parent_->graphs[root];
    if(!root_g_ptr){
        std::cerr << "[ERROR] root_graph is nullptr => cannot proceed\n";
        return;
    }
    Graph* root_graph = root_g_ptr.get();

    std::cout << "[INFO] find_within_tau_and_save_csv => root=" << root
              << ", tau=" << T << ", total graphs=" << parent_->graphs.size()
              << "\n   => csv=" << csv_filename << "\n";

    // Collect target graph IDs
    std::vector<int> targets;
    targets.reserve(parent_->graphs.size() - 1);
    for(int i = 0; i < (int)parent_->graphs.size(); i++){
        if(i == root) continue;
        targets.push_back(i);
    }

    size_t total_tasks = targets.size();
    if(total_tasks == 0){
        std::cerr << "[WARN] No other graphs => skip.\n";
        return;
    }

    std::atomic<size_t> done_count(0), wrote_count(0), infplus_count(0), exceed_count(0);
    bool keepRunning = true;

    // Open CSV file in append mode
    std::ofstream ofs(csv_filename, std::ios::app);
    if(!ofs.is_open()){
        std::cerr << "[ERROR] fail to open " << csv_filename << " in append\n";
        return;
    }

    // Threading setup
    size_t num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4;
    num_threads = (num_threads > 16) ? 16 : num_threads;

    std::atomic<size_t> nextIdx(0);
    static std::mutex g_write_mutex;

    // Progress monitoring thread
    std::thread progressThread([&](){
        while(keepRunning){
            std::this_thread::sleep_for(std::chrono::seconds(1));
            size_t d = done_count.load(std::memory_order_relaxed);
            double ratio = 100.0 * (double)d / (double)total_tasks;
            size_t w = wrote_count.load(std::memory_order_relaxed);
            size_t i = infplus_count.load(std::memory_order_relaxed);
            size_t e = exceed_count.load(std::memory_order_relaxed);

            std::cerr << "\r[Progress] done=" << d << "/" << total_tasks
                      << "(" << (int)ratio << "%), wrote=" << w
                      << ", INF+1=" << i << ", exceed=" << e << "..." << std::flush;
        }
    });

    auto t0 = std::chrono::steady_clock::now();

    // Worker function
    auto worker = [&](int tid){
        const int CHUNK_SIZE = 50;
        std::vector<std::tuple<int,int,int,std::string>> local_cache;
        local_cache.reserve(CHUNK_SIZE);

        while(true){
            size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
            if(idx >= total_tasks) break;

            int other_id = targets[idx];
            auto other_g_ptr = parent_->graphs[other_id];
            if(!other_g_ptr){
                done_count.fetch_add(1);
                continue;
            }
            Graph* other_graph = other_g_ptr.get();

            // Compute GED using AppForComputation
            Application app((ui)T, "BMao");
            app.set_all_edge_labels_same(parent_->all_edge_labels_same);
            app.init(root_graph, other_graph);
            int ged_res = app.AppForComputation();

            std::vector<std::pair<ui,ui>> mapping;
            app.get_mapping(mapping);

            if(ged_res == INF+1){
                infplus_count.fetch_add(1);
            } else if(ged_res > (int)T){
                exceed_count.fetch_add(1);
            } else {
                wrote_count.fetch_add(1);
                // Serialize mapping
                std::ostringstream oss;
                oss << "\"[";
                for(size_t k = 0; k < mapping.size(); k++){
                    oss << "(" << mapping[k].first << "," << mapping[k].second << ")";
                    if(k + 1 < mapping.size()) oss << ",";
                }
                oss << "]\"";

                local_cache.emplace_back(root, other_id, ged_res, oss.str());
            }

            done_count.fetch_add(1);

            if((int)local_cache.size() >= CHUNK_SIZE){
                std::lock_guard<std::mutex> lk(g_write_mutex);
                for(auto &tup: local_cache){
                    int r, o, dist;
                    std::string maps;
                    std::tie(r, o, dist, maps) = tup;
                    ofs << r << "," << o << "," << dist << "," << maps << "\n";
                }
                local_cache.clear();
            }
        }

        // Write remaining cached results
        if(!local_cache.empty()){
            std::lock_guard<std::mutex> lk(g_write_mutex);
            for(auto &tup: local_cache){
                int r, o, dist;
                std::string maps;
                std::tie(r, o, dist, maps) = tup;
                ofs << r << "," << o << "," << dist << "," << maps << "\n";
            }
        }
    };

    // Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t i = 0; i < num_threads; i++){
        threads.emplace_back(worker, (int)i);
    }
    for(auto &th: threads){
        th.join();
    }

    keepRunning = false;
    progressThread.join();
    ofs.close();

    auto t1 = std::chrono::steady_clock::now();
    double dur = std::chrono::duration<double>(t1 - t0).count();

    size_t d = done_count.load();
    size_t w = wrote_count.load();
    size_t inf = infplus_count.load();
    size_t e = exceed_count.load();
    std::cout << "\n[INFO] find_within_tau_and_save_csv done:\n"
              << " total=" << total_tasks << ", done=" << d << ", wrote=" << w
              << ", INF+1=" << inf << ", exceed=" << e << "\n"
              << " Time=" << dur << "s\n"
              << " Results => " << csv_filename << "\n";
}

void EditPathForestConstructor::build_one_ball_EPT_from_csv(
    int EPT_root_ind,
    double alpha,
    const std::string &csv_path,
    const std::string &ept_filename) {

    // Check EPT_root_ind validity
    if(EPT_root_ind < 0 || EPT_root_ind >= (int)parent_->graphs.size()){
        std::cerr << "[ERROR] EPT_root_ind=" << EPT_root_ind
                  << " out_of_range (graphs.size()="
                  << parent_->graphs.size() << ")\n";
        return;
    }

    double T = alpha; // Use alpha as threshold
    auto root_graph_ptr = parent_->graphs[EPT_root_ind];
    if(!root_graph_ptr){
        std::cerr << "[ERROR] root_graph is nullptr => can't build EPT\n";
        return;
    }
    Graph* root_graph = root_graph_ptr.get();

    // Read CSV file
    std::ifstream ifs(csv_path);
    if(!ifs.is_open()){
        std::cerr << "[ERROR] cannot open CSV: " << csv_path << "\n";
        return;
    }

    std::cout << "[INFO] build_one_ball_EPT_from_csv => root_ind="
              << EPT_root_ind << ", alpha=" << alpha
              << ", reading " << csv_path << "\n";

    // Collect (other_id, ged_val, mapping) where r_id==EPT_root_ind && ged_val <= T
    std::vector<std::tuple<ui,int,std::vector<std::pair<ui,ui>>>> cluster_nodes;

    std::string line;
    while(std::getline(ifs,line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string r_str, o_str, ged_str, map_str;
        if(!std::getline(ss,r_str,',')) continue;
        if(!std::getline(ss,o_str,',')) continue;
        if(!std::getline(ss,ged_str,',')) continue;
        if(!std::getline(ss,map_str))    continue;

        try {
            int r_id = std::stoi(r_str);
            int o_id = std::stoi(o_str);
            int ged_val = std::stoi(ged_str);

            // Filter: only include rows for our root with ged <= threshold
            if(r_id == EPT_root_ind && ged_val <= (int)T){
                // Parse mapping string
                while(!map_str.empty() && std::isspace((unsigned char)map_str.front()))
                    map_str.erase(0,1);
                while(!map_str.empty() && std::isspace((unsigned char)map_str.back()))
                    map_str.pop_back();
                if(map_str.size() >= 2 && map_str.front() == '"' && map_str.back() == '"'){
                    map_str = map_str.substr(1, map_str.size()-2);
                }

                std::vector<std::pair<ui,ui>> mapping;
                size_t pos = 0;
                while(true){
                    size_t start = map_str.find('(',pos);
                    if(start == std::string::npos) break;
                    size_t end = map_str.find(')',start);
                    if(end == std::string::npos) break;
                    std::string sub = map_str.substr(start+1,end-(start+1));
                    pos = end+1;

                    size_t cPos = sub.find(',');
                    if(cPos == std::string::npos) continue;
                    std::string u_str = sub.substr(0,cPos);
                    std::string v_str = sub.substr(cPos+1);
                    ui u = (ui)std::stoul(u_str);
                    ui v = (ui)std::stoul(v_str);
                    mapping.emplace_back(u,v);
                }
                cluster_nodes.push_back({(ui)o_id, ged_val, mapping});
            }
        } catch(...){
            continue; // Skip lines with parsing errors
        }
    }
    ifs.close();

    // If no nodes found, build empty EPT with just root
    if(cluster_nodes.empty()){
        std::cout << "[INFO] No node found in CSV for root=" << EPT_root_ind
                  << " & ged<= " << T << ". Build empty EPT.\n";
        EditPathTree ept((ui)EPT_root_ind);
        TreeNode root_node;
        root_node.anchor_id = (ui)EPT_root_ind;
        root_node.ept_node_id = 0;
        root_node.completed_db_graph_ids.push_back(EPT_root_ind);
        root_node.tree_node_graph_id = EPT_root_ind;
        root_node.level = 0;
        root_node.pseudo_graph = PseudoGraph(*root_graph);
        root_node.graph_ptr = std::make_shared<Graph>(*root_graph);
        root_node.graph_ptr->id = std::to_string(EPT_root_ind);

        ept.tree_nodes.push_back(root_node);
        ept.root_index = 0;

        ept.save_to_file(ept_filename);
        std::cout << "[INFO] Saved empty EPT => " << ept_filename << "\n";
        return;
    }

    // Prepare edit_operations_list for EPT construction
    std::vector<std::vector<EditOperation>> edit_operations_list;
    edit_operations_list.reserve(cluster_nodes.size());
    std::vector<ui> target_graph_ids;
    target_graph_ids.reserve(cluster_nodes.size());

    // Compute edit operations from mappings
    for(auto &info: cluster_nodes){
        ui other_id = std::get<0>(info);
        auto &mapping = std::get<2>(info);

        if(other_id >= parent_->graphs.size()){
            std::cerr << "[WARN] other_id=" << other_id << " out_of_range\n";
            continue;
        }
        auto other_g_ptr = parent_->graphs[other_id];
        if(!other_g_ptr){
            std::cerr << "[WARN] other_graph is nullptr => skip\n";
            continue;
        }
        Graph* other_graph = other_g_ptr.get();

        std::vector<EditOperation> edit_ops;
        root_graph->compute_mapping_cost(*other_graph, mapping, edit_ops);

        if(!edit_ops.empty()){
            edit_operations_list.push_back(std::move(edit_ops));
            target_graph_ids.push_back(other_id);
        }
    }

    // Build EPT using the helper function
    EditPathTree ept((ui)EPT_root_ind);
    build_edit_path_tree(ept, edit_operations_list, target_graph_ids,
                         (ui)EPT_root_ind, *root_graph);

    // Save EPT to file
    ept.save_to_file(ept_filename);
    std::cout << "[INFO] One-ball EPT built & saved => " << ept_filename << "\n";
}

void EditPathForestConstructor::update_one_exact_anchor() {
    std::cout << "[INFO] Starting update_one_exact_anchor..." << std::endl;

    // Find anchor with most nodes to refine
    std::shared_ptr<Anchor> target_anchor = nullptr;
    size_t max_cluster_size = 0;

    for (auto& anchor : parent_->net_dag->anchors) {
        if (!anchor) continue;

        size_t cluster_size = anchor->nodes_in_cluster.size();
        if (cluster_size > max_cluster_size) {
            max_cluster_size = cluster_size;
            target_anchor = anchor;
        }
    }

    if (!target_anchor || max_cluster_size == 0) {
        std::cout << "[INFO] No suitable anchor found for updating." << std::endl;
        return;
    }

    std::cout << "[INFO] Updating anchor " << target_anchor->node_id
              << " with " << max_cluster_size << " nodes in cluster." << std::endl;

    // Process nodes in the cluster to update exact distances
    std::vector<std::pair<ui, ui>> nodes_to_process;
    auto temp_cluster = target_anchor->nodes_in_cluster;

    while (!temp_cluster.empty() && nodes_to_process.size() < 100) {
        auto [dist, node_id] = temp_cluster.top();
        temp_cluster.pop();

        if (node_id != target_anchor->node_id) {
            nodes_to_process.emplace_back(dist, node_id);
        }
    }

    if (nodes_to_process.empty()) {
        std::cout << "[INFO] No nodes to process for anchor " << target_anchor->node_id << std::endl;
        return;
    }

    std::cout << "[INFO] Processing " << nodes_to_process.size() << " nodes..." << std::endl;

    // Compute exact GED distances
    int updated_count = 0;
    auto anchor_graph = target_anchor->graph.get();

    for (auto& [old_dist, node_id] : nodes_to_process) {
        auto other_node = parent_->net_dag->nodes[node_id];
        if (!other_node || !other_node->graph) continue;

        auto other_graph = other_node->graph.get();

        // Compute exact GED using AppForComputation
        Application app(8u, "BMao");
        app.set_all_edge_labels_same(parent_->all_edge_labels_same);
        app.init(anchor_graph, other_graph);
        int exact_ged = app.AppForComputation();

        std::vector<std::pair<ui,ui>> mapping;
        app.get_mapping(mapping);

        // Update if within threshold
        if (exact_ged != INF+1 && exact_ged <= 8) {
            // Add to exact cluster
            Anchor::ExactClusterNode cluster_node;
            cluster_node.dist = exact_ged;
            cluster_node.node_id = node_id;
            cluster_node.mapping = mapping;
            target_anchor->nodes_in_exact_cluster.push(cluster_node);
            updated_count++;
        }
    }

    std::cout << "[INFO] Updated " << updated_count << " nodes in exact cluster for anchor "
              << target_anchor->node_id << std::endl;
}

void EditPathForestConstructor::update_one_exact_anchor_parallel() {
    std::cout << "[INFO] Starting update_one_exact_anchor_parallel..." << std::endl;

    // Find anchor with most nodes to refine
    std::shared_ptr<Anchor> target_anchor = nullptr;
    size_t max_cluster_size = 0;

    for (auto& anchor : parent_->net_dag->anchors) {
        if (!anchor) continue;

        size_t cluster_size = anchor->nodes_in_cluster.size();
        if (cluster_size > max_cluster_size) {
            max_cluster_size = cluster_size;
            target_anchor = anchor;
        }
    }

    if (!target_anchor || max_cluster_size == 0) {
        std::cout << "[INFO] No suitable anchor found for updating." << std::endl;
        return;
    }

    std::cout << "[INFO] Updating anchor " << target_anchor->node_id
              << " with " << max_cluster_size << " nodes in cluster (parallel)." << std::endl;

    // Collect nodes to process
    std::vector<std::pair<ui, ui>> nodes_to_process;
    auto temp_cluster = target_anchor->nodes_in_cluster;

    while (!temp_cluster.empty() && nodes_to_process.size() < 1000) {
        auto [dist, node_id] = temp_cluster.top();
        temp_cluster.pop();

        if (node_id != target_anchor->node_id) {
            nodes_to_process.emplace_back(dist, node_id);
        }
    }

    if (nodes_to_process.empty()) {
        std::cout << "[INFO] No nodes to process for anchor " << target_anchor->node_id << std::endl;
        return;
    }

    std::cout << "[INFO] Processing " << nodes_to_process.size() << " nodes in parallel..." << std::endl;

    // Threading setup
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads > 16) num_threads = 16;
    if (num_threads == 0) num_threads = 4;

    std::atomic<size_t> nextIdx(0);
    std::atomic<int> updated_count(0);
    std::mutex update_mutex;

    // Progress monitoring
    std::atomic<size_t> processed_count(0);
    size_t total_tasks = nodes_to_process.size();

    auto t0 = std::chrono::steady_clock::now();

    // Worker function
    auto worker = [&](int thread_id) {
        std::vector<std::tuple<ui, ui, std::vector<std::pair<ui,ui>>>> local_updates;

        while (true) {
            size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_tasks) break;

            auto [old_dist, node_id] = nodes_to_process[idx];
            auto other_node = parent_->net_dag->nodes[node_id];

            if (!other_node || !other_node->graph) {
                processed_count.fetch_add(1);
                continue;
            }

            auto anchor_graph = target_anchor->graph.get();
            auto other_graph = other_node->graph.get();

            // Compute exact GED using AppForComputation
            Application app(8u, "BMao");
            app.set_all_edge_labels_same(parent_->all_edge_labels_same);
            app.init(anchor_graph, other_graph);
            int exact_ged = app.AppForComputation();

            std::vector<std::pair<ui,ui>> mapping;
            app.get_mapping(mapping);

            // Store valid results
            if (exact_ged != INF+1 && exact_ged <= 8) {
                local_updates.emplace_back(exact_ged, node_id, mapping);
            }

            processed_count.fetch_add(1);

            // Periodic progress report
            if (processed_count.load() % 100 == 0) {
                double progress = 100.0 * processed_count.load() / total_tasks;
                std::cout << "[PROGRESS] " << processed_count.load() << "/" << total_tasks
                         << " (" << (int)progress << "%)\n";
            }
        }

        // Apply updates
        if (!local_updates.empty()) {
            std::lock_guard<std::mutex> lock(update_mutex);
            for (auto& [ged, node_id, mapping] : local_updates) {
                Anchor::ExactClusterNode cluster_node;
                cluster_node.dist = ged;
                cluster_node.node_id = node_id;
                cluster_node.mapping = mapping;
                target_anchor->nodes_in_exact_cluster.push(cluster_node);
                updated_count.fetch_add(1);
            }
        }
    };

    // Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    auto t1 = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[INFO] Parallel update completed in " << duration << " seconds." << std::endl;
    std::cout << "[INFO] Updated " << updated_count.load() << " nodes in exact cluster for anchor "
              << target_anchor->node_id << std::endl;
}

int EditPathForestConstructor::select_queries_from_csv(
    int root_ind,
    int need_count,
    const std::string &csv_path,
    const std::string &out_queries_txt,
    std::string &out_filtered_csv) {

    namespace fs = std::filesystem;

    // Read CSV data
    std::ifstream ifs(csv_path);
    if(!ifs.is_open()){
        std::cerr << "[ERROR] cannot open " << csv_path << "\n";
        return 0;
    }

    std::vector<std::tuple<int,int,int,std::string>> data; // (root, other, ged, mapping)
    {
        std::string line;
        while(std::getline(ifs,line)){
            if(line.empty()) continue;
            std::stringstream ss(line);
            std::string r_str, o_str, g_str, map_str;
            if(!std::getline(ss, r_str, ',')) continue;
            if(!std::getline(ss, o_str, ',')) continue;
            if(!std::getline(ss, g_str, ',')) continue;
            if(!std::getline(ss, map_str))    continue;

            try{
                int r = std::stoi(r_str);
                int o = std::stoi(o_str);
                int g = std::stoi(g_str);
                data.push_back({r,o,g,map_str});
            }catch(...){
                continue;
            }
        }
    }
    ifs.close();

    if(data.empty()){
        std::cerr << "[WARN] CSV is empty => " << csv_path << "\n";
        return 0;
    }
    std::cout << "[INFO] CSV(" << csv_path << ") has " << data.size() << " lines.\n";

    // Collect candidate indices where root matches root_ind
    std::vector<int> candidate_idx;
    candidate_idx.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        if(std::get<0>(data[i]) == root_ind){
            candidate_idx.push_back((int)i);
        }
    }

    std::cout << "[INFO] lines with root=" << root_ind << ": " << candidate_idx.size() << "\n";
    if(candidate_idx.empty()){
        out_filtered_csv = csv_path;
        return 0;
    }

    // Randomly shuffle candidates
    std::shuffle(candidate_idx.begin(), candidate_idx.end(), std::default_random_engine{});

    // Select up to need_count entries
    int pick_count = std::min((int)candidate_idx.size(), need_count);
    std::vector<int> chosen_idx(candidate_idx.begin(),
                               candidate_idx.begin() + pick_count);

    // Write selected queries
    {
        std::ofstream ofsQ(out_queries_txt);
        if(!ofsQ.is_open()){
            std::cerr << "[ERROR] cannot create " << out_queries_txt << "\n";
        } else {
            for(int idx : chosen_idx){
                auto &row = data[idx]; // (root, other, ged, mapping)
                ofsQ << std::get<0>(row) << ","
                     << std::get<1>(row) << ","
                     << std::get<2>(row) << ","
                     << std::get<3>(row) << "\n";
            }
        }
    }
    std::cout << "[INFO] wrote " << chosen_idx.size()
              << " queries => " << out_queries_txt << "\n";

    // Create filtered data (unselected rows)
    std::unordered_set<int> chosenSet(chosen_idx.begin(), chosen_idx.end());
    std::vector<std::tuple<int,int,int,std::string>> filtered;
    filtered.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        if(chosenSet.find((int)i) == chosenSet.end()){
            filtered.push_back(data[i]);
        }
    }

    std::cout << "[INFO] old data size=" << data.size()
              << ", filtered size=" << filtered.size()
              << ", removed=" << (data.size() - filtered.size()) << "\n";

    // Write filtered CSV
    {
        std::ofstream ofsF(out_filtered_csv);
        if(!ofsF.is_open()){
            std::cerr << "[ERROR] cannot create " << out_filtered_csv << "\n";
        } else {
            for(auto &row : filtered){
                ofsF << std::get<0>(row) << ","
                     << std::get<1>(row) << ","
                     << std::get<2>(row) << ","
                     << std::get<3>(row) << "\n";
            }
        }
    }
    std::cout << "[INFO] new CSV => " << out_filtered_csv << "\n";

    return pick_count; // Return actual number of queries selected
}

int EditPathForestConstructor::select_queries_by_ged_ranges(
    int root_ind,
    const std::string &csv_path,
    const std::string &output_dir) {

    namespace fs = std::filesystem;

    // Read CSV data
    std::ifstream ifs(csv_path);
    if(!ifs.is_open()){
        std::cerr << "[ERROR] cannot open " << csv_path << "\n";
        return 0;
    }

    std::vector<std::tuple<int,int,int,std::string>> data; // (root, other, ged, mapping)
    {
        std::string line;
        while(std::getline(ifs,line)){
            if(line.empty()) continue;
            std::stringstream ss(line);
            std::string r_str, o_str, g_str, map_str;
            if(!std::getline(ss, r_str, ',')) continue;
            if(!std::getline(ss, o_str, ',')) continue;
            if(!std::getline(ss, g_str, ',')) continue;
            if(!std::getline(ss, map_str))    continue;

            try {
                int r = std::stoi(r_str);
                int o = std::stoi(o_str);
                int g = std::stoi(g_str);
                data.push_back({r,o,g,map_str});
            } catch(...) {
                continue;
            }
        }
    }
    ifs.close();

    if(data.empty()){
        std::cerr << "[WARN] CSV is empty => " << csv_path << "\n";
        return 0;
    }
    std::cout << "[INFO] CSV(" << csv_path << ") has " << data.size() << " lines.\n";

    // Select rows with root == root_ind
    std::vector<int> candidate_idx;
    candidate_idx.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        if(std::get<0>(data[i]) == root_ind){
            candidate_idx.push_back((int)i);
        }
    }
    std::cout << "[INFO] lines with root=" << root_ind << ": " << candidate_idx.size() << "\n";
    if(candidate_idx.empty()){
        return 0;
    }

    // Group by GED ranges: <=5, =6, =7, =8, =9, =10, others(>10)
    std::vector<int> grp_le5, grp_eq6, grp_eq7, grp_eq8, grp_eq9, grp_eq10, grp_others;
    grp_le5.reserve(candidate_idx.size());
    grp_eq6.reserve(candidate_idx.size());
    grp_eq7.reserve(candidate_idx.size());
    grp_eq8.reserve(candidate_idx.size());
    grp_eq9.reserve(candidate_idx.size());
    grp_eq10.reserve(candidate_idx.size());
    grp_others.reserve(candidate_idx.size());

    for(int idx : candidate_idx) {
        int ged = std::get<2>(data[idx]);
        if(ged <= 5) {
            grp_le5.push_back(idx);
        } else if(ged == 6) {
            grp_eq6.push_back(idx);
        } else if(ged == 7) {
            grp_eq7.push_back(idx);
        } else if(ged == 8) {
            grp_eq8.push_back(idx);
        } else if(ged == 9) {
            grp_eq9.push_back(idx);
        } else if(ged == 10){
            grp_eq10.push_back(idx);
        } else {
            grp_others.push_back(idx);
        }
    }

    // Helper function: randomly pick up to 100 from each group
    auto pick_random_100 = [&](std::vector<int> &grp) {
        std::shuffle(grp.begin(), grp.end(), std::default_random_engine{});
        int pick_count = std::min((int)grp.size(), 100);
        std::vector<int> chosen(grp.begin(), grp.begin() + pick_count);
        grp.erase(grp.begin(), grp.begin() + pick_count);
        return chosen;
    };

    // Pick 100 from each group
    std::vector<int> chosen_le5  = pick_random_100(grp_le5);
    std::vector<int> chosen_eq6  = pick_random_100(grp_eq6);
    std::vector<int> chosen_eq7  = pick_random_100(grp_eq7);
    std::vector<int> chosen_eq8  = pick_random_100(grp_eq8);
    std::vector<int> chosen_eq9  = pick_random_100(grp_eq9);
    std::vector<int> chosen_eq10 = pick_random_100(grp_eq10);
    std::vector<int> chosen_others = pick_random_100(grp_others);

    // Create output directory
    fs::create_directories(output_dir);

    // Helper function to write CSV files
    auto write_csv = [&](const std::string &outpath, const std::vector<int> &indices){
        std::ofstream ofs(outpath);
        if(!ofs.is_open()){
            std::cerr << "[ERROR] cannot create " << outpath << "\n";
            return;
        }
        for(int idx : indices){
            auto &row = data[idx];
            ofs << std::get<0>(row) << ","
                << std::get<1>(row) << ","
                << std::get<2>(row) << ","
                << std::get<3>(row) << "\n";
        }
        ofs.close();
    };

    // Write 7 separate CSV files for different GED ranges
    write_csv(output_dir + "/selected_ged_le5.csv",  chosen_le5);
    write_csv(output_dir + "/selected_ged_6.csv",    chosen_eq6);
    write_csv(output_dir + "/selected_ged_7.csv",    chosen_eq7);
    write_csv(output_dir + "/selected_ged_8.csv",    chosen_eq8);
    write_csv(output_dir + "/selected_ged_9.csv",    chosen_eq9);
    write_csv(output_dir + "/selected_ged_10.csv",   chosen_eq10);
    write_csv(output_dir + "/selected_random.csv",   chosen_others);

    size_t total_chosen = chosen_le5.size() + chosen_eq6.size() + chosen_eq7.size() +
                          chosen_eq8.size() + chosen_eq9.size() + chosen_eq10.size() +
                          chosen_others.size();
    std::cout << "[INFO] total chosen from root=" << root_ind
              << " => " << total_chosen << " lines\n";

    // Create filtered dataset (remove selected entries)
    std::unordered_set<int> chosen_set;
    chosen_set.insert(chosen_le5.begin(),  chosen_le5.end());
    chosen_set.insert(chosen_eq6.begin(),  chosen_eq6.end());
    chosen_set.insert(chosen_eq7.begin(),  chosen_eq7.end());
    chosen_set.insert(chosen_eq8.begin(),  chosen_eq8.end());
    chosen_set.insert(chosen_eq9.begin(),  chosen_eq9.end());
    chosen_set.insert(chosen_eq10.begin(), chosen_eq10.end());
    chosen_set.insert(chosen_others.begin(), chosen_others.end());

    std::vector<std::tuple<int,int,int,std::string>> filtered;
    filtered.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        if(chosen_set.find((int)i) == chosen_set.end()){
            filtered.push_back(data[i]);
        }
    }
    std::cout << "[INFO] old data size=" << data.size()
              << ", new filtered size=" << filtered.size()
              << ", removed=" << (data.size() - filtered.size()) << "\n";

    // Write filtered CSV
    std::string new_csv = output_dir + "/WithinTau_"
                        + std::to_string(root_ind)
                        + "_filtered.csv";
    {
        std::ofstream ofs3(new_csv);
        if(!ofs3.is_open()){
            std::cerr << "[ERROR] cannot create " << new_csv << "\n";
        } else {
            for(auto &row : filtered){
                ofs3 << std::get<0>(row) << ","
                     << std::get<1>(row) << ","
                     << std::get<2>(row) << ","
                     << std::get<3>(row) << "\n";
            }
        }
    }
    std::cout << "[INFO] new CSV => " << new_csv << "\n";

    return (int)total_chosen; // Return total number of selected entries
}

// Helper function implementation
void EditPathForestConstructor::build_edit_path_tree(
    EditPathTree& ept,
    const std::vector<std::vector<EditOperation>>& edit_operations_list,
    const std::vector<ui>& db_graph_ids,
    ui anchor_id,
    const Graph& anchor_graph)
{
    int ept_node_counter = 0;
    ept.tree_nodes.clear();

    PseudoGraph root_pseudo_graph(anchor_graph);
    std::shared_ptr<Graph> root_graph_ptr = std::make_shared<Graph>(anchor_graph);

    EditOperation dummy_op;
    dummy_op.type = static_cast<EditOperation::OperationType>(-1);

    // Create root node
    TreeNode root_node(dummy_op, 0, ept_node_counter, anchor_id, ept_node_counter, SIZE_MAX, root_pseudo_graph);
    root_node.graph_ptr = root_graph_ptr;
    root_node.graph_ptr->id = std::to_string(root_node.tree_node_graph_id);
    ept_node_counter++;

    root_node.accumulated_ops.clear();

    ept.tree_nodes.push_back(root_node);
    ept.root_index = 0;

    size_t num_graphs = edit_operations_list.size();
    std::vector<std::unordered_set<EditOperation, EditOperationHash, EditOperationEqual>> undone_ops_lists(num_graphs);
    for (size_t i = 0; i < num_graphs; ++i) {
        for (const auto& op : edit_operations_list[i]) {
            undone_ops_lists[i].insert(op);
        }
    }

    std::vector<ui> indices(num_graphs);
    for (size_t i = 0; i < num_graphs; ++i) {
        indices[i] = (ui)i;
    }

    build_edit_path_tree_recursive(ept, undone_ops_lists, db_graph_ids, anchor_id,
                                   ept.root_index, indices, 0, ept_node_counter);

    // print_tree(ept); // Debug function removed
    simplify_EPT(ept);
    // print_tree(ept); // Debug function removed

    ept.reorder();
    // print_tree(ept); // Debug function removed

    // Coverage check
    std::queue<size_t> q;
    q.push(ept.root_index);

    std::vector<int> all_completed_ids_list;
    while (!q.empty()) {
        size_t nid = q.front(); q.pop();
        const TreeNode& node = ept.tree_nodes[nid];

        for (int cid : node.completed_db_graph_ids) {
            all_completed_ids_list.push_back(cid);
        }

        for (auto child_idx : node.children_indices) {
            q.push(child_idx);
        }
    }

    std::unordered_set<int> all_completed_ids(all_completed_ids_list.begin(), all_completed_ids_list.end());

    std::unordered_set<int> input_db_set;
    for (auto id : db_graph_ids) {
        input_db_set.insert((int)id);
    }

    size_t should_contain_count = input_db_set.size();
    size_t actually_contain_count = 0;

    for (auto cid : input_db_set) {
        if (all_completed_ids.find(cid) != all_completed_ids.end()) {
            actually_contain_count++;
        }
    }

    if (actually_contain_count < should_contain_count) {
        std::cout << "Missing these db_graph_ids:\n";
        std::vector<int> missing_ids;
        std::unordered_map<int, size_t> dbid_to_index;
        for (size_t i = 0; i < db_graph_ids.size(); ++i) {
            dbid_to_index[(int)db_graph_ids[i]] = i;
        }

        for (auto cid : input_db_set) {
            if (all_completed_ids.find(cid) == all_completed_ids.end()) {
                missing_ids.push_back(cid);
                std::cout << cid << " ";
            }
        }
        std::cout << std::endl;

        std::cout << "Printing op_list for missing db_graph_ids:\n";
        for (int mid : missing_ids) {
            if (dbid_to_index.find(mid) != dbid_to_index.end()) {
                size_t idx = dbid_to_index[mid];
                const auto& op_list = edit_operations_list[idx];
                std::cout << "DB Graph ID: " << mid << ", op_list size: " << op_list.size() << "\n";
                for (const auto& op : op_list) {
                    std::cout << op.to_string() << "\n";
                }
            } else {
                std::cout << "DB Graph ID " << mid << " not found in dbid_to_index map.\n";
            }
        }
    }
}