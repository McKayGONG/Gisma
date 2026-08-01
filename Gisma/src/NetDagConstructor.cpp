#include "NetDagConstructor.h"
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
#include <mutex>
#include <iomanip>
#include <random>
#include <algorithm>
#include <sstream>
#include <set>
#include <vector>
#include <utility>

namespace fs = std::filesystem;
static std::mutex g_write_mutex;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

NetDagConstructor::NetDagConstructor(GismaIndexBuilder* parent) : parent_(parent) {}

void NetDagConstructor::initialize() {
    // 打印 graphs.size() 和 has_ged_matrix 的值
    std::cout << "Graphs size: " << parent_->graphs.size() << std::endl;
    std::cout << "Has GED Matrix: " << (parent_->has_ged_matrix ? "True" : "False") << std::endl;

    // 获取root节点的embedding向量
    std::vector<float> root_embedding;
    if (parent_->root_ind < parent_->embeddings.size()) {
        root_embedding = parent_->embeddings[parent_->root_ind];
    }

    // 创建根节点
    auto root_node = std::make_shared<Anchor>(
        parent_->root_ind,                   // node_id
        parent_->graphs[parent_->root_ind],     // graph
        parent_->nameList[parent_->root_ind],   // file_name
        parent_->root_ind,                   // nearest_anchor
        0.0,                              // nearest_anchor_dist
        root_embedding,                   // 第6个参：使用正确的embedding向量
        static_cast<int>(parent_->anchors.size()) // 第7个参：anchor_id
    );
    

    if (!root_node) {
        std::cerr << "Error: root_node is nullptr!" << std::endl;
        return;
    }

    std::cout << "Root node created with ID: " << root_node->node_id << std::endl;
    std::cout << "index_name: " << parent_->index_name << std::endl;
    std::cout << "alpha: " << parent_->alpha << std::endl;
    std::cout << "tau: " << parent_->tau << std::endl;

    if (root_node->graph) {
        // 若 graph 还没初始化，则先初始化向量数据
        root_node->graph->initialize_vectors_from_arrays();
        // Graph vectors initialized for embedding-based distance calculation
    } else {
        std::cerr << "Warning: root_node->graph is nullptr, cannot initialize vectors.\n";
    }

    // 创建 NetDag 并将根节点添加到 anchors 和 nodes 中
    parent_->net_dag = std::make_shared<NetDag>(parent_->index_name, root_node, parent_->alpha, parent_->tau);
    parent_->anchors.push_back(root_node);
    parent_->net_dag->add_anchor(root_node);

    

    if (root_node->node_id >= parent_->nodes.size()) {
        parent_->nodes.resize(root_node->node_id + 1);
    }
    parent_->nodes[root_node->node_id] = root_node;

    parent_->child_range = 3 * parent_->r_k + 2 * parent_->tau;
    parent_->r_list.push_back(parent_->r_k);

    // 遍历所有图，创建每个节点并添加到 nodes 和根节点的 nodes_in_cluster 中
    for (size_t i = 0; i < parent_->graphs.size(); ++i) {
        if (i != parent_->root_ind) {
            // 获取对应的embedding向量
            std::vector<float> node_embedding;
            if (i < parent_->embeddings.size()) {
                node_embedding = parent_->embeddings[i];
            }

            auto new_node = std::make_shared<Node>(
                i,
                parent_->graphs[i],
                parent_->nameList[i],
                false,
                root_node->node_id,
                0.0,
                node_embedding
            );
            if (new_node->graph) {
                new_node->graph->initialize_vectors_from_arrays();
                // Graph vectors initialized for embedding-based distance calculation
            } else {
                std::cerr << "Warning: new_node->graph is nullptr, cannot initialize vectors.\n";
            }

            if (!new_node) {
                std::cerr << "Error: new_node is nullptr!" << std::endl;
                continue;
            }
            double nearest_anchor_dist = this->compute_ged_online(root_node, new_node);

            new_node->nearest_anchor_dist = nearest_anchor_dist;
            
            if (new_node->node_id >= parent_->nodes.size()) {
                parent_->nodes.resize(new_node->node_id + 1);
            }
            parent_->nodes[new_node->node_id] = new_node;

            

            // 输出每个 new_node 的信息 (commented out for performance)
            // std::cout << "New node created: ID = " << new_node->node_id
            //           << ", Nearest anchor dist = " << nearest_anchor_dist << std::endl;

            root_node->nodes_in_cluster.push({nearest_anchor_dist, i});
        }
    }

    if (!root_node->nodes_in_cluster.empty()) {
        double furthest_node_in_cluster_dist = root_node->nodes_in_cluster.top().first;
        // std::cout << "Furthest node distance in cluster: " << furthest_node_in_cluster_dist << std::endl;
        // BUG FIX: Don't pop the furthest node - this loses coverage!
        // The node is lost because it's not saved anywhere else.
        // root_node->nodes_in_cluster.pop();  // 弹出最大距离的节点
    }

    parent_->phaseList = getStateList(parent_->r_k, parent_->alpha);

    // 更新 phaseList 和其他信息
    this->update_phase();
    this->find_max_dist();
    this->update_child_and_parent_by_phase_dict();

    // 验证初始化后的覆盖情况
    verify_coverage("after_initialize");
}

std::vector<double> NetDagConstructor::getStateList(double maxRadius, double alpha) {
    std::vector<double> stateList;
    stateList.push_back(alpha);
    double state = alpha;

    while (state < maxRadius) {
        state *= 2;
        stateList.push_back(state);
    }
    std::reverse(stateList.begin(), stateList.end());
    return stateList;
}

void NetDagConstructor::update_phase() {
    if (parent_->phaseList.empty()) {
        std::cerr << "Error: phaseList is empty!" << std::endl;
        return;
    }

    // 使用 std::upper_bound 查找第一个 <= max_dist 的元素
    auto it = std::upper_bound(
        parent_->phaseList.begin(),
        parent_->phaseList.end(),
        parent_->max_dist,
        [](double value, double element) {
            // 自定义比较器，适应降序排序
            return value > element;
        }
    );

    if (it == parent_->phaseList.begin()) {
        // 所有元素都小于 max_dist，选择第一个元素
        parent_->phase = parent_->phaseList.front();
    } else {
        // 设置 phase 为找到的位置前一个元素
        parent_->phase = *(it - 1);
    }

    // ========== Trigger callback for statistics collection (select-alpha mode) ==========
    // Check if r_k crossed an integer boundary
    static int last_integer_r = -1;
    int current_integer_r = static_cast<int>(std::floor(parent_->r_k));

    if (current_integer_r != last_integer_r && current_integer_r >= 0) {
        // r_k crossed an integer boundary, trigger callback if set
        const auto& callback = parent_->get_cluster_stats_callback();
        if (callback) {
            callback(parent_->r_k, parent_->anchors, parent_->phase);
        }
        last_integer_r = current_integer_r;
    }
    // ==================================================================================

    // 可选：打印更新后的 phase 以调试
    // std::cout << "Updated phase to: " << parent_->phase << std::endl;
}

double NetDagConstructor::compute_ged_online(const std::shared_ptr<Node> &node1, const std::shared_ptr<Node> &node2) {
    parent_->NDC++;

    double ged = 0;
    if (parent_->has_ged_matrix) {
        ged = parent_->get_ged_from_matrix(node1->node_id, node2->node_id);
    } else {
        ged = Utility::euclidean_distance(node1->embedding, node2->embedding);

        // Debug: Print first 5 pairs involving node 0
        static int debug_count = 0;
        if (node1->node_id == 0 || node2->node_id == 0) {
            if (debug_count < 5) {
                std::cout << "[DEBUG_GED] Pair (" << node1->node_id << ", " << node2->node_id << "):" << std::endl;
                std::cout << "  has_ged_matrix=" << parent_->has_ged_matrix << std::endl;
                std::cout << "  emb_dist=" << ged << std::endl;
                debug_count++;
            }
        }
    }
    return ged;
}

// Note: This implementation will be quite large. I need to add placeholder implementations for now
// and then extract the real implementations from the original file in subsequent steps.

void NetDagConstructor::overall_NetDag_const() {
    // 创建目录（如果不存在）
    // 从index_name提取数据集名称（格式：dataset_size_alpha_tau_error）
    std::string dataset_name = parent_->index_name.substr(0, parent_->index_name.find('_'));
    std::string directory = "./NetDags/" + dataset_name + "/configs";
    if (!fs::exists(directory)) {
        fs::create_directories(directory);
    }
    std::string file_path = directory + "/" + parent_->index_name + ".dat";

    // 初始化计时器
    std::map<std::string, double> timing = {
        {"initialize", 0},
        {"find_max_dist", 0},
        {"update_phase", 0},
        {"update_friends", 0},
        {"update_child_and_parent_by_phase_dict", 0},
        {"link_parent_and_child", 0},
        {"update_one_anchor", 0},
        {"get_nodes_in_cover_range", 0},
        {"update_one_exact_anchor", 0}
    };

    // 记录 overall_NetDag_const 函数开始执行的时间
    auto start_overall = std::chrono::high_resolution_clock::now();

    // initialize
    auto start = std::chrono::high_resolution_clock::now();
    initialize();
    auto end = std::chrono::high_resolution_clock::now();
    timing["initialize"] += std::chrono::duration<double>(end - start).count();
    std::cout << "Initialization completed." << std::endl;
    std::cout << "max_dist = " << parent_->max_dist << std::endl;
    std::cout << "alpha = " << parent_->alpha << std::endl;
    
    while (parent_->max_dist > parent_->alpha) {
        int phase_before = parent_->phase;

        // 打印调试信息
        std::cout << "Entering loop with max_dist = " << parent_->max_dist 
                  << ", alpha = " << parent_->alpha << std::endl;
        std::cout << "Number of anchors: " << parent_->anchors.size() << std::endl;

        // find_max_dist
        start = std::chrono::high_resolution_clock::now();
        find_max_dist();
        end = std::chrono::high_resolution_clock::now();
        timing["find_max_dist"] += std::chrono::duration<double>(end - start).count();
        std::cout << "find_max_dist completed. max_dist = " << parent_->max_dist << std::endl;

        // update_phase
        start = std::chrono::high_resolution_clock::now();
        update_phase();
        end = std::chrono::high_resolution_clock::now();
        timing["update_phase"] += std::chrono::duration<double>(end - start).count();
        std::cout << "update_phase completed. Current phase = " << parent_->phase << std::endl;

        if (parent_->phase < phase_before) {
            std::cout << "Phase decreased from " << phase_before << " to " << parent_->phase
                      << ", entering update block..." << std::endl;
            // update_friends
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Starting update_friends..." << std::endl;
            update_friends();
            end = std::chrono::high_resolution_clock::now();
            timing["update_friends"] += std::chrono::duration<double>(end - start).count();
            std::cout << "update_friends completed in "
                      << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

            // 更新 parent_by_phase_dict
            parent_->parent_by_phase_dict[parent_->phase] = std::vector<int>();
            for (const auto& anchor : parent_->anchors) {
                parent_->parent_by_phase_dict[parent_->phase].push_back(anchor->node_id);
            }
            std::cout << "Updated parent_by_phase_dict for phase = " << parent_->phase << std::endl;
            std::cout << "Number of anchors: " << parent_->anchors.size() << std::endl;

            // update_child_and_parent_by_phase_dict
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Starting update_child_and_parent_by_phase_dict..." << std::endl;
            update_child_and_parent_by_phase_dict();
            end = std::chrono::high_resolution_clock::now();
            timing["update_child_and_parent_by_phase_dict"] += std::chrono::duration<double>(end - start).count();
            std::cout << "update_child_and_parent_by_phase_dict completed in "
                      << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

            // link_parent_and_child
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Starting link_parent_and_child..." << std::endl;
            link_parent_and_child();
            end = std::chrono::high_resolution_clock::now();
            timing["link_parent_and_child"] += std::chrono::duration<double>(end - start).count();
            std::cout << "link_parent_and_child completed in "
                      << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
        } else {
            std::cout << "Phase unchanged or increased (phase_before=" << phase_before
                      << ", current phase=" << parent_->phase << "), skipping update block" << std::endl;
        }

        std::cout << "Number of anchors = " << parent_->anchors.size() << std::endl;
        std::cout << "Current r_k = " << parent_->r_k << std::endl;

        if (parent_->max_dist < parent_->alpha) {
            std::cout << "Breaking loop as max_dist < alpha." << std::endl;
            break;
        }

        // update_one_anchor
        start = std::chrono::high_resolution_clock::now();
        update_one_anchor();
        end = std::chrono::high_resolution_clock::now();
        timing["update_one_anchor"] += std::chrono::duration<double>(end - start).count();
        std::cout << "update_one_anchor completed." << std::endl;

        // 每次update_one_anchor后验证覆盖情况，追踪新增丢失
        static int last_uncovered = 0;
        int current_uncovered = verify_coverage("after_anchor_" + std::to_string(parent_->anchors.size()));
        if (current_uncovered > last_uncovered) {
            std::cout << "[WARNING] Uncovered increased by " << (current_uncovered - last_uncovered)
                      << " after creating anchor " << parent_->anchors.back()->node_id << std::endl;
        }
        last_uncovered = current_uncovered;
    }

    // 循环结束后最终验证
    verify_coverage("after_main_loop");

    // 重新分配节点到精确簇
    // reassign_nodes_in_cluster_with_csv();

    // 使用 update_one_exact_anchor 处理剩余的 nodes_in_cluster
    start = std::chrono::high_resolution_clock::now();
    end = std::chrono::high_resolution_clock::now();
    timing["update_one_exact_anchor"] += std::chrono::duration<double>(end - start).count();
    std::cout << "All nodes have been assigned to exact clusters." << std::endl;

    
    // 更新 net_dag 对象的成员变量
    parent_->net_dag->parent_by_phase_dict = parent_->parent_by_phase_dict;
    parent_->net_dag->anchors = parent_->anchors;
    parent_->net_dag->nodes = parent_->nodes;

    // construct_cluster_EPT();

    // 打印每个函数的总运行时间
    for (const auto& [func, total_time] : timing) {
        std::cout << "Total time for " << func << ": " << total_time << " seconds" << std::endl;
    }

    // 记录 overall_NetDag_const 函数结束执行的时间，并计算总时间
    auto end_overall = std::chrono::high_resolution_clock::now();
    double total_time_overall = std::chrono::duration<double>(end_overall - start_overall).count();
    std::cout << "Total time for overall_NetDag_const: " << total_time_overall << " seconds" << std::endl;
    std::cout << "Construct NDC = " << parent_->NDC << std::endl;
    // E1: NetDAG structure build is essentially serial here, so wall == work.
    std::cout << "[E1_ND] wall_seconds=" << total_time_overall
              << " work_seconds=" << total_time_overall
              << " ndc=" << parent_->NDC << std::endl;
    Utility::append_e1_timing("index_name=" + parent_->index_name + "\tdataset=" + parent_->dataset
        + "\tstage=construct_ND\tthreads=1\twall_seconds=" + std::to_string(total_time_overall)
        + "\twork_seconds=" + std::to_string(total_time_overall)
        + "\tndc=" + std::to_string(parent_->NDC));

    // BEFORE FILTERING: Save Anchor 0's friends to debug file
    std::string friends_debug_dir = "./NetDags/" + parent_->dataset + "/debug/";
    std::filesystem::create_directories(friends_debug_dir);
    std::string friends_file = friends_debug_dir + "anchor0_friends_before_filter.txt";
    std::ofstream friends_ofs(friends_file);
    if (friends_ofs) {
        for (const auto& anchor : parent_->anchors) {
            if (anchor && anchor->node_id == 0) {
                friends_ofs << "Anchor 0 friends count: " << anchor->friends.size() << "\n";
                friends_ofs << "Friends (distance, node_id):\n";
                for (const auto& [dist, friend_id] : anchor->friends) {
                    friends_ofs << dist << " " << friend_id << "\n";
                }
                std::cout << "[DEBUG] Saved Anchor 0's " << anchor->friends.size()
                          << " friends (BEFORE filter) to " << friends_file << std::endl;
                break;
            }
        }
        friends_ofs.close();
    }

    // 在保存之前筛选朋友关系，只保留距离 <= 4*alpha 的朋友
    // filter_friends_before_saving();  // DISABLED: Not needed since friends are not used in search

    // 保存 net_dag 对象到文件
    std::cout << "Saving NetDag object to file..." << std::endl;
    parent_->net_dag->save_to_file(file_path);
    std::cout << "NetDag object saved to file." << std::endl;

    std::cout << "Printing NetDag information:" << std::endl;
}

void NetDagConstructor::net_tree_const() {
    // ================================
    // 1) 创建输出目录、文件路径
    // ================================
    std::string directory = "./NetTrees";
    if (!std::filesystem::exists(directory)) {
        std::filesystem::create_directories(directory);
    }
    std::string file_path = directory + "/" + parent_->index_name;

    // ================================
    // 2) 需要的数据结构
    //    boundary2anchors ：记录在"跨过某个整数"时有哪些 anchor（ID）
    //    boundary2clusterSize ：记录在"跨过某个整数"时，每个 anchor 的 cluster 大小
    // ================================
    std::map<int, std::vector<int>> boundary2anchors;
    std::map<int, std::vector<std::pair<int,int>>> boundary2clusterSize;

    // 记录上一次 floor(max_dist) 值，在 initialize() 后设置
    int last_recorded_intDist = -1;

    // ================================
    // 3) 初始化计时器
    // ================================
    std::map<std::string, double> timing = {
        {"initialize", 0},
        {"find_max_dist", 0},
        {"update_phase", 0},
        {"update_friends", 0},
        {"update_child_and_parent_by_phase_dict", 0},
        {"link_parent_and_child", 0},
        {"update_one_anchor", 0},
        {"get_nodes_in_cover_range", 0},
        {"update_one_exact_anchor", 0}
    };
    auto start_overall = std::chrono::high_resolution_clock::now();

    // ================================
    // 4) initialize
    // ================================
    auto start = std::chrono::high_resolution_clock::now();
    this->initialize();  // <-- 你的初始化逻辑
    auto end = std::chrono::high_resolution_clock::now();
    timing["initialize"] += std::chrono::duration<double>(end - start).count();

    std::cout << "Initialization completed." << std::endl;
    std::cout << "max_dist = " << parent_->max_dist << std::endl;
    std::cout << "alpha = " << parent_->alpha << std::endl;

    // 在初始化完成后，记录下当前的 floor(max_dist)
    last_recorded_intDist = static_cast<int>(std::floor(parent_->max_dist));
    if (last_recorded_intDist >= 0) {
        // 先把此时的 anchors 记录到 boundary2anchors
        std::vector<int> current_anchors;
        current_anchors.reserve(parent_->anchors.size());

        // 也收集所有 anchor 的 (anchorID, cluster_size)
        std::vector<std::pair<int,int>> anchor_cluster_sizes;
        anchor_cluster_sizes.reserve(parent_->anchors.size());

        for (auto &anc : parent_->anchors) {
            current_anchors.push_back(anc->node_id);

            // 这里假设 nodes_in_cluster 是一个 priority_queue，
            // size() 即为堆里元素数量
            int cluster_size = static_cast<int>(anc->nodes_in_cluster.size());
            anchor_cluster_sizes.push_back({anc->node_id, cluster_size});
        }
        boundary2anchors[last_recorded_intDist] = current_anchors;
        boundary2clusterSize[last_recorded_intDist] = anchor_cluster_sizes;

        // 触发回调（如果设置了的话）- for select-alpha mode
        const auto& callback = parent_->get_cluster_stats_callback();
        if (callback) {
            callback(parent_->max_dist, parent_->anchors, parent_->phase);
        }
    }

    // ================================
    // 5) 进入主循环
    // ================================
    while (parent_->max_dist > parent_->alpha) {
        int phase_before = parent_->phase;

        // ============= find_max_dist =============
        start = std::chrono::high_resolution_clock::now();
        this->find_max_dist(); // <-- 你的逻辑：更新 max_dist
        end = std::chrono::high_resolution_clock::now();
        timing["find_max_dist"] += std::chrono::duration<double>(end - start).count();
        std::cout << "find_max_dist completed. max_dist = " << parent_->max_dist << std::endl;

        // ============= update_phase =============
        start = std::chrono::high_resolution_clock::now();
        this->update_phase();  // <-- 更新 phase
        end = std::chrono::high_resolution_clock::now();
        timing["update_phase"] += std::chrono::duration<double>(end - start).count();
        std::cout << "update_phase completed. Current phase = " << parent_->phase << std::endl;

        // 如果阶段下降了，才去更新 friends、父子链接等
        if (parent_->phase < phase_before) {
            // ---------- update_friends ----------
            start = std::chrono::high_resolution_clock::now();
            this->update_friends();  // <-- 你的 update_friends 逻辑
            end = std::chrono::high_resolution_clock::now();
            timing["update_friends"] += std::chrono::duration<double>(end - start).count();
            std::cout << "update_friends completed." << std::endl;

            // ---------- 更新 parent_by_phase_dict ----------
            parent_->parent_by_phase_dict[parent_->phase] = std::vector<int>();
            for (const auto& anchor : parent_->anchors) {
                parent_->parent_by_phase_dict[parent_->phase].push_back(anchor->node_id);
            }
            std::cout << "Updated parent_by_phase_dict for phase = " << parent_->phase << std::endl;

            // ---------- update_child_and_parent_by_phase_dict ----------
            start = std::chrono::high_resolution_clock::now();
            this->update_child_and_parent_by_phase_dict(); // <-- 若需要可以解开
            end = std::chrono::high_resolution_clock::now();
            timing["update_child_and_parent_by_phase_dict"] += std::chrono::duration<double>(end - start).count();
            std::cout << "update_child_and_parent_by_phase_dict completed." << std::endl;

            // ---------- link_parent_and_child ----------
            start = std::chrono::high_resolution_clock::now();
            this->link_parent_and_child(); // <-- 若需要可以解开
            end = std::chrono::high_resolution_clock::now();
            timing["link_parent_and_child"] += std::chrono::duration<double>(end - start).count();
            std::cout << "link_parent_and_child completed." << std::endl;
        }

        std::cout << "Number of anchors = " << parent_->anchors.size() << std::endl;
        std::cout << "Current r_k = " << parent_->r_k << std::endl;

        if (parent_->max_dist < parent_->alpha) {
            std::cout << "Breaking loop as max_dist < alpha." << std::endl;
            break;
        }

        // ============= update_one_anchor =============
        start = std::chrono::high_resolution_clock::now();
        this->update_one_anchor(); // <-- 你的逻辑：更新/新增 anchor
        end = std::chrono::high_resolution_clock::now();
        timing["update_one_anchor"] += std::chrono::duration<double>(end - start).count();
        std::cout << "update_one_anchor completed." << std::endl;

        // 在本次 iteration 结束后，看 max_dist 是否跨过了新的整数
        int new_floor = static_cast<int>(std::floor(parent_->max_dist));
        while (new_floor < last_recorded_intDist) {
            // （比如 from 21.2 -> 19.8，说明跨过 21 和 20）

            // 1) 收集当前所有 anchor 的 ID
            std::vector<int> current_anchors;
            current_anchors.reserve(parent_->anchors.size());

            // 2) 收集所有 anchor 的 (anchorID, cluster_size)
            std::vector<std::pair<int,int>> anchor_cluster_sizes;
            anchor_cluster_sizes.reserve(parent_->anchors.size());

            for (auto &anc : parent_->anchors) {
                current_anchors.push_back(anc->node_id);
                int cluster_size = static_cast<int>(anc->nodes_in_cluster.size());
                anchor_cluster_sizes.push_back({anc->node_id, cluster_size});
            }

            // 3) 保存到数据结构
            boundary2anchors[last_recorded_intDist - 1] = current_anchors;
            boundary2clusterSize[last_recorded_intDist - 1] = anchor_cluster_sizes;

            // 4) 触发回调（如果设置了的话）- for select-alpha mode
            //    注意：这里传递的是 last_recorded_intDist - 1，表示刚刚跨过的整数边界
            const auto& callback = parent_->get_cluster_stats_callback();
            if (callback) {
                // 传递整数边界值而不是 max_dist，避免重复记录
                callback(static_cast<double>(last_recorded_intDist - 1), parent_->anchors, parent_->phase);
            }

            // 5) 递减 last_recorded_intDist
            last_recorded_intDist--;
        }
    }

    // 最后保存NetTree到文件
    std::cout << "NetDag construction completed. Saving to " << file_path << std::endl;
    // TODO: 可以在这里添加保存NetTree到文件的逻辑
}

void NetDagConstructor::find_max_dist() {
    double tmp_max_dist = -1.0;
    std::shared_ptr<Node> tmp_max_node = nullptr;
    int tmp_max_anchor_id = -1;
    
    // 遍历所有 anchors
    for (const auto& anchor : parent_->anchors) { 
        if (!anchor->nodes_in_cluster.empty()) {          
            // 使用最大堆的堆顶（最远距离）
            auto furthest_node_tuple = anchor->nodes_in_cluster.top(); // 堆顶元素
            double furthest_dist_of_this_anchor = furthest_node_tuple.first; // 取负值恢复原始距离
            if (furthest_dist_of_this_anchor > tmp_max_dist) {
                tmp_max_dist = furthest_dist_of_this_anchor;
                tmp_max_node = parent_->nodes[furthest_node_tuple.second];
                tmp_max_anchor_id = anchor->node_id;
               
            }
        }
    }
    
    parent_->max_dist = tmp_max_dist;
    parent_->max_node = tmp_max_node;
    parent_->max_anchor_node_id = tmp_max_anchor_id;
}

void NetDagConstructor::update_friends() {
    for (auto& anchor : parent_->anchors) {
        double threshold = 8 * parent_->r_k + 4 * parent_->error_tolerance_index;

        // 确保 friends 是一个最大堆（默认行为，堆顶是最大值）
        std::make_heap(anchor->friends.begin(), anchor->friends.end());

        // 当堆顶元素的距离大于阈值时，移除堆顶元素
        while (!anchor->friends.empty() && anchor->friends.front().first > threshold) {
            std::pop_heap(anchor->friends.begin(), anchor->friends.end());
            anchor->friends.pop_back();
        }
    }
}

void NetDagConstructor::update_child_and_parent_by_phase_dict() {
    for (const auto& anchor_ptr : parent_->anchors) {
        // 创建一个副本来遍历 priority_queue
        std::priority_queue<std::pair<double, int>> nodes_copy = anchor_ptr->nodes_in_cluster;

        while (!nodes_copy.empty()) {
            auto node_tuple = nodes_copy.top();
            nodes_copy.pop();

            double distance = node_tuple.first;
            int node_id = node_tuple.second;

            // 更新 child_and_parent_by_phase_dict
            parent_->child_and_parent_by_phase_dict[parent_->phase][node_id] = anchor_ptr->node_id;
        }
    }
}

void NetDagConstructor::link_parent_and_child() {
    // 计算 child_range
    parent_->child_range = 3 * parent_->r_k + 2 * parent_->tau + 4 * parent_->error_tolerance_index;

    // 计算 last_phase
    int last_phase = parent_->phase * 2;
    std::cout << "last_phase= " << last_phase << std::endl;

    // 获取 last_phase_anchors
    auto it = parent_->parent_by_phase_dict.find(last_phase);
    if (it == parent_->parent_by_phase_dict.end()) {
        std::cerr << "No anchors found for last_phase: " << last_phase << std::endl;
        return;
    }
    std::vector<int> last_phase_anchors = it->second;

    // 创建 tmp_dist_dict，键为 (a, b) 对，值为距离
    std::map<std::pair<int, int>, double> tmp_dist_dict;

    for (int last_phase_anchor_id : last_phase_anchors) {
        auto last_phase_anchor_node = parent_->nodes[last_phase_anchor_id];

        // 尝试将 Node 转换为 Anchor
        auto last_phase_anchor = std::dynamic_pointer_cast<Anchor>(last_phase_anchor_node);
        if (!last_phase_anchor) {
            std::cerr << "Node " << last_phase_anchor_id << " is not an Anchor." << std::endl;
            continue;
        }

        for (const auto& friend_tuple : last_phase_anchor->friends) {
            int node_id = friend_tuple.second;

            double tmp_dist;
            int a = std::min(node_id, last_phase_anchor_id);
            int b = std::max(node_id, last_phase_anchor_id);
            std::pair<int, int> key = {a, b};
            auto dist_it = tmp_dist_dict.find(key);
            if (dist_it != tmp_dist_dict.end()) {
                tmp_dist = dist_it->second;
            } else {
                tmp_dist = compute_ged_online(last_phase_anchor, parent_->nodes[node_id]);
                tmp_dist_dict[key] = tmp_dist;
            }

            if (tmp_dist <= parent_->child_range) {
                auto node = parent_->nodes[node_id];
                this->add_edge(last_phase_anchor, node, parent_->phase, tmp_dist);
            }
        }

        // 添加自连接
        this->add_edge(last_phase_anchor, last_phase_anchor, parent_->phase, 0.0);
    }
}

void NetDagConstructor::update_one_anchor() {
    // 确保 max_node 是有效的
    if (!parent_->max_node) {
        std::cerr << "Error: max_node is nullptr!" << std::endl;
        return;
    }

    // 尝试将 max_node 转换为 Anchor 类型
    std::shared_ptr<Anchor> existing_anchor = std::dynamic_pointer_cast<Anchor>(parent_->max_node);

    if (existing_anchor) {
        // max_node 已经是 Anchor，直接使用它
        std::cout << "max_node is already an Anchor with ID: " << existing_anchor->node_id << std::endl;
    } else {
        // 创建新的 Anchor 对象
        auto new_anchor = std::make_shared<Anchor>(
            parent_->max_node->node_id,
            parent_->max_node->graph,
            parent_->max_node->file_name,
            parent_->max_node->nearest_anchor,
            parent_->max_node->nearest_anchor_dist,
            parent_->max_node->embedding,
            static_cast<int>(parent_->anchors.size())
        );

        // 更新节点列表，将 max_node 更新为新的 Anchor
        parent_->nodes[parent_->max_node->node_id] = new_anchor;

        // 将新的 Anchor 添加到 anchors 列表中
        parent_->anchors.push_back(new_anchor);

        // 打印当前锚点数量
        std::cout << "Current number of anchors: " << parent_->anchors.size() << std::endl;

        // 重新赋值 existing_anchor 为新创建的 Anchor
        existing_anchor = new_anchor;
    }

    // 从原 anchor 的 cluster 中移除新成为 anchor 的节点
    // 因为 anchor 不应该出现在其他 anchor 的 cluster 中
    auto old_anchor_node = std::dynamic_pointer_cast<Anchor>(parent_->nodes[parent_->max_anchor_node_id]);
    if (old_anchor_node && old_anchor_node->node_id != existing_anchor->node_id) {
        // 重建 nodes_in_cluster，跳过新 anchor
        std::priority_queue<std::pair<double, int>> new_cluster;
        while (!old_anchor_node->nodes_in_cluster.empty()) {
            auto node_pair = old_anchor_node->nodes_in_cluster.top();
            old_anchor_node->nodes_in_cluster.pop();
            if (node_pair.second != existing_anchor->node_id) {
                new_cluster.push(node_pair);
            }
        }
        old_anchor_node->nodes_in_cluster = std::move(new_cluster);
    }

    // 更新 r_k 和 child_range
    parent_->r_k = parent_->max_dist;
    parent_->child_range = 3 * parent_->r_k + 2 * parent_->tau + 4 * parent_->error_tolerance_index;
    parent_->r_list.push_back(parent_->r_k);

    // 更新 phase
    this->update_phase();

    // Debug: Track specific nodes
    std::vector<int> tracked_nodes = {15981, 1534, 17480, 17220, 20381};
    bool is_tracked = std::find(tracked_nodes.begin(), tracked_nodes.end(), existing_anchor->node_id) != tracked_nodes.end();

    std::vector<int> friend_candidates;

    if (parent_->child_and_parent_by_phase_dict.find(parent_->phase * 2) != parent_->child_and_parent_by_phase_dict.end()) {
        const std::unordered_map<int, int>& parents_map = parent_->child_and_parent_by_phase_dict[parent_->phase * 2];

        if (parents_map.find(parent_->max_node->node_id) != parents_map.end()) {
            int parent_2_phase_ago_id = parents_map.at(parent_->max_node->node_id);
            auto parent_2_phase_ago_node = parent_->nodes[parent_2_phase_ago_id];
            auto parent_anchor_node = std::dynamic_pointer_cast<Anchor>(parent_2_phase_ago_node);

            if (parent_anchor_node) {
                if (is_tracked) {
                    std::cout << "[TRACK] Anchor " << existing_anchor->node_id
                              << " using LOCAL search (情况A), parent=" << parent_2_phase_ago_id
                              << ", phase=" << parent_->phase
                              << ", r_k=" << parent_->r_k
                              << ", threshold=" << (4 * parent_->r_k + parent_->error_tolerance_index) << std::endl;
                    std::cout << "[TRACK]   Parent has " << parent_anchor_node->friends.size() << " friends" << std::endl;
                }

                for (const auto& friend_tuple : parent_anchor_node->friends) {
                    friend_candidates.push_back(friend_tuple.second);
                }
                friend_candidates.push_back(parent_anchor_node->node_id);
            }
        }
    } else {
        if (is_tracked) {
            std::cout << "[TRACK] Anchor " << existing_anchor->node_id
                      << " using GLOBAL search (情况B)"
                      << ", phase=" << parent_->phase
                      << ", r_k=" << parent_->r_k
                      << ", threshold=" << (4 * parent_->r_k + parent_->error_tolerance_index) << std::endl;
            std::cout << "[TRACK]   Checking all " << parent_->anchors.size() << " existing anchors" << std::endl;
        }

        for (const auto& anchor : parent_->anchors) {
            friend_candidates.push_back(anchor->node_id);
        }
    }

    // 打印候选人的数量
    std::cout << "Number of friend candidates before removal: " << friend_candidates.size() << std::endl;

    // 从候选人中移除自身节点
    friend_candidates.erase(std::remove(friend_candidates.begin(), friend_candidates.end(), existing_anchor->node_id), friend_candidates.end());
    std::cout << "Number of friend candidates after removal: " << friend_candidates.size() << std::endl;

    // 计算与候选朋友之间的 GED 距离
    int friends_made = 0;
    for (int friend_candidate : friend_candidates) {
        double tmp_dist = compute_ged_online(existing_anchor, parent_->nodes[friend_candidate]);

        // Check if this is a tracked interaction
        bool candidate_is_tracked = std::find(tracked_nodes.begin(), tracked_nodes.end(), friend_candidate) != tracked_nodes.end();
        if (is_tracked && candidate_is_tracked) {
            std::cout << "[TRACK] Distance between " << existing_anchor->node_id
                      << " and " << friend_candidate << " = " << tmp_dist
                      << " (threshold=" << (4 * parent_->r_k + parent_->error_tolerance_index) << ")" << std::endl;
        }

        if (tmp_dist <= 4 * parent_->r_k + parent_->error_tolerance_index) {
            auto friend_candidate_node = std::dynamic_pointer_cast<Anchor>(parent_->nodes[friend_candidate]);
            if (friend_candidate_node) {
                if (is_tracked && candidate_is_tracked) {
                    std::cout << "[TRACK] Making friends between " << existing_anchor->node_id
                              << " and " << friend_candidate << std::endl;
                }
                this->make_friends(existing_anchor, friend_candidate_node, tmp_dist);
                friends_made++;
            }
        } else if (is_tracked && candidate_is_tracked) {
            std::cout << "[TRACK] Distance too large, NOT making friends" << std::endl;
        }
    }

    if (is_tracked) {
        std::cout << "[TRACK] Anchor " << existing_anchor->node_id
                  << " made " << friends_made << " friends total (now has "
                  << existing_anchor->friends.size() << " friends)" << std::endl;

        // Check if any tracked nodes are in existing anchors but not in candidates
        for (int tracked : tracked_nodes) {
            if (tracked == existing_anchor->node_id) continue;

            bool in_candidates = std::find(friend_candidates.begin(), friend_candidates.end(), tracked) != friend_candidates.end();
            bool is_existing_anchor = false;

            for (const auto& anchor : parent_->anchors) {
                if (anchor->node_id == tracked) {
                    is_existing_anchor = true;
                    break;
                }
            }

            if (is_existing_anchor && !in_candidates) {
                // Compute distance to this missed anchor
                double missed_dist = compute_ged_online(existing_anchor, parent_->nodes[tracked]);
                std::cout << "[TRACK] WARNING: Anchor " << tracked
                          << " exists but NOT in candidate pool! Distance=" << missed_dist
                          << " (would be friend? " << (missed_dist <= 4 * parent_->r_k + parent_->error_tolerance_index ? "YES" : "NO")
                          << ")" << std::endl;
            }
        }
    }

    // 更新节点在朋友的 nodes_in_cluster 中的距离
    for (const auto& friend_tuple : existing_anchor->friends) {
        double friend_dist = friend_tuple.first;
        if (friend_dist > 2 * parent_->r_k + 3 * parent_->error_tolerance_index) {
            continue;
        }

        auto friend_node = std::dynamic_pointer_cast<Anchor>(parent_->nodes[friend_tuple.second]);
        if (!friend_node) {
            continue;
        }

        std::vector<std::pair<double, int>> tmp_waste_node_list;
        while (!friend_node->nodes_in_cluster.empty()) {
            auto node_tuple = friend_node->nodes_in_cluster.top();
            int node_id = node_tuple.second;
            double dist_to_friend = node_tuple.first; // 使用正数，因为是最大堆

            if (friend_dist > 2 * dist_to_friend + 3 * parent_->error_tolerance_index) {
                break;
            }

            double dist_to_new_anchor = compute_ged_online(existing_anchor, parent_->nodes[node_id]);

            // 从 nodes_in_cluster 中移除该节点
            friend_node->nodes_in_cluster.pop();

            if (dist_to_new_anchor <= dist_to_friend) {
                auto node = parent_->nodes[node_id];
                node->nearest_anchor = existing_anchor->node_id;  // 更新最近锚点 ID
                node->nearest_anchor_dist = dist_to_new_anchor;
                existing_anchor->nodes_in_cluster.push(std::make_pair(dist_to_new_anchor, node_id));
            } else {
                tmp_waste_node_list.push_back(node_tuple);
            }
        }

        // 将未处理的节点重新插入堆中
        for (const auto& node_tuple : tmp_waste_node_list) {
            friend_node->nodes_in_cluster.push(node_tuple);
        }
    }

    // 打印新添加的 anchor 的朋友数量
    std::cout << "New anchor ID " << existing_anchor->node_id << " has "
              << existing_anchor->friends.size() << " friends." << std::endl;
}

void NetDagConstructor::get_nodes_in_cover_range() {
    std::map<std::pair<int, int>, double> tmp_dist_dict;

    for (const auto& anchor : parent_->anchors) {
        std::unordered_set<int> seen_nodes;  // 用于跟踪已经插入的节点

        for (const auto& friend_tuple : anchor->friends) {
            double friend_dist = friend_tuple.first;
            int friend_id = friend_tuple.second;
            auto friend_node = parent_->nodes[friend_id];

            // 如果 friend_node 实际是 Anchor，进行 dynamic_pointer_cast
            auto friend_anchor = std::dynamic_pointer_cast<Anchor>(friend_node);
            if (!friend_anchor) {
                // 如果不是 Anchor 类型，跳过
                continue;
            }

            if (friend_dist <= 2 * parent_->alpha + 2 * parent_->tau + 6 * parent_->error_tolerance_index) {
                // 将 priority_queue 中的元素复制到 std::vector 中
                std::priority_queue<std::pair<double, int>> nodes_in_cluster_copy = friend_anchor->nodes_in_cluster;
                std::vector<std::pair<double, int>> combined;

                // 将所有元素从 priority_queue 移动到 vector
                while (!nodes_in_cluster_copy.empty()) {
                    combined.push_back(nodes_in_cluster_copy.top());
                    nodes_in_cluster_copy.pop();
                }

                // 将 anchor 自身作为一个特殊元素加入
                combined.emplace_back(0.0, friend_anchor->node_id);

                // 遍历 combined
                for (const auto& possible_cover_node_tuple : combined) {
                    int possible_cover_node_id = possible_cover_node_tuple.second;

                    // 检查是否已经插入该节点
                    if (seen_nodes.find(possible_cover_node_id) != seen_nodes.end()) {
                        continue;  // 如果该节点已经插入，跳过
                    }

                    double tmp_dist;
                    int a = std::min(anchor->node_id, possible_cover_node_id);
                    int b = std::max(anchor->node_id, possible_cover_node_id);
                    std::pair<int, int> key = {a, b};
                    auto dist_it = tmp_dist_dict.find(key);
                    if (dist_it != tmp_dist_dict.end()) {
                        tmp_dist = dist_it->second;
                    } else {
                        tmp_dist = compute_ged_online(anchor, parent_->nodes[possible_cover_node_id]);
                        tmp_dist_dict[key] = tmp_dist;
                    }

                    if (tmp_dist <= parent_->alpha + 2 * parent_->tau + 3 * parent_->error_tolerance_index) {
                        anchor->nodes_in_cover_range.emplace_back(possible_cover_node_id, tmp_dist);
                        seen_nodes.insert(possible_cover_node_id);  // 插入节点ID到 seen_nodes 中
                    }
                }
            }
        }

        // 添加 nodes_in_cluster 和自身到 nodes_in_cover_range
        std::priority_queue<std::pair<double, int>> nodes_in_cluster_copy = anchor->nodes_in_cluster;
        std::vector<std::pair<double, int>> temp_vector;

        while (!nodes_in_cluster_copy.empty()) {
            auto [first, second] = nodes_in_cluster_copy.top();  // 拆解 pair/tuple
            temp_vector.push_back(std::make_pair(second, first));  // 交换 first 和 second 的顺序
            nodes_in_cluster_copy.pop();
        }

        for (const auto& node_tuple : temp_vector) {
            int node_id = node_tuple.first;
            if (seen_nodes.find(node_id) == seen_nodes.end()) {
                anchor->nodes_in_cover_range.emplace_back(node_tuple.first, node_tuple.second);
                seen_nodes.insert(node_id);  // 确保每个节点只插入一次
            }
        }

        // 最后再添加 anchor 自身
        if (seen_nodes.find(anchor->node_id) == seen_nodes.end()) {
            anchor->nodes_in_cover_range.emplace_back(anchor->node_id, 0.0);
            seen_nodes.insert(anchor->node_id);
        }
    }
}

void NetDagConstructor::make_friends(std::shared_ptr<Anchor> anchor1, std::shared_ptr<Anchor> anchor2, double dist) {
    // 添加 anchor1 的 friend
    anchor1->friends.push_back({dist, anchor2->node_id});
    
    // 添加 anchor2 的 friend
    anchor2->friends.push_back({dist, anchor1->node_id});
}

void NetDagConstructor::add_edge(std::shared_ptr<Anchor> parent, std::shared_ptr<Node> child, int child_phase, double child_dist) {
    auto& children = parent->children;
    if (children.find(child_phase) == children.end()) {
        children[child_phase] = { std::make_pair(child->node_id, child_dist) };
    } else {
        children[child_phase].emplace_back(child->node_id, child_dist);
    }
}

void NetDagConstructor::add_edge(std::shared_ptr<Node> parent, std::shared_ptr<Node> child, int child_phase, double child_dist) {
    // TODO: Extract implementation from original file
    // This method adds an edge between two nodes - implementation not found in original
    std::cout << "add_edge(Node, Node) called - implementation needed" << std::endl;
}

// Only compute GED and save to csv, no NetDag modification
void NetDagConstructor::compute_ged_to_csv() {
    std::cout << "[compute_ged_to_csv] Starting GED computation..." << std::endl;

    // Define csv file paths
    std::string dataset_dir = "./NetDags/" + parent_->dataset + "/";
    std::string ged_results_dir = dataset_dir + "ged_results/";
    std::filesystem::create_directories(ged_results_dir);

    std::string result_filename = ged_results_dir + parent_->index_name + "_exact_ged_results.csv";
    std::string timeout_filename = ged_results_dir + parent_->index_name + "_timeout_ged_results.csv";
    std::string over_threshold_filename = ged_results_dir + parent_->index_name + "_over_threshold_ged_results.csv";

    // Create empty files if they don't exist
    namespace fs = std::filesystem;
    if (!fs::exists(result_filename)) {
        std::ofstream create_empty_file(result_filename);
        std::cout << "[INFO] Created empty CSV file: " << result_filename << std::endl;
    }
    if (!fs::exists(timeout_filename)) {
        std::ofstream create_empty_file(timeout_filename);
        std::cout << "[INFO] Created empty timeout CSV file: " << timeout_filename << std::endl;
    }
    if (!fs::exists(over_threshold_filename)) {
        std::ofstream create_empty_file(over_threshold_filename);
        std::cout << "[INFO] Created empty over_threshold CSV file: " << over_threshold_filename << std::endl;
    }

    // Load existing GED results
    std::unordered_map<std::pair<int, int>, std::tuple<int, std::string>, pair_hash> computed_results;
    std::unordered_set<std::pair<int, int>, pair_hash> timeout_results;
    std::unordered_set<std::pair<int, int>, pair_hash> over_threshold_results;

    std::ifstream existing_csv_file(result_filename);
    if (existing_csv_file.is_open()) {
        std::string line;
        while (std::getline(existing_csv_file, line)) {
            std::stringstream ss(line);
            std::string id1_str, id2_str, ged_str, mapping_str;
            if (!std::getline(ss, id1_str, ',')) continue;
            if (!std::getline(ss, id2_str, ',')) continue;
            if (!std::getline(ss, ged_str, ',')) continue;
            if (!std::getline(ss, mapping_str)) continue;

            try {
                int id1 = std::stoi(id1_str);
                int id2 = std::stoi(id2_str);
                int ged = std::stoi(ged_str);
                int min_id = std::min(id1, id2);
                int max_id = std::max(id1, id2);
                computed_results[{min_id, max_id}] = std::make_tuple(ged, mapping_str);
            } catch (...) {
                continue;
            }
        }
        existing_csv_file.close();
        std::cout << "[compute_ged_to_csv] Loaded " << computed_results.size() << " existing results from CSV" << std::endl;
    }

    // Load timeout results
    std::ifstream timeout_csv_file(timeout_filename);
    if (timeout_csv_file.is_open()) {
        std::string line;
        while (std::getline(timeout_csv_file, line)) {
            std::stringstream ss(line);
            std::string id1_str, id2_str;
            if (!std::getline(ss, id1_str, ',')) continue;
            if (!std::getline(ss, id2_str, ',')) continue;

            try {
                int id1 = std::stoi(id1_str);
                int id2 = std::stoi(id2_str);
                int min_id = std::min(id1, id2);
                int max_id = std::max(id1, id2);
                timeout_results.insert({min_id, max_id});
            } catch (...) {
                continue;
            }
        }
        timeout_csv_file.close();
        std::cout << "[compute_ged_to_csv] Loaded " << timeout_results.size() << " timeout results" << std::endl;
    }

    // Load over_threshold results
    std::ifstream over_threshold_csv_file(over_threshold_filename);
    if (over_threshold_csv_file.is_open()) {
        std::string line;
        while (std::getline(over_threshold_csv_file, line)) {
            std::stringstream ss(line);
            std::string id1_str, id2_str;
            if (!std::getline(ss, id1_str, ',')) continue;
            if (!std::getline(ss, id2_str, ',')) continue;

            try {
                int id1 = std::stoi(id1_str);
                int id2 = std::stoi(id2_str);
                int min_id = std::min(id1, id2);
                int max_id = std::max(id1, id2);
                over_threshold_results.insert({min_id, max_id});
            } catch (...) {
                continue;
            }
        }
        over_threshold_csv_file.close();
        std::cout << "[compute_ged_to_csv] Loaded " << over_threshold_results.size() << " over_threshold results" << std::endl;
    }

    // Collect pairs to compute from nodes_in_cluster (without modifying it)
    std::vector<std::tuple<int, int, double>> pairs_to_compute;
    size_t total_cluster_nodes = 0;
    size_t already_computed = 0;
    size_t already_timeout = 0;
    size_t already_over_threshold = 0;

    for (const auto& anchor : parent_->anchors) {
        int anchor_id = anchor->node_id;
        auto nodes_in_cluster_copy = anchor->nodes_in_cluster;  // Copy, don't modify

        while (!nodes_in_cluster_copy.empty()) {
            auto [dist, node_id] = nodes_in_cluster_copy.top();
            nodes_in_cluster_copy.pop();
            total_cluster_nodes++;

            if (anchor_id == node_id) continue;

            int min_id = std::min(anchor_id, node_id);
            int max_id = std::max(anchor_id, node_id);
            std::pair<int, int> pair = {min_id, max_id};

            if (computed_results.find(pair) != computed_results.end()) {
                already_computed++;
            } else if (timeout_results.find(pair) != timeout_results.end()) {
                already_timeout++;
            } else if (over_threshold_results.find(pair) != over_threshold_results.end()) {
                already_over_threshold++;
            } else {
                pairs_to_compute.emplace_back(anchor_id, node_id, dist);
            }
        }
    }

    std::cout << "\n=== GED Computation Statistics ===" << std::endl;
    std::cout << "Total cluster nodes: " << total_cluster_nodes << std::endl;
    std::cout << "Already computed (success): " << already_computed << std::endl;
    std::cout << "Already timeout: " << already_timeout << std::endl;
    std::cout << "Already over_threshold: " << already_over_threshold << std::endl;
    std::cout << "Need to compute: " << pairs_to_compute.size() << std::endl;
    std::cout << "=================================\n" << std::endl;

    if (pairs_to_compute.empty()) {
        std::cout << "[compute_ged_to_csv] No new pairs to compute." << std::endl;
        return;
    }

    // Open files for appending
    std::ofstream result_file(result_filename, std::ios::app);
    std::ofstream timeout_file(timeout_filename, std::ios::app);
    std::ofstream over_threshold_file(over_threshold_filename, std::ios::app);
    if (!result_file.is_open() || !timeout_file.is_open() || !over_threshold_file.is_open()) {
        std::cerr << "[ERROR] Unable to open output files" << std::endl;
        return;
    }

    // Parallel computation
    std::mutex result_mutex;
    std::mutex computed_mutex;
    std::atomic<size_t> task_index(0);
    std::atomic<size_t> processed_tasks(0);
    size_t total_tasks = pairs_to_compute.size();
    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(200));
    if (num_threads == 0) num_threads = 4;

    auto start_time = std::chrono::steady_clock::now();

    // E1: accumulate per-task GED compute time across threads (= single-thread-equivalent work).
    std::mutex work_mutex;
    double total_ged_work_seconds = 0.0;

    auto worker = [&]() {
        size_t index = task_index.fetch_add(1);
        double local_work = 0.0;  // E1: this thread's accumulated GED compute time

        while (index < total_tasks) {
            auto [anchor_id, node_id, dist] = pairs_to_compute[index];
            int min_id = std::min(anchor_id, node_id);
            int max_id = std::max(anchor_id, node_id);
            std::pair<int, int> pair = {min_id, max_id};

            // Check if already computed by another thread
            {
                std::lock_guard<std::mutex> lock(computed_mutex);
                if (computed_results.find(pair) != computed_results.end()) {
                    processed_tasks.fetch_add(1);
                    index = task_index.fetch_add(1);
                    continue;
                }
            }

            // Get graphs
            if (anchor_id < 0 || anchor_id >= static_cast<int>(parent_->nodes.size()) ||
                node_id < 0 || node_id >= static_cast<int>(parent_->nodes.size())) {
                processed_tasks.fetch_add(1);
                index = task_index.fetch_add(1);
                continue;
            }

            auto graph1 = parent_->nodes[anchor_id]->graph.get();
            auto graph2 = parent_->nodes[node_id]->graph.get();

            if (!graph1 || !graph2) {
                processed_tasks.fetch_add(1);
                index = task_index.fetch_add(1);
                continue;
            }

            try {
                auto _ged_c0 = std::chrono::steady_clock::now();  // E1: time GED compute (LB + AppForComputation) only, excl. IO
                // Lower bound filter
                ui LB = graph1->ged_lower_bound_filter(
                    graph2,
                    static_cast<ui>(parent_->max_exact_ged_for_EPT),
                    parent_->vM.size(),
                    parent_->eM.size(),
                    parent_->max_n
                );

                int ged_res = -1;
                bool hit_limit = false;  // 记号: A* 耗尽 epf_exact_iter 而退出 => 归 timeout 档
                std::vector<std::pair<ui, ui>> mapping;

                if (LB <= parent_->max_exact_ged_for_EPT) {
                    // 迭代预算改为可配(parent_->epf_exact_iter, 默认 1e6 = 原行为)
                    const int exact_max_iter = parent_->epf_exact_iter;
                    Application app(static_cast<ui>(parent_->max_exact_ged_for_EPT), "BMao",
                                    exact_max_iter, exact_max_iter);
                    app.set_all_edge_labels_same(parent_->all_edge_labels_same);
                    app.init(graph1, graph2);
                    int lb_astar = -1;
                    ged_res = app.AppForComputation(&mapping, &lb_astar);
                    hit_limit = app.hit_iter_limit();
                    if (mapping.empty()) {
                        app.get_mapping(mapping);
                    }
                } else {
                    ged_res = LB;
                }
                local_work += std::chrono::duration<double>(std::chrono::steady_clock::now() - _ged_c0).count();  // E1

                if (ged_res != -1 && ged_res <= parent_->max_exact_ged_for_EPT) {
                    // Build mapping string
                    std::stringstream mapping_ss;
                    if (!mapping.empty()) {
                        mapping_ss << "\"[";
                        for (size_t i = 0; i < mapping.size(); ++i) {
                            mapping_ss << "(" << mapping[i].first << "," << mapping[i].second << ")";
                            if (i != mapping.size() - 1) mapping_ss << ",";
                        }
                        mapping_ss << "]\"";
                    } else {
                        mapping_ss << "\"[]\"";
                    }

                    // Save to csv
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        result_file << min_id << "," << max_id << "," << ged_res << "," << mapping_ss.str() << "\n";
                        result_file.flush();
                    }

                    // Update cache
                    {
                        std::lock_guard<std::mutex> lock(computed_mutex);
                        computed_results[pair] = std::make_tuple(ged_res, mapping_ss.str());
                    }
                } else {
                    // Failed (timeout or over threshold)
                    // hit_limit: 预算耗尽 => 未定, 归 timeout 档(原先靠 ged_res==-1, 但那条
                    // 路径被死宏封住, 导致两种退出都落进 over_threshold)。
                    if (ged_res == -1 || hit_limit) {
                        // Timeout
                        std::lock_guard<std::mutex> lock(result_mutex);
                        timeout_file << min_id << "," << max_id << "\n";
                        timeout_file.flush();
                    } else {
                        // Over threshold
                        std::lock_guard<std::mutex> lock(result_mutex);
                        over_threshold_file << min_id << "," << max_id << "," << ged_res << "\n";
                        over_threshold_file.flush();
                    }
                }

            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Exception for pair (" << anchor_id << ", " << node_id << "): " << e.what() << std::endl;
            }

            // Update progress
            size_t current = processed_tasks.fetch_add(1) + 1;
            if (current % 100 == 0 || current == total_tasks) {
                double progress = static_cast<double>(current) / total_tasks;
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double remaining = (elapsed / progress) - elapsed;
                std::cout << "\r[compute_ged_to_csv] Progress: " << std::fixed << std::setprecision(2)
                          << (progress * 100) << "% (" << current << "/" << total_tasks << ")"
                          << ", Elapsed: " << elapsed << "s, Remaining: " << std::max(0.0, remaining) << "s" << std::flush;
            }

            index = task_index.fetch_add(1);
        }
        { std::lock_guard<std::mutex> lk(work_mutex); total_ged_work_seconds += local_work; }  // E1
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    result_file.close();
    timeout_file.close();
    over_threshold_file.close();

    double ged_wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[E1_GED] threads=" << num_threads << " wall_seconds=" << ged_wall_seconds
              << " work_seconds=" << total_ged_work_seconds
              << " pairs=" << total_tasks << std::endl;
    Utility::append_e1_timing("index_name=" + parent_->index_name + "\tdataset=" + parent_->dataset
        + "\tstage=compute_paths\tthreads=" + std::to_string(num_threads)
        + "\twall_seconds=" + std::to_string(ged_wall_seconds)
        + "\twork_seconds=" + std::to_string(total_ged_work_seconds)
        + "\tpairs=" + std::to_string(total_tasks));
    std::cout << "\n[compute_ged_to_csv] Completed!" << std::endl;
    std::cout << "[compute_ged_to_csv] Success results saved to: " << result_filename << std::endl;
    std::cout << "[compute_ged_to_csv] Timeout results saved to: " << timeout_filename << std::endl;
    std::cout << "[compute_ged_to_csv] Over threshold results saved to: " << over_threshold_filename << std::endl;
}

void NetDagConstructor::reassign_nodes_in_cluster() {
    // 定义用于存储已计算的 GED 结果的 CSV 文件名
    std::string dataset_dir = "./NetDags/" + parent_->dataset + "/";
    std::string ged_results_dir = dataset_dir + "ged_results/";

    // 确保目录存在
    std::filesystem::create_directories(ged_results_dir);

    std::string result_filename = ged_results_dir + parent_->index_name + "_exact_ged_results.csv";
    std::string failed_result_filename = ged_results_dir + parent_->index_name + "_failed_ged_results.csv";
    namespace fs = std::filesystem;
    if (!fs::exists(result_filename)) {
        // 用默认构造方式打开，写空内容
        std::ofstream create_empty_file(result_filename);
        if (!create_empty_file.is_open()) {
            std::cerr << "[ERROR] Unable to create an empty file: " << result_filename << std::endl;
        } else {
            std::cout << "[INFO] Created empty CSV file: " << result_filename << std::endl;
        }
    }
    if (!fs::exists(failed_result_filename)) {
        std::ofstream create_empty_file(failed_result_filename);
        if (!create_empty_file.is_open()) {
            std::cerr << "[ERROR] Unable to create an empty file: " << failed_result_filename << std::endl;
        } else {
            std::cout << "[INFO] Created empty failed results CSV file: " << failed_result_filename << std::endl;
        }
    }
    // 定义用于加载已有的 GED 结果的 CSV 文件名
    std::string existing_csv_filename = result_filename;

    // 加载已有的 GED 结果
    std::unordered_map<std::pair<int, int>, std::tuple<int, std::string>, pair_hash> computed_results;
    std::unordered_set<std::pair<int, int>, pair_hash> failed_results;

    std::ifstream existing_csv_file(existing_csv_filename);
    if (existing_csv_file.is_open()) {
        std::string line;
        int line_number = 0;
        while (std::getline(existing_csv_file, line)) {
            ++line_number;
            std::stringstream ss(line);
            std::string id1_str, id2_str, ged_str, mapping_str;
            if (!std::getline(ss, id1_str, ',')) continue;
            if (!std::getline(ss, id2_str, ',')) continue;
            if (!std::getline(ss, ged_str, ',')) continue;
            if (!std::getline(ss, mapping_str)) continue;

            try {
                int id1 = std::stoi(id1_str);
                int id2 = std::stoi(id2_str);
                int ged = std::stoi(ged_str);

                int min_id = std::min(id1, id2);
                int max_id = std::max(id1, id2);
                std::pair<int, int> pair = std::make_pair(min_id, max_id);

                // 保存 ged 和 mapping_str
                computed_results[pair] = std::make_tuple(ged, mapping_str);
            } catch (const std::exception& e) {
                // 跳过此行
                continue;
            }
        }
        existing_csv_file.close();
        std::cout << "Loaded existing GED results from " << existing_csv_filename << std::endl;
    } else {
        std::cerr << "Warning: Unable to open existing CSV file: " << existing_csv_filename << std::endl;
    }

    // 加载失败的计算结果
    std::ifstream failed_csv_file(failed_result_filename);
    if (failed_csv_file.is_open()) {
        std::string line;
        int failed_count = 0;
        while (std::getline(failed_csv_file, line)) {
            std::stringstream ss(line);
            std::string id1_str, id2_str, type_str;
            if (!std::getline(ss, id1_str, ',')) continue;
            if (!std::getline(ss, id2_str, ',')) continue;
            if (!std::getline(ss, type_str)) continue;

            try {
                int id1 = std::stoi(id1_str);
                int id2 = std::stoi(id2_str);
                int min_id = std::min(id1, id2);
                int max_id = std::max(id1, id2);
                std::pair<int, int> pair = std::make_pair(min_id, max_id);
                failed_results.insert(pair);
                failed_count++;
            } catch (const std::exception& e) {
                continue;
            }
        }
        failed_csv_file.close();
        std::cout << "Loaded " << failed_count << " failed results from " << failed_result_filename << std::endl;
    }

    // **第一步：收集需要计算的节点对**
    std::vector<std::tuple<int, int, double>> pairs_to_compute; // (anchor_id, node_id, dist)

    // 临时存储计算结果的结构
    std::unordered_map<int, std::priority_queue<std::pair<double, int>>> temp_nodes_in_exact_cluster;

    // 定义互斥锁
    std::mutex temp_mutex;
    std::mutex computed_results_mutex;
    std::mutex result_mutex;
    std::mutex cout_mutex;

    // DEBUG: 统计总cluster node数量
    size_t total_cluster_nodes = 0;
    size_t total_skipped_self = 0;
    size_t total_has_result = 0;
    size_t total_has_failed = 0;
    size_t total_needs_compute = 0;

    // 遍历 anchors，收集需要计算的节点对
    for (size_t idx = 0; idx < parent_->anchors.size(); ++idx) {
        auto anchor = parent_->anchors[idx];
        int anchor_id = anchor->node_id;

        // 复制 nodes_in_cluster
        auto nodes_in_cluster_pq = anchor->nodes_in_cluster;
        std::priority_queue<std::pair<double, int>> updated_nodes_in_cluster;

        while (!nodes_in_cluster_pq.empty()) {
            auto [dist, node_id] = nodes_in_cluster_pq.top();
            nodes_in_cluster_pq.pop();

            total_cluster_nodes++;  // DEBUG: 统计所有cluster nodes

            if (anchor_id != node_id) {
                int min_id = std::min(anchor_id, node_id);
                int max_id = std::max(anchor_id, node_id);
                std::pair<int, int> pair = std::make_pair(min_id, max_id);

                // 检查是否已有计算结果或失败记录
                bool has_result = false;
                bool has_failed = false;
                int ged = -1;
                std::string mapping_str;

                {
                    std::lock_guard<std::mutex> lock(computed_results_mutex);
                    auto it = computed_results.find(pair);
                    if (it != computed_results.end()) {
                        has_result = true;
                        ged = std::get<0>(it->second);
                        mapping_str = std::get<1>(it->second);
                    } else if (failed_results.find(pair) != failed_results.end()) {
                        has_failed = true;
                    }
                }

                if (has_result) {
                    total_has_result++;  // DEBUG
                    // 已有结果，判断 ged 是否满足条件
                    if (ged != -1 && ged <= parent_->max_exact_ged_for_EPT) {
                        // 将结果加入临时的 temp_nodes_in_exact_cluster
                        {
                            std::lock_guard<std::mutex> lock(temp_mutex);
                            temp_nodes_in_exact_cluster[anchor_id].push({static_cast<double>(ged), node_id});
                        }
                        // 不将节点加入 updated_nodes_in_cluster，从 nodes_in_cluster 中移除
                    } else {
                        // GED 超过阈值，保留在 updated_nodes_in_cluster 中
                        updated_nodes_in_cluster.push({dist, node_id});
                    }
                } else if (has_failed) {
                    total_has_failed++;  // DEBUG
                    // 已知失败的计算（超时或超过阈值），不重复计算，保留在cluster中
                    updated_nodes_in_cluster.push({dist, node_id});
                } else {
                    total_needs_compute++;  // DEBUG
                    // 需要计算的节点对
                    pairs_to_compute.emplace_back(anchor_id, node_id, dist);
                    // 暂时保留节点在 updated_nodes_in_cluster 中，等待计算后再决定
                    updated_nodes_in_cluster.push({dist, node_id});
                }
            } else {
                total_skipped_self++;  // DEBUG
                // 将自身（锚点）保留在 updated_nodes_in_cluster 中
                updated_nodes_in_cluster.push({dist, node_id});
            }
        }

        // 更新 anchor 的 nodes_in_cluster
        anchor->nodes_in_cluster = updated_nodes_in_cluster;
    }

    // **DEBUG: 打印统计信息**
    std::cout << "\n=== Cluster Node Statistics ===" << std::endl;
    std::cout << "Total cluster nodes: " << total_cluster_nodes << std::endl;
    std::cout << "  - Skipped (anchor_id == node_id): " << total_skipped_self << std::endl;
    std::cout << "  - Has result (from CSV): " << total_has_result << std::endl;
    std::cout << "  - Has failed (from failed CSV): " << total_has_failed << std::endl;
    std::cout << "  - Needs compute: " << total_needs_compute << std::endl;
    std::cout << "Total pairs in CSV should be: " << (total_has_result + total_has_failed + total_needs_compute) << std::endl;
    std::cout << "================================\n" << std::endl;

    // **第二步：并行计算需要计算的节点对**
    std::cout << "Parallel computing..." << std::endl;
    // 创建结果文件输出流（多线程写入，追加模式）
    std::ofstream result_file(result_filename, std::ios::app);
    if (!result_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << result_filename << std::endl;
        return;
    }

    std::ofstream failed_file(failed_result_filename, std::ios::app);
    if (!failed_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << failed_result_filename << std::endl;
        return;
    }

    // 将结果文件流设为线程安全的
    result_file.sync_with_stdio(false);
    failed_file.sync_with_stdio(false);

    // 创建线程并处理节点对
    std::vector<std::thread> threads;
    std::atomic<size_t> task_index(0);       // 任务索引
    std::atomic<size_t> processed_tasks(0);  // 已处理的任务数
    size_t total_tasks = pairs_to_compute.size();
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = (num_threads <= 200) ? num_threads : 200; // 最多 200 个线程
    // 开始时间
    auto start_time = std::chrono::steady_clock::now();

    // 定义工作函数
    auto worker = [&]() {
        size_t index = task_index.fetch_add(1);

        while (index < total_tasks) {
            auto [anchor_id, node_id, dist] = pairs_to_compute[index];

            int min_id = std::min(anchor_id, node_id);
            int max_id = std::max(anchor_id, node_id);
            std::pair<int, int> pair = std::make_pair(min_id, max_id);

            // 获取图
            if (anchor_id < 0 || anchor_id >= parent_->nodes.size() || node_id < 0 || node_id >= parent_->nodes.size()) {
                std::cerr << "Error: Node IDs out of bounds: " << anchor_id << ", " << node_id << std::endl;
                // 增加已处理的任务数
                size_t current_processed = processed_tasks.fetch_add(1) + 1;
                index = task_index.fetch_add(1);
                continue;
            }

            auto node_anchor = parent_->nodes[anchor_id];
            auto node_other = parent_->nodes[node_id];
            auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(node_anchor);
            if (!anchor_ptr) {
                std::cerr << "Error: Node is not an Anchor for node ID: " << anchor_id << std::endl;
                // 增加已处理的任务数
                size_t current_processed = processed_tasks.fetch_add(1) + 1;
                index = task_index.fetch_add(1);
                continue;
            }

            auto graph1 = node_anchor->graph.get();
            auto graph2 = node_other->graph.get();

            if (!graph1 || !graph2) {
                std::cerr << "Error: Graph is null for node IDs: " << anchor_id << ", " << node_id << std::endl;
                // 增加已处理的任务数
                size_t current_processed = processed_tasks.fetch_add(1) + 1;
                index = task_index.fetch_add(1);
                continue;
            }

            // 先检查是否已有计算结果
            bool has_result = false;
            int ged_res = -1;
            std::string mapping_str;

            {
                std::lock_guard<std::mutex> lock(computed_results_mutex);
                auto it = computed_results.find(pair);
                if (it != computed_results.end()) {
                    has_result = true;
                    ged_res = std::get<0>(it->second);
                    mapping_str = std::get<1>(it->second);
                }
            }

            if (has_result) {
                if (ged_res != -1 && ged_res <= parent_->max_exact_ged_for_EPT) {
                    // 将结果加入临时的 temp_nodes_in_exact_cluster
                    {
                        std::lock_guard<std::mutex> lock(temp_mutex);
                        temp_nodes_in_exact_cluster[anchor_id].push({static_cast<double>(ged_res), node_id});
                    }
                    // 从 anchor 的 nodes_in_cluster 中移除该节点
                    {
                        std::lock_guard<std::mutex> lock(temp_mutex);
                        auto& nodes_in_cluster = anchor_ptr->nodes_in_cluster;
                        // 创建一个新的优先队列，过滤掉已处理的节点
                        std::priority_queue<std::pair<double, int>> updated_nodes_in_cluster_inner;
                        while (!nodes_in_cluster.empty()) {
                            auto [d, nid] = nodes_in_cluster.top();
                            nodes_in_cluster.pop();
                            if (nid != node_id) {
                                updated_nodes_in_cluster_inner.push({d, nid});
                            }
                        }
                        // 更新 nodes_in_cluster
                        nodes_in_cluster = updated_nodes_in_cluster_inner;
                    }
                } else {
                    // GED 超过阈值，尝试通过 friends 进行归入 (可通过开关控制)
                    bool assigned_via_friend = false;

                    // 检查是否启用friends收留功能
                    if (!parent_->enable_friends_reassign) {
                        // 功能被禁用，跳过friends收留逻辑
                        // 该节点将在后续被归为无法覆盖的节点
                    } else {
                    for (const auto& [friend_dist, friend_id] : anchor_ptr->friends) {
                        // 检查是否已有计算结果
                        int min_friend_id = std::min(friend_id, node_id);
                        int max_friend_id = std::max(friend_id, node_id);
                        std::pair<int, int> friend_pair = std::make_pair(min_friend_id, max_friend_id);

                        bool has_friend_result = false;
                        int friend_ged_res = -1;
                        std::string friend_mapping_str;

                        {
                            std::lock_guard<std::mutex> lock(computed_results_mutex);
                            auto it_friend = computed_results.find(friend_pair);
                            if (it_friend != computed_results.end()) {
                                has_friend_result = true;
                                friend_ged_res = std::get<0>(it_friend->second);
                                friend_mapping_str = std::get<1>(it_friend->second);
                            }
                        }

                        if (has_friend_result) {
                            if (friend_ged_res != -1 && friend_ged_res <= parent_->max_exact_ged_for_EPT) {
                                // 将节点加入朋友的 temp_nodes_in_exact_cluster
                                {
                                    std::lock_guard<std::mutex> lock(temp_mutex);
                                    temp_nodes_in_exact_cluster[friend_id].push({static_cast<double>(friend_ged_res), node_id});
                                }
                                assigned_via_friend = true;
                                break; // 已成功归入，跳出 friends 循环
                            }
                        } else {
                            // 需要计算与朋友的 GED
                            auto friend_node = parent_->nodes[friend_id];
                            auto friend_anchor = std::dynamic_pointer_cast<Anchor>(friend_node);
                            if (!friend_anchor) continue;

                            auto friend_graph = friend_anchor->graph.get();

                            if (!friend_graph) continue;

                            try {

                                ui friend_LB = friend_graph->ged_lower_bound_filter(
                                    graph2,
                                    static_cast<ui>(parent_->max_exact_ged_for_EPT),
                                    parent_->vM.size(),    // 顶点标签种类数量
                                    parent_->eM.size(),    // 边标签种类数量
                                    parent_->max_n         // 最大节点数
                                );
                                if (friend_LB > parent_->max_exact_ged_for_EPT) {
                                    // LB 大于阈值，不需要计算
                                    continue;
                                }
                                // double friend_ML_ged = ML_graph::get_ML_ged(friend_anchor->ml_graph, node_other->ml_graph);
                                // if (friend_ML_ged > parent_->max_exact_ged_for_EPT + 3 * parent_->error_tolerance_index) {
                                //     // ML GED 大于阈值，不需要计算
                                //     continue;
                                // }
                                // 调用 AppForComputation
                                Application app(static_cast<ui>(parent_->max_exact_ged_for_EPT), "BMao");
                                app.set_all_edge_labels_same(parent_->all_edge_labels_same);
                                app.init(friend_graph, graph2);
                                int lb_astar_friend = -1;
                                std::vector<std::pair<ui, ui>> mapping_friend;
                                int friend_ged_res = app.AppForComputation(&mapping_friend, &lb_astar_friend);
                                if (mapping_friend.empty()) {
                                    app.get_mapping(mapping_friend);
                                }

                                if (friend_ged_res != -1 && friend_ged_res <= parent_->max_exact_ged_for_EPT) {
                                    // 将节点映射转换为字符串
                                    std::stringstream mapping_ss_friend;
                                    if (!mapping_friend.empty()) {
                                        mapping_ss_friend << "\"[";
                                        for (size_t i = 0; i < mapping_friend.size(); ++i) {
                                            mapping_ss_friend << "(" << mapping_friend[i].first << "," << mapping_friend[i].second << ")";
                                            if (i != mapping_friend.size() - 1) {
                                                mapping_ss_friend << ",";
                                            }
                                        }
                                        mapping_ss_friend << "]\"";
                                    } else {
                                        mapping_ss_friend << "\"[]\"";
                                    }
                                    friend_mapping_str = mapping_ss_friend.str();

                                    // 保存新的计算结果
                                    {
                                        std::lock_guard<std::mutex> lock(computed_results_mutex);
                                        computed_results[friend_pair] = std::make_tuple(friend_ged_res, friend_mapping_str);
                                    }

                                    // 将节点加入朋友的 temp_nodes_in_exact_cluster
                                    {
                                        std::lock_guard<std::mutex> lock(temp_mutex);
                                        temp_nodes_in_exact_cluster[friend_id].push({static_cast<double>(friend_ged_res), node_id});
                                    }

                                    // 将结果写入结果文件
                                    {
                                        std::lock_guard<std::mutex> lock(result_mutex);
                                        result_file << min_friend_id << "," << max_friend_id << "," << friend_ged_res << "," << friend_mapping_str << "\n";
                                        result_file.flush();
                                    }

                                    assigned_via_friend = true;
                                    break; // 已成功归入，跳出 friends 循环
                                } else {
                                    // GED 超过阈值，不需要写入结果文件
                                }

                            } catch (const std::exception& e) {
                                // 捕获异常，输出错误信息
                                std::cerr << "Exception occurred while processing node IDs: " << friend_id << ", " << node_id << ". Exception: " << e.what() << std::endl;
                                // 不需要写入错误信息到结果文件
                            }
                        }
                    }

                    if (!assigned_via_friend) {
                        // 无法通过 friends 归入，节点保留在 anchor 的 nodes_in_cluster 中
                        // 由于在主线程中已经保留，无需额外处理
                    } else {
                        // 从 anchor 的 nodes_in_cluster 中移除该节点
                        {
                            std::lock_guard<std::mutex> lock(temp_mutex);
                            auto& nodes_in_cluster = anchor_ptr->nodes_in_cluster;
                            // 创建一个新的优先队列，过滤掉已处理的节点
                            std::priority_queue<std::pair<double, int>> updated_nodes_in_cluster_inner;
                            while (!nodes_in_cluster.empty()) {
                                auto [d, nid] = nodes_in_cluster.top();
                                nodes_in_cluster.pop();
                                if (nid != node_id) {
                                    updated_nodes_in_cluster_inner.push({d, nid});
                                }
                            }
                            // 更新 nodes_in_cluster
                            nodes_in_cluster = updated_nodes_in_cluster_inner;
                        }
                    }
                    } // 结束 enable_friends_reassign 条件检查
                }
            } else {
                // 需要计算
                try {
                    // 调用 AppForComputation
                    ui LB = graph1->ged_lower_bound_filter(
                        graph2,
                        static_cast<ui>(parent_->max_exact_ged_for_EPT),
                        parent_->vM.size(),    // 顶点标签种类数量
                        parent_->eM.size(),    // 边标签种类数量
                        parent_->max_n         // 最大节点数
                    );
                    std::vector<std::pair<ui, ui>> mapping;
                    if (LB <= parent_->max_exact_ged_for_EPT){
                        // double ML_ged = ML_graph::get_ML_ged(anchor_ptr->ml_graph, node_other->ml_graph);
                        // if (ML_ged <= parent_->max_exact_ged_for_EPT + 3 * parent_->error_tolerance_index){
                            Application app(static_cast<ui>(parent_->max_exact_ged_for_EPT), "BMao");
                            app.set_all_edge_labels_same(parent_->all_edge_labels_same);
                            app.init(graph1, graph2);
                            int lb_astar = -1;

                            ged_res = app.AppForComputation(&mapping, &lb_astar);
                            if (mapping.empty()) {
                                app.get_mapping(mapping);
                            }
                        // }else{
                        //     ged_res = LB;
                        // }
                    }else{
                        ged_res = LB;
                    }


                    // 检查是否为超时结果(-1)，如果是则跳过
                    if (ged_res != -1 && ged_res <= parent_->max_exact_ged_for_EPT) {
                        // 将节点映射转换为字符串
                        std::stringstream mapping_ss;
                        if (!mapping.empty()) {
                            mapping_ss << "\"[";
                            for (size_t i = 0; i < mapping.size(); ++i) {
                                mapping_ss << "(" << mapping[i].first << "," << mapping[i].second << ")";
                                if (i != mapping.size() - 1) {
                                    mapping_ss << ",";
                                }
                            }
                            mapping_ss << "]\"";
                        } else {
                            mapping_ss << "\"[]\"";
                        }
                        mapping_str = mapping_ss.str();

                        // 保存新的计算结果
                        {
                            std::lock_guard<std::mutex> lock(computed_results_mutex);
                            computed_results[pair] = std::make_tuple(ged_res, mapping_str);
                        }

                        // 将结果加入临时的 temp_nodes_in_exact_cluster
                        {
                            std::lock_guard<std::mutex> lock(temp_mutex);
                            temp_nodes_in_exact_cluster[anchor_id].push({static_cast<double>(ged_res), node_id});
                        }
                        // 从 anchor 的 nodes_in_cluster 中移除该节点
                        {
                            std::lock_guard<std::mutex> lock(temp_mutex);
                            auto& nodes_in_cluster = anchor_ptr->nodes_in_cluster;
                            // 创建一个新的优先队列，过滤掉已处理的节点
                            std::priority_queue<std::pair<double, int>> updated_nodes_in_cluster_inner;
                            while (!nodes_in_cluster.empty()) {
                                auto [d, nid] = nodes_in_cluster.top();
                                nodes_in_cluster.pop();
                                if (nid != node_id) {
                                    updated_nodes_in_cluster_inner.push({d, nid});
                                }
                            }
                            // 更新 nodes_in_cluster
                            nodes_in_cluster = updated_nodes_in_cluster_inner;
                        }

                        // 将结果写入结果文件
                        {
                            std::lock_guard<std::mutex> lock(result_mutex);
                            result_file << min_id << "," << max_id << "," << ged_res << "," << mapping_str << "\n";
                            result_file.flush();
                        }
                    } else {
                        // GED超过阈值或超时，写入failed results
                        std::string result_type = (ged_res == -1) ? "TIMEOUT" : "OVER_THRESHOLD";

                        // 写入failed CSV文件
                        {
                            std::lock_guard<std::mutex> lock(result_mutex);
                            failed_file << min_id << "," << max_id << "," << result_type << "\n";
                            failed_file.flush();
                        }

                        // 如果不是超时，也保存到内存缓存（用于friend归入）
                        if (ged_res != -1) {
                            std::stringstream mapping_ss;
                            if (!mapping.empty()) {
                                mapping_ss << "\"[";
                                for (size_t i = 0; i < mapping.size(); ++i) {
                                    mapping_ss << "(" << mapping[i].first << "," << mapping[i].second << ")";
                                    if (i != mapping.size() - 1) {
                                        mapping_ss << ",";
                                    }
                                }
                                mapping_ss << "]\"";
                            } else {
                                mapping_ss << "\"[]\"";
                            }
                            mapping_str = mapping_ss.str();

                            // 保存计算结果到缓存（用于friend归入）
                            {
                                std::lock_guard<std::mutex> lock(computed_results_mutex);
                                computed_results[pair] = std::make_tuple(ged_res, mapping_str);
                            }
                        }

                        // 尝试通过 friends 进行归入 (可通过开关控制)
                        bool assigned_via_friend = false;

                        // 检查是否启用friends收留功能
                        if (!parent_->enable_friends_reassign) {
                            // 功能被禁用，跳过friends收留逻辑
                        } else {
                        for (const auto& [friend_dist, friend_id] : anchor_ptr->friends) {
                            // 检查是否已有计算结果
                            int min_friend_id = std::min(friend_id, node_id);
                            int max_friend_id = std::max(friend_id, node_id);
                            std::pair<int, int> friend_pair = std::make_pair(min_friend_id, max_friend_id);

                            bool has_friend_result = false;
                            int friend_ged_res = -1;
                            std::string friend_mapping_str;

                            {
                                std::lock_guard<std::mutex> lock(computed_results_mutex);
                                auto it_friend = computed_results.find(friend_pair);
                                if (it_friend != computed_results.end()) {
                                    has_friend_result = true;
                                    friend_ged_res = std::get<0>(it_friend->second);
                                    friend_mapping_str = std::get<1>(it_friend->second);
                                }
                            }

                            if (has_friend_result) {
                                if (friend_ged_res != -1 && friend_ged_res <= parent_->max_exact_ged_for_EPT) {
                                    // 将节点加入朋友的 temp_nodes_in_exact_cluster
                                    {
                                        std::lock_guard<std::mutex> lock(temp_mutex);
                                        temp_nodes_in_exact_cluster[friend_id].push({static_cast<double>(friend_ged_res), node_id});
                                    }
                                    assigned_via_friend = true;
                                    break; // 已成功归入，跳出 friends 循环
                                }
                            } else {
                                // 需要计算与朋友的 GED
                                auto friend_node = parent_->nodes[friend_id];
                                auto friend_anchor = std::dynamic_pointer_cast<Anchor>(friend_node);
                                if (!friend_anchor) continue;

                                auto friend_graph = friend_anchor->graph.get();

                                if (!friend_graph) continue;

                                try {
                                    ui friend_LB = friend_graph->ged_lower_bound_filter(
                                        graph2,
                                        static_cast<ui>(parent_->max_exact_ged_for_EPT),
                                        parent_->vM.size(),    // 顶点标签种类数量
                                        parent_->eM.size(),    // 边标签种类数量
                                        parent_->max_n         // 最大节点数
                                    );
                                    if (friend_LB > parent_->max_exact_ged_for_EPT) {
                                        // LB 大于阈值，不需要计算
                                        continue;
                                    }
                                    // double friend_ML_ged = ML_graph::get_ML_ged(friend_anchor->ml_graph, node_other->ml_graph);
                                    // if (friend_ML_ged > parent_->max_exact_ged_for_EPT + 3 * parent_->error_tolerance_index) {
                                    //     // ML GED 大于阈值，不需要计算
                                    //     continue;
                                    // }
                                    // 调用 AppForComputation
                                    Application app(static_cast<ui>(parent_->max_exact_ged_for_EPT), "BMao");
                                app.set_all_edge_labels_same(parent_->all_edge_labels_same);
                                    app.init(friend_graph, graph2);
                                    int lb_astar_friend = -1;
                                    std::vector<std::pair<ui, ui>> mapping_friend;
                                    int friend_ged_res = app.AppForComputation(&mapping_friend, &lb_astar_friend);
                                    if (mapping_friend.empty()) {
                                        app.get_mapping(mapping_friend);
                                    }

                                    if (friend_ged_res != -1 && friend_ged_res <= parent_->max_exact_ged_for_EPT) {
                                        // 将节点映射转换为字符串
                                        std::stringstream mapping_ss_friend;
                                        if (!mapping_friend.empty()) {
                                            mapping_ss_friend << "\"[";
                                            for (size_t i = 0; i < mapping_friend.size(); ++i) {
                                                mapping_ss_friend << "(" << mapping_friend[i].first << "," << mapping_friend[i].second << ")";
                                                if (i != mapping_friend.size() - 1) {
                                                    mapping_ss_friend << ",";
                                                }
                                            }
                                            mapping_ss_friend << "]\"";
                                        } else {
                                            mapping_ss_friend << "\"[]\"";
                                        }
                                        friend_mapping_str = mapping_ss_friend.str();

                                        // 保存新的计算结果
                                        {
                                            std::lock_guard<std::mutex> lock(computed_results_mutex);
                                            computed_results[friend_pair] = std::make_tuple(friend_ged_res, friend_mapping_str);
                                        }

                                        // 将节点加入朋友的 temp_nodes_in_exact_cluster
                                        {
                                            std::lock_guard<std::mutex> lock(temp_mutex);
                                            temp_nodes_in_exact_cluster[friend_id].push({static_cast<double>(friend_ged_res), node_id});
                                        }

                                        // 将结果写入结果文件
                                        {
                                            std::lock_guard<std::mutex> lock(result_mutex);
                                            result_file << min_friend_id << "," << max_friend_id << "," << friend_ged_res << "," << friend_mapping_str << "\n";
                                            result_file.flush();
                                        }

                                        assigned_via_friend = true;
                                        break; // 已成功归入，跳出 friends 循环
                                    } else {
                                        // GED 超过阈值，不需要写入结果文件
                                    }

                                } catch (const std::exception& e) {
                                    // 捕获异常，输出错误信息
                                    std::cerr << "Exception occurred while processing node IDs: " << friend_id << ", " << node_id << ". Exception: " << e.what() << std::endl;
                                    // 不需要写入错误信息到结果文件
                                }
                            }
                        }

                        if (!assigned_via_friend) {
                            // 无法通过 friends 归入，节点保留在 anchor 的 nodes_in_cluster 中
                            // 由于在主线程中已经保留，无需额外处理
                        } else {
                            // 从 anchor 的 nodes_in_cluster 中移除该节点
                            {
                                std::lock_guard<std::mutex> lock(temp_mutex);
                                auto& nodes_in_cluster = anchor_ptr->nodes_in_cluster;
                                // 创建一个新的优先队列，过滤掉已处理的节点
                                std::priority_queue<std::pair<double, int>> updated_nodes_in_cluster_inner;
                                while (!nodes_in_cluster.empty()) {
                                    auto [d, nid] = nodes_in_cluster.top();
                                    nodes_in_cluster.pop();
                                    if (nid != node_id) {
                                        updated_nodes_in_cluster_inner.push({d, nid});
                                    }
                                }
                                // 更新 nodes_in_cluster
                                nodes_in_cluster = updated_nodes_in_cluster_inner;
                            }
                        }
                        } // 结束 enable_friends_reassign 条件检查
                    }

                } catch (const std::exception& e) {
                    // 捕获异常，输出错误信息
                    std::cerr << "Exception occurred while processing node IDs: " << anchor_id << ", " << node_id << ". Exception: " << e.what() << std::endl;
                    // 不需要写入错误信息到结果文件
                }
            }

            // 增加已处理的任务数
            size_t current_processed = processed_tasks.fetch_add(1) + 1;

            // 更新进度显示
            if (current_processed % 100 == 0 || current_processed == total_tasks) {
                double progress = (double)current_processed / total_tasks;

                // 计算已用时间和剩余时间
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = now - start_time;
                double estimated_total_time = elapsed.count() / progress;
                double remaining_time = estimated_total_time - elapsed.count();

                // 输出进度信息
                {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "\rProgress: " << std::fixed << std::setprecision(2)
                              << (progress * 100) << "% (" << current_processed << "/" << total_tasks << ")"
                              << ", Elapsed Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s"
                              << ", Estimated Remaining Time: " << std::fixed << std::setprecision(2)
                              << std::max(0.0, remaining_time) << "s"
                              << std::flush;
                }
            }

            // 获取下一个任务索引
            index = task_index.fetch_add(1);
        }
    };

    // 启动线程
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // 关闭结果文件
    result_file.close();

    // **第三步：将临时结果更新到 anchors 的 nodes_in_exact_cluster**
    for (const auto& [anchor_id, pq] : temp_nodes_in_exact_cluster) {
        if (anchor_id >= 0 && anchor_id < parent_->nodes.size()) {
            auto node_anchor = parent_->nodes[anchor_id];
            auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(node_anchor);
            if (!anchor_ptr) {
                std::cerr << "[ERROR] Node " << anchor_id << " is not an Anchor.\n";
                continue;
            }

            // 1) 先把 pq 中的 (dist, node_id) 转存到一个新的队列 new_queue
            std::priority_queue<Anchor::ExactClusterNode> new_queue;

            // 由于优先队列只能逐个 pop，我们要复制/重建
            auto tmp = pq; // 拷贝一份，避免破坏原 pq
            while (!tmp.empty()) {
                auto [dist, node_id] = tmp.top();
                tmp.pop();

                Anchor::ExactClusterNode ecn;
                ecn.dist = dist;
                ecn.node_id = node_id;
                // 若你没有mapping，就留空；若有 mapping，需要另行存储

                new_queue.push(ecn);
            }

            // 2) 最后再赋值给 anchor_ptr->nodes_in_exact_cluster
            anchor_ptr->nodes_in_exact_cluster = std::move(new_queue);
        } else {
            std::cerr << "[ERROR] Anchor ID out of range: " << anchor_id << std::endl;
        }
    }

    // 最终显示进度100%
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\rProgress: 100.00% (" << total_tasks << "/" << total_tasks << ")"
                  << ", Elapsed Time: " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() << "s"
                  << ", Estimated Remaining Time: 0.00s" << std::endl;
    }

    result_file.close();
    failed_file.close();

    std::cout << "Finished reassign_nodes_in_cluster." << std::endl;
    std::cout << "Valid results saved to: " << result_filename << std::endl;
    std::cout << "Failed results saved to: " << failed_result_filename << std::endl;
    // 保存 net_dag 对象到文件
    parent_->net_dag->anchors = parent_->anchors;
    parent_->net_dag->nodes = parent_->nodes;
    std::string reassigned_dir = dataset_dir + "reassigned/";
    std::filesystem::create_directories(reassigned_dir);
    std::string resigned_file_path = reassigned_dir + parent_->index_name + ".dat";
    std::cout << "Saving NetDag object to file..." << std::endl;
    parent_->net_dag->save_to_file(resigned_file_path);
    std::cout << "NetDag object saved to file." << std::endl;
}

void NetDagConstructor::reassign_nodes_in_cluster_with_csv() {
    // 验证reassign前的覆盖情况
    verify_coverage("before_reassign_csv");

    // CSV 文件路径
    std::string dataset_dir = "./NetDags/" + parent_->dataset + "/";
    std::string ged_results_dir = dataset_dir + "ged_results/";
    std::string csv_filename = ged_results_dir + parent_->index_name + "_exact_ged_results.csv";

    // 检查 CSV 文件是否存在
    std::ifstream csv_check(csv_filename);
    if (!csv_check.is_open()) {
        std::cout << "[INFO] CSV file " << csv_filename << " not found. Skipping reassignment." << std::endl;
        return;
    }
    csv_check.close();

    // ============================================================
    // Step 1: 加载 CSV 到 map: (min_id, max_id) -> (ged, mapping_str)
    // ============================================================
    struct GedResult {
        int ged;
        std::string mapping_str;
    };
    std::unordered_map<uint64_t, GedResult> ged_map;  // key = min_id * 1000000 + max_id

    auto make_key = [](int id1, int id2) -> uint64_t {
        int min_id = std::min(id1, id2);
        int max_id = std::max(id1, id2);
        return static_cast<uint64_t>(min_id) * 1000000ULL + static_cast<uint64_t>(max_id);
    };

    std::ifstream csv_file(csv_filename);
    std::cout << "[reassign] Loading GED results from " << csv_filename << "..." << std::endl;

    std::string line;
    int loaded_count = 0;
    while (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string id1_str, id2_str, ged_str, mapping_str;
        if (!std::getline(ss, id1_str, ',')) continue;
        if (!std::getline(ss, id2_str, ',')) continue;
        if (!std::getline(ss, ged_str, ',')) continue;
        std::getline(ss, mapping_str);

        try {
            int id1 = std::stoi(id1_str);
            int id2 = std::stoi(id2_str);
            int ged = std::stoi(ged_str);

            uint64_t key = make_key(id1, id2);
            ged_map[key] = {ged, mapping_str};
            loaded_count++;
        } catch (...) {
            continue;
        }
    }
    csv_file.close();
    std::cout << "[reassign] Loaded " << loaded_count << " GED results." << std::endl;

    // ============================================================
    // Step 2: 遍历每个 anchor 的 nodes_in_cluster，根据 csv 决定分配
    // ============================================================
    int total_to_exact = 0;
    int total_to_cluster = 0;

    auto parse_mapping = [](const std::string& mapping_str) -> std::vector<std::pair<ui, ui>> {
        std::vector<std::pair<ui, ui>> mapping;
        std::string str = mapping_str;

        // 去除首尾空白和引号
        str.erase(0, str.find_first_not_of(" \t\r\n\""));
        str.erase(str.find_last_not_of(" \t\r\n\"") + 1);

        size_t pos = 0;
        while (true) {
            size_t start = str.find('(', pos);
            if (start == std::string::npos) break;
            size_t end = str.find(')', start);
            if (end == std::string::npos) break;

            std::string pair_str = str.substr(start + 1, end - start - 1);
            pos = end + 1;

            size_t comma = pair_str.find(',');
            if (comma == std::string::npos) continue;

            try {
                ui u = std::stoul(pair_str.substr(0, comma));
                ui v = std::stoul(pair_str.substr(comma + 1));
                mapping.emplace_back(u, v);
            } catch (...) {
                continue;
            }
        }
        return mapping;
    };

    // Build a set of all anchor IDs to skip them in clusters
    std::unordered_set<int> anchor_ids;
    for (const auto& a : parent_->anchors) {
        anchor_ids.insert(a->node_id);
    }
    int total_skipped_anchors = 0;

    for (auto& anchor : parent_->anchors) {
        int anchor_id = anchor->node_id;

        // 清空 exact_cluster
        while (!anchor->nodes_in_exact_cluster.empty()) {
            anchor->nodes_in_exact_cluster.pop();
        }

        // 处理 nodes_in_cluster
        std::priority_queue<std::pair<double, int>> old_cluster = anchor->nodes_in_cluster;
        std::priority_queue<std::pair<double, int>> new_cluster;

        while (!old_cluster.empty()) {
            auto [dist, node_id] = old_cluster.top();
            old_cluster.pop();

            // Skip if this node is also an anchor (including self)
            // Anchors should not appear in other anchors' clusters
            if (anchor_ids.count(node_id)) {
                total_skipped_anchors++;
                continue;
            }

            // 查找 csv 中是否有这个 pair
            uint64_t key = make_key(anchor_id, node_id);
            auto it = ged_map.find(key);

            if (it != ged_map.end() && it->second.ged <= parent_->max_exact_ged_for_EPT) {
                // 在 csv 中找到，且 GED <= 阈值 -> exact_cluster
                Anchor::ExactClusterNode ecn;
                ecn.dist = static_cast<double>(it->second.ged);
                ecn.node_id = node_id;
                ecn.mapping = parse_mapping(it->second.mapping_str);
                anchor->nodes_in_exact_cluster.push(ecn);
                total_to_exact++;
            } else {
                // 不在 csv 中，或 GED > 阈值 -> 留在 cluster
                new_cluster.push({dist, node_id});
                total_to_cluster++;
            }
        }

        anchor->nodes_in_cluster = new_cluster;
    }
    std::cout << "[reassign] Skipped " << total_skipped_anchors << " anchor nodes from clusters." << std::endl;

    std::cout << "[reassign] Completed: " << total_to_exact << " nodes to exact_cluster, "
              << total_to_cluster << " nodes remain in cluster." << std::endl;

    // 验证reassign后的覆盖情况
    verify_coverage("after_reassign_csv");
}

void NetDagConstructor::filter_friends_before_saving() {
    // 距离阈值：4 * alpha
    double friend_distance_threshold = 4.0 * parent_->alpha;

    // 遍历所有锚点，筛选朋友关系
    for (auto& anchor : parent_->anchors) {
        if (!anchor) continue;

        // 创建新的朋友列表，只保留距离 <= 4*alpha 的朋友
        std::vector<std::pair<double, int>> filtered_friends;
        filtered_friends.reserve(anchor->friends.size());

        for (const auto& friend_pair : anchor->friends) {
            if (friend_pair.first <= friend_distance_threshold) {
                filtered_friends.push_back(friend_pair);
            }
        }

        // 更新朋友列表
        anchor->friends = std::move(filtered_friends);
    }

    // 同步更新 net_dag 中的 anchors
    parent_->net_dag->anchors = parent_->anchors;
}

int NetDagConstructor::verify_coverage(const std::string& checkpoint_name) {
    // 检查所有节点是否都被覆盖（要么是anchor，要么在某个anchor的cluster中）
    std::unordered_set<int> covered;

    // 1. 所有anchor本身都是覆盖的
    for (const auto& anchor : parent_->anchors) {
        covered.insert(anchor->node_id);

        // 2. anchor的nodes_in_cluster中的节点也是覆盖的
        std::priority_queue<std::pair<double, int>> nodes_copy = anchor->nodes_in_cluster;
        while (!nodes_copy.empty()) {
            covered.insert(nodes_copy.top().second);
            nodes_copy.pop();
        }

        // 3. anchor的nodes_in_exact_cluster中的节点也是覆盖的
        auto exact_copy = anchor->nodes_in_exact_cluster;
        while (!exact_copy.empty()) {
            covered.insert(exact_copy.top().node_id);
            exact_copy.pop();
        }
    }

    // 统计未覆盖的节点
    int total_nodes = static_cast<int>(parent_->nodes.size());
    int uncovered_count = 0;
    std::vector<int> uncovered_samples;

    for (int i = 0; i < total_nodes; i++) {
        if (parent_->nodes[i] && covered.find(i) == covered.end()) {
            uncovered_count++;
            if (uncovered_samples.size() < 10) {
                uncovered_samples.push_back(i);
            }
        }
    }

    // 统计各类节点和重复
    size_t anchor_count = 0, cluster_count = 0, exact_count = 0;
    std::unordered_map<int, int> cluster_node_counts, exact_node_counts;
    for (const auto& anchor : parent_->anchors) {
        anchor_count++;
        std::priority_queue<std::pair<double, int>> nodes_copy = anchor->nodes_in_cluster;
        while (!nodes_copy.empty()) {
            cluster_node_counts[nodes_copy.top().second]++;
            cluster_count++;
            nodes_copy.pop();
        }
        auto exact_copy = anchor->nodes_in_exact_cluster;
        while (!exact_copy.empty()) {
            exact_node_counts[exact_copy.top().node_id]++;
            exact_count++;
            exact_copy.pop();
        }
    }
    // 统计重复
    int cluster_dups = 0, exact_dups = 0;
    for (const auto& [id, cnt] : cluster_node_counts) {
        if (cnt > 1) cluster_dups++;
    }
    for (const auto& [id, cnt] : exact_node_counts) {
        if (cnt > 1) exact_dups++;
    }
    std::cout << "[COVERAGE CHECK] " << checkpoint_name << ": "
              << covered.size() << "/" << total_nodes << " covered, "
              << uncovered_count << " uncovered (anchors=" << anchor_count
              << ", cluster=" << cluster_count << "[dups=" << cluster_dups << "]"
              << ", exact=" << exact_count << "[dups=" << exact_dups << "])";

    if (!uncovered_samples.empty()) {
        std::cout << " (samples:";
        for (int id : uncovered_samples) {
            auto node = parent_->nodes[id];
            std::cout << " " << id << "[na=" << node->nearest_anchor << "]";
        }
        std::cout << ")";
    }
    std::cout << std::endl;

    return uncovered_count;
}