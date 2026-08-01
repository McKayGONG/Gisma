#include "GismaSearchEngine.h"
#include "EditPathTree.h"
#include "Application.h"
#include "OrigApp.h"
#include "Utility.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <chrono>
#include <unordered_set>
#include <atomic>
#include <future>
#include <thread>
#include <mutex>
#include <stack>
#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <filesystem>
#include <cstdlib>  // for std::atexit

// 全局覆盖集合（在 experiment_mode_impl.cpp 中定义）
extern std::unordered_set<int> g_ept_coverage;
extern std::unordered_set<int> g_extra_coverage;

// Reuse logging (defined in experiment_mode_impl.cpp)
extern FILE* g_reuse_log;
extern std::mutex g_reuse_log_mutex;

// 跨平台 localtime 函数
#ifdef _WIN32
    #define LOCALTIME_SAFE(tm_ptr, time_ptr) localtime_s(tm_ptr, time_ptr)
#else
    #define LOCALTIME_SAFE(tm_ptr, time_ptr) localtime_r(time_ptr, tm_ptr)
#endif


// 构造函数的实现 - Test modification
GismaSearchEngine::GismaSearchEngine(
    std::shared_ptr<NetDag> net_dag,
    double tau_index,
    // double tau,  // 移除
    double error_tolerance_search,
    int q_start,
    int q_end,
    bool has_ged_matrix,
    const std::vector<double> &ged_matrix,
    const std::string &search_method,
    const std::string &dataset_name,
    const std::vector<std::shared_ptr<Node>> &db_node_list,
    const std::vector<std::shared_ptr<Node>> &query_node_list,
    int N,
    std::vector<Graph *> db,
    const std::map<int, std::map<double, std::vector<int>>> &ground_truth,
    const std::map<std::string, ui> &vM,
    const std::map<std::string, ui> &eM,
    ui max_n,
    EditPathTreeManager *ept_manager,
    const std::string& nd_mode,
    const std::string& dfs_mode,
    bool use_ept_filters,
    bool only_compute_db_graph,
    int _app_max_iter,
    bool _fast_down,
    int _exact_max_iter,
    double _nd_filter_ratio,
    bool _disable_lsa_pruning,
    bool _disable_reuse_lsa,
    bool _verify_reuse,
    bool _chain_reuse) : net_dag(net_dag),
                                        tau_index(tau_index),
                                        // tau(tau),  // 移除
                                        error_tolerance_search(error_tolerance_search),
                                        has_ged_matrix(has_ged_matrix),
                                        ged_matrix(ged_matrix),
                                        search_method(search_method),
                                        dataset_name(dataset_name),
                                        db_node_list(db_node_list),
                                        query_node_list(query_node_list),
                                        N(N),
                                        db(db),
                                        ground_truth(ground_truth),
                                        vM(vM),
                                        eM(eM),
                                        max_n(max_n),
                                        ept_manager(ept_manager),
                                        q_start(q_start),
                                        q_end(q_end),
                                        nd_mode(nd_mode),
                                        dfs_mode(dfs_mode),
                                        use_ept_filters(use_ept_filters),
                                        only_compute_db_graph(only_compute_db_graph),
                                        app_max_iter(_app_max_iter),
                                        fast_down(_fast_down),
                                        exact_max_iter(_exact_max_iter),
                                        nd_filter_ratio(_nd_filter_ratio),
                                        disable_lsa_pruning(_disable_lsa_pruning),
                                        disable_reuse_lsa(_disable_reuse_lsa),
                                        verify_reuse(_verify_reuse),
                                        chain_reuse(_chain_reuse),
                                        disable_subtree_pruning(false),
                                        disable_lb_propagation(false)
{
}

GismaSearchEngine::~GismaSearchEngine() {
    // 显式清理EPT管理器相关资源
    if (ept_manager) {
        // 让EPT管理器清理所有EPT资源
        ept_manager->clear_all_epts();
    }
}

double GismaSearchEngine::get_ged_from_matrix(int i, int j)
{
    if (i < 0 || j < 0 || i >= N || j >= N)
    {
        throw std::out_of_range("Index out of bounds in get_ged_from_matrix");
    }

    auto get_upper_tri_index = [](int N, int i, int j) -> size_t
    {
        if (i > j)
            std::swap(i, j);
        return static_cast<size_t>(i * N - (i * (i - 1)) / 2 + (j - i));
    };

    size_t index = get_upper_tri_index(N, i, j);
    return ged_matrix[index];
}



std::vector<int> GismaSearchEngine::BMao_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // 结果
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.EPT_ndc_count++;

        // ========== LB 计算 ==========
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau) {
            continue;
        }

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter, exact_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);

        app.init(db_graph, query_graph);

        int ged_res = app.App_baseline(nullptr, nullptr);

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

std::vector<int> GismaSearchEngine::App_LSa_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;
        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);
        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();
        if (lb > tau) { continue; }
#endif

        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "LSa", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        int ged_res = app.App_baseline(nullptr, nullptr);
        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

std::vector<int> GismaSearchEngine::AStar_LSa_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;
        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);
        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();
        if (lb > tau) { continue; }

        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "LSa", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        int ged_res = app.AStar_baseline(nullptr, nullptr);
        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

std::vector<int> GismaSearchEngine::App_BMao_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // 结果
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB 计算 ==========
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau) {
            continue;
        }
#endif

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        int ged_res = app.App_baseline(nullptr, nullptr);

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

// 原版 App-BMao: 全量扫 db + Gisma 的 LB filter + origbmao 引擎(精简 State + 原版 AStar)
// 与 App_BMao_search 结构相同, 唯一区别 verify 用 origbmao::Application (逐字节原版), 无 LSa/reuse/snapshot 开销
std::vector<int> GismaSearchEngine::App_BMao_orig_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;
        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);
        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();
        if (lb > tau) continue;
#endif

        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;
        origbmao::Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.init(db_graph, query_graph);     // origbmao::init 读 Gisma Graph 的 n/pstarts/edges/vlabels/elabels
        int ged_res = app.AStar();           // 原版精确 A* (app_max_iter>0 时近似截断)
        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res >= 0 && ged_res <= tau)
            exact_results_within_tau.push_back(static_cast<int>(node_id));
    }

    return exact_results_within_tau;
}

// 纯Base方法，使用AStar_baseline()（精确A*）
std::vector<int> GismaSearchEngine::AStar_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB 计算 ==========
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau) {
            continue;
        }
#endif

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        int ged_res = app.AStar_baseline();  // 使用精确A*

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

// 纯Base方法，使用AStar()但禁用LSa剪枝（公平对比App_baseline）
std::vector<int> GismaSearchEngine::AStar_scan_no_lsa_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB 计算 ==========
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau) {
            continue;
        }
#endif

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        app.set_disable_lsa_pruning(true);  // 禁用LSa剪枝
        int ged_res = app.App();

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

std::vector<int> GismaSearchEngine::AStar_BMao_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // 结果
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // ========== LB 计算 ==========
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau) {
            continue;
        }

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);

        app.init(db_graph, query_graph);
        int ged_res = app.AStar_baseline(nullptr, nullptr);

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

// Base+GS: 用GS选anchors，然后遍历这些anchors的cluster和exact_cluster，用App-BMao验证
std::vector<int> GismaSearchEngine::Base_GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    // 1) 使用 GS_search 获取候选 anchors
    auto candidate_anchors_with_results = GS_search(query_node, tau, stats);

    if (candidate_anchors_with_results.empty()) {
        return exact_results_within_tau;
    }

    // 2) 收集所有候选anchor的complete_ids（从EPT中收集）和cluster中未进EPT的图ID
    std::unordered_set<int> candidate_graph_ids;

    for (const auto& [anchor_id, netdag_lb, netdag_ged] : candidate_anchors_with_results) {
        auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
        if (!anchor_ptr) {
            continue;
        }

        // 首先加入anchor本身
        candidate_graph_ids.insert(anchor_id);

        // 从nodes_in_cluster_vec中收集所有图ID（这些是没进EPT的图）
        for (const auto& [dist, graph_id] : anchor_ptr->nodes_in_cluster_vec) {
            candidate_graph_ids.insert(graph_id);
        }

        // 从EPT中收集complete_ids（已经计算过精确GED的图）
        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
        if (ept && !ept->tree_nodes.empty()) {
            // 遍历EPT的所有节点，收集complete_ids
            for (const auto& tree_node : ept->tree_nodes) {
                for (int complete_id : tree_node.completed_db_graph_ids) {
                    candidate_graph_ids.insert(complete_id);
                }
            }
        }
    }

    // 3) 排序候选图ID以提升cache locality，然后使用App-BMao方式验证
    std::vector<int> sorted_candidates(candidate_graph_ids.begin(), candidate_graph_ids.end());
    std::sort(sorted_candidates.begin(), sorted_candidates.end());
    for (int node_id : sorted_candidates) {
        // 检查node_id是否有效
        if (node_id < 0 || node_id >= static_cast<int>(db.size())) {
            continue;
        }

        Graph *db_graph = db[node_id];
        if (!db_graph) {
            continue;
        }

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB 计算 ==========
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;
        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);
        auto lb_end = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau) {
            continue;
        }
#endif

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        int ged_res = app.App_baseline(nullptr, nullptr);

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau) {
            exact_results_within_tau.push_back(node_id);
        }
    }

    return exact_results_within_tau;
}

// Base+SS: 跳过GS层次化导航，直接对所有anchors做距离计算，然后用SS
std::vector<int> GismaSearchEngine::Base_SS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    // Debug: Store selected anchors for Query 9
    std::vector<int> base_ss_selected_anchors;
    std::unordered_set<int> result_set;  // 用于去重

    // Base+SS: 类似Gisma但跳过GS层次化导航
    // 1. 计算所有anchor到query的距离
    // 2. 选择距离在范围内的anchor
    // 3. 对选中的anchor：搜索EPT + 搜索cluster

    std::vector<std::tuple<int, double, int>> candidate_anchors;  // (anchor_id, netdag_lb, netdag_ged)

    // 遍历所有anchor，计算距离
    for (const auto& anchor_ptr : net_dag->anchors) {
        if (!anchor_ptr) continue;

        int anchor_id = anchor_ptr->node_id;
        Graph *anchor_graph = anchor_ptr->graph.get();
        if (!anchor_graph) continue;

        // 计算anchor到query的距离
        ui netdag_lb, netdag_ged;

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.ND_ndc_count++;

        // 三种NetDag模式（与GS_search一致）
        if (nd_mode == "filters") {
            // 使用传统下界过滤器
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_lb_count++;
            netdag_lb = query_graph->ged_lower_bound_filter(
                anchor_graph, static_cast<ui>(net_dag->alpha + tau), vM.size(), eM.size(), max_n);
            netdag_ged = INF;

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;
        } else if (nd_mode == "astar") {
            // 使用App_test计算精确距离
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_astar_count++;
            Application app(static_cast<ui>(tau), "BMao", app_max_iter);
            app.set_all_edge_labels_same(all_edge_labels_same);
            app.init(anchor_graph, query_graph);
            netdag_ged = app.App_test(nullptr, nullptr);
            netdag_lb = app.get_overall_lb();
            stats.ND_app_test_count++;

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_astar_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;
        } else if (nd_mode == "filters_astar") {
            // 先用filters计算lb，如果lb<=tau则用AStar获取精确GED
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_lb_count++;
            netdag_lb = query_graph->ged_lower_bound_filter(
                anchor_graph, static_cast<ui>(net_dag->alpha + tau), vM.size(), eM.size(), max_n);

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;

            // 如果lb<=tau，调用AStar获取精确GED
            if (netdag_lb <= tau) {
                auto t2 = std::chrono::high_resolution_clock::now();

                stats.ND_astar_count++;
                Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                app.set_all_edge_labels_same(all_edge_labels_same);
                app.init(anchor_graph, query_graph);
                netdag_ged = app.App_test(nullptr, nullptr);
                netdag_lb = app.get_overall_lb();
                stats.ND_app_test_count++;

                auto t3 = std::chrono::high_resolution_clock::now();
                stats.ND_astar_time += std::chrono::duration<double>(t3 - t2).count();
            } else {
                netdag_ged = INF;
            }
        } else {
            // 未知模式，使用filters模式作为默认
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_lb_count++;
            netdag_lb = query_graph->ged_lower_bound_filter(
                anchor_graph, static_cast<ui>(net_dag->alpha + tau), vM.size(), eM.size(), max_n);
            netdag_ged = INF;

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;
        }

        candidate_anchors.emplace_back(anchor_id, netdag_lb, netdag_ged);
    }

    // 对选中的anchor进行EPT搜索和cluster搜索
    for (const auto& [anchor_id, netdag_lb, netdag_ged] : candidate_anchors) {
        auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
        if (!anchor_ptr) continue;

        // 0. 检查anchor本身是否满足条件
        {
            stats.EPT_ndc_count++;  // NDC: anchor本身的检查
            auto lb_start = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_count++;
            ui lb = query_graph->ged_lower_bound_filter(
                db[anchor_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);
            auto lb_end = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

            if (lb <= tau) {
                // 进一步用A*验证
                stats.EPT_astar_count++;
                auto astar_start = std::chrono::high_resolution_clock::now();

                Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                app.set_all_edge_labels_same(all_edge_labels_same);
                app.init(db[anchor_id], query_graph);
                app.set_disable_lsa_pruning(disable_lsa_pruning);
                int ged_res = app.App();

                auto astar_end = std::chrono::high_resolution_clock::now();
                stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

                if (ged_res <= tau) {
                    result_set.insert(anchor_id);

                    // IMPORTANT: 如果anchor自己满足条件（GED=0），也要把EPT root节点中的completed_db_graph_ids加入
                    // 这些是与anchor完全相同（GED=0）的图
                    if (ged_res == 0) {
                        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
                        if (ept && !ept->tree_nodes.empty()) {
                            const auto &root_completed = ept->tree_nodes[0].completed_db_graph_ids;
                            for (int gid : root_completed) {
                                result_set.insert(gid);
                            }
                        }
                    }
                }
            }
        }

        // 1. 搜索EPT
        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
        if (ept && !ept->tree_nodes.empty()) {
            auto ss_results = SS_search(query_node, anchor_id, netdag_lb, netdag_ged, tau, stats);
            for (int gid : ss_results) {
                result_set.insert(gid);
            }
        }

        // 2. 搜索cluster（extra_cluster_search）
        auto cluster_results = extra_cluster_search(query_node, anchor_id, tau, stats);
        for (int gid : cluster_results) {
            result_set.insert(gid);
        }
    }

    // 转换为vector返回
    exact_results_within_tau.assign(result_set.begin(), result_set.end());

    return exact_results_within_tau;
}

// Base_All_EPT: 遍历所有anchor，不做任何过滤，直接做SS_search和extra_search
std::vector<int> GismaSearchEngine::Base_All_EPT_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;
    std::unordered_set<int> result_set;  // 用于去重

    // 遍历所有anchor，不做任何过滤
    for (const auto& anchor_ptr : net_dag->anchors) {
        if (!anchor_ptr) continue;

        int anchor_id = anchor_ptr->node_id;

        // 0. 检查anchor本身是否满足条件
        {
            stats.EPT_ndc_count++;  // NDC: anchor本身的检查
            auto lb_start = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_count++;
            ui lb = query_graph->ged_lower_bound_filter(
                db[anchor_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);
            auto lb_end = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

            if (lb <= tau) {
                // 进一步用A*验证
                stats.EPT_astar_count++;
                auto astar_start = std::chrono::high_resolution_clock::now();

                Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                app.set_all_edge_labels_same(all_edge_labels_same);
                app.init(db[anchor_id], query_graph);
                app.set_disable_lsa_pruning(disable_lsa_pruning);
                int ged_res = app.App();

                auto astar_end = std::chrono::high_resolution_clock::now();
                stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

                if (ged_res <= tau) {
                    result_set.insert(anchor_id);

                    // IMPORTANT: 如果anchor自己满足条件（GED=0），也要把EPT root节点中的completed_db_graph_ids加入
                    // 这些是与anchor完全相同（GED=0）的图
                    if (ged_res == 0) {
                        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
                        if (ept && !ept->tree_nodes.empty()) {
                            const auto &root_completed = ept->tree_nodes[0].completed_db_graph_ids;
                            for (int gid : root_completed) {
                                result_set.insert(gid);
                            }
                        }
                    }
                }
            }
        }

        // 1. 搜索EPT (使用SS_search)
        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
        if (ept && !ept->tree_nodes.empty()) {
            // 传递netdag_lb=0, netdag_ged=-1表示不使用NetDag过滤
            auto ss_results = SS_search(query_node, anchor_id, 0.0, -1, tau, stats);
            for (int gid : ss_results) {
                result_set.insert(gid);
            }
        }

        // 2. 搜索cluster（extra_cluster_search）
        auto cluster_results = extra_cluster_search(query_node, anchor_id, tau, stats);
        for (int gid : cluster_results) {
            result_set.insert(gid);
        }
    }

    // 转换为vector返回
    exact_results_within_tau.assign(result_set.begin(), result_set.end());

    return exact_results_within_tau;
}


std::vector<int> GismaSearchEngine::answer_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    std::vector<int> ids_within_tau;
    {
        std::lock_guard<std::mutex> lock(ground_truth_mutex);
        auto it = ground_truth.find(query_node->node_id);
        if (it != ground_truth.end())
        {
            const std::map<double, std::vector<int>> &distances = it->second;

            // 1) 找到 ground truth 中实际 ged <= tau 的所有节点
            for (const auto &distance_pair : distances)
            {
                double ged = distance_pair.first;
                const std::vector<int> &graph_ids = distance_pair.second;
                if (ged <= tau)
                {
                    ids_within_tau.insert(ids_within_tau.end(), graph_ids.begin(), graph_ids.end());
                }
            }
        }
    }  // lock 在此释放，A* 计算不再持锁
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // 结果
    std::vector<int> exact_results_within_tau;

    for (auto node_id : ids_within_tau)
    {
        Graph *db_graph = db[node_id];

        // NDC统计：每个被访问的节点计数一次（去重）
        stats.EPT_ndc_count++;

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_graph, query_graph);
        app.set_disable_lsa_pruning(disable_lsa_pruning);
        int ged_res = app.App();

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

std::vector<std::tuple<int, double, int>> GismaSearchEngine::GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    // ========== 1) 基本检查及初始化 ==========

    // candidate_anchor_ids 已被 candidate_anchors_with_lb 替代

    // 获取根节点 (Anchor)
    auto current_node = std::dynamic_pointer_cast<Anchor>(net_dag->root);
    if (!current_node)
    {
        std::cerr << "[GS_search] Error: root node is not Anchor or is null.\n";
        return {};
    }

    // 设定初始相位
    double current_phase = static_cast<double>(current_node->children.rbegin()->first);

    // 存储候选anchor及其对应的NetDag计算结果，避免DFS中重复计算
    std::vector<std::tuple<int, double, int>> candidate_anchors_with_results;  // (anchor_id, lb, ged_result)
    // 注意：当USE_ND_FILTERS=0时，使用App_test，ged_result是精确GED；当=1时，ged_result=-1表示无精确结果

    // 定义用于记录最优节点的变量（在循环外部，便于后续使用）
    double min_dist = std::numeric_limits<double>::infinity();
    int min_child_node_id = -1;

    // ========== 特殊情况：根节点直接连接到anchor层 ==========
    // 当 current_phase <= alpha 时，根节点的子节点就是anchor层，
    // 无需层级导航，直接对根节点children做候选收集
    if (current_phase <= net_dag->alpha)
    {
        auto it = current_node->children.find(static_cast<int>(current_phase));
        if (it == current_node->children.end())
        {
            return {};
        }

        Graph *query_graph = (query_node && query_node->graph) ? query_node->graph.get() : nullptr;
        if (!query_graph) return {};

        for (const auto &[child_node_id, child_node_dist] : it->second)
        {
            auto childNode = net_dag->nodes[child_node_id];
            if (!childNode || !childNode->graph) continue;

            Graph *db_graph = childNode->graph.get();
            ui netdag_lb, netdag_ged;

            stats.ND_ndc_count++;
            stats.ND_lb_count++;

            auto t0 = std::chrono::high_resolution_clock::now();
            ui threshold = static_cast<ui>(current_phase + tau);
            netdag_lb = query_graph->ged_lower_bound_filter(
                db_graph, threshold, vM.size(), eM.size(), max_n);
            netdag_ged = INF;
            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

            double range_threshold = net_dag->alpha + tau + error_tolerance_search;
            if (netdag_lb <= range_threshold)
            {
                candidate_anchors_with_results.emplace_back(child_node_id, netdag_lb, netdag_ged);
            }
        }

        return candidate_anchors_with_results;
    }

    // ========== 2) 循环：phase 不断折半，直到 <= alpha ==========

    while (current_phase > net_dag->alpha)
    {
        double child_phase = current_phase / 2.0;
        // 重置每一轮的最小距离和节点ID
        min_dist = std::numeric_limits<double>::infinity();
        min_child_node_id = -1;

        // Thread-safe access to children map
        // Use shared_lock for concurrent read access
        std::vector<std::pair<int, double>> children_at_phase_copy;
        {
            std::shared_lock<std::shared_mutex> lock(net_dag->children_mutex);

            // 找到对应 child_phase
            auto it = current_node->children.find(static_cast<int>(child_phase));
            if (it == current_node->children.end())
            {
                // 若无对应 child_phase，退出
                return {};
            }

            // Copy children_at_phase to avoid holding lock during iteration
            children_at_phase_copy = it->second;
        } // Release lock here

        const auto &children_at_phase = children_at_phase_copy;

        // ========== 3) 遍历 children_at_phase，LB检查 ==========

        for (const auto &[child_node_id, child_node_dist] : children_at_phase)
        {
            // 先简单检查 child_node_dist
            // TEMPORARILY DISABLED: This pre-filtering may cause recall loss
            // if (child_node_dist <= 1.5 * current_phase + 2 * tau + 3 * error_tolerance_search)
            if (true)  // Check all children without pre-filtering
            {
                // 获取 childNode
                auto childNode = net_dag->nodes[child_node_id];
                if (!childNode || !childNode->graph)
                {
                    // 略过无效节点
                    continue;
                }

                // 拿到要做 LB 检查的图
                Graph *db_graph = childNode->graph.get();
                Graph *query_graph = (query_node && query_node->graph)
                                         ? query_node->graph.get()
                                         : nullptr;
                if (!query_graph)
                {
                    continue;
                }

                // 声明在外部以便后续使用
                ui netdag_lb, netdag_ged;
                bool is_last_layer = (child_phase <= net_dag->alpha);

                // NDC统计：每个被访问的节点计数一次（去重）
                stats.ND_ndc_count++;

                // ========== 三种NetDag模式 ==========
                if (nd_mode == "filters") {
                    // ========== Mode 1: ND_only_filters (只用传统LB过滤器，不用AStar) ==========
                    auto t0 = std::chrono::high_resolution_clock::now();

                    // 使用统一的ged_lower_bound_filter
                    stats.ND_lb_count++;
                    ui threshold = static_cast<ui>(child_phase + tau);
                    netdag_lb = query_graph->ged_lower_bound_filter(
                        db_graph, threshold, vM.size(), eM.size(), max_n);
                    netdag_ged = INF; // filters模式没有精确GED，设为INF

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio)
                    {
                        continue;
                    }
                }
                else if (nd_mode == "astar") {
                    // ========== Mode 2: ND_only_AStar (跳过传统过滤器，直接用AStar) ==========
                    auto t0 = std::chrono::high_resolution_clock::now();

                    stats.ND_astar_count++; // App_test是完整A*搜索，计入astar_count
                    Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                    app.set_all_edge_labels_same(all_edge_labels_same);
                    app.init(db_graph, query_graph);
                    netdag_ged = app.App_test(nullptr, nullptr);
                    netdag_lb = app.get_overall_lb();
                    stats.ND_app_test_count++; // 统计App_test调用次数

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_astar_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio)
                    {
                        continue;
                    }
                }
                else if (nd_mode == "filters_astar") {
                    // ========== Mode 3: ND_filters_AStar (先用filters，最后一层且lb<=tau时才用AStar) ==========
                    // 非最后一层：只用filters
                    // 最后一层：先用filters，如果lb<=tau，再调用AStar获取精确GED

                    auto t0 = std::chrono::high_resolution_clock::now();

                    // 使用统一的ged_lower_bound_filter
                    stats.ND_lb_count++;
                    ui threshold = static_cast<ui>(child_phase + tau);
                    netdag_lb = query_graph->ged_lower_bound_filter(
                        db_graph, threshold, vM.size(), eM.size(), max_n);

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio)
                    {
                        continue;
                    }

                    // 关键判断：只有在最后一层且lb<=tau时才调用AStar
                    if (is_last_layer && netdag_lb <= tau)
                    {
                        auto t2 = std::chrono::high_resolution_clock::now();

                        stats.ND_astar_count++;
                        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                        app.set_all_edge_labels_same(all_edge_labels_same);
                        app.init(db_graph, query_graph);
                        netdag_ged = app.App_test(nullptr, nullptr);
                        netdag_lb = app.get_overall_lb();  // 使用AStar的lb更新
                        stats.ND_app_test_count++;

                        auto t3 = std::chrono::high_resolution_clock::now();
                        stats.ND_astar_time += std::chrono::duration<double>(t3 - t2).count();
                    }
                    else
                    {
                        // 非最后一层，或者最后一层但lb>tau，不调用AStar
                        netdag_ged = INF;
                    }
                }
                else {
                    // 未知模式，默认使用filters模式
                    std::cerr << "[GS_search] Warning: Unknown nd_mode '" << nd_mode << "', using 'filters' mode\n";

                    auto t0 = std::chrono::high_resolution_clock::now();

                    // 使用统一的ged_lower_bound_filter
                    stats.ND_lb_count++;
                    ui threshold = static_cast<ui>(child_phase + tau);
                    netdag_lb = query_graph->ged_lower_bound_filter(
                        db_graph, threshold, vM.size(), eM.size(), max_n);
                    netdag_ged = INF;

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio) continue;
                }

                // ========== 4) 根据策略选择节点 ==========
                // 如果是最后一层，收集所有符合条件的anchor，同时记录最优节点以继续遍历
                if (child_phase <= net_dag->alpha)
                {
                    // 最后一层：收集所有符合下界条件的anchor
                    // 使用 error_tolerance_search 参数扩展候选范围
                    double range_threshold = net_dag->alpha + tau + error_tolerance_search;
                    if (netdag_lb <= range_threshold)
                    {
                        // 存储anchor_id、下界和精确GED结果（当前使用App_test，所以有精确结果）
                        candidate_anchors_with_results.emplace_back(child_node_id, netdag_lb, netdag_ged);
                    }

                    // 同时记录最优节点以继续while循环
                    if (netdag_lb < min_dist)
                    {
                        min_dist = netdag_lb;
                        min_child_node_id = child_node_id;
                    }
                }
                else
                {
                    // 非最后一层：根据fast_down标志选择策略
                    if (fast_down)
                    {
                        // 快速下降策略：找到第一个满足条件的就立即选择
                        if (netdag_lb <= (child_phase + tau) * nd_filter_ratio)
                        {
                            min_dist = netdag_lb;
                            min_child_node_id = child_node_id;
                            break;  // 立即跳出for循环，选择此节点往下走
                        }
                    }
                    else
                    {
                        // 贪心策略：选择最小距离的节点
                        if (netdag_lb < min_dist)
                        {
                            min_dist = netdag_lb;
                            min_child_node_id = child_node_id;
                        }
                    }
                }

            } // end if (child_node_dist <= ...)
        }

        // 若没找到合适子节点，则退出
        if (min_child_node_id == -1)
        {
            return {};
        }

        // 转到下一个 childNode
        current_node = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[min_child_node_id]);
        if (!current_node)
        {
            return {};
        }
        current_phase = child_phase;
    }

    // ========== 5) 若退出 while 说明 current_phase <= alpha ==========

    // [diagnostic] env-gated dump of the giant-step candidate anchors (no production overhead unless set)
    if (std::getenv("GISMA_GS_DEBUG")) {
        std::cerr << "[GS-DBG] candidate anchors (" << candidate_anchors_with_results.size() << "): ";
        for (const auto& t : candidate_anchors_with_results) std::cerr << std::get<0>(t) << " ";
        std::cerr << "\n";
    }

    return candidate_anchors_with_results;
}

std::vector<int> GismaSearchEngine::SS_search(std::shared_ptr<Node> query_node, int anchor_id, double netdag_lb, int netdag_ged, double tau, SearchStats &stats, const std::string &dfs_mode_override)
{
    int query_id = query_node->node_id;

    // if (tau >= 6) use_ML = true;
    std::vector<int> exact_results_within_tau;

    EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);

    // 统计使用的EPT的总节点数
    if (ept) {
        stats.EPT_total_nodes_in_used_epts += ept->tree_nodes.size();
        if (e7_stats) stats.e7_ept_trees++;  // E7: count this EPT tree as entered
    }

    // 直接调用 traverse_ept_and_search(...)，传递NetDag计算结果，避免DFS根节点重复计算
    traverse_ept_and_search(*ept, query_node, exact_results_within_tau, tau, stats, netdag_lb, netdag_ged, dfs_mode_override);
    return exact_results_within_tau;
}


















// COMPUTE_ASTAR_ONLY_FOR_DATA_GRAPH macro is now defined in Utility.h

// 计算EPT子树大小的辅助函数
size_t GismaSearchEngine::calculate_subtree_size(const EditPathTree& ept, size_t node_index) {
    if (node_index >= ept.tree_nodes.size()) {
        return 0;
    }

    const TreeNode& node = ept.tree_nodes[node_index];
    size_t subtree_size = 1; // 当前节点本身

    // 递归计算所有子节点的子树大小
    for (size_t child_index : node.children_indices) {
        subtree_size += calculate_subtree_size(ept, child_index);
    }

    return subtree_size;
}

bool GismaSearchEngine::use_orig_verifier = false;

bool GismaSearchEngine::dfs_traverse_no_reuse(
    size_t                node_index,
    const EditPathTree   &ept,
    std::shared_ptr<Node> query_node,
    std::vector<int>     &exact_results_within_tau,
    double                tau,
    SearchStats          &stats,
    ui                    estimate_lb,
    double                anchor_netdag_lb,
    int                   anchor_netdag_ged)
{

    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();
    
    // 将tau转换为ui类型，避免类型问题
    ui tau_ui = (ui)tau;
    

    // ========== 节点类型统计 ==========
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }
    
    if (!node.completed_db_graph_ids.empty()) {
        stats.EPT_nodes_with_completed_ids++;
    }
    
    // 统计编辑操作类型
    if (node.op.type != EditOperation::NONE) {
        stats.EPT_op_type_count[node.op.type]++;
    }
    
    // ========== 用于传递给子节点的estimate_lb ==========
    ui new_estimate_lb = estimate_lb;  // 默认使用传入的值

    // 判断是否是db图（根节点或有completed_db_graph_ids的节点）
    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    // ========== Lower Bound 检查 ==========
    bool should_compute_ged = true;

    // NDC计数标志：确保每个节点只在实际做计算时计数一次
    bool ndc_counted = false;

    // 统计访问的节点总数
    stats.EPT_total_nodes_visited++;

    // 调试计数器
    static size_t debug_total_nodes = 0;
    static size_t debug_estimate_lb_skip = 0;
    static size_t debug_filter_skip = 0;
    static size_t debug_estimate_lb_skip_at_root = 0;
    static size_t debug_filter_skip_at_root = 0;
    debug_total_nodes++;

    bool is_root = (node_index == ept.root_index);

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    // LB Propagation: 如果estimate_lb > tau则跳过计算
    if (estimate_lb > tau_ui) {
        should_compute_ged = false;
        stats.lb_pruning_count++;  // 统计因estimate_lb > tau跳过的App_test计算
        debug_estimate_lb_skip++;
        if (is_root) {
            debug_estimate_lb_skip_at_root++;
        }
#ifdef DEBUG_PRUNING
        printf("[STATS] LB propagation skip: count=%zu, estimate_lb=%u, tau=%u\n",
               stats.lb_pruning_count, estimate_lb, tau_ui);
#endif
    }
#endif

    if (use_ept_filters && should_compute_ged) {
        // NDC统计
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        // 标记：进入filter计算，说明做了计算
        stats.EPT_nodes_computed++;

        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        // 分开统计db图和中间图的filter时间
        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }

        new_estimate_lb = std::max(new_estimate_lb, lb);  // 更新 estimate_lb

        if (lb > tau_ui) {
            should_compute_ged = false;
            stats.EPT_filter_pruned_nodes++;  // 统计被filter剪枝的节点
            debug_filter_skip++;
            if (is_root) {
                debug_filter_skip_at_root++;
            }
        }
    }
    

    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF) {
        // 只有当 netdag_ged 是有效值时（不是 INF）才使用缓存
        // nd_mode="filters" 时 netdag_ged=INF，此时不能跳过计算
        // nd_mode="astar"/"filters_astar" 时可能有有效的netdag_ged

        // 如果NetDag计算的GED <= tau，直接收入answer
        if (anchor_netdag_ged <= (int)tau_ui) {
            // 根节点：确保anchor本身被添加，再添加completed_db_graph_ids中的其他图
            int anchor_id = static_cast<int>(node.anchor_id);

            // 首先检查anchor_id是否已经在completed_db_graph_ids中
            bool anchor_in_completed = false;
            for (int id : node.completed_db_graph_ids) {
                if (id == anchor_id) {
                    anchor_in_completed = true;
                    break;
                }
            }

            // 如果anchor不在completed_db_graph_ids中，需要单独添加
            if (!anchor_in_completed) {
                exact_results_within_tau.push_back(anchor_id);
                stats.EPT_results_from_astar++;
            }

            // 添加completed_db_graph_ids中的所有图
            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                           node.completed_db_graph_ids.begin(),
                                           node.completed_db_graph_ids.end());
            stats.EPT_results_from_astar += node.completed_db_graph_ids.size();

            found_here = true;
            stats.root_netdag_ged_reuse_count++;  // 统计根节点复用次数
        }
        // else: netdag_ged > tau，不收入answer，但继续遍历子节点
        // 因为子节点通过编辑操作可能使GED减小到<=tau

        // 更新estimate_lb为NetDag的overall_lb（用于传递给子节点）
        new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));

        // 根节点已经使用NetDag的GED，不需要再计算
        should_compute_ged = false;
    } else if (node_index == ept.root_index && anchor_netdag_lb >= 0) {
        // nd_mode="filters" 时：netdag_ged=INF，但 netdag_lb 有效
        // 传递 netdag_lb 给子节点，但不跳过根节点的 GED 计算
        new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));
    }

    // ========== 检查是否需要计算GED ==========
    // 如果启用了EPT filters或only_compute_db_graph，则只对有completed_db_graph_ids的节点计算GED
    // 根节点（anchor本身）即使completed_db_graph_ids为空也需要计算GED，因为anchor本身可能就是一个结果
    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== 如果通过LB检查，执行App_test ==========
    if (should_compute_ged) {
        // NDC统计：如果跳过LB但做A*计算，也要计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        // 标记：执行了GED计算
        stats.EPT_nodes_computed++;

        // 执行App_test计算（无reuse，有overall_lb设置）
        auto t0 = std::chrono::high_resolution_clock::now();
        // E8 fix noreuse LSa: gisma-lsa / app-lsa 在 no_reuse 路径也须用 LSa 验证器（原硬编码 BMao）。
        const char* v_lb = (ged_algorithm == "gisma-lsa" || ged_algorithm == "app-lsa") ? "LSa" : "BMao";
        int ged;
        ui overall_lb_val = 0, overall_ub_val = 0;
        if (use_orig_verifier) {
            // 原作者引擎(48B State / 2 级堆比较 / 无 reuse 载荷), 与全扫 baseline 同一份验证器代码。
            origbmao::Application oapp(tau_ui, v_lb, (long long)app_max_iter);
            oapp.init(db_g, qry_g);
            ged = (int)oapp.AStar();
            overall_lb_val = oapp.get_overall_lb();
            overall_ub_val = (ui)ged;
        } else {
            Application app(tau_ui, v_lb, app_max_iter);
            app.set_all_edge_labels_same(all_edge_labels_same);
            app.init(db_g, qry_g);
            ged = app.App_test(nullptr, nullptr);
            overall_lb_val = app.get_overall_lb();
            overall_ub_val = app.get_overall_ub();
        }
        (void)overall_ub_val;
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double>(t1 - t0).count();

        stats.EPT_astar_count++;
        stats.EPT_astar_time += elapsed_time;

        // 分开统计db图和中间图的verification时间
        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        // 处理结果
        if (ged <= (int)tau_ui) {
            // 对于根节点，需要确保anchor本身被添加到结果中
            if (node_index == ept.root_index) {
                int anchor_id = static_cast<int>(ept.anchor_id);

                // 检查anchor是否已在completed_db_graph_ids中
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                     node.completed_db_graph_ids.end(),
                                                     anchor_id) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id);
                    stats.EPT_results_from_astar++;
                }
            }

            // 添加completed_db_graph_ids中的所有图
            size_t num_results = node.completed_db_graph_ids.size();
            stats.EPT_results_from_astar += num_results;

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                          node.completed_db_graph_ids.begin(),
                                          node.completed_db_graph_ids.end());
            found_here = true;
        }

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
        // LB Propagation: 更新estimate_lb
        new_estimate_lb = overall_lb_val;
#endif

        // ========== Subtree Pruning 剪枝检查 ==========
#if USE_SUBTREE_PRUNING
        // 获取当前图对的overall_lb（局部变量）
        ui overall_lb = overall_lb_val;

        // Subtree Pruning: 叶子节点跳过（没有子节点可以剪枝）
        if (!node.children_indices.empty() && overall_lb > 0 && overall_lb < 10000) {  // 假设合理的GED不会超过10000
            // 计算从当前节点还能走的最大步数
            int max_step_more = node.max_subtree_depth - node.level;
            
            // 调试输出（通过DEBUG_PRUNING宏控制）
#ifdef DEBUG_PRUNING
            printf("Subtree pruning check: node.level=%d, max_step_more=%d, tau=%u, overall_lb=%u\n",
                   node.level, max_step_more, tau_ui, overall_lb);
            printf("  -> Condition: %d + %u < %u ? %s\n",
                   max_step_more, tau_ui, overall_lb,
                   (max_step_more + (int)tau_ui < (int)overall_lb) ? "YES (prune)" : "NO (continue)");
#endif
            
            // 剪枝条件：即使接下来每步都是最优编辑（减少1），
            // 最终GED也至少是 overall_lb - max_step_more
            // 如果这个值 > tau，则无法找到满足条件的解
            if (max_step_more + (int)tau_ui < (int)overall_lb) {
#ifdef DEBUG_PRUNING
                printf("[SUBTREE_PRUNE] Node %zu (level=%d): max_steps=%d, tau=%u, overall_lb=%u, overall_ub=%u, is_leaf=%s -> PRUNED\n",
                       node_index, node.level, max_step_more, tau_ui, overall_lb, overall_ub_val,
                       node.children_indices.empty() ? "yes" : "no");
#endif

                // 统计subtree pruning效果
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // 减去当前节点本身
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                // 统计：有多少是叶子节点触发的
                if (node.children_indices.empty()) {
                    stats.subtree_pruning_on_leaf_nodes++;
                }
#ifdef DEBUG_PRUNING
                printf("[STATS] Subtree pruning: decisions=%zu, avoided_nodes=%zu (is_leaf=%s)\n",
                       stats.subtree_pruning_decisions, avoided_nodes,
                       node.children_indices.empty() ? "yes" : "no");
#endif

                return found_here;
            }
        }
#ifdef DEBUG_PRUNING
        else {
            printf("Abnormal overall_lb value (%u), skipping pruning\n", overall_lb);
        }
#endif
#endif
    }
#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    else {
        // 跳过了App_test计算（可能是因为estimate_lb或filter），但仍然要检查是否可以剪枝
#if USE_SUBTREE_PRUNING
        // Subtree Pruning: 叶子节点跳过（没有子节点可以剪枝）
        if (!node.children_indices.empty() && new_estimate_lb > 0 && new_estimate_lb < 10000) {
            int max_step_more;
            
            max_step_more = node.max_subtree_depth - node.level;
            
            if (max_step_more + (int)tau_ui < (int)new_estimate_lb) {
                // 剪枝：不递归处理子节点

                // 统计subtree pruning效果（estimate_lb分支）
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // 减去当前节点本身
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                return found_here;
            }
        }
#endif
    }
#endif


    // 递归处理子节点
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }
        
        // 计算子节点的estimate_lb
        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        ui child_estimate_lb = (level_diff >= (int)new_estimate_lb) ? 0 : new_estimate_lb - level_diff;

        // 统计传递estimate_lb到子节点的次数
        stats.lb_propagation_count++;

        if (dfs_traverse_no_reuse(ch, ept, query_node,
                               exact_results_within_tau, tau, stats,
                               child_estimate_lb, anchor_netdag_lb, anchor_netdag_ged)) {
            found_child = true;
        }
    }

    // 返回当前节点或其子树是否找到答案
    return found_here || found_child;
}

// 纯DFS遍历 - 使用App_baseline，无LB传递，无lookahead剪枝
// 这是最简单的遍历方式，用于基线对比
bool GismaSearchEngine::dfs_traverse_only_dfs(
    size_t                node_index,
    const EditPathTree   &ept,
    std::shared_ptr<Node> query_node,
    std::vector<int>     &exact_results_within_tau,
    double                tau,
    SearchStats          &stats,
    double                anchor_netdag_lb,
    int                   anchor_netdag_ged)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;

    // NDC计数标志：确保每个节点只在实际做计算时计数一次
    bool ndc_counted = false;

    // 节点类型统计
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }

    // 检查节点是否关联数据库图（非中间图）
    // 根节点是anchor本身（db图），或者有completed_db_graph_ids的节点也是db图
    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    // 决定是否需要计算GED
    bool should_compute_ged = true;

    // ========== 根节点特殊处理：使用NetDag缓存的GED值 ==========
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF) {
        // 只有当 netdag_ged 是有效值时（不是 INF）才使用缓存
        if (anchor_netdag_ged <= (int)tau_ui) {
            int anchor_id = static_cast<int>(node.anchor_id);

            bool anchor_in_completed = false;
            for (int id : node.completed_db_graph_ids) {
                if (id == anchor_id) {
                    anchor_in_completed = true;
                    break;
                }
            }

            if (!anchor_in_completed) {
                exact_results_within_tau.push_back(anchor_id);
                stats.EPT_results_from_astar++;
            }

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                           node.completed_db_graph_ids.begin(),
                                           node.completed_db_graph_ids.end());
            stats.EPT_results_from_astar += node.completed_db_graph_ids.size();

            found_here = true;
            stats.root_netdag_ged_reuse_count++;
        }
        // 根节点已经使用NetDag的GED，不需要再计算
        should_compute_ged = false;
    }

    // ========== LB Filter (与simple模式一致) ==========
    if (use_ept_filters && should_compute_ged) {
        // NDC统计：只在实际做LB计算时计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        stats.EPT_nodes_computed++;

        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        // 分开统计db图和中间图的filter时间
        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }

        if (lb > tau_ui) {
            should_compute_ged = false;
            stats.EPT_filter_pruned_nodes++;
        }
    }

    // ========== 检查是否需要计算GED（use_ept_filters或only_compute_db_graph控制）==========
    // 如果启用了use_ept_filters或only_compute_db_graph，则只对有completed_db_graph_ids的节点计算GED
    // 根节点即使completed_db_graph_ids为空也需要计算GED
    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== 如果需要计算GED，执行App_baseline ==========
    if (should_compute_ged) {
        // NDC统计：如果跳过LB但做A*计算，也要计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        stats.EPT_nodes_computed++;

        auto t0 = std::chrono::high_resolution_clock::now();
        Application app(tau_ui, "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_g, qry_g);

        // 使用App_baseline - 最基础的计算方法
        int ged = app.App_baseline(nullptr, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double>(t1 - t0).count();

        stats.EPT_astar_count++;
        stats.EPT_astar_time += elapsed_time;

        // 分开统计db图和中间图的verification时间
        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        // 处理结果
        if (ged <= (int)tau_ui) {
            if (node_index == ept.root_index) {
                int anchor_id = static_cast<int>(ept.anchor_id);

                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                     node.completed_db_graph_ids.end(),
                                                     anchor_id) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id);
                    stats.EPT_results_from_astar++;
                }
            }

            size_t num_results = node.completed_db_graph_ids.size();
            stats.EPT_results_from_astar += num_results;

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                          node.completed_db_graph_ids.begin(),
                                          node.completed_db_graph_ids.end());
            found_here = true;
        }
        // 注意：无LB传递，无lookahead剪枝
    }

    // 递归处理子节点（纯DFS，不传递estimate_lb）
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }

        if (dfs_traverse_only_dfs(ch, ept, query_node,
                                  exact_results_within_tau, tau, stats,
                                  anchor_netdag_lb, anchor_netdag_ged)) {
            found_child = true;
        }
    }

    return found_here || found_child;
}

bool GismaSearchEngine::dfs_traverse(
    size_t                          node_index,
    const EditPathTree&             ept,
    std::shared_ptr<Node>           query_node,
    std::vector<int>&               exact_results_within_tau,
    double                          tau,
    SearchStats&                    stats,
    std::shared_ptr<SearchSnapshot> parent_snapshot,
    int                             parent_db_vertex_count,
    ui                              estimate_lb,
    double                          anchor_netdag_lb,
    int                             anchor_netdag_ged,
    int                             parent_no_snapshot_reason,
    size_t                          parent_node_index)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;
    ui new_estimate_lb = estimate_lb;

    
    std::shared_ptr<SearchSnapshot> current_snapshot = nullptr;
    enum NoSnapshotReason { SNAP_UNKNOWN=0, SNAP_LP_SKIPPED, SNAP_FILTER_SKIPPED, SNAP_REUSE_NO_CHAIN } no_snapshot_reason = SNAP_UNKNOWN;

    // 动态深度探测函数

    // NDC计数标志：确保每个节点只在实际做计算时计数一次
    bool ndc_counted = false;

    // 节点类型统计
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }

    if (!node.completed_db_graph_ids.empty()) {
        stats.EPT_nodes_with_completed_ids++;
    }

    if (node.op.type != EditOperation::NONE) {
        stats.EPT_op_type_count[node.op.type]++;
    }

    // 判断是否是db图（根节点或有completed_db_graph_ids的节点）
    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    bool should_compute_ged = true;

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    // LB Propagation: 如果estimate_lb > tau则跳过计算
    if (estimate_lb > tau_ui) {
        should_compute_ged = false;
        stats.lb_pruning_count++;
        no_snapshot_reason = SNAP_LP_SKIPPED;
        // 交叉统计：这个被 LP 跳过的 node，结构上本来是否 reuse-able？
        // (mode-independent: 非root + 父子vertex数相同 + accumulated_ops<=max_ged_gap)
        bool structurally_reuseable =
            (node_index != ept.root_index) &&
            (parent_db_vertex_count >= 0 && (int)db_g->n == parent_db_vertex_count) &&
            (node.accumulated_ops.size() <= (size_t)max_ged_gap);
        if (structurally_reuseable) stats.lp_skip_reuseable++;
        else                        stats.lp_skip_not_reuseable++;
    }
#endif

    if (use_ept_filters && should_compute_ged) {
        // NDC统计：只在实际做LB计算时计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }

        auto lb_start = std::chrono::high_resolution_clock::now();

        stats.EPT_lb_count++;
        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);
        if (lb > tau_ui) {
            should_compute_ged = false;
            no_snapshot_reason = SNAP_FILTER_SKIPPED;
            // 即使跳过GED计算，也用LB更新estimate_lb供子节点使用
            new_estimate_lb = std::max(new_estimate_lb, lb);
        }

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        // 分开统计db图和中间图的filter时间
        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }
    }

    // 如果启用了EPT filters或only_compute_db_graph，则只对有completed_db_graph_ids的节点计算GED
    // 根节点（anchor本身）即使completed_db_graph_ids为空也需要计算GED，因为anchor本身可能就是一个结果
    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== 根节点优化：利用NetDag缓存的GED，避免重复A*计算 ==========
    // GS_search阶段已经对anchor计算过exact GED，若无需为子节点生成snapshot则可跳过A*
    bool root_used_netdag_cache = false;
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF && should_compute_ged) {
        // 检查是否有近距离子节点需要snapshot
        bool needs_snapshot = false;
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            const TreeNode &c_node = ept.tree_nodes[ch];
            int distance = c_node.level - node.level;
            if (distance <= 0 || distance > max_ged_gap) continue;
            // 只有 net_n == 0 的 child 會用 reuse (新政策：父子 n 不同走 baseline)。
            int net_n = 0;
            for (const auto& op : c_node.accumulated_ops) {
                if (op.type == EditOperation::NODE_INSERTION) net_n++;
                else if (op.type == EditOperation::NODE_DELETION) net_n--;
            }
            if (net_n == 0) { needs_snapshot = true; break; }
        }
#endif

        if (!needs_snapshot) {
            // 无需snapshot，直接使用NetDag缓存的GED
            if (anchor_netdag_ged <= (int)tau_ui) {
                int anchor_id_local = static_cast<int>(ept.anchor_id);
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                      node.completed_db_graph_ids.end(),
                                                      anchor_id_local) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id_local);
                    stats.EPT_results_from_astar++;
                }
                exact_results_within_tau.insert(exact_results_within_tau.end(),
                                               node.completed_db_graph_ids.begin(),
                                               node.completed_db_graph_ids.end());
                stats.EPT_results_from_astar += node.completed_db_graph_ids.size();
                found_here = true;
            }
            new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));
            should_compute_ged = false;
            root_used_netdag_cache = true;
            stats.root_netdag_ged_reuse_count++;
        }
        // else: 需要snapshot供子节点复用，让后续A*正常运行
    }

    // GED计算
    if (should_compute_ged) {
        // NDC统计：如果跳过LB但做A*计算，也要计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        bool use_baseline = false;
        int optimal_margin = 0;
        bool has_close_child = false;  // 移到外层作用域，方便后面复用
        
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        // 使用level字段而不是accumulated_ops.size()
        int current_level = node.level;

        // 计算optimal_margin 并设置 has_close_child
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;

            const TreeNode &child_node = ept.tree_nodes[ch];
            int child_level = child_node.level;
            int distance = child_level - current_level;
            if (distance <= 0 || distance > max_ged_gap) continue;
            // 只有 net_n == 0 的 child 會用 reuse (新政策：父子 n 不同走 baseline)。
            int net_n = 0;
            for (const auto& op : child_node.accumulated_ops) {
                if (op.type == EditOperation::NODE_INSERTION) net_n++;
                else if (op.type == EditOperation::NODE_DELETION) net_n--;
            }
            if (net_n != 0) continue;
            has_close_child = true;
            optimal_margin = std::max(optimal_margin, distance);
        }
        
        // 限制margin不超过max_margin
        optimal_margin = std::min(optimal_margin, max_margin);

        // 如果没有近距离的子节点，使用baseline（但根节点除外）
        if (!has_close_child && !node.children_indices.empty() && node_index != ept.root_index) {
            use_baseline = true;
            optimal_margin = 0;  // baseline相当于margin=0
        }
#endif
        
        // 判断是否可以重用
        bool can_reuse = false;
        
        enum ReuseFailReason {
            NO_FAILURE = 0,
            NO_PARENT_SNAPSHOT,           // 父节点没用AStar（或Astar reuse）计算GED
            PARENT_SNAPSHOT_EMPTY,        // 父节点计算了但extract为空（v.size()==0）
            PARENT_SNAPSHOT_SIZE_ONE,     // 父节点计算了但只有1个节点（v.size()==1）
            ROOT_NODE,
            MULTI_OPS,
            VERTEX_COUNT_CHANGED,
            MO_INCOMPATIBLE,
            USE_BASELINE_DISTANT
        } fail_reason = NO_FAILURE;
        
        if (use_baseline) {
            fail_reason = USE_BASELINE_DISTANT;
        } else {
            if (node_index == ept.root_index) {
                stats.EPT_reuse_fail_root_node++;
                fail_reason = ROOT_NODE;
            } else if (parent_db_vertex_count >= 0 &&
                       (int)db_g->n != (int)parent_db_vertex_count) {
                // 父子 vertex 數不同 → 直接 fallback baseline。
                stats.EPT_reuse_fail_vertex_count_changed++;
                fail_reason = VERTEX_COUNT_CHANGED;
            } else if (node.accumulated_ops.size() > max_ged_gap) {
                stats.EPT_reuse_fail_multi_ops++;
                fail_reason = MULTI_OPS;
            } else if (parent_snapshot == nullptr || parent_db_vertex_count < 0) {
                stats.EPT_reuse_fail_no_parent_snapshot++;
                switch (parent_no_snapshot_reason) {
                    case SNAP_LP_SKIPPED:     stats.EPT_reuse_fail_no_parent_lp_skipped++; break;
                    case SNAP_FILTER_SKIPPED: stats.EPT_reuse_fail_no_parent_filter_skipped++; break;
                    case SNAP_REUSE_NO_CHAIN: stats.EPT_reuse_fail_no_parent_reuse_no_chain++; break;
                    default:                  stats.EPT_reuse_fail_no_parent_other++; break;
                }
                fail_reason = NO_PARENT_SNAPSHOT;
            } else {
                // 检查mo数组兼容性
                size_t parent_mo_size = parent_snapshot->mo.size();
                size_t current_min_size = std::min(static_cast<size_t>(qry_g->n),
                                                  static_cast<size_t>(db_g->n));
                size_t current_max_size = std::max(static_cast<size_t>(qry_g->n),
                                                  static_cast<size_t>(db_g->n));

                bool mo_size_compatible = (parent_mo_size <= current_min_size);
                bool mo_content_valid = true;

                if (mo_size_compatible && !parent_snapshot->mo.empty()) {
                    for (size_t i = 0; i < parent_snapshot->mo.size(); ++i) {
                        if (parent_snapshot->mo[i] != static_cast<size_t>(-1) &&
                            parent_snapshot->mo[i] >= current_max_size) {
                            mo_content_valid = false;
                            break;
                        }
                    }
                }

                if (!mo_size_compatible || !mo_content_valid) {
                    fail_reason = MO_INCOMPATIBLE;
                } else if (parent_snapshot->v.empty()) {
                    fail_reason = PARENT_SNAPSHOT_EMPTY;
                } else if (parent_snapshot->v.size() >= 1) {
                    // size=1 (dummy or real) and size>=2 all go to reuse
                    can_reuse = true;
                    stats.EPT_reuse_attempt++;
                    fail_reason = NO_FAILURE;
                }
            }
        }
        
        // E8 gisma-lsa: 默认 reuse 路径，但把验证器下界从 BMao 换成 LSa。
        //   其余（reuse / snapshot / App_test / App）逻辑完全不变。
        //   ged_algorithm=="App"（真正的默认）仍然用 BMao，主结果字节级不变。
        const char* default_path_lb = (ged_algorithm == "gisma-lsa" || ged_algorithm == "app-lsa") ? "LSa" : "BMao";
        if (ged_algorithm == "app-lsa") can_reuse = false;  // E8 app-lsa: App()+LSa, no reuse
        Application app(tau_ui, default_path_lb, app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_g, qry_g);
        app.set_disable_reuse_lsa(disable_reuse_lsa);
        app.set_exact_value_mode(exact_value_mode);
        app.set_early_stop_at_tau(early_stop_at_tau);
        app.set_skip_intersection_in_reuse(early_stop_at_tau);

        int ged = -1;
        bool used_reuse = false;
        double elapsed_time = 0.0;

        // E8: 非默认 verifier（"AStar"=精确A*=Gisma-AStar）→ 绕过 reuse + App-BMao，直接精确验证。
        //     默认 ged_algorithm=="App" 时 alt_verifier=false，原 reuse/App 逻辑完全不变。
        //     gisma-lsa 走默认 reuse 路径（只换 LB），因此 NOT alt_verifier。
        bool alt_verifier = (ged_algorithm == "AStar" || ged_algorithm == "LSa");
        if (alt_verifier) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // E8 备用精确验证器（Gisma-AStar / Gisma-LSa）。
            // 必须用 AppForComputation（精确）：它会正确设置 overall_lb，供下方
            // LB Propagation（new_estimate_lb = app.get_overall_lb()）与 Subtree Pruning 使用。
            // 旧实现调用 AStar_baseline，但 AStar_baseline 从不写 overall_lb，导致
            // get_overall_lb() 返回初始值 INF → 子节点 estimate_lb≈INF > tau → should_compute_ged=false，
            // 子树里的所有 db 图被静默跳过（不计算 GED）。这同时解释了 recall 暴跌且更快。
            // AppForComputation 跑到最优（exact_max_iter=1e6），upper_bound=tau+1 时
            // GED<=tau 返回精确值、GED>tau 返回 tau+1，作为 verifier 判定完全正确。
            //
            // ged_algorithm=="AStar" → BMao 下界的精确 A*（用上面的 app，lb=BMao）。
            // ged_algorithm=="LSa"   → LSa 下界的精确 A*（"Gisma-LSa"）：另建一个 lb=LSa 的
            //                          Application，同样跑 AppForComputation 取精确 GED。
            if (ged_algorithm == "LSa") {
                Application app_lsa(tau_ui, "LSa", app_max_iter);
                app_lsa.set_all_edge_labels_same(all_edge_labels_same);
                app_lsa.init(db_g, qry_g);
                ged = (int)app_lsa.AppForComputation(nullptr, nullptr);
                // overall_lb 来自 LSa A*，传给下方 LB Propagation（仍是合法下界）。
                app.set_overall_lb_for_propagation(app_lsa.get_overall_lb());
            } else if (ged_algorithm == "app-lsa") {
                // E8 app-lsa: 近似 App（受 app_max_iter 限制）+ LSa 下界，无 reuse。
                //   与 LSa 分支同样另建 lb=LSa 的 Application，但用受迭代上限约束的近似搜索
                //   而非 AppForComputation()（精确）。这里用 App_test()（_test A* 机制：
                //   generate_best_extension_test + construct_sibling_test + compute_best_extension_LSa_baseline），
                //   它与 AppForComputation 共享同一套 LSa 实现，因此对 lb_method==LSa 稳定可靠；
                //   而 App()（非 _test）走 compute_best_extension_LSa，那条 LSa 路径未实现/不稳定。
                //   App_test() 同样受 app_max_iter 约束并在结束时写好 overall_lb（活跃/终止前沿）。
                Application app_applsa(tau_ui, "LSa", app_max_iter);
                app_applsa.set_all_edge_labels_same(all_edge_labels_same);
                app_applsa.init(db_g, qry_g);
                ged = (int)app_applsa.App_test(nullptr, nullptr);  // 近似 App（capped, _test LSa），无 reuse
                app.set_overall_lb_for_propagation(app_applsa.get_overall_lb());
            } else {
                ged = (int)app.AppForComputation(nullptr, nullptr);  // 精确 A* 验证器（BMao, exact）
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            stats.EPT_astar_count++;
        }

        if (!alt_verifier && can_reuse) {
            // chain_reuse=false时，使用reuse的节点不会保存snapshot，因此不需要margin
            if (chain_reuse) {
                app.set_margin(optimal_margin);  // 链式复用时需要为后续保存snapshot
            } else {
                app.set_margin(0);  // 不需要为后续保存snapshot，使用margin=0减少计算
            }
            // 正常reuse路径（dummy-only已在前面处理）
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.app_reuse(*parent_snapshot, tau_ui, node.accumulated_ops.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            used_reuse = (ged >= 0);
            if (used_reuse) {
                stats.EPT_astar_count++;
                if (!chain_reuse) {
                    no_snapshot_reason = SNAP_REUSE_NO_CHAIN;
                }
            }
        }

        if (!alt_verifier && !used_reuse) {
            // AStar需要根据children距离设置margin（可能需要保存snapshot）
            app.set_margin(optimal_margin);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            stats.EPT_astar_count++;
            auto t0 = std::chrono::high_resolution_clock::now();
            // E8 gisma-lsa (Phase 2): lb=LSa 现在与默认 BMao 路径走同一套 margin 逻辑。
            //   App()+LSa 的越界崩溃已修复：compute_best_extension_LSa_baseline 现在设置
            //   siblings_total_n = siblings_n（见 Application.cpp step-14），使 App() 的
            //   construct_sibling intersection 索引正确。App() 因此正常产出 boundary_nodes
            //   snapshot，gisma-lsa 获得真正的 search-tree reuse（上方 app_reuse），只有 LB 不同。
            //   optimal_margin==0 时用 App_test 省去 snapshot 开销（无 close child，不会 reuse）。
            //   ged_algorithm=="App"（默认 BMao）行为字节级不变。
            ged = (ged_algorithm == "app-lsa") ? (int)app.App()
                : ((optimal_margin == 0) ? (int)app.App_test() : (int)app.App());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();

            // Verify margin overhead: compare App(margin) vs App_baseline(no margin)
            if (verify_reuse && optimal_margin > 0) {
                Application app_no_margin(tau_ui, "BMao", app_max_iter);
                app_no_margin.init(db_g, qry_g);
                auto t_nm_start = std::chrono::high_resolution_clock::now();
                int ged_no_margin = app_no_margin.App_baseline(nullptr, nullptr);
                auto t_nm_end = std::chrono::high_resolution_clock::now();
                double no_margin_time = std::chrono::duration<double>(t_nm_end - t_nm_start).count();
                stats.margin_overhead_count++;
                stats.margin_overhead_with_margin_time += elapsed_time;
                stats.margin_overhead_without_margin_time += no_margin_time;
                bool margin_correct = ((ged > (int)tau_ui && ged_no_margin > (int)tau_ui) ||
                                      (ged <= (int)tau_ui && ged_no_margin <= (int)tau_ui));
                if (margin_correct) stats.margin_overhead_correct++;
                else stats.margin_overhead_incorrect++;
            }
        }

        if (used_reuse) {
            stats.EPT_reuse_count++;
            if (node.op.type != EditOperation::NONE) {
                stats.EPT_reuse_by_op_type[node.op.type]++;
            }
            
            // 验证reuse效果（通过命令行参数控制）
            if (verify_reuse) {
                // 使用 App_baseline 作为 baseline（verification版本，与App-BMao一致）
                Application app_verify(tau_ui, "BMao", app_max_iter);
                app_verify.init(db_g, qry_g);

                auto t_verify_start = std::chrono::high_resolution_clock::now();
                int ged_standard = app_verify.App_baseline(nullptr, nullptr);
                auto t_verify_end = std::chrono::high_resolution_clock::now();

                double verify_time = std::chrono::duration<double>(t_verify_end - t_verify_start).count();
                stats.EPT_verification_astar_time += verify_time;
                stats.EPT_reuse_verifications++;

                // App_baseline 统计
                stats.EPT_baseline_app_count++;
                stats.EPT_baseline_app_time += verify_time;
                stats.EPT_baseline_reuse_time += elapsed_time;  // 记录对应的 reuse 时间

                // 收集样本对 (最多 MAX_BASELINE_SAMPLES 个)
                if (stats.baseline_samples.size() < SearchStats::MAX_BASELINE_SAMPLES) {
                    double reuse_ms = elapsed_time * 1000.0;
                    double baseline_ms = verify_time * 1000.0;
                    stats.baseline_samples.push_back({reuse_ms, baseline_ms});
                }

                if (ged <= (int)tau_ui) {
                    stats.EPT_reuse_found_ged_le_tau++;
                }
                if (ged_standard <= (int)tau_ui) {
                    stats.EPT_astar_found_ged_le_tau++;
                }

                bool reuse_correct = ((ged > (int)tau_ui && ged_standard > (int)tau_ui) ||
                                     (ged <= (int)tau_ui && ged_standard <= (int)tau_ui));

                if (reuse_correct) {
                    stats.EPT_reuse_correct++;
                } else {
                    stats.EPT_reuse_incorrect++;
                }

                // Speedup distribution
                double sp = (elapsed_time > 0) ? verify_time / elapsed_time : 0.0;
                int sp_bucket;
                if (sp > 3.0) { stats.reuse_speedup_gt3x++; sp_bucket = 0; }
                else if (sp > 2.0) { stats.reuse_speedup_2x_3x++; sp_bucket = 1; }
                else if (sp > 1.0) { stats.reuse_speedup_1x_2x++; sp_bucket = 2; }
                else if (sp > 0.5) { stats.reuse_speedup_05x_1x++; sp_bucket = 3; }
                else { stats.reuse_speedup_lt05x++; sp_bucket = 4; }

                // Snapshot size
                int size_bucket = -1;
                if (parent_snapshot) {
                    size_t sz = parent_snapshot->v.size();
                    if (sz <= 1) { stats.reuse_snapshot_size_1++; size_bucket = 0; }
                    else if (sz <= 10) { stats.reuse_snapshot_size_2_10++; size_bucket = 1; }
                    else { stats.reuse_snapshot_size_gt10++; size_bucket = 2; }
                }

                // Cross-tab [size][speedup]
                if (size_bucket >= 0) {
                    stats.reuse_xtab[size_bucket][sp_bucket]++;
                    stats.reuse_xtime[size_bucket] += elapsed_time;
                    stats.baseline_xtime[size_bucket] += verify_time;
                }

                // Chain depth statistics: depth = parent_snapshot->chain_depth
                int chain_d = parent_snapshot ? parent_snapshot->chain_depth : 0;
                if (chain_d < 0) chain_d = 0;
                if (chain_d >= SearchStats::MAX_CHAIN_DEPTH) chain_d = SearchStats::MAX_CHAIN_DEPTH - 1;
                stats.chain_depth_count[chain_d]++;
                stats.chain_depth_reuse_time[chain_d]    += elapsed_time;
                stats.chain_depth_baseline_time[chain_d] += verify_time;
                if (reuse_correct) stats.chain_depth_correct[chain_d]++;
                if (ged_standard <= (int)tau_ui) {
                    stats.chain_depth_pos[chain_d]++;
                    if (ged <= (int)tau_ui) stats.chain_depth_tp[chain_d]++;
                }

                // Per-reuse log
                if (g_reuse_log) {
                    std::lock_guard<std::mutex> _lk(g_reuse_log_mutex);
                    fprintf(g_reuse_log,
                            "%d,%u,%zu,%zu,%zu,%d,%.3f,%.3f,%d,%d,%d\n",
                            query_node->node_id,
                            ept.anchor_id,
                            parent_node_index,
                            node_index,
                            parent_snapshot ? parent_snapshot->v.size() : (size_t)0,
                            (int)node.accumulated_ops.size(),
                            elapsed_time * 1e6,
                            verify_time * 1e6,
                            ged,
                            ged_standard,
                            (int)tau_ui);
                }
            }
        }

        stats.EPT_astar_time += elapsed_time;

        // 分开统计db图和中间图的verification时间
        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        if (used_reuse) {
            stats.EPT_reuse_success_time += elapsed_time;
        } else {
            switch (fail_reason) {
                case NO_PARENT_SNAPSHOT:
                    stats.EPT_astar_time_no_parent_snapshot += elapsed_time;
                    break;
                case ROOT_NODE:
                    stats.EPT_astar_time_root_node += elapsed_time;
                    break;
                case MULTI_OPS:
                    stats.EPT_astar_time_multi_ops += elapsed_time;
                    break;
                case PARENT_SNAPSHOT_EMPTY:
                    stats.EPT_astar_time_parent_snapshot_empty += elapsed_time;
                    break;
                case PARENT_SNAPSHOT_SIZE_ONE:
                    stats.EPT_astar_time_parent_snapshot_size_one += elapsed_time;
                    break;
                case VERTEX_COUNT_CHANGED:
                    stats.EPT_astar_time_vertex_count_changed += elapsed_time;
                    break;
                case MO_INCOMPATIBLE:
                    stats.EPT_astar_time_mo_incompatible += elapsed_time;
                    break;
                case USE_BASELINE_DISTANT:
                    break;
                default:
                    break;
            }
        }
        
#ifdef USE_ESTIMATE_LB_OPTIMIZATION
        // LB Propagation: 更新estimate_lb
        new_estimate_lb = app.get_overall_lb();
        // 根节点：结合NetDag的lb，取更紧的值用于子节点传播
        if (node_index == ept.root_index && anchor_netdag_lb >= 0) {
            new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));
        }
#endif

#if USE_SUBTREE_PRUNING
        ui overall_lb = app.get_overall_lb();
        // 根节点：结合NetDag的lb，取更紧的值用于Subtree Pruning
        if (node_index == ept.root_index && anchor_netdag_lb >= 0) {
            overall_lb = std::max(overall_lb, static_cast<ui>(anchor_netdag_lb));
        }

        // Subtree Pruning: 叶子节点跳过（没有子节点可以剪枝）
        if (!node.children_indices.empty() && overall_lb > 0 && overall_lb < 10000) {
            int max_step_more;

            max_step_more = node.max_subtree_depth - node.level;

            if (max_step_more + (int)tau_ui < (int)overall_lb) {
                // 统计subtree pruning效果（unified版本）
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // 减去当前节点本身
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                return found_here;
            }
        }
#endif

        if (ged <= (int)tau_ui) {
            // 对于根节点，确保anchor本身被添加到结果中（旧EPT文件可能不包含anchor_id）
            if (node_index == ept.root_index) {
                int anchor_id_local = static_cast<int>(ept.anchor_id);
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                      node.completed_db_graph_ids.end(),
                                                      anchor_id_local) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id_local);
                    if (used_reuse) {
                        stats.EPT_results_from_reuse += 1;
                    } else {
                        stats.EPT_results_from_astar += 1;
                    }
                }
            }

            // 添加completed_db_graph_ids中的所有图
            size_t num_results = node.completed_db_graph_ids.size();
            if (used_reuse) {
                stats.EPT_results_from_reuse += num_results;
            } else {
                stats.EPT_results_from_astar += num_results;
            }

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                          node.completed_db_graph_ids.begin(),
                                          node.completed_db_graph_ids.end());
            if (e7_stats) { stats.e7_answer_depth_sum += (long long)node.level * (long long)num_results; stats.e7_answer_count += (long long)num_results; }
            found_here = true;
        } else {
            // GED > tau, rejected
        }
        
        // ========== 提取 snapshot（供子节点复用）==========
        // 只有当有近距离children时才生成snapshot
        // chain_reuse=true时，使用reuse的节点也保存snapshot供后续节点复用
        // E8 alt_verifier（AStar/LSa/app-lsa）：ged 来自独立的 app_lsa/app_applsa 对象，
        //   默认 app 从未运行过 A*，其内部 open_heap/boundary_nodes 为空，
        //   extract_snapshot(app) 会读到未初始化状态而崩溃。这些验证器不复用 snapshot，直接跳过。
        bool should_save_snapshot = (!alt_verifier && ged >= 0 && ged < INT_MAX && has_close_child);
        if (!chain_reuse) {
            should_save_snapshot = should_save_snapshot && !used_reuse;
        }
        if (should_save_snapshot) {
            current_snapshot = std::make_shared<SearchSnapshot>();
            
            try {
                app.extract_snapshot(*current_snapshot);
                current_snapshot->ub = ged;
                current_snapshot->margin = optimal_margin;  // 保存margin信息
                // chain_depth: 0 if this node used fresh A*, parent_snapshot->chain_depth + 1 if it used reuse
                current_snapshot->chain_depth = used_reuse
                    ? ((parent_snapshot ? parent_snapshot->chain_depth : 0) + 1)
                    : 0;

                if (current_snapshot->v.empty()) {
                    current_snapshot = nullptr;
                }
            } catch (...) {
                current_snapshot = nullptr;
            }
        }
    }
#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    else {
        // 跳过了计算（LB Propagation等），但仍要检查是否可以剪枝
#if USE_SUBTREE_PRUNING
        // Subtree Pruning: 叶子节点跳过（没有子节点可以剪枝）
        if (!node.children_indices.empty() && new_estimate_lb > 0 && new_estimate_lb < 10000) {
            int max_step_more;

            max_step_more = node.max_subtree_depth - node.level;

            if (max_step_more + (int)tau_ui < (int)new_estimate_lb) {
                // 统计subtree pruning效果（unified版本，estimate_lb分支）
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // 减去当前节点本身
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                return found_here;
            }
        }
#endif
    }
#endif
    
    // 递归处理子节点
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }
        
        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        ui child_estimate_lb = (level_diff >= (int)new_estimate_lb) ?
                              0 : new_estimate_lb - level_diff;

        // 统计传递estimate_lb到子节点的次数
        stats.lb_propagation_count++;

        if (dfs_traverse(ch, ept, query_node,
                                exact_results_within_tau, tau, stats,
                                current_snapshot,
                                db_g->n,
                                child_estimate_lb,
                                anchor_netdag_lb,
                                anchor_netdag_ged,
                                (current_snapshot == nullptr) ? no_snapshot_reason : 0,
                                node_index)) {
            found_child = true;
        }
    }

    return found_here || found_child;
}

// ========== dfs_traverse_no_SP: 与 unified 相同但禁用 Subtree Pruning ==========
// 保留 Reuse + Distance Propagation，但不使用 Subtree Pruning
bool GismaSearchEngine::dfs_traverse_no_SP(
    size_t                          node_index,
    const EditPathTree&             ept,
    std::shared_ptr<Node>           query_node,
    std::vector<int>&               exact_results_within_tau,
    double                          tau,
    SearchStats&                    stats,
    std::shared_ptr<SearchSnapshot> parent_snapshot,
    int                             parent_db_vertex_count,
    ui                              estimate_lb,
    double                          anchor_netdag_lb,
    int                             anchor_netdag_ged,
    int                             parent_no_snapshot_reason,
    size_t                          parent_node_index)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;
    ui new_estimate_lb = estimate_lb;

    std::shared_ptr<SearchSnapshot> current_snapshot = nullptr;
    enum NoSnapshotReason { SNAP_UNKNOWN=0, SNAP_LP_SKIPPED, SNAP_FILTER_SKIPPED, SNAP_REUSE_NO_CHAIN } no_snapshot_reason = SNAP_UNKNOWN;

    // NDC计数标志：确保每个节点只在实际做计算时计数一次
    bool ndc_counted = false;

    // 节点类型统计
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }

    if (!node.completed_db_graph_ids.empty()) {
        stats.EPT_nodes_with_completed_ids++;
    }

    if (node.op.type != EditOperation::NONE) {
        stats.EPT_op_type_count[node.op.type]++;
    }

    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    bool should_compute_ged = true;

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    if (estimate_lb > tau_ui) {
        should_compute_ged = false;
        stats.lb_pruning_count++;
        no_snapshot_reason = SNAP_LP_SKIPPED;
    }
#endif

    if (use_ept_filters && should_compute_ged) {
        // NDC统计：只在实际做LB计算时计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }

        auto lb_start = std::chrono::high_resolution_clock::now();

        stats.EPT_lb_count++;
        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);
        if (lb > tau_ui) {
            should_compute_ged = false;
            no_snapshot_reason = SNAP_FILTER_SKIPPED;
            new_estimate_lb = std::max(new_estimate_lb, lb);
        }

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }
    }

    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== 根节点优化：利用NetDag缓存的GED，避免重复A*计算 ==========
    bool root_used_netdag_cache = false;
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF && should_compute_ged) {
        bool needs_snapshot = false;
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            const TreeNode &c_node = ept.tree_nodes[ch];
            int distance = c_node.level - node.level;
            if (distance <= 0 || distance > max_ged_gap) continue;
            // 只有 net_n == 0 的 child 會用 reuse (新政策：父子 n 不同走 baseline)。
            int net_n = 0;
            for (const auto& op : c_node.accumulated_ops) {
                if (op.type == EditOperation::NODE_INSERTION) net_n++;
                else if (op.type == EditOperation::NODE_DELETION) net_n--;
            }
            if (net_n == 0) { needs_snapshot = true; break; }
        }
#endif
        if (!needs_snapshot) {
            if (anchor_netdag_ged <= (int)tau_ui) {
                int anchor_id_local = static_cast<int>(ept.anchor_id);
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                      node.completed_db_graph_ids.end(),
                                                      anchor_id_local) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id_local);
                    stats.EPT_results_from_astar++;
                }
                exact_results_within_tau.insert(exact_results_within_tau.end(),
                                               node.completed_db_graph_ids.begin(),
                                               node.completed_db_graph_ids.end());
                stats.EPT_results_from_astar += node.completed_db_graph_ids.size();
                found_here = true;
            }
            new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));
            should_compute_ged = false;
            root_used_netdag_cache = true;
            stats.root_netdag_ged_reuse_count++;
        }
    }

    // GED计算 (与unified相同，但无Subtree Pruning)
    if (should_compute_ged) {
        // NDC统计：如果跳过LB但做A*计算，也要计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        bool use_baseline = false;
        int optimal_margin = 0;
        bool has_close_child = false;

#if USE_BASELINE_FOR_DISTANT_CHILDREN
        int current_level = node.level;
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            const TreeNode &child_node = ept.tree_nodes[ch];
            int child_level = child_node.level;
            int distance = child_level - current_level;
            if (distance <= 0 || distance > max_ged_gap) continue;
            // 只有 net_n == 0 的 child 會用 reuse (新政策：父子 n 不同走 baseline)。
            int net_n = 0;
            for (const auto& op : child_node.accumulated_ops) {
                if (op.type == EditOperation::NODE_INSERTION) net_n++;
                else if (op.type == EditOperation::NODE_DELETION) net_n--;
            }
            if (net_n != 0) continue;
            has_close_child = true;
            optimal_margin = std::max(optimal_margin, distance);
        }
        optimal_margin = std::min(optimal_margin, max_margin);
        if (!has_close_child && !node.children_indices.empty() && node_index != ept.root_index) {
            use_baseline = true;
            optimal_margin = 0;
        }
#endif

        bool can_reuse = false;

        if (!use_baseline) {
            if (node_index == ept.root_index) {
                stats.EPT_reuse_fail_root_node++;
            } else if (parent_db_vertex_count >= 0 &&
                       (int)db_g->n != (int)parent_db_vertex_count) {
                // 父子 vertex 數不同 → 直接 fallback baseline。
                stats.EPT_reuse_fail_vertex_count_changed++;
            } else if (node.accumulated_ops.size() > max_ged_gap) {
                stats.EPT_reuse_fail_multi_ops++;
            } else if (parent_snapshot == nullptr || parent_db_vertex_count < 0) {
                stats.EPT_reuse_fail_no_parent_snapshot++;
                switch (parent_no_snapshot_reason) {
                    case SNAP_LP_SKIPPED:     stats.EPT_reuse_fail_no_parent_lp_skipped++; break;
                    case SNAP_FILTER_SKIPPED: stats.EPT_reuse_fail_no_parent_filter_skipped++; break;
                    case SNAP_REUSE_NO_CHAIN: stats.EPT_reuse_fail_no_parent_reuse_no_chain++; break;
                    default:                  stats.EPT_reuse_fail_no_parent_other++; break;
                }
            } else {
                size_t parent_mo_size = parent_snapshot->mo.size();
                size_t current_min_size = std::min(static_cast<size_t>(qry_g->n), static_cast<size_t>(db_g->n));
                size_t current_max_size = std::max(static_cast<size_t>(qry_g->n), static_cast<size_t>(db_g->n));
                bool mo_size_compatible = (parent_mo_size <= current_min_size);
                bool mo_content_valid = true;
                if (mo_size_compatible && !parent_snapshot->mo.empty()) {
                    for (size_t i = 0; i < parent_snapshot->mo.size(); ++i) {
                        if (parent_snapshot->mo[i] != static_cast<size_t>(-1) &&
                            parent_snapshot->mo[i] >= current_max_size) {
                            mo_content_valid = false;
                            break;
                        }
                    }
                }
                if (!mo_size_compatible || !mo_content_valid) {
                } else if (parent_snapshot->v.empty()) {
                } else if (parent_snapshot->v.size() >= 1) {
                    can_reuse = true;
                    stats.EPT_reuse_attempt++;
                }
            }
        }

        Application app(tau_ui, "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_g, qry_g);
        app.set_disable_reuse_lsa(disable_reuse_lsa);
        app.set_exact_value_mode(exact_value_mode);
        app.set_early_stop_at_tau(early_stop_at_tau);
        app.set_skip_intersection_in_reuse(early_stop_at_tau);

        int ged = -1;
        bool used_reuse = false;
        double elapsed_time = 0.0;

        if (can_reuse) {
            // chain_reuse=false时，使用reuse的节点不会保存snapshot，因此不需要margin
            if (chain_reuse) {
                app.set_margin(optimal_margin);  // 链式复用时需要为后续保存snapshot
            } else {
                app.set_margin(0);  // 不需要为后续保存snapshot，使用margin=0减少计算
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.app_reuse(*parent_snapshot, tau_ui, node.accumulated_ops.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            used_reuse = (ged >= 0);
            if (used_reuse) {
                stats.EPT_astar_count++;
                if (!chain_reuse) {
                    no_snapshot_reason = SNAP_REUSE_NO_CHAIN;
                }
            }
        }

        if (!used_reuse) {
            app.set_margin(optimal_margin);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            stats.EPT_astar_count++;
            auto t0 = std::chrono::high_resolution_clock::now();
            // optimal_margin==0: 不会产出有用 snapshot, 用 App_test 省掉 boundary_nodes + protect_ancestor_chain 开销
            ged = (optimal_margin == 0) ? app.App_test() : app.App();
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();

            if (verify_reuse && optimal_margin > 0) {
                Application app_no_margin(tau_ui, "BMao", app_max_iter);
                app_no_margin.init(db_g, qry_g);
                auto t_nm_start = std::chrono::high_resolution_clock::now();
                int ged_no_margin = app_no_margin.App_baseline(nullptr, nullptr);
                auto t_nm_end = std::chrono::high_resolution_clock::now();
                double no_margin_time = std::chrono::duration<double>(t_nm_end - t_nm_start).count();
                stats.margin_overhead_count++;
                stats.margin_overhead_with_margin_time += elapsed_time;
                stats.margin_overhead_without_margin_time += no_margin_time;
                bool margin_correct = ((ged > (int)tau_ui && ged_no_margin > (int)tau_ui) ||
                                      (ged <= (int)tau_ui && ged_no_margin <= (int)tau_ui));
                if (margin_correct) stats.margin_overhead_correct++;
                else stats.margin_overhead_incorrect++;
            }
        }

        if (used_reuse) {
            stats.EPT_reuse_count++;
            if (node.op.type != EditOperation::NONE) {
                stats.EPT_reuse_by_op_type[node.op.type]++;
            }
            if (verify_reuse) {
                Application app_verify(tau_ui, "BMao", app_max_iter);
                app_verify.init(db_g, qry_g);
                auto t_verify_start = std::chrono::high_resolution_clock::now();
                int ged_standard = app_verify.App_baseline(nullptr, nullptr);
                auto t_verify_end = std::chrono::high_resolution_clock::now();
                double verify_time = std::chrono::duration<double>(t_verify_end - t_verify_start).count();
                stats.EPT_verification_astar_time += verify_time;
                stats.EPT_reuse_verifications++;
                stats.EPT_baseline_app_count++;
                stats.EPT_baseline_app_time += verify_time;
                stats.EPT_baseline_reuse_time += elapsed_time;
                if (stats.baseline_samples.size() < SearchStats::MAX_BASELINE_SAMPLES) {
                    stats.baseline_samples.push_back({elapsed_time * 1000.0, verify_time * 1000.0});
                }
                bool reuse_correct = ((ged > (int)tau_ui && ged_standard > (int)tau_ui) ||
                                     (ged <= (int)tau_ui && ged_standard <= (int)tau_ui));
                if (reuse_correct) stats.EPT_reuse_correct++;
                else stats.EPT_reuse_incorrect++;

                // Speedup distribution
                double sp = (elapsed_time > 0) ? verify_time / elapsed_time : 0.0;
                int sp_bucket;
                if (sp > 3.0) { stats.reuse_speedup_gt3x++; sp_bucket = 0; }
                else if (sp > 2.0) { stats.reuse_speedup_2x_3x++; sp_bucket = 1; }
                else if (sp > 1.0) { stats.reuse_speedup_1x_2x++; sp_bucket = 2; }
                else if (sp > 0.5) { stats.reuse_speedup_05x_1x++; sp_bucket = 3; }
                else { stats.reuse_speedup_lt05x++; sp_bucket = 4; }

                // Snapshot size
                int size_bucket = -1;
                if (parent_snapshot) {
                    size_t sz = parent_snapshot->v.size();
                    if (sz <= 1) { stats.reuse_snapshot_size_1++; size_bucket = 0; }
                    else if (sz <= 10) { stats.reuse_snapshot_size_2_10++; size_bucket = 1; }
                    else { stats.reuse_snapshot_size_gt10++; size_bucket = 2; }
                }

                // Cross-tab [size][speedup]
                if (size_bucket >= 0) {
                    stats.reuse_xtab[size_bucket][sp_bucket]++;
                    stats.reuse_xtime[size_bucket] += elapsed_time;
                    stats.baseline_xtime[size_bucket] += verify_time;
                }

                // Chain depth statistics: depth = parent_snapshot->chain_depth
                int chain_d = parent_snapshot ? parent_snapshot->chain_depth : 0;
                if (chain_d < 0) chain_d = 0;
                if (chain_d >= SearchStats::MAX_CHAIN_DEPTH) chain_d = SearchStats::MAX_CHAIN_DEPTH - 1;
                stats.chain_depth_count[chain_d]++;
                stats.chain_depth_reuse_time[chain_d]    += elapsed_time;
                stats.chain_depth_baseline_time[chain_d] += verify_time;
                if (reuse_correct) stats.chain_depth_correct[chain_d]++;
                if (ged_standard <= (int)tau_ui) {
                    stats.chain_depth_pos[chain_d]++;
                    if (ged <= (int)tau_ui) stats.chain_depth_tp[chain_d]++;
                }

                // Per-reuse log
                if (g_reuse_log) {
                    std::lock_guard<std::mutex> _lk(g_reuse_log_mutex);
                    fprintf(g_reuse_log,
                            "%d,%u,%zu,%zu,%zu,%d,%.3f,%.3f,%d,%d,%d\n",
                            query_node->node_id,
                            ept.anchor_id,
                            parent_node_index,
                            node_index,
                            parent_snapshot ? parent_snapshot->v.size() : (size_t)0,
                            (int)node.accumulated_ops.size(),
                            elapsed_time * 1e6,
                            verify_time * 1e6,
                            ged,
                            ged_standard,
                            (int)tau_ui);
                }
            }
        }

        stats.EPT_astar_time += elapsed_time;
        if (used_reuse) {
            stats.EPT_reuse_success_time += elapsed_time;
        }

        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
        // LB Propagation: 更新estimate_lb
        new_estimate_lb = app.get_overall_lb();
#endif

        // 注意：这里故意跳过 Subtree Pruning 检查

        if (ged <= (int)tau_ui) {
            if (node_index == ept.root_index) {
                int anchor_id_local = static_cast<int>(ept.anchor_id);
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                      node.completed_db_graph_ids.end(),
                                                      anchor_id_local) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id_local);
                    if (used_reuse) {
                        stats.EPT_results_from_reuse += 1;
                    } else {
                        stats.EPT_results_from_astar += 1;
                    }
                }
            }

            size_t num_results = node.completed_db_graph_ids.size();
            if (used_reuse) {
                stats.EPT_results_from_reuse += num_results;
            } else {
                stats.EPT_results_from_astar += num_results;
            }

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                          node.completed_db_graph_ids.begin(),
                                          node.completed_db_graph_ids.end());
            found_here = true;
        }

        // 提取 snapshot（供子节点复用）
        // chain_reuse=true时，使用reuse的节点也保存snapshot供后续节点复用
        bool should_save_snapshot = (ged >= 0 && ged < INT_MAX && has_close_child);
        if (!chain_reuse) {
            should_save_snapshot = should_save_snapshot && !used_reuse;
        }
        if (should_save_snapshot) {
            current_snapshot = std::make_shared<SearchSnapshot>();
            try {
                app.extract_snapshot(*current_snapshot);
                current_snapshot->ub = ged;
                current_snapshot->margin = optimal_margin;
                current_snapshot->chain_depth = used_reuse
                    ? ((parent_snapshot ? parent_snapshot->chain_depth : 0) + 1)
                    : 0;
                if (current_snapshot->v.empty()) {
                    current_snapshot = nullptr;
                }
            } catch (...) {
                current_snapshot = nullptr;
            }
        }
    }
#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    // 注意：这里也跳过 Subtree Pruning 检查（else分支）
#endif

    // 递归处理子节点
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }

        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        ui child_estimate_lb = (level_diff >= (int)new_estimate_lb) ?
                              0 : new_estimate_lb - level_diff;

        stats.lb_propagation_count++;

        if (dfs_traverse_no_SP(ch, ept, query_node,
                              exact_results_within_tau, tau, stats,
                              current_snapshot, db_g->n, child_estimate_lb,
                              anchor_netdag_lb, anchor_netdag_ged,
                              (current_snapshot == nullptr) ? no_snapshot_reason : 0,
                              node_index)) {
            found_child = true;
        }
    }

    return found_here || found_child;
}

// ========== dfs_traverse_no_LP: 与 unified 相同但禁用 Distance Propagation ==========
// 保留 Reuse + Subtree Pruning，但不使用 Distance Propagation (LB Propagation)
bool GismaSearchEngine::dfs_traverse_no_LP(
    size_t                          node_index,
    const EditPathTree&             ept,
    std::shared_ptr<Node>           query_node,
    std::vector<int>&               exact_results_within_tau,
    double                          tau,
    SearchStats&                    stats,
    std::shared_ptr<SearchSnapshot> parent_snapshot,
    int                             parent_db_vertex_count,
    ui                              estimate_lb,
    double                          anchor_netdag_lb,
    int                             anchor_netdag_ged,
    int                             parent_no_snapshot_reason,
    size_t                          parent_node_index)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;
    ui new_estimate_lb = estimate_lb;  // 不会被更新（禁用DP）

    std::shared_ptr<SearchSnapshot> current_snapshot = nullptr;
    enum NoSnapshotReason { SNAP_UNKNOWN=0, SNAP_LP_SKIPPED, SNAP_FILTER_SKIPPED, SNAP_REUSE_NO_CHAIN } no_snapshot_reason = SNAP_UNKNOWN;


    // NDC计数标志：确保每个节点只在实际做计算时计数一次
    bool ndc_counted = false;

    // 节点类型统计
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }

    if (!node.completed_db_graph_ids.empty()) {
        stats.EPT_nodes_with_completed_ids++;
    }

    if (node.op.type != EditOperation::NONE) {
        stats.EPT_op_type_count[node.op.type]++;
    }

    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    bool should_compute_ged = true;

    // 注意：这里故意跳过 LB Propagation 的 estimate_lb > tau 检查

    ui ept_filter_lb = 0;  // 保存EPT filter的lb，供当前节点SP使用（不传给子节点）

    if (use_ept_filters && should_compute_ged) {
        // NDC统计：只在实际做LB计算时计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }

        auto lb_start = std::chrono::high_resolution_clock::now();

        stats.EPT_lb_count++;
        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);
        ept_filter_lb = lb;  // 保存lb供SP使用
        if (lb > tau_ui) {
            should_compute_ged = false;
            no_snapshot_reason = SNAP_FILTER_SKIPPED;
        }

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }
    }

    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== 根节点优化：利用NetDag缓存的GED，避免重复A*计算 ==========
    bool root_used_netdag_cache = false;
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF && should_compute_ged) {
        bool needs_snapshot = false;
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            const TreeNode &c_node = ept.tree_nodes[ch];
            int distance = c_node.level - node.level;
            if (distance <= 0 || distance > max_ged_gap) continue;
            // 只有 net_n == 0 的 child 會用 reuse (新政策：父子 n 不同走 baseline)。
            int net_n = 0;
            for (const auto& op : c_node.accumulated_ops) {
                if (op.type == EditOperation::NODE_INSERTION) net_n++;
                else if (op.type == EditOperation::NODE_DELETION) net_n--;
            }
            if (net_n == 0) { needs_snapshot = true; break; }
        }
#endif
        if (!needs_snapshot) {
            if (anchor_netdag_ged <= (int)tau_ui) {
                int anchor_id_local = static_cast<int>(ept.anchor_id);
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                      node.completed_db_graph_ids.end(),
                                                      anchor_id_local) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id_local);
                    stats.EPT_results_from_astar++;
                }
                exact_results_within_tau.insert(exact_results_within_tau.end(),
                                               node.completed_db_graph_ids.begin(),
                                               node.completed_db_graph_ids.end());
                stats.EPT_results_from_astar += node.completed_db_graph_ids.size();
                found_here = true;
            }
            // 注意：这里故意不更新 new_estimate_lb（禁用LP）
            should_compute_ged = false;
            root_used_netdag_cache = true;
            stats.root_netdag_ged_reuse_count++;
        }
    }

    // GED计算 (与unified相同，但无LB Propagation更新)
    if (should_compute_ged) {
        // NDC统计：如果跳过LB但做A*计算，也要计数
        if (!ndc_counted ) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        bool use_baseline = false;
        int optimal_margin = 0;
        bool has_close_child = false;

#if USE_BASELINE_FOR_DISTANT_CHILDREN
        int current_level = node.level;
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            const TreeNode &child_node = ept.tree_nodes[ch];
            int child_level = child_node.level;
            int distance = child_level - current_level;
            if (distance <= 0 || distance > max_ged_gap) continue;
            // 只有 net_n == 0 的 child 會用 reuse (新政策：父子 n 不同走 baseline)。
            int net_n = 0;
            for (const auto& op : child_node.accumulated_ops) {
                if (op.type == EditOperation::NODE_INSERTION) net_n++;
                else if (op.type == EditOperation::NODE_DELETION) net_n--;
            }
            if (net_n != 0) continue;
            has_close_child = true;
            optimal_margin = std::max(optimal_margin, distance);
        }
        optimal_margin = std::min(optimal_margin, max_margin);
        if (!has_close_child && !node.children_indices.empty() && node_index != ept.root_index) {
            use_baseline = true;
            optimal_margin = 0;
        }
#endif

        bool can_reuse = false;

        if (!use_baseline) {
            if (node_index == ept.root_index) {
                stats.EPT_reuse_fail_root_node++;
            } else if (parent_db_vertex_count >= 0 &&
                       (int)db_g->n != (int)parent_db_vertex_count) {
                // 父子 vertex 數不同 → 直接 fallback baseline。
                stats.EPT_reuse_fail_vertex_count_changed++;
            } else if (node.accumulated_ops.size() > max_ged_gap) {
                stats.EPT_reuse_fail_multi_ops++;
            } else if (parent_snapshot == nullptr || parent_db_vertex_count < 0) {
                stats.EPT_reuse_fail_no_parent_snapshot++;
                switch (parent_no_snapshot_reason) {
                    case SNAP_LP_SKIPPED:     stats.EPT_reuse_fail_no_parent_lp_skipped++; break;
                    case SNAP_FILTER_SKIPPED: stats.EPT_reuse_fail_no_parent_filter_skipped++; break;
                    case SNAP_REUSE_NO_CHAIN: stats.EPT_reuse_fail_no_parent_reuse_no_chain++; break;
                    default:                  stats.EPT_reuse_fail_no_parent_other++; break;
                }
            } else {
                size_t parent_mo_size = parent_snapshot->mo.size();
                size_t current_min_size = std::min(static_cast<size_t>(qry_g->n), static_cast<size_t>(db_g->n));
                size_t current_max_size = std::max(static_cast<size_t>(qry_g->n), static_cast<size_t>(db_g->n));
                bool mo_size_compatible = (parent_mo_size <= current_min_size);
                bool mo_content_valid = true;
                if (mo_size_compatible && !parent_snapshot->mo.empty()) {
                    for (size_t i = 0; i < parent_snapshot->mo.size(); ++i) {
                        if (parent_snapshot->mo[i] != static_cast<size_t>(-1) &&
                            parent_snapshot->mo[i] >= current_max_size) {
                            mo_content_valid = false;
                            break;
                        }
                    }
                }
                if (!mo_size_compatible || !mo_content_valid) {
                } else if (parent_snapshot->v.empty()) {
                } else if (parent_snapshot->v.size() >= 1) {
                    can_reuse = true;
                    stats.EPT_reuse_attempt++;
                }
            }
        }

        Application app(tau_ui, "BMao", app_max_iter);
        app.set_all_edge_labels_same(all_edge_labels_same);
        app.init(db_g, qry_g);
        app.set_disable_reuse_lsa(disable_reuse_lsa);
        app.set_exact_value_mode(exact_value_mode);
        app.set_early_stop_at_tau(early_stop_at_tau);
        app.set_skip_intersection_in_reuse(early_stop_at_tau);

        int ged = -1;
        bool used_reuse = false;
        double elapsed_time = 0.0;

        if (can_reuse) {
            // chain_reuse=false时，使用reuse的节点不会保存snapshot，因此不需要margin
            if (chain_reuse) {
                app.set_margin(optimal_margin);  // 链式复用时需要为后续保存snapshot
            } else {
                app.set_margin(0);  // 不需要为后续保存snapshot，使用margin=0减少计算
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.app_reuse(*parent_snapshot, tau_ui, node.accumulated_ops.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            used_reuse = (ged >= 0);
            if (used_reuse) {
                stats.EPT_astar_count++;
                if (!chain_reuse) {
                    no_snapshot_reason = SNAP_REUSE_NO_CHAIN;
                }
            }
        }

        if (!used_reuse) {
            app.set_margin(optimal_margin);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            stats.EPT_astar_count++;
            auto t0 = std::chrono::high_resolution_clock::now();
            // optimal_margin==0: 不会产出有用 snapshot, 用 App_test 省掉 boundary_nodes + protect_ancestor_chain 开销
            ged = (optimal_margin == 0) ? app.App_test() : app.App();
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();

            if (verify_reuse && optimal_margin > 0) {
                Application app_no_margin(tau_ui, "BMao", app_max_iter);
                app_no_margin.init(db_g, qry_g);
                auto t_nm_start = std::chrono::high_resolution_clock::now();
                int ged_no_margin = app_no_margin.App_baseline(nullptr, nullptr);
                auto t_nm_end = std::chrono::high_resolution_clock::now();
                double no_margin_time = std::chrono::duration<double>(t_nm_end - t_nm_start).count();
                stats.margin_overhead_count++;
                stats.margin_overhead_with_margin_time += elapsed_time;
                stats.margin_overhead_without_margin_time += no_margin_time;
                bool margin_correct = ((ged > (int)tau_ui && ged_no_margin > (int)tau_ui) ||
                                      (ged <= (int)tau_ui && ged_no_margin <= (int)tau_ui));
                if (margin_correct) stats.margin_overhead_correct++;
                else stats.margin_overhead_incorrect++;
            }
        }

        if (used_reuse) {
            stats.EPT_reuse_count++;
            if (node.op.type != EditOperation::NONE) {
                stats.EPT_reuse_by_op_type[node.op.type]++;
            }
            if (verify_reuse) {
                Application app_verify(tau_ui, "BMao", app_max_iter);
                app_verify.init(db_g, qry_g);
                auto t_verify_start = std::chrono::high_resolution_clock::now();
                int ged_standard = app_verify.App_baseline(nullptr, nullptr);
                auto t_verify_end = std::chrono::high_resolution_clock::now();
                double verify_time = std::chrono::duration<double>(t_verify_end - t_verify_start).count();
                stats.EPT_verification_astar_time += verify_time;
                stats.EPT_reuse_verifications++;
                stats.EPT_baseline_app_count++;
                stats.EPT_baseline_app_time += verify_time;
                stats.EPT_baseline_reuse_time += elapsed_time;
                if (stats.baseline_samples.size() < SearchStats::MAX_BASELINE_SAMPLES) {
                    stats.baseline_samples.push_back({elapsed_time * 1000.0, verify_time * 1000.0});
                }
                bool reuse_correct = ((ged > (int)tau_ui && ged_standard > (int)tau_ui) ||
                                     (ged <= (int)tau_ui && ged_standard <= (int)tau_ui));
                if (reuse_correct) stats.EPT_reuse_correct++;
                else stats.EPT_reuse_incorrect++;

                // Speedup distribution
                double sp = (elapsed_time > 0) ? verify_time / elapsed_time : 0.0;
                int sp_bucket;
                if (sp > 3.0) { stats.reuse_speedup_gt3x++; sp_bucket = 0; }
                else if (sp > 2.0) { stats.reuse_speedup_2x_3x++; sp_bucket = 1; }
                else if (sp > 1.0) { stats.reuse_speedup_1x_2x++; sp_bucket = 2; }
                else if (sp > 0.5) { stats.reuse_speedup_05x_1x++; sp_bucket = 3; }
                else { stats.reuse_speedup_lt05x++; sp_bucket = 4; }

                // Snapshot size
                int size_bucket = -1;
                if (parent_snapshot) {
                    size_t sz = parent_snapshot->v.size();
                    if (sz <= 1) { stats.reuse_snapshot_size_1++; size_bucket = 0; }
                    else if (sz <= 10) { stats.reuse_snapshot_size_2_10++; size_bucket = 1; }
                    else { stats.reuse_snapshot_size_gt10++; size_bucket = 2; }
                }

                // Cross-tab [size][speedup]
                if (size_bucket >= 0) {
                    stats.reuse_xtab[size_bucket][sp_bucket]++;
                    stats.reuse_xtime[size_bucket] += elapsed_time;
                    stats.baseline_xtime[size_bucket] += verify_time;
                }

                // Chain depth statistics: depth = parent_snapshot->chain_depth
                int chain_d = parent_snapshot ? parent_snapshot->chain_depth : 0;
                if (chain_d < 0) chain_d = 0;
                if (chain_d >= SearchStats::MAX_CHAIN_DEPTH) chain_d = SearchStats::MAX_CHAIN_DEPTH - 1;
                stats.chain_depth_count[chain_d]++;
                stats.chain_depth_reuse_time[chain_d]    += elapsed_time;
                stats.chain_depth_baseline_time[chain_d] += verify_time;
                if (reuse_correct) stats.chain_depth_correct[chain_d]++;
                if (ged_standard <= (int)tau_ui) {
                    stats.chain_depth_pos[chain_d]++;
                    if (ged <= (int)tau_ui) stats.chain_depth_tp[chain_d]++;
                }

                // Per-reuse log
                if (g_reuse_log) {
                    std::lock_guard<std::mutex> _lk(g_reuse_log_mutex);
                    fprintf(g_reuse_log,
                            "%d,%u,%zu,%zu,%zu,%d,%.3f,%.3f,%d,%d,%d\n",
                            query_node->node_id,
                            ept.anchor_id,
                            parent_node_index,
                            node_index,
                            parent_snapshot ? parent_snapshot->v.size() : (size_t)0,
                            (int)node.accumulated_ops.size(),
                            elapsed_time * 1e6,
                            verify_time * 1e6,
                            ged,
                            ged_standard,
                            (int)tau_ui);
                }
            }
        }

        stats.EPT_astar_time += elapsed_time;
        if (used_reuse) {
            stats.EPT_reuse_success_time += elapsed_time;
        }

        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        // 注意：这里故意不更新 new_estimate_lb（禁用DP）

        // Subtree Pruning 检查（保留）
#if USE_SUBTREE_PRUNING
        ui overall_lb = app.get_overall_lb();

        if (!node.children_indices.empty() && overall_lb > 0 && overall_lb < 10000) {
            int max_step_more;

            max_step_more = node.max_subtree_depth - node.level;

            if (max_step_more + (int)tau_ui < (int)overall_lb) {
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1;
                stats.subtree_pruning_avoided_nodes += avoided_nodes;
                return found_here;
            }
        }
#endif

        if (ged <= (int)tau_ui) {
            if (node_index == ept.root_index) {
                int anchor_id_local = static_cast<int>(ept.anchor_id);
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                      node.completed_db_graph_ids.end(),
                                                      anchor_id_local) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id_local);
                    if (used_reuse) {
                        stats.EPT_results_from_reuse += 1;
                    } else {
                        stats.EPT_results_from_astar += 1;
                    }
                }
            }

            size_t num_results = node.completed_db_graph_ids.size();
            if (used_reuse) {
                stats.EPT_results_from_reuse += num_results;
            } else {
                stats.EPT_results_from_astar += num_results;
            }

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                          node.completed_db_graph_ids.begin(),
                                          node.completed_db_graph_ids.end());
            found_here = true;
        }

        // 提取 snapshot（供子节点复用）
        // chain_reuse=true时，使用reuse的节点也保存snapshot供后续节点复用
        bool should_save_snapshot = (ged >= 0 && ged < INT_MAX && has_close_child);
        if (!chain_reuse) {
            should_save_snapshot = should_save_snapshot && !used_reuse;
        }
        if (should_save_snapshot) {
            current_snapshot = std::make_shared<SearchSnapshot>();
            try {
                app.extract_snapshot(*current_snapshot);
                current_snapshot->ub = ged;
                current_snapshot->margin = optimal_margin;
                current_snapshot->chain_depth = used_reuse
                    ? ((parent_snapshot ? parent_snapshot->chain_depth : 0) + 1)
                    : 0;
                if (current_snapshot->v.empty()) {
                    current_snapshot = nullptr;
                }
            } catch (...) {
                current_snapshot = nullptr;
            }
        }
    }
    else {
        // 跳过了计算，但仍要检查Subtree Pruning（保留）
        // 使用当前节点EPT filter算出的lb（不是从父节点传播的estimate_lb）
#if USE_SUBTREE_PRUNING
        if (!node.children_indices.empty() && ept_filter_lb > 0 && ept_filter_lb < 10000) {
            int max_step_more;

            max_step_more = node.max_subtree_depth - node.level;

            if (max_step_more + (int)tau_ui < (int)ept_filter_lb) {
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1;
                stats.subtree_pruning_avoided_nodes += avoided_nodes;
                return found_here;
            }
        }
#endif
    }

    // 递归处理子节点
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }

        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        // 注意：因为禁用DP，new_estimate_lb不会被更新，这里传0
        ui child_estimate_lb = 0;

        stats.lb_propagation_count++;

        if (dfs_traverse_no_LP(ch, ept, query_node,
                              exact_results_within_tau, tau, stats,
                              current_snapshot, db_g->n, child_estimate_lb,
                              anchor_netdag_lb, anchor_netdag_ged,
                              (current_snapshot == nullptr) ? no_snapshot_reason : 0,
                              node_index)) {
            found_child = true;
        }
    }

    return found_here || found_child;
}

void GismaSearchEngine::traverse_ept_and_search(
    const EditPathTree &ept,
    std::shared_ptr<Node> query_node,
    std::vector<int> &exact_results_within_tau,
    double tau,
    SearchStats &stats,
    double anchor_netdag_lb,
    int anchor_netdag_ged,
    const std::string &dfs_mode_override)
{
    if (ept.tree_nodes.empty()) {
        return;
    }

    if (ept.root_index >= ept.tree_nodes.size()) {
        return;
    }

    // 使用override参数（如有），否则使用成员变量，避免并行模式下的数据竞争
    const std::string &effective_dfs_mode = dfs_mode_override.empty() ? this->dfs_mode : dfs_mode_override;

    // 根据dfs_mode参数选择DFS遍历方式（默认使用dfs_traverse，包含所有优化）
    if (effective_dfs_mode == "no_reuse") {
        // 使用no_reuse版本（SP + DP优化，但无搜索树复用）
        ui initial_estimate_lb = 0;

        dfs_traverse_no_reuse(ept.root_index, ept, query_node,
                           exact_results_within_tau, tau, stats,
                           initial_estimate_lb, anchor_netdag_lb, anchor_netdag_ged);
    } else if (effective_dfs_mode == "only_dfs") {
        // 使用纯DFS遍历（App_baseline，无优化）
        dfs_traverse_only_dfs(ept.root_index, ept, query_node,
                             exact_results_within_tau, tau, stats,
                             anchor_netdag_lb, anchor_netdag_ged);
    } else if (effective_dfs_mode == "no_SP") {
        // 使用no_SP版本（reuse + DP优化，但无Subtree Pruning）
        ui initial_estimate_lb = 0;
        dfs_traverse_no_SP(ept.root_index, ept, query_node,
                          exact_results_within_tau, tau, stats,
                          nullptr, -1, initial_estimate_lb,
                          anchor_netdag_lb, anchor_netdag_ged);
    } else if (effective_dfs_mode == "no_LP") {
        // 使用no_LP版本（reuse + SP优化，但无Distance Propagation）
        ui initial_estimate_lb = 0;
        dfs_traverse_no_LP(ept.root_index, ept, query_node,
                          exact_results_within_tau, tau, stats,
                          nullptr, -1, initial_estimate_lb,
                          anchor_netdag_lb, anchor_netdag_ged);
    } else {
        // 默认：使用dfs_traverse（包含所有优化：Reuse + SP + LP）
        ui initial_estimate_lb = 0;
        dfs_traverse(ept.root_index, ept, query_node,
                            exact_results_within_tau, tau, stats,
                            nullptr, -1, initial_estimate_lb,
                            anchor_netdag_lb, anchor_netdag_ged);
    }
}


std::vector<int> GismaSearchEngine::extra_cluster_search(std::shared_ptr<Node> query_node,
                                                    int anchor_id,
                                                    double tau,
                                                    SearchStats &stats)
{
    std::vector<int> results;

    // 根据 anchor_id 获取对应 Anchor 对象
    auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
    if (!anchor_ptr)
    {
        // 若拿不到或不是 Anchor，返回空结果
        return results;
    }

    Graph *query_graph = query_node->graph.get();
    if (!query_graph)
    {
        return results;
    }

    // 1) 直接用 anchor_ptr->nodes_in_cluster_vec
    const auto &clusterCopy = anchor_ptr->nodes_in_cluster_vec;

    // 2) 遍历 clusterCopy
    for (auto &pair_item : clusterCopy)
    {
        double dist = pair_item.first;
        int cluster_node_id = pair_item.second;

        // ========== LB ==========
        stats.extra_ndc_count++;
        stats.EPT_ndc_count++;
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;
        stats.extra_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db[cluster_node_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;
        stats.extra_lb_time += lb_duration;

        if (lb > tau)
        {
            continue;
        }

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;
        stats.extra_astar_count++;

        // E8 fix extra_cluster LSa: gisma-lsa / app-lsa 的 cluster 成员验证也须用 LSa（原硬编码 BMao）。
        const char* c_lb = (ged_algorithm == "gisma-lsa" || ged_algorithm == "app-lsa") ? "LSa" : "BMao";
        int ged_res;
        if (use_orig_verifier) {
            origbmao::Application oapp((ui)tau, c_lb, (long long)app_max_iter);
            oapp.init(db[cluster_node_id], query_graph);
            ged_res = (int)oapp.AStar();
        } else {
            Application app((ui)tau, c_lb, app_max_iter);
            app.set_all_edge_labels_same(all_edge_labels_same);
            app.init(db[cluster_node_id], query_graph);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            // LSa 走 App_test（_test 机制对 lb_method==LSa 稳定；App() 的 LSa 路径不稳）。
            ged_res = (c_lb[0] == 'L') ? (int)app.App_test(nullptr, nullptr) : (int)app.App();
        }

        auto astar_end = std::chrono::high_resolution_clock::now();
        double astar_duration = std::chrono::duration<double>(astar_end - astar_start).count();
        stats.EPT_astar_time += astar_duration;
        stats.extra_astar_time += astar_duration;

        if (ged_res <= (int)tau)
        {
            results.push_back(cluster_node_id);
        }
    }

    return results;
}

std::vector<int> GismaSearchEngine::Gisma_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats, const std::string &dfs_mode_override)
{
    int query_id = query_node->node_id;

    // 1) GS => 得到 anchor IDs with NetDag LB distances
    auto nd_start = std::chrono::high_resolution_clock::now();
    std::vector<std::tuple<int, double, int>> candidate_anchors_with_results = GS_search(query_node, tau, stats);

    if (candidate_anchors_with_results.empty())
    {
        // ND_total_time is now computed as ND_lb_time + ND_astar_time in print_summary()
        // auto nd_end = std::chrono::high_resolution_clock::now();
        // stats.ND_total_time += std::chrono::duration<double>(nd_end - nd_start).count();
        return {};
    }
    auto EPT_start = std::chrono::high_resolution_clock::now();

    // 用于汇总所有 SS_search 的耗时，以及对应的 anchor_id
    std::vector<std::pair<int, double>> ss_time_records;
    ss_time_records.reserve(candidate_anchors_with_results.size());

    std::vector<int> results;
    std::vector<int> ept_results_all;      // DEBUG: 收集所有EPT结果
    std::vector<int> extra_results_all;    // DEBUG: 收集所有extra_cluster结果
    std::unordered_set<int> ept_coverage;    // DEBUG: EPT覆盖的所有图ID
    std::unordered_set<int> extra_coverage;  // DEBUG: Extra cluster覆盖的所有图ID

    for (const auto& [anchor_id, netdag_lb, netdag_ged] : candidate_anchors_with_results)
    {
        // DEBUG: 收集EPT和Extra的覆盖范围
        auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
        if (anchor_ptr) {
            // EPT覆盖：从EPT树本身获取所有图ID
            // EPT结果只来自: ept.anchor_id 和 node.completed_db_graph_ids
            EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
            if (ept) {
                ept_coverage.insert(ept->anchor_id);  // anchor本身
                for (const auto& tree_node : ept->tree_nodes) {
                    for (int id : tree_node.completed_db_graph_ids) {
                        ept_coverage.insert(id);
                    }
                }
            }
            // Extra覆盖：nodes_in_cluster
            for (const auto& p : anchor_ptr->nodes_in_cluster_vec) {
                extra_coverage.insert(p.second);
            }
        }

        // （a）统计 SS_search
        auto ss_start = std::chrono::high_resolution_clock::now();
        auto anchor_results = SS_search(query_node, anchor_id, netdag_lb, netdag_ged, tau, stats, dfs_mode_override);
        auto ss_end = std::chrono::high_resolution_clock::now();

        double ss_duration = std::chrono::duration<double>(ss_end - ss_start).count();
        // 记录 (anchor_id, SS_search 耗时)
        ss_time_records.emplace_back(anchor_id, ss_duration);

        // （b）合并结果
        results.insert(results.end(), anchor_results.begin(), anchor_results.end());
        ept_results_all.insert(ept_results_all.end(), anchor_results.begin(), anchor_results.end());  // DEBUG

        // ========== (可选) 补充搜索: 对 anchor->nodes_in_cluster 做 BMao-scan-like ==========
        auto cluster_results = extra_cluster_search(query_node, anchor_id, tau, stats);
        results.insert(results.end(), cluster_results.begin(), cluster_results.end());
        extra_results_all.insert(extra_results_all.end(), cluster_results.begin(), cluster_results.end());  // DEBUG
    }

    // ========== DEBUG: 累积EPT和extra_cluster各自的recall统计 ==========
#if 0  // 设为0可关闭调试输出
    {
        // 静态变量用于累积统计
        static std::mutex debug_mutex;
        static int debug_query_count = 0;
        static long long debug_total_gt = 0;
        static long long debug_total_ept_gt = 0;      // EPT覆盖范围内的GT数
        static long long debug_total_extra_gt = 0;    // Extra覆盖范围内的GT数
        static long long debug_total_ept_hit = 0;
        static long long debug_total_extra_hit = 0;
        static long long debug_total_all_hit = 0;
        static bool registered_atexit = false;

        // 注册atexit回调，程序结束时输出汇总
        if (!registered_atexit) {
            registered_atexit = true;
            std::atexit([]() {
                if (debug_query_count > 0) {
                    double ept_recall = debug_total_ept_gt > 0 ? 100.0 * debug_total_ept_hit / debug_total_ept_gt : 0;
                    double extra_recall = debug_total_extra_gt > 0 ? 100.0 * debug_total_extra_hit / debug_total_extra_gt : 0;
                    double total_recall = debug_total_gt > 0 ? 100.0 * debug_total_all_hit / debug_total_gt : 0;
                    printf("\n========== EPT vs Extra Cluster Recall Summary ==========\n");
                    printf("Queries: %d, Total GT: %lld\n", debug_query_count, debug_total_gt);
                    printf("EPT:   %lld / %lld = %.2f%% (GT in EPT coverage)\n", debug_total_ept_hit, debug_total_ept_gt, ept_recall);
                    printf("Extra: %lld / %lld = %.2f%% (GT in Extra coverage)\n", debug_total_extra_hit, debug_total_extra_gt, extra_recall);
                    printf("Total: %lld / %lld = %.2f%% (overall)\n", debug_total_all_hit, debug_total_gt, total_recall);
                    printf("==========================================================\n");
                }
            });
        }

        std::lock_guard<std::mutex> lock(ground_truth_mutex);
        auto it = ground_truth.find(query_id);
        if (it != ground_truth.end()) {
            // 收集所有 ged <= tau 的ground truth（和compute_recall一致）
            std::vector<int> gt_ids;
            for (const auto& distance_pair : it->second) {
                double ged = distance_pair.first;
                if (ged <= tau) {
                    gt_ids.insert(gt_ids.end(), distance_pair.second.begin(), distance_pair.second.end());
                }
            }

            if (!gt_ids.empty()) {
                std::unordered_set<int> gt_set(gt_ids.begin(), gt_ids.end());
                int gt_size = static_cast<int>(gt_set.size());

                // 去重
                std::unordered_set<int> ept_set(ept_results_all.begin(), ept_results_all.end());
                std::unordered_set<int> extra_set(extra_results_all.begin(), extra_results_all.end());
                std::unordered_set<int> all_set(results.begin(), results.end());

                // 计算GT在各覆盖范围内的数量，以及命中数
                // 使用全局覆盖集合（整个数据库的覆盖），而不是查询访问到的覆盖
                int ept_gt = 0, extra_gt = 0;
                int ept_hit = 0, extra_hit = 0, all_hit = 0;

                // DEBUG: 收集当前查询访问到的anchor集合
                std::unordered_set<int> visited_anchor_set;
                for (const auto& [anchor_id, lb, ged] : candidate_anchors_with_results) {
                    visited_anchor_set.insert(anchor_id);
                }

                for (int id : gt_set) {
                    bool in_ept = g_ept_coverage.count(id) > 0;
                    bool in_extra = g_extra_coverage.count(id) > 0;

                    if (in_ept) ept_gt++;
                    if (in_extra) extra_gt++;

                    if (ept_set.count(id)) ept_hit++;
                    if (extra_set.count(id)) extra_hit++;
                    if (all_set.count(id)) all_hit++;

                    // DEBUG: 如果GT在EPT覆盖范围但没被找到，输出原因
                    if (in_ept && !ept_set.count(id)) {
                        // 找出这个GT属于哪个anchor
                        int owner_anchor = -1;
                        bool anchor_was_visited = false;

                        for (const auto& anchor : net_dag->anchors) {
                            if (!anchor) continue;
                            // 检查是否是anchor本身
                            if (anchor->node_id == id) {
                                owner_anchor = anchor->node_id;
                                break;
                            }
                            // 检查是否在anchor的EPT中
                            EditPathTree* ept_check = ept_manager->get_ept_no_lock(anchor->node_id);
                            if (ept_check) {
                                for (const auto& tn : ept_check->tree_nodes) {
                                    for (int db_id : tn.completed_db_graph_ids) {
                                        if (db_id == id) {
                                            owner_anchor = anchor->node_id;
                                            break;
                                        }
                                    }
                                    if (owner_anchor >= 0) break;
                                }
                            }
                            if (owner_anchor >= 0) break;
                        }

                        if (owner_anchor >= 0) {
                            anchor_was_visited = visited_anchor_set.count(owner_anchor) > 0;
                        }

                        printf("[EPT_MISS] Query=%d, MissedGT=%d, OwnerAnchor=%d, AnchorVisited=%s\n",
                               query_id, id, owner_anchor, anchor_was_visited ? "YES" : "NO");
                    }
                }

                // 累积统计
                {
                    std::lock_guard<std::mutex> dlock(debug_mutex);
                    debug_query_count++;
                    debug_total_gt += gt_size;
                    debug_total_ept_gt += ept_gt;
                    debug_total_extra_gt += extra_gt;
                    debug_total_ept_hit += ept_hit;
                    debug_total_extra_hit += extra_hit;
                    debug_total_all_hit += all_hit;
                }
            }
        }
    }
#endif



    return results;
}

// Gisma但禁用reuse（使用no_reuse模式）
std::vector<int> GismaSearchEngine::Gisma_no_reuse_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "no_reuse");
}

// Gisma纯DFS遍历（使用App_baseline，无优化）
std::vector<int> GismaSearchEngine::Gisma_only_dfs_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "only_dfs");
}

// Gisma禁用Subtree Pruning（消融实验）
std::vector<int> GismaSearchEngine::Gisma_no_SP_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "no_SP");
}

// Gisma禁用Distance Propagation/LB Propagation（消融实验）
std::vector<int> GismaSearchEngine::Gisma_no_LP_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "no_LP");
}



std::tuple<bool, double, double, double, int, int> GismaSearchEngine::compute_recall_precision_IoU(int query_id, const std::vector<int> &exact_results_within_tau, double tau)
{
    std::lock_guard<std::mutex> lock(ground_truth_mutex);
    auto it = ground_truth.find(query_id);
    if (it != ground_truth.end())
    {
        const std::map<double, std::vector<int>> &distances = it->second;
        std::vector<int> ids_within_tau;

        // 1) 找到 ground truth 中实际 ged <= tau 的所有节点
        for (const auto &distance_pair : distances)
        {
            double ged = distance_pair.first;
            const std::vector<int> &graph_ids = distance_pair.second;
            if (ged <= tau)
            {
                ids_within_tau.insert(ids_within_tau.end(), graph_ids.begin(), graph_ids.end());
            }
        }

        int ground_truth_size = static_cast<int>(ids_within_tau.size());
        if (ground_truth_size > 0)
        {
            // 2) 构造 set 方便求交
            std::unordered_set<int> exact_result_set(exact_results_within_tau.begin(), exact_results_within_tau.end());
            std::unordered_set<int> ground_truth_set(ids_within_tau.begin(), ids_within_tau.end());

            int intersection_count = 0;
            for (int id : ground_truth_set)
            {
                if (exact_result_set.find(id) != exact_result_set.end())
                {
                    intersection_count++;
                }
            }

            // 3) 计算 recall, precision, iou
            double recall = static_cast<double>(intersection_count) / ground_truth_size;

            int exact_size = static_cast<int>(exact_result_set.size());  // 用去重后的大小
            double precision = 0.0;
            if (exact_size > 0)
            {
                precision = static_cast<double>(intersection_count) / exact_size;
            }
            // union_size = ground_truth_size + exact_size - intersection_count
            int union_size = ground_truth_size + exact_size - intersection_count;
            double iou = 0.0;
            if (union_size > 0)
            {
                iou = static_cast<double>(intersection_count) / union_size;
            }

            return std::make_tuple(true, recall, precision, iou, intersection_count, ground_truth_size);
        }
        else
        {
            // ground_truth_size == 0 => 表示 tau 下无任何匹配
            // 仍返回 false
            return std::make_tuple(false, 0.0, 0.0, 0.0, 0, 0);
        }
    }
    else
    {
        // 查无 ground_truth
        return std::make_tuple(false, 0.0, 0.0, 0.0, 0, 0);
    }
}


void GismaSearchEngine::perform_search(double tau)
{
    // ========== 0) 全局统计变量 (Recall/Precision/IoU 相关) ==========
    int total_queries_processed = 0;
    int total_results_found = 0;
    int queries_with_non_empty_results = 0;

    double total_query_time_sum = 0.0;
    double total_recall = 0.0;
    double total_precision = 0.0;
    double total_iou = 0.0;
    int valid_query_count = 0;

    int total_intersection_count = 0;
    int total_ground_truth_count = 0;

    // ========== 1) 全局统计 (ND/EPT) ==========
    SearchStats global_stats;

    // ========== 2) 记录每个 query 的耗时 ==========
    std::vector<double> all_query_times;
    all_query_times.reserve(query_node_list.size());

    // 新增：记录每个query的详细信息
    std::vector<QueryDetails> query_details_list;
    query_details_list.reserve(query_node_list.size());

    size_t total_queries = q_end - q_start + 1;

    // ========== 3) 遍历查询节点  ==========
    for (size_t idx = 0; idx <= q_end - q_start; ++idx)
    {
        const auto &query_node = query_node_list[idx];

        auto total_query_start = std::chrono::high_resolution_clock::now();

        SearchStats local_stats;
        std::vector<int> results;

        // (a) 调用对应搜索方法
        if (this->search_method == "Gisma" || this->search_method == "Gisma-default")
        {
            results = this->Gisma_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "Gisma-no-reuse")
        {
            results = this->Gisma_no_reuse_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "BMao_scan")
        {
            results = this->BMao_scan_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "App-BMao")
        {
            results = this->App_BMao_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "AStar-BMao")
        {
            results = this->AStar_BMao_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "answer_search")
        {
            results = this->answer_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "Base+GS")
        {
            results = this->Base_GS_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "Base+SS")
        {
            results = this->Base_SS_search(query_node, tau, local_stats);
        }
        else if (this->search_method == "Base_All_EPT")
        {
            results = this->Base_All_EPT_search(query_node, tau, local_stats);
        }
        else
        {
            std::cout << "[WARN] Invalid search method: " << this->search_method
                      << ". fallback to Gisma.\n";
            results = this->Gisma_search(query_node, tau, local_stats);
        }

        // (b) 累加 local_stats 到全局
        global_stats.add(local_stats);

        auto total_query_end = std::chrono::high_resolution_clock::now();
        double total_query_time = std::chrono::duration<double>(total_query_end - total_query_start).count();

        // (c) 加入统计
        total_query_time_sum += total_query_time;
        all_query_times.push_back(total_query_time);

        // 新增：把 (query_id, total_query_time) 存起来
        int query_id = query_node->node_id;

        // (d) 统计结果
        int num_results = static_cast<int>(results.size());
        total_results_found += num_results;
        if (num_results > 0)
        {
            queries_with_non_empty_results++;
        }

        // (e) recall / precision / iou
        auto [has_ground_truth, recall, precision, iou, intersection_count, ground_truth_size]
            = this->compute_recall_precision_IoU(query_id, results, tau);

        // 保存query详情（包含recall等信息）
        query_details_list.emplace_back(query_id, total_query_time, recall, precision, iou, has_ground_truth);

        if (has_ground_truth)
        {
            total_recall += recall;
            total_precision += precision;
            total_iou += iou;
            valid_query_count++;

            total_intersection_count += intersection_count;
            total_ground_truth_count += ground_truth_size;
        }

        total_queries_processed++;

        // 显示进度条
        if (total_queries_processed % (std::max(total_queries / 10, size_t(1))) == 0 ||
            total_queries_processed == total_queries) {

            int progress = (100 * total_queries_processed) / total_queries;
            int bar_width = 50;
            int filled = (bar_width * progress) / 100;

            std::cout << "\r[";
            for (int i = 0; i < filled; ++i) std::cout << "=";
            if (filled < bar_width) std::cout << ">";
            for (int i = filled + 1; i < bar_width; ++i) std::cout << " ";
            std::cout << "] " << progress << "% (" << total_queries_processed << "/" << total_queries << ")";
            std::cout.flush();

            if (total_queries_processed == total_queries) {
                std::cout << std::endl;
            }
        }
    }
    std::cout << "\nCompleted!\n";
    
    // ========== 4) 调用统一的输出函数 ==========
    // 计算总时间（对于非并行版本，使用查询时间总和）
    double total_time = total_query_time_sum;
    
    // 为了兼容性，从query_details_list生成query_time_pairs
    std::vector<std::pair<int, double>> query_time_pairs;
    for (const auto& qd : query_details_list) {
        query_time_pairs.emplace_back(qd.query_id, qd.time);
    }

    print_search_statistics(
        global_stats,
        total_queries_processed,
        total_results_found,
        queries_with_non_empty_results,
        total_query_time_sum,
        total_time,
        total_recall,
        total_precision,
        total_iou,
        valid_query_count,
        total_intersection_count,
        total_ground_truth_count,
        query_time_pairs,
        false  // is_parallel = false
    );

    // 保存实验结果到文件（仅在启用save_logs时）
    if (save_logs) {
        save_experiment_results(
            dataset_name,
            tau,
            global_stats,
            total_queries_processed,
            total_results_found,
            queries_with_non_empty_results,
            total_query_time_sum,
            total_time,
            total_recall,
            total_precision,
            total_iou,
            valid_query_count,
            total_intersection_count,
            total_ground_truth_count,
            query_details_list,
            false  // is_parallel = false
        );
    }

    std::cout << "[perform_search] Sequential search completed.\n";
}

void GismaSearchEngine::perform_search_parallel(double tau)
{
    // ========== 0) 定义并行统计变量 ==========
    std::atomic<int> total_queries_processed(0);
    std::atomic<int> total_results_found(0);
    std::atomic<int> queries_with_non_empty_results(0);

    double total_query_time_sum = 0.0;
    double total_search_time = 0.0;

    double total_recall = 0.0;
    double total_precision = 0.0;
    double total_iou = 0.0;
    std::atomic<int> valid_query_count(0);

    std::atomic<int> total_intersection_count(0);
    std::atomic<int> total_ground_truth_count(0);

    std::mutex recall_mutex;  // 保护 total_recall, total_precision, total_iou

    // ========== 1) 全局统计 (ND/EPT) ==========
    SearchStats global_stats;

    // ========== 2) 记录每个 query 的耗时 ==========
    std::vector<double> all_query_times;
    all_query_times.reserve(query_node_list.size());

    // 新增：记录每个query的详细信息
    std::vector<QueryDetails> query_details_list;
    query_details_list.reserve(query_node_list.size());

    // 互斥锁，用于保护对"浮点"或"多字段同时更新"的操作
    std::mutex stats_mutex;

    // ========== 3) 并行开始计时 ==========
    auto total_start = std::chrono::high_resolution_clock::now();

    // ========== 4) 确定线程数 & 平均分配查询给各线程 ==========
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = (num_threads <= 200) ? num_threads : 200; // 最多200线程
    std::cout << "[perform_search_parallel] Using " << num_threads << " threads.\n";

    size_t queries_per_thread = (query_node_list.size() + num_threads - 1) / num_threads;

    // ========== 5) 创建多线程任务 (std::async) ==========
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (unsigned int i = 0; i < num_threads; ++i)
    {
        size_t start_index = i * queries_per_thread;
        size_t end_index = std::min(start_index + queries_per_thread, query_node_list.size());

        // 捕获所有我们需要的引用/值
        futures.emplace_back(std::async(std::launch::async,
            [=,
             &stats_mutex,
             &recall_mutex,
             &total_queries_processed,
             &total_results_found,
             &queries_with_non_empty_results,
             &total_query_time_sum,
             &total_search_time,
             &total_recall,
             &total_precision,
             &total_iou,
             &valid_query_count,
             &total_intersection_count,
             &total_ground_truth_count,
             &global_stats,
             &all_query_times,
             &query_details_list]()
            {
                // 处理本线程负责的查询 [start_index, end_index)
                for (size_t idx = start_index; idx < end_index; ++idx)
                {

                    auto &query_node = query_node_list[idx];
                    int query_id = query_node->node_id;

                    // 整个查询计时
                    auto total_query_start = std::chrono::high_resolution_clock::now();

                    SearchStats local_stats; // 每次查询都有局部统计
                    std::vector<int> results;

                    // 根据 search_method 调用不同函数
                    if (this->search_method == "Gisma" || this->search_method == "Gisma-default")
                    {
                        results = this->Gisma_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "Gisma-no-reuse")
                    {
                        results = this->Gisma_no_reuse_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "BMao_scan")
                    {
                        results = this->BMao_scan_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "App-BMao")
                    {
                        results = this->App_BMao_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "AStar-BMao")
                    {
                        results = this->AStar_BMao_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "answer_search")
                    {
                        results = this->answer_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "Base+GS")
                    {
                        results = this->Base_GS_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "Base+SS")
                    {
                        results = this->Base_SS_search(query_node, tau, local_stats);
                    }
                    else if (this->search_method == "Base_All_EPT")
                    {
                        results = this->Base_All_EPT_search(query_node, tau, local_stats);
                    }
                    else
                    {
                        results = this->Gisma_search(query_node, tau, local_stats);
                    }

                    auto total_query_end = std::chrono::high_resolution_clock::now();
                    double total_query_time = std::chrono::duration<double>(total_query_end - total_query_start).count();

                    // ========== 计算 Recall/Precision/IoU ==========
                    auto [has_ground_truth, recall, precision, iou, intersection_count, ground_truth_size]
                        = this->compute_recall_precision_IoU(query_id, results, tau);
                    // printf("results.size() = %d, ground_truth_size = %d\n", results.size(), ground_truth_size);

                    // ========== 创建QueryDetails对象 ==========
                    QueryDetails query_detail(query_id, total_query_time, recall, precision, iou, has_ground_truth);
                    query_detail.query_nodes = query_node->graph->n;
                    query_detail.query_edges = query_node->graph->m;
                    query_detail.result_count = (int)results.size();
                    query_detail.lb_time = local_stats.total_lb_time();
                    query_detail.astar_time = local_stats.total_astar_time();
                    // EPT统计信息需要从local_stats中提取
                    query_detail.total_ept_nodes = local_stats.EPT_total_nodes_visited;
                    query_detail.nodes_computed = local_stats.EPT_nodes_computed;
                    query_detail.nodes_pruned = local_stats.EPT_filter_pruned_nodes;
                    query_detail.lb_pruning_count = local_stats.lb_pruning_count;
                    query_detail.subtree_pruned = local_stats.subtree_pruning_avoided_nodes;

                    // ========== 实时保存JSON（在锁外，避免阻塞） ==========
                    if (save_logs && !experiment_base_dir.empty()) {
                        this->save_single_query_json(query_detail, tau, experiment_base_dir);
                    }

                    // ========== 更新统计信息 (需要加锁) ==========
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);

                        // 添加 query 耗时
                        all_query_times.push_back(total_query_time);
                        query_details_list.push_back(query_detail);

                        // 累加各种时间
                        total_query_time_sum += total_query_time;

                        // 累加 local_stats 到 global_stats
                        global_stats.add(local_stats);

                        // 如果有 ground truth，累加 recall/precision/IoU
                        if (has_ground_truth)
                        {
                            {
                                std::lock_guard<std::mutex> lock(recall_mutex);
                                total_recall += recall;
                                total_precision += precision;
                                total_iou += iou;
                            }
                            valid_query_count++;
                        }

                        // 总是累加 intersection 和 ground_truth 计数
                        total_intersection_count += intersection_count;
                        total_ground_truth_count += ground_truth_size;
                    }

                    // 原子更新计数
                    total_queries_processed++;
                    total_results_found += (int)results.size();
                    if (!results.empty())
                    {
                        queries_with_non_empty_results++;
                    }
                }
            }));
    }

    // ========== 6) 等待所有线程完成 ==========
    for (auto &f : futures)
    {
        f.get();
    }

    // ========== 7) 并行结束计时 ==========
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    // ========== 8) 调用统一的输出函数 ==========
    // 为了兼容性，从query_details_list生成query_time_pairs
    std::vector<std::pair<int, double>> query_time_pairs;
    for (const auto& qd : query_details_list) {
        query_time_pairs.emplace_back(qd.query_id, qd.time);
    }

    print_search_statistics(
        global_stats,
        total_queries_processed.load(),
        total_results_found.load(),
        queries_with_non_empty_results.load(),
        total_query_time_sum,
        total_time,  // 并行版本的总时间
        total_recall,
        total_precision,
        total_iou,
        valid_query_count.load(),
        total_intersection_count.load(),
        total_ground_truth_count.load(),
        query_time_pairs,
        true  // is_parallel = true
    );

    // 保存实验结果到文件（仅在启用save_logs时）
    if (save_logs) {
        save_experiment_results(
            dataset_name,
            tau,
            global_stats,
            total_queries_processed.load(),
            total_results_found.load(),
            queries_with_non_empty_results.load(),
            total_query_time_sum,
            total_time,
            total_recall,
            total_precision,
            total_iou,
            valid_query_count.load(),
            total_intersection_count.load(),
            total_ground_truth_count.load(),
            query_details_list,
            true  // is_parallel = true
        );
    }

    std::cout << "[perform_search_parallel] Parallel search completed.\n";
}

void GismaSearchEngine::print_search_statistics(
    const SearchStats& global_stats,
    int total_queries_processed,
    int total_results_found,
    int queries_with_non_empty_results,
    double total_query_time_sum,
    double total_time,  // 并行版本的总时间
    double total_recall,
    double total_precision,
    double total_iou,
    int valid_query_count,
    int total_intersection_count,
    int total_ground_truth_count,
    const std::vector<std::pair<int, double>>& query_time_pairs,
    bool is_parallel)
{
    printf("\n=================== Search Results ===================\n");
    if (is_parallel) {
        printf("[Parallel Search Mode]\n");
    }
    
    // ========== 1. 基本统计信息 ==========
    printf("Total queries processed: %d\n", total_queries_processed);
    printf("Total results found: %d / %d\n", total_results_found, total_ground_truth_count);
    
    // 结果来源分析（如果有）
    if (global_stats.EPT_results_from_reuse + global_stats.EPT_results_from_astar > 0) {
        printf("  - Results from reuse: %zu", global_stats.EPT_results_from_reuse);
        if (total_results_found > 0) {
            double reuse_percentage = 100.0 * global_stats.EPT_results_from_reuse / total_results_found;
            printf(" (%.1f%%)", reuse_percentage);
        }
        printf("\n");
        
        printf("  - Results from AStar: %zu", global_stats.EPT_results_from_astar);
        if (total_results_found > 0) {
            double astar_percentage = 100.0 * global_stats.EPT_results_from_astar / total_results_found;
            printf(" (%.1f%%)", astar_percentage);
        }
        printf("\n");
    }
    
    printf("Queries with non-empty results: %d\n", queries_with_non_empty_results);
    
    double sum_of_query_times = global_stats.total_lb_time() + global_stats.total_astar_time();
    printf("Total query time: %f seconds.\n", sum_of_query_times);
    
    // ========== 2. 准确性指标 ==========
    printf("\nPrecision Metrics:\n");
    int valid_q = valid_query_count;

    double avg_recall = (valid_q > 0) ? (total_recall / valid_q) : 0.0;
    double avg_precision = (valid_q > 0) ? (total_precision / valid_q) : 0.0;
    double avg_iou = (valid_q > 0) ? (total_iou / valid_q) : 0.0;

    printf("Average recall (over %d queries):    %f\n", valid_q, avg_recall);
    printf("Average precision (over %d queries): %f\n", valid_q, avg_precision);
    printf("Average IoU (over %d queries):      %f\n", valid_q, avg_iou);
    
    // 总体召回率
    double recall_overall = (total_ground_truth_count > 0) ? 
        ((double)total_intersection_count / total_ground_truth_count) : 0.0;
    printf("Overall recall (all queries combined): %.6f\n", recall_overall);
    
    // ========== 3. 复用统计 ==========
    if (global_stats.EPT_reuse_attempt > 0 || global_stats.EPT_astar_count > 0)
    {
        printf("\n========== Overall Reuse Statistics ==========\n");
        size_t total_ept_nodes = global_stats.EPT_astar_count + global_stats.EPT_reuse_count;
        printf("Total EPT nodes processed: %zu\n", total_ept_nodes);
        printf("  - Standard AStar calls: %zu\n", global_stats.EPT_astar_count);
        printf("  - Reuse attempts: %zu\n", global_stats.EPT_reuse_attempt);
        printf("  - Successful reuses: %zu\n", global_stats.EPT_reuse_count);
        
        // 结果来源统计
        printf("\nResults breakdown:\n");
        printf("  - Total results found: %zu\n", (global_stats.EPT_results_from_reuse + global_stats.EPT_results_from_astar));
        printf("    - From successful reuse: %zu", global_stats.EPT_results_from_reuse);
        if (global_stats.EPT_results_from_reuse + global_stats.EPT_results_from_astar > 0) {
            double reuse_result_pct = 100.0 * global_stats.EPT_results_from_reuse / 
                                     (global_stats.EPT_results_from_reuse + global_stats.EPT_results_from_astar);
            printf(" (%.1f%%)", reuse_result_pct);
        }
        printf("\n");
        printf("    - From standard AStar: %zu", global_stats.EPT_results_from_astar);
        if (global_stats.EPT_results_from_reuse + global_stats.EPT_results_from_astar > 0) {
            double astar_result_pct = 100.0 * global_stats.EPT_results_from_astar / 
                                     (global_stats.EPT_results_from_reuse + global_stats.EPT_results_from_astar);
            printf(" (%.1f%%)", astar_result_pct);
        }
        printf("\n");
        
        if (global_stats.EPT_reuse_attempt > 0) {
            double reuse_success_rate = 100.0 * global_stats.EPT_reuse_count / global_stats.EPT_reuse_attempt;
            printf("  - Reuse success rate: %.1f%%\n", reuse_success_rate);
            
            double reuse_attempt_rate = 100.0 * global_stats.EPT_reuse_attempt / total_ept_nodes;
            printf("  - Reuse attempt rate: %.1f%% (attempts / total nodes)\n", reuse_attempt_rate);
        }
        
        if (total_ept_nodes > 0) {
            double reuse_coverage = 100.0 * global_stats.EPT_reuse_count / total_ept_nodes;
            printf("  - Reuse coverage: %.1f%% of all EPT GED computations\n", reuse_coverage);
        }
        
        // ========== 4. 节点类型统计 ==========
        printf("\n---------- Node Type Statistics ----------\n");
        printf("  - Leaf nodes processed: %zu\n", global_stats.EPT_leaf_nodes_processed);
        printf("  - Internal nodes processed: %zu\n", global_stats.EPT_internal_nodes_processed);
        printf("  - Nodes with completed IDs: %zu\n", global_stats.EPT_nodes_with_completed_ids);
        
        size_t total_nodes_processed = global_stats.EPT_leaf_nodes_processed + 
                                      global_stats.EPT_internal_nodes_processed;
        if (total_nodes_processed > 0) {
            double leaf_percentage = 100.0 * global_stats.EPT_leaf_nodes_processed / total_nodes_processed;
            printf("  - Leaf node percentage: %.1f%%\n", leaf_percentage);
        }
        
        // ========== 7. 操作类型统计 ==========
        printf("\n---------- Operation Type Statistics ----------\n");
        for (const auto& [op_type, count] : global_stats.EPT_op_type_count) {
            std::string op_name;
            switch(op_type) {
                case EditOperation::NODE_SUBSTITUTION: op_name = "NODE_SUBSTITUTION"; break;
                case EditOperation::NODE_DELETION: op_name = "NODE_DELETION"; break;
                case EditOperation::NODE_INSERTION: op_name = "NODE_INSERTION"; break;
                case EditOperation::EDGE_SUBSTITUTION: op_name = "EDGE_SUBSTITUTION"; break;
                case EditOperation::EDGE_DELETION: op_name = "EDGE_DELETION"; break;
                case EditOperation::EDGE_INSERTION: op_name = "EDGE_INSERTION"; break;
                default: op_name = "UNKNOWN"; break;
            }
            
            // 修复：使用 find() 而不是 [] 操作符来避免 const 限定符问题
            size_t reused = 0;
            auto reuse_it = global_stats.EPT_reuse_by_op_type.find(op_type);
            if (reuse_it != global_stats.EPT_reuse_by_op_type.end()) {
                reused = reuse_it->second;
            }
            
            double reuse_rate = (count > 0) ? (100.0 * reused / count) : 0.0;
            
            printf("  - %s: %zu (reused: %zu, %.1f%%)\n", 
                   op_name.c_str(), count, reused, reuse_rate);
        }
        
        // ========== 8. 验证统计（如果启用了验证）==========
        if (global_stats.EPT_reuse_verifications > 0) {
            printf("\n---------- Reuse Correctness Verification ----------\n");
            printf("Verifications performed: %zu\n", global_stats.EPT_reuse_verifications);
            printf("  - Correct: %zu", global_stats.EPT_reuse_correct);
            
            double correct_rate = 100.0 * global_stats.EPT_reuse_correct / 
                                 global_stats.EPT_reuse_verifications;
            printf(" (%.2f%%)\n", correct_rate);
            
            printf("  - Incorrect: %zu", global_stats.EPT_reuse_incorrect);
            printf(" (%.2f%%)\n", (100.0 - correct_rate));
            
            // ========== 新增：简化的GED<=tau统计 ==========
            printf("\n---------- GED <= tau Statistics ----------\n");
            printf("Reuse found GED <= tau: %zu times\n", global_stats.EPT_reuse_found_ged_le_tau);
            printf("AStar found GED <= tau: %zu times\n", global_stats.EPT_astar_found_ged_le_tau);
            
            if (global_stats.EPT_astar_found_ged_le_tau > 0) {
                double ratio = (double)global_stats.EPT_reuse_found_ged_le_tau / 
                              global_stats.EPT_astar_found_ged_le_tau;
                printf("Ratio (Reuse/AStar): %.4f (%.2f%%)\n", ratio, ratio * 100);
                
                if (ratio < 1.0) {
                    printf("=> Reuse found %.1f%% fewer results than AStar\n", (1.0 - ratio) * 100);
                } else if (ratio > 1.0) {
                    printf("=> Reuse found %.1f%% more results than AStar\n", (ratio - 1.0) * 100);
                } else {
                    printf("=> Reuse found exactly the same number of results as AStar\n");
                }
            }
            
            // ========== 验证时间和加速比 ==========
            printf("\n---------- Verification Time Analysis ----------\n");
            printf("Total verification AStar time: %.3f seconds\n", global_stats.EPT_verification_astar_time);

            if (global_stats.EPT_reuse_verifications > 0) {
                double avg_verify_time = global_stats.EPT_verification_astar_time / global_stats.EPT_reuse_verifications;
                printf("Average time per verification: %.3f ms\n", avg_verify_time * 1000);
            }

            // 计算总的复用时间与验证时间的比较
            if (global_stats.EPT_reuse_success_time > 0 && global_stats.EPT_verification_astar_time > 0) {
                double reuse_vs_verify_speedup = global_stats.EPT_verification_astar_time / global_stats.EPT_reuse_success_time;
                printf("\n*** Reuse Speedup Comparison ***\n");
                printf("Total reuse time: %.3f seconds\n", global_stats.EPT_reuse_success_time);
                printf("Total verification time (standard AStar): %.3f seconds\n", global_stats.EPT_verification_astar_time);
                printf("=> Reuse is %.2fx faster than standard AStar\n", reuse_vs_verify_speedup);

                // 计算如果不使用复用，需要的总时间
                double time_without_reuse = global_stats.EPT_astar_time - global_stats.EPT_reuse_success_time +
                                          global_stats.EPT_verification_astar_time;
                double total_speedup = time_without_reuse / global_stats.EPT_astar_time;
                printf("\nTotal time with reuse: %.3f seconds\n", global_stats.EPT_astar_time);
                printf("Total time without reuse (estimated): %.3f seconds\n", time_without_reuse);
                printf("=> Overall speedup with reuse: %.2fx\n", total_speedup);
            }

            // ========== EXP-5: Search Tree Reuse Effectiveness (表格输出) ==========
            if (global_stats.EPT_baseline_app_count > 0 && global_stats.EPT_reuse_count > 0) {
                double avg_reuse_time_ms = (global_stats.EPT_reuse_success_time / global_stats.EPT_reuse_count) * 1000.0;
                double avg_baseline_time_ms = (global_stats.EPT_baseline_app_time / global_stats.EPT_baseline_app_count) * 1000.0;
                double speedup = avg_baseline_time_ms / avg_reuse_time_ms;

                printf("\n");
                printf("================================================================================\n");
                printf("                    EXP-5: Search Tree Reuse Effectiveness                      \n");
                printf("================================================================================\n");
                printf("| %-20s | %-15s | %-15s | %-10s |\n", "Method", "Count", "Avg Time (ms)", "Speedup");
                printf("|----------------------|-----------------|-----------------|------------|\n");
                printf("| %-20s | %15zu | %15.4f | %10s |\n",
                       "With Reuse", global_stats.EPT_reuse_count, avg_reuse_time_ms, "-");
                printf("| %-20s | %15zu | %15.4f | %9.2fx |\n",
                       "Baseline (AppForComp)", global_stats.EPT_baseline_app_count, avg_baseline_time_ms, speedup);
                printf("================================================================================\n");
                printf("=> Search tree reuse achieves %.2fx speedup\n", speedup);
                printf("================================================================================\n");
            }
            
            if (!global_stats.EPT_ged_diff_distribution.empty()) {
                printf("\nGED difference distribution (reuse - standard):\n");
                for (const auto& [diff, count] : global_stats.EPT_ged_diff_distribution) {
                    printf("  - Diff %3d: %6zu times", diff, count);
                    
                    if (global_stats.EPT_reuse_incorrect > 0) {
                        double percent = 100.0 * count / global_stats.EPT_reuse_incorrect;
                        printf(" (%.1f%% of errors)", percent);
                    }
                    printf("\n");
                }
                
                // 计算平均差异
                if (global_stats.EPT_reuse_incorrect > 0) {
                    double total_diff = 0;
                    for (const auto& [diff, count] : global_stats.EPT_ged_diff_distribution) {
                        total_diff += diff * count;
                    }
                    double avg_diff = total_diff / global_stats.EPT_reuse_incorrect;
                    printf("\nAverage GED difference: %.3f\n", avg_diff);
                }
            }
        }
        
        printf("=============================================\n");
    }
    
    // ========== 9. LB和AStar时间分析 ==========
    if (global_stats.total_lb_time() > 0 || global_stats.total_astar_time() > 0) {
        printf("\n========== Time Breakdown (LB vs AStar) ==========\n");
        
        // 计算总的算法时间（LB + AStar）
        double total_algorithm_time = global_stats.total_lb_time() + global_stats.total_astar_time();
        
        printf("Total algorithm time: %.3f seconds\n", total_algorithm_time);
        
        // LB时间统计
        printf("\nLower Bound (LB) Time:\n");
        printf("  - Total LB time: %.3f seconds", global_stats.total_lb_time());
        if (total_algorithm_time > 0) {
            double lb_percentage = 100.0 * global_stats.total_lb_time() / total_algorithm_time;
            printf(" (%.1f%% of algorithm time)", lb_percentage);
        }
        printf("\n");
        
        if (global_stats.ND_lb_time > 0 || global_stats.EPT_lb_time > 0) {
            printf("    - ND LB time:  %.3f seconds\n", global_stats.ND_lb_time);
            printf("    - EPT LB time: %.3f seconds\n", global_stats.EPT_lb_time);
        }
        
        // AStar时间统计
        printf("\nAStar Time:\n");
        printf("  - Total AStar time: %.3f seconds", global_stats.total_astar_time());
        if (total_algorithm_time > 0) {
            double astar_percentage = 100.0 * global_stats.total_astar_time() / total_algorithm_time;
            printf(" (%.1f%% of algorithm time)", astar_percentage);
        }
        printf("\n");
        
        if (global_stats.ND_astar_time > 0 || global_stats.EPT_astar_time > 0) {
            printf("    - ND AStar time:  %.3f seconds\n", global_stats.ND_astar_time);
            printf("    - EPT AStar time: %.3f seconds\n", global_stats.EPT_astar_time);
        }
        
        // 占总算法时间的比例
        if (total_algorithm_time > 0) {
            printf("\nAs percentage of total algorithm time:\n");
            printf("  - LB time:    %.1f%% of total\n",
                   (100.0 * global_stats.total_lb_time() / total_algorithm_time));
            printf("  - AStar time: %.1f%% of total\n",
                   (100.0 * global_stats.total_astar_time() / total_algorithm_time));
        }
        
        // 平均每次调用的时间
        if (global_stats.total_lb_count() > 0) {
            double avg_lb_time = global_stats.total_lb_time() / global_stats.total_lb_count();
            printf("\n  - Average time per LB call: %.3f ms\n", (avg_lb_time * 1000));
        }
        
        if (global_stats.total_astar_count() > 0) {
            double avg_astar_time = global_stats.total_astar_time() / global_stats.total_astar_count();
            printf("  - Average time per AStar call: %.3f ms\n", (avg_astar_time * 1000));
        }

        // ========== 优化效果统计 ==========
        printf("\n========== Optimization Effectiveness Statistics ==========\n");

        // 与EPT总节点数对比（重点：避免了多少计算）
        if (global_stats.EPT_total_nodes_in_used_epts > 0) {
            size_t nodes_pruned = global_stats.subtree_pruning_avoided_nodes +
                                  global_stats.lb_pruning_count;
            size_t nodes_computed = global_stats.EPT_nodes_computed;

            printf("\nComputation Reduction (vs EPT Total Nodes):\n");
            printf("  - Total nodes in used EPTs: %zu\n", global_stats.EPT_total_nodes_in_used_epts);
            printf("  - Nodes that did computation: %zu (%.1f%% of EPT total)  [includes filter + GED]\n",
                   nodes_computed,
                   100.0 * nodes_computed / global_stats.EPT_total_nodes_in_used_epts);
            printf("    - Of which computed full GED: %zu (%.1f%% of computed, %.1f%% of EPT total)\n",
                   global_stats.EPT_astar_count,
                   nodes_computed > 0 ? 100.0 * global_stats.EPT_astar_count / nodes_computed : 0.0,
                   100.0 * global_stats.EPT_astar_count / global_stats.EPT_total_nodes_in_used_epts);
            printf("  - Nodes pruned (avoided computation): %zu (%.1f%% of EPT total)\n",
                   nodes_pruned,
                   100.0 * nodes_pruned / global_stats.EPT_total_nodes_in_used_epts);

            printf("\n  Pruning Breakdown:\n");
            printf("    - LB Propagation: %zu (%.1f%% of EPT total)\n",
                   global_stats.lb_pruning_count,
                   100.0 * global_stats.lb_pruning_count / global_stats.EPT_total_nodes_in_used_epts);
            printf("    - Subtree pruned: %zu (%.1f%% of EPT total)\n",
                   global_stats.subtree_pruning_avoided_nodes,
                   100.0 * global_stats.subtree_pruning_avoided_nodes / global_stats.EPT_total_nodes_in_used_epts);
        }

        // Subtree pruning 统计
        printf("\nSubtree Pruning:\n");
        printf("  - Pruning decisions: %zu\n", global_stats.subtree_pruning_decisions);
        printf("    - On leaf nodes: %zu (%.1f%% of decisions)\n",
               global_stats.subtree_pruning_on_leaf_nodes,
               global_stats.subtree_pruning_decisions > 0 ?
               100.0 * global_stats.subtree_pruning_on_leaf_nodes / global_stats.subtree_pruning_decisions : 0.0);
        printf("    - On internal nodes: %zu (%.1f%% of decisions)\n",
               global_stats.subtree_pruning_decisions - global_stats.subtree_pruning_on_leaf_nodes,
               global_stats.subtree_pruning_decisions > 0 ?
               100.0 * (global_stats.subtree_pruning_decisions - global_stats.subtree_pruning_on_leaf_nodes) / global_stats.subtree_pruning_decisions : 0.0);
        printf("  - Avoided descendant nodes: %zu\n", global_stats.subtree_pruning_avoided_nodes);
        if (global_stats.subtree_pruning_decisions > 0) {
            printf("  - Average descendants avoided per decision: %.1f nodes\n",
                   (double)global_stats.subtree_pruning_avoided_nodes / global_stats.subtree_pruning_decisions);
        }


        // NetDag缓存复用统计
        printf("\nNetDag GED Cache Reuse:\n");
        printf("  - Root nodes reused NetDag GED: %zu\n", global_stats.root_netdag_ged_reuse_count);
        if (global_stats.root_netdag_ged_reuse_count > 0 && global_stats.EPT_results_from_astar > 0) {
            printf("  - Percentage of results from NetDag cache: %.1f%%\n",
                   100.0 * global_stats.root_netdag_ged_reuse_count / (global_stats.EPT_results_from_astar + global_stats.root_netdag_ged_reuse_count));
        }

        // 总体优化效果
        printf("\nTotal Optimization Impact:\n");
        printf("  - Subtree pruning decisions: %zu\n", global_stats.subtree_pruning_decisions);
        printf("  - Avoided descendant nodes: %zu\n", global_stats.subtree_pruning_avoided_nodes);
        printf("  - App_test computations skipped by LB propagation: %zu\n", global_stats.lb_pruning_count);
        printf("  - Root nodes reused NetDag GED: %zu\n", global_stats.root_netdag_ged_reuse_count);

        if (global_stats.subtree_pruning_decisions > 0) {
            printf("  - Average descendants avoided per decision: %.1f nodes\n",
                   (double)global_stats.subtree_pruning_avoided_nodes / global_stats.subtree_pruning_decisions);
        }

        printf("==================================================\n");
    }
    
    // ========== 10. NetDag & EPT 详细统计 ==========
    printf("\n=== NetDag & EPT Detailed Stats ===\n");
    global_stats.print_summary(std::cout);  // 这个需要保留cout，因为是调用其他函数
    
    // ========== 11. 最慢查询排序（可选）==========
    if (!query_time_pairs.empty() && query_time_pairs.size() <= 100) {
        // 创建副本进行排序，避免修改原始数据
        std::vector<std::pair<int, double>> sorted_pairs = query_time_pairs;
        std::sort(sorted_pairs.begin(), sorted_pairs.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        printf("\n=== Top 10 Slowest Queries ===\n");
        for (size_t i = 0; i < std::min(size_t(10), sorted_pairs.size()); ++i) {
            printf("Query %d: %f seconds\n", sorted_pairs[i].first, sorted_pairs[i].second);
        }
    }
    
    printf("\n=============================================\n");
}

void GismaSearchEngine::save_experiment_results(
    const std::string& dataset_name,
    double tau,
    const SearchStats& global_stats,
    int total_queries_processed,
    int total_results_found,
    int queries_with_non_empty_results,
    double total_query_time_sum,
    double total_time,
    double total_recall,
    double total_precision,
    double total_iou,
    int valid_query_count,
    int total_intersection_count,
    int total_ground_truth_count,
    const std::vector<QueryDetails>& query_details,
    bool is_parallel)
{
    // 创建结果目录
    std::string exp_dir;
    std::string exp_name;
    char timestamp[64] = "N/A";

    // 如果设置了 experiment_base_dir，直接使用它作为输出目录
    if (!experiment_base_dir.empty()) {
        exp_dir = experiment_base_dir;
        std::filesystem::create_directories(exp_dir);
        printf("\n[Saving Results] Using experiment base directory: %s\n", exp_dir.c_str());

        // 使用目录名作为实验名称
        exp_name = std::filesystem::path(exp_dir).filename().string();

        // 生成时间戳
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        LOCALTIME_SAFE(&tm_now, &time_t_now);
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_now);
    } else {
        // 否则使用默认的目录命名方式
        std::string base_dir = "./experiment_results";
        std::filesystem::create_directories(base_dir);

        // 生成时间戳
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        LOCALTIME_SAFE(&tm_now, &time_t_now);
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_now);

        // 生成实验名称：dataset_method_tau_timestamp
        std::ostringstream exp_name_stream;
        exp_name_stream << dataset_name << "_" << search_method
                        << "_tau" << static_cast<int>(tau)
                        << "_" << timestamp;
        exp_name = exp_name_stream.str();

        // 创建实验专属目录
        exp_dir = base_dir + "/" + exp_name;
        std::filesystem::create_directories(exp_dir);

        printf("\n[Saving Results] Experiment directory: %s\n", exp_dir.c_str());
    }

    // ========== 1. 保存总体统计 (summary.txt) ==========
    std::string summary_path = exp_dir + "/summary.txt";
    std::ofstream summary_file(summary_path);

    if (!summary_file.is_open()) {
        std::cerr << "Failed to create summary file: " << summary_path << std::endl;
        return;
    }

    summary_file << "=================== Experiment Summary ===================\n";
    summary_file << "Experiment Name: " << exp_name << "\n";
    summary_file << "Timestamp: " << timestamp << "\n\n";

    summary_file << "--- Configuration ---\n";
    summary_file << "Dataset: " << dataset_name << "\n";
    summary_file << "Search Method: " << search_method << "\n";
    summary_file << "Tau Search: " << tau << "\n";
    summary_file << "Tau Index: " << tau_index << "\n";
    summary_file << "Query Range: [" << q_start << ", " << q_end << "]\n";
    summary_file << "Parallel Mode: " << (is_parallel ? "Yes" : "No") << "\n\n";

    summary_file << "--- Performance Metrics ---\n";
    summary_file << "Total Queries Processed: " << total_queries_processed << "\n";
    summary_file << "Total Results Found: " << total_results_found << " / " << total_ground_truth_count << "\n";
    summary_file << "Queries with Non-Empty Results: " << queries_with_non_empty_results << "\n";

    double sum_of_query_times = global_stats.total_lb_time() + global_stats.total_astar_time();
    summary_file << "Total Query Time: " << sum_of_query_times << " seconds\n";
    summary_file << "Average Query Time: " << (total_queries_processed > 0 ? sum_of_query_times / total_queries_processed : 0.0) << " seconds\n\n";

    // 判断是否使用两层索引结构 (Gisma)
    bool uses_two_layer_index = (search_method == "Gisma" || search_method == "Gisma-default" || search_method == "Gisma-no-reuse");

    summary_file << "--- Time Breakdown ---\n";
    if (uses_two_layer_index) {
        // Gisma: 显示 ND + EPT 分解
        double nd_total = global_stats.ND_lb_time + global_stats.ND_astar_time;
        double ept_total = global_stats.EPT_lb_time + global_stats.EPT_astar_time;
        summary_file << "ND Total Time: " << nd_total << " seconds\n";
        summary_file << "  - ND LB Time: " << global_stats.ND_lb_time << " seconds\n";
        summary_file << "  - ND AStar Time: " << global_stats.ND_astar_time << " seconds\n";
        summary_file << "EPT Total Time: " << ept_total << " seconds\n";
        summary_file << "  - EPT LB Time: " << global_stats.EPT_lb_time << " seconds\n";
        summary_file << "  - EPT AStar Time: " << global_stats.EPT_astar_time << " seconds\n\n";
    } else {
        // App-BMao, AStar-BMao 等: 只显示总的 LB + AStar
        double lb_time = global_stats.ND_lb_time + global_stats.EPT_lb_time;
        double astar_time = global_stats.ND_astar_time + global_stats.EPT_astar_time;
        summary_file << "Total LB Time: " << lb_time << " seconds\n";
        summary_file << "Total AStar Time: " << astar_time << " seconds\n\n";
    }

    summary_file << "--- Precision Metrics ---\n";
    double avg_recall = (valid_query_count > 0) ? (total_recall / valid_query_count) : 0.0;
    double avg_precision = (valid_query_count > 0) ? (total_precision / valid_query_count) : 0.0;
    double avg_iou = (valid_query_count > 0) ? (total_iou / valid_query_count) : 0.0;
    double overall_recall = (total_ground_truth_count > 0) ? ((double)total_intersection_count / total_ground_truth_count) : 0.0;

    summary_file << "Average Recall (per query): " << avg_recall << " (over " << valid_query_count << " queries)\n";
    summary_file << "Average Precision (per query): " << avg_precision << "\n";
    summary_file << "Average IoU (per query): " << avg_iou << "\n";
    summary_file << "Overall Recall (combined): " << overall_recall << "\n\n";

    if (uses_two_layer_index) {
        summary_file << "--- Candidate Set Statistics ---\n";
        summary_file << "Avg EPT Nodes Processed: " << (total_queries_processed > 0 ? (double)(global_stats.EPT_leaf_nodes_processed + global_stats.EPT_internal_nodes_processed) / total_queries_processed : 0.0) << "\n\n";
    }

    summary_file << "--- LB & AStar Call Counts ---\n";
    if (uses_two_layer_index) {
        // Gisma: 分别显示 ND 和 EPT
        summary_file << "ND LB Calls:\n";
        summary_file << "  - Size LB: " << global_stats.ND_size_lb_count << "\n";
        summary_file << "  - Vertex LB: " << global_stats.ND_vertex_lb_count << "\n";
        summary_file << "  - Edge LB: " << global_stats.ND_edge_lb_degree_count << "\n";
        summary_file << "ND AStar Calls: " << global_stats.ND_astar_count << "\n\n";

        summary_file << "EPT LB Calls:\n";
        summary_file << "  - Size LB: " << global_stats.EPT_size_lb_count << "\n";
        summary_file << "  - Vertex LB: " << global_stats.EPT_vertex_lb_count << "\n";
        summary_file << "  - Edge LB: " << global_stats.EPT_edge_lb_degree_count << "\n";
        summary_file << "EPT AStar Calls: " << global_stats.EPT_astar_count << "\n\n";
    } else {
        // App-BMao, AStar-BMao: 显示总计
        int total_size_lb = global_stats.ND_size_lb_count + global_stats.EPT_size_lb_count;
        int total_vertex_lb = global_stats.ND_vertex_lb_count + global_stats.EPT_vertex_lb_count;
        int total_edge_lb = global_stats.ND_edge_lb_degree_count + global_stats.EPT_edge_lb_degree_count;
        int total_astar = global_stats.ND_astar_count + global_stats.EPT_astar_count;

        summary_file << "Total LB Calls:\n";
        summary_file << "  - Size LB: " << total_size_lb << "\n";
        summary_file << "  - Vertex LB: " << total_vertex_lb << "\n";
        summary_file << "  - Edge LB: " << total_edge_lb << "\n";
        summary_file << "Total AStar Calls: " << total_astar << "\n\n";
    }

    summary_file.close();
    printf("[Saved] %s\n", summary_path.c_str());

    // ========== 2. 保存每个query的详细信息 (query_details.csv) ==========
    std::string details_path = exp_dir + "/query_details.csv";
    std::ofstream details_file(details_path);

    if (!details_file.is_open()) {
        std::cerr << "Failed to create query details file: " << details_path << std::endl;
        return;
    }

    details_file << "Query_ID,Time(s),Recall,Precision,IoU\n";

    for (const auto& qd : query_details) {
        details_file << qd.query_id << "," << qd.time;
        if (qd.has_ground_truth) {
            details_file << "," << qd.recall << "," << qd.precision << "," << qd.iou;
        } else {
            details_file << ",N/A,N/A,N/A";
        }
        details_file << "\n";
    }

    details_file.close();
    printf("[Saved] %s\n", details_path.c_str());

    // ========== 3. 保存配置参数 (config.txt) ==========
    std::string config_path = exp_dir + "/config.txt";
    std::ofstream config_file(config_path);

    if (!config_file.is_open()) {
        std::cerr << "Failed to create config file: " << config_path << std::endl;
        return;
    }

    config_file << "dataset=" << dataset_name << "\n";
    config_file << "search_method=" << search_method << "\n";
    config_file << "tau=" << tau << "\n";
    config_file << "tau_index=" << tau_index << "\n";
    config_file << "error_tolerance_search=" << error_tolerance_search << "\n";
    config_file << "q_start=" << q_start << "\n";
    config_file << "q_end=" << q_end << "\n";
    config_file << "is_parallel=" << is_parallel << "\n";

    config_file.close();
    printf("[Saved] %s\n", config_path.c_str());

    printf("[Saving Complete] All results saved to: %s\n\n", exp_dir.c_str());
}

// 保存单个query的JSON结果（实时保存）
void GismaSearchEngine::save_single_query_json(
    const QueryDetails& query_detail,
    double tau,
    const std::string& output_dir
) {
    // 生成文件名：query_<id>.json
    std::ostringstream filename;
    filename << "query_" << query_detail.query_id << ".json";
    std::string filepath = output_dir + "/" + filename.str();

    // 调用QueryDetails的save_to_json方法
    query_detail.save_to_json(filepath, tau);
}

// QueryDetails::save_to_json implementation
void QueryDetails::save_to_json(const std::string& filepath, double tau) const {
    std::ofstream json_file(filepath);
    if (!json_file.is_open()) {
        std::cerr << "Failed to create JSON file: " << filepath << std::endl;
        return;
    }

    json_file << "{\n";
    json_file << "  \"query_id\": " << query_id << ",\n";
    json_file << "  \"tau\": " << tau << ",\n";
    json_file << "  \"query_size\": {\n";
    json_file << "    \"nodes\": " << query_nodes << ",\n";
    json_file << "    \"edges\": " << query_edges << "\n";
    json_file << "  },\n";
    json_file << "  \"time\": {\n";
    json_file << "    \"total\": " << std::fixed << std::setprecision(6) << time << ",\n";
    json_file << "    \"lb_time\": " << lb_time << ",\n";
    json_file << "    \"astar_time\": " << astar_time << ",\n";
    json_file << "    \"nd_lb_time\": " << nd_lb_time << ",\n";
    json_file << "    \"nd_astar_time\": " << nd_astar_time << ",\n";
    json_file << "    \"ept_lb_time\": " << ept_lb_time << ",\n";
    json_file << "    \"ept_astar_time\": " << ept_astar_time << "\n";
    json_file << "  },\n";
    json_file << "  \"results\": {\n";
    json_file << "    \"count\": " << result_count << ",\n";
    json_file << "    \"has_ground_truth\": " << (has_ground_truth ? "true" : "false") << ",\n";
    json_file << "    \"has_results\": " << (has_results ? "true" : "false") << ",\n";
    json_file << "    \"recall\": " << (recall >= 0 ? recall : -1) << ",\n";
    json_file << "    \"precision\": " << (precision >= 0 ? precision : -1) << ",\n";
    json_file << "    \"iou\": " << (iou >= 0 ? iou : -1) << "\n";
    json_file << "  },\n";
    json_file << "  \"ept_stats\": {\n";
    json_file << "    \"total_ept_nodes\": " << total_ept_nodes << ",\n";
    json_file << "    \"nodes_computed\": " << nodes_computed << ",\n";
    json_file << "    \"nodes_pruned\": " << nodes_pruned << ",\n";
    json_file << "    \"lb_pruning_count\": " << lb_pruning_count << ",\n";
    json_file << "    \"lb_propagation_count\": " << lb_propagation_count << ",\n";
    json_file << "    \"subtree_pruned\": " << subtree_pruned << "\n";
    json_file << "  }\n";
    json_file << "}\n";

    json_file.close();
}

// 单个query搜索实现
