#include "GismaSearchEngine.h"
#include "EditPathTree.h"
#include "Application.h"
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

// Global coverage sets (defined in experiment_mode_impl.cpp)
extern std::unordered_set<int> g_ept_coverage;
extern std::unordered_set<int> g_extra_coverage;

// Cross-platform localtime function
#ifdef _WIN32
    #define LOCALTIME_SAFE(tm_ptr, time_ptr) localtime_s(tm_ptr, time_ptr)
#else
    #define LOCALTIME_SAFE(tm_ptr, time_ptr) localtime_r(time_ptr, tm_ptr)
#endif


// Constructor implementation - Test modification
GismaSearchEngine::GismaSearchEngine(
    std::shared_ptr<NetDag> net_dag,
    double tau_index,
    // double tau,  // removed
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
    bool _verify_reuse_baseline,
    bool _chain_reuse) : net_dag(net_dag),
                                        tau_index(tau_index),
                                        // tau(tau),  // removed
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
                                        verify_reuse_baseline(_verify_reuse_baseline),
                                        chain_reuse(_chain_reuse),
                                        disable_subtree_pruning(false),
                                        disable_lb_propagation(false)
{
}

GismaSearchEngine::~GismaSearchEngine() {
    // Explicitly clean up EPT manager resources
    if (ept_manager) {
        // Let EPT manager clean up all EPT resources
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

double GismaSearchEngine::compute_ged_online_by_pyg(int i, int query_index)
{
    return get_ged_from_matrix(i, query_index);
}

std::vector<int> GismaSearchEngine::BMao_scan_on_subset(
    std::shared_ptr<Node> query_node,
    const std::vector<int> &subset,
    double tau,
    SearchStats &stats)
{
    // 1) Get the query graph
    Graph *query_graph = query_node->graph.get();
    std::vector<int> exact_results;

    // 2) Iterate graph IDs in subset
    for(int node_id : subset)
    {
        if(node_id < 0 || node_id >= (int)db.size()) {
            continue; // guard
        }
        Graph *db_graph = db[node_id];
        if(!db_graph) {
            continue;
        }

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

        // ========== LB computation ==========
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

        Application app((ui)tau, "BMao", app_max_iter);
        app.init(db_graph, query_graph);
        app.set_disable_lsa_pruning(disable_lsa_pruning);
        int ged_res = app.App();

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= (int)tau) {
            exact_results.push_back(node_id);
        }
    }

    return exact_results;
}

std::vector<int> GismaSearchEngine::BMao_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // Results
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

        // ========== LB computation ==========
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

std::vector<int> GismaSearchEngine::App_BMao_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // Results
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB computation ==========
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

// Pure Base method using AStar_baseline() (exact A*)
std::vector<int> GismaSearchEngine::AStar_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB computation ==========
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
        app.init(db_graph, query_graph);
        int ged_res = app.AStar_baseline();  // using exact A*

        auto astar_end = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        if (ged_res <= tau)
        {
            exact_results_within_tau.push_back(static_cast<int>(node_id));
        }
    }

    return exact_results_within_tau;
}

// Pure Base method using AStar() with LSa pruning disabled (fair comparison with App_baseline)
std::vector<int> GismaSearchEngine::AStar_scan_no_lsa_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB computation ==========
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
        app.init(db_graph, query_graph);
        app.set_disable_lsa_pruning(true);  // Disable LSa pruning
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

    // Results
    std::vector<int> exact_results_within_tau;

    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];

        // ========== LB computation ==========
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

// Base+GS: use GS to select anchors, then iterate their cluster and exact_cluster, verify with App-BMao
std::vector<int> GismaSearchEngine::Base_GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    // 1) Use GS_search to get candidate anchors
    auto candidate_anchors_with_results = GS_search(query_node, tau, stats);

    if (candidate_anchors_with_results.empty()) {
        return exact_results_within_tau;
    }

    // 2) Collect all candidate anchor complete_ids (from EPT) and graph IDs in cluster not in EPT
    std::unordered_set<int> candidate_graph_ids;

    for (const auto& [anchor_id, netdag_lb, netdag_ged] : candidate_anchors_with_results) {
        auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
        if (!anchor_ptr) {
            continue;
        }

        // First add anchor itself
        candidate_graph_ids.insert(anchor_id);

        // Collect all graph IDs from nodes_in_cluster_vec (these are graphs not in EPT)
        for (const auto& [dist, graph_id] : anchor_ptr->nodes_in_cluster_vec) {
            candidate_graph_ids.insert(graph_id);
        }

        // Collect complete_ids from EPT (graphs with exact GED already computed)
        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
        if (ept && !ept->tree_nodes.empty()) {
            // Iterate all EPT nodes, collect complete_ids
            for (const auto& tree_node : ept->tree_nodes) {
                for (int complete_id : tree_node.completed_db_graph_ids) {
                    candidate_graph_ids.insert(complete_id);
                }
            }
        }
    }

    // 3) Sort candidate graph IDs to improve cache locality, then verify using App-BMao
    std::vector<int> sorted_candidates(candidate_graph_ids.begin(), candidate_graph_ids.end());
    std::sort(sorted_candidates.begin(), sorted_candidates.end());
    for (int node_id : sorted_candidates) {
        // Check if node_id is valid
        if (node_id < 0 || node_id >= static_cast<int>(db.size())) {
            continue;
        }

        Graph *db_graph = db[node_id];
        if (!db_graph) {
            continue;
        }

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

#if USE_FILTERS_FOR_APP_BMAO
        // ========== LB computation ==========
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

// Base+SS: skip GS hierarchical navigation, compute distance to all anchors directly, then use SS
std::vector<int> GismaSearchEngine::Base_SS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;

    // Debug: Store selected anchors for Query 9
    std::vector<int> base_ss_selected_anchors;
    std::unordered_set<int> result_set;  // for deduplication

    // Base+SS: Similar to Gisma but skips GS hierarchical navigation
    // 1. Compute distance from all anchors to query
    // 2. Select anchors within distance range
    // 3. For selected anchors:Search EPT + Search cluster

    std::vector<std::tuple<int, double, int>> candidate_anchors;  // (anchor_id, netdag_lb, netdag_ged)

    // Iterate all anchors, compute distance
    for (const auto& anchor_ptr : net_dag->anchors) {
        if (!anchor_ptr) continue;

        int anchor_id = anchor_ptr->node_id;
        Graph *anchor_graph = anchor_ptr->graph.get();
        if (!anchor_graph) continue;

        // Compute distance from anchor to query
        ui netdag_lb, netdag_ged;

        // NDC statistics: count each visited node once (deduplicated)
        stats.ND_ndc_count++;

        // Three NetDag modes (consistent with GS_search)
        if (nd_mode == "filters") {
            // Use traditional lower bound filters
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_lb_count++;
            netdag_lb = query_graph->ged_lower_bound_filter(
                anchor_graph, static_cast<ui>(net_dag->alpha + tau), vM.size(), eM.size(), max_n);
            netdag_ged = INF;

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;
        } else if (nd_mode == "astar") {
            // Use App_test to compute exact distance
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_astar_count++;
            Application app(static_cast<ui>(tau), "BMao", app_max_iter);
            app.init(anchor_graph, query_graph);
            netdag_ged = app.App_test(nullptr, nullptr);
            netdag_lb = app.get_overall_lb();
            stats.ND_app_test_count++;

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_astar_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;
        } else if (nd_mode == "filters_astar") {
            // First compute lb using filters, if lb<=tau then use AStar for exact GED
            auto t0 = std::chrono::high_resolution_clock::now();

            stats.ND_lb_count++;
            netdag_lb = query_graph->ged_lower_bound_filter(
                anchor_graph, static_cast<ui>(net_dag->alpha + tau), vM.size(), eM.size(), max_n);

            auto t1 = std::chrono::high_resolution_clock::now();
            stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

            if (netdag_lb > (net_dag->alpha + tau) * nd_filter_ratio) continue;

            // If lb<=tau, call AStar for exact GED
            if (netdag_lb <= tau) {
                auto t2 = std::chrono::high_resolution_clock::now();

                stats.ND_astar_count++;
                Application app(static_cast<ui>(tau), "BMao", app_max_iter);
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
            // Unknown mode, use filters mode as default
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

    // Perform EPT search and cluster search on selected anchors
    for (const auto& [anchor_id, netdag_lb, netdag_ged] : candidate_anchors) {
        auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
        if (!anchor_ptr) continue;

        // 0. Check if anchor itself satisfies the condition
        {
            stats.EPT_ndc_count++;  // NDC: checking anchor itself
            auto lb_start = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_count++;
            ui lb = query_graph->ged_lower_bound_filter(
                db[anchor_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);
            auto lb_end = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

            if (lb <= tau) {
                // Further verify with A*
                stats.EPT_astar_count++;
                auto astar_start = std::chrono::high_resolution_clock::now();

                Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                app.init(db[anchor_id], query_graph);
                app.set_disable_lsa_pruning(disable_lsa_pruning);
                int ged_res = app.App();

                auto astar_end = std::chrono::high_resolution_clock::now();
                stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

                if (ged_res <= tau) {
                    result_set.insert(anchor_id);

                    // IMPORTANT: if anchor itself satisfies condition (GED=0), also add completed_db_graph_ids from EPT root node
                    // These are graphs identical to anchor (GED=0)
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

        // 1. Search EPT
        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
        if (ept && !ept->tree_nodes.empty()) {
            auto ss_results = SS_search(query_node, anchor_id, netdag_lb, netdag_ged, tau, stats);
            for (int gid : ss_results) {
                result_set.insert(gid);
            }
        }

        // 2. Search cluster (extra_cluster_search)
        auto cluster_results = extra_cluster_search(query_node, anchor_id, tau, stats);
        for (int gid : cluster_results) {
            result_set.insert(gid);
        }
    }

    // Convert to vector and return
    exact_results_within_tau.assign(result_set.begin(), result_set.end());

    return exact_results_within_tau;
}

// Base_All_EPT: iterate all anchors without any filtering, directly do SS_search and extra_search
std::vector<int> GismaSearchEngine::Base_All_EPT_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    std::vector<int> exact_results_within_tau;
    std::unordered_set<int> result_set;  // for deduplication

    // Iterate all anchors without any filtering
    for (const auto& anchor_ptr : net_dag->anchors) {
        if (!anchor_ptr) continue;

        int anchor_id = anchor_ptr->node_id;

        // 0. Check if anchor itself satisfies the condition
        {
            stats.EPT_ndc_count++;  // NDC: checking anchor itself
            auto lb_start = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_count++;
            ui lb = query_graph->ged_lower_bound_filter(
                db[anchor_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);
            auto lb_end = std::chrono::high_resolution_clock::now();
            stats.EPT_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

            if (lb <= tau) {
                // Further verify with A*
                stats.EPT_astar_count++;
                auto astar_start = std::chrono::high_resolution_clock::now();

                Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                app.init(db[anchor_id], query_graph);
                app.set_disable_lsa_pruning(disable_lsa_pruning);
                int ged_res = app.App();

                auto astar_end = std::chrono::high_resolution_clock::now();
                stats.EPT_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

                if (ged_res <= tau) {
                    result_set.insert(anchor_id);

                    // IMPORTANT: if anchor itself satisfies condition (GED=0), also add completed_db_graph_ids from EPT root node
                    // These are graphs identical to anchor (GED=0)
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

        // 1. Search EPT (using SS_search)
        EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
        if (ept && !ept->tree_nodes.empty()) {
            // Pass netdag_lb=0, netdag_ged=-1 to indicate no NetDag filtering
            auto ss_results = SS_search(query_node, anchor_id, 0.0, -1, tau, stats);
            for (int gid : ss_results) {
                result_set.insert(gid);
            }
        }

        // 2. Search cluster (extra_cluster_search)
        auto cluster_results = extra_cluster_search(query_node, anchor_id, tau, stats);
        for (int gid : cluster_results) {
            result_set.insert(gid);
        }
    }

    // Convert to vector and return
    exact_results_within_tau.assign(result_set.begin(), result_set.end());

    return exact_results_within_tau;
}

std::vector<int> GismaSearchEngine::BMao_export_candidates(std::shared_ptr<Node> query_node, double tau, const std::string& output_file)
{
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;
    
    std::vector<int> candidates;
    
    auto filter_start = std::chrono::high_resolution_clock::now();
    
    // If output file is specified, open file stream
    std::ofstream outfile;
    bool write_to_file = !output_file.empty();
    if (write_to_file) {
        outfile.open(output_file, std::ios::app);  // append mode
        if (!outfile.is_open()) {
            printf("[ERROR] Cannot open output file: %s\n", output_file.c_str());
            return candidates;
        }
    }
    
    printf("[INFO] Query %d: Running BMao filtering for candidate export...\n", query_id);
    
    for (size_t node_id = 0; node_id < db.size(); ++node_id)
    {
        Graph *db_graph = db[node_id];
        
        // BMao lower bound filtering (same logic as in BMao_scan_search)
        ui lb = query_graph->ged_lower_bound_filter(
            db_graph, static_cast<ui>(tau), vM.size(), eM.size(), max_n);
        
        // If lower bound passes filter, add to candidate set
        if (lb <= tau) {
            candidates.push_back(static_cast<int>(node_id));
        }
    }
    
    auto filter_end = std::chrono::high_resolution_clock::now();
    double filter_time = std::chrono::duration<double, std::milli>(filter_end - filter_start).count();
    
    printf("[TIMING] Query %d BMao_Filtering: %.2fms, Candidates: %d/%d (%.2f%% reduction)\n", 
           query_id, filter_time, (int)candidates.size(), (int)db.size(), 
           (1.0 - (double)candidates.size() / db.size()) * 100.0);
    
    // Write to file
    if (write_to_file) {
        outfile << query_id << "," << tau << ",";
        for (size_t i = 0; i < candidates.size(); ++i) {
            outfile << candidates[i];
            if (i < candidates.size() - 1) outfile << ";";
        }
        outfile << "\n";
        outfile.close();
    }
    
    return candidates;
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

            // 1) Find all nodes in ground truth with actual ged <= tau
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
    }  // lock released here, A* computation no longer holds lock
    Graph *query_graph = query_node->graph.get();
    int query_id = query_node->node_id;

    // Results
    std::vector<int> exact_results_within_tau;

    for (auto node_id : ids_within_tau)
    {
        Graph *db_graph = db[node_id];

        // NDC statistics: count each visited node once (deduplicated)
        stats.EPT_ndc_count++;

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();

        stats.EPT_astar_count++;
        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
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
    // ========== 1) Basic checks and initialization ==========

    // candidate_anchor_ids has been replaced by candidate_anchors_with_lb

    // Get root node (Anchor)
    auto current_node = std::dynamic_pointer_cast<Anchor>(net_dag->root);
    if (!current_node)
    {
        std::cerr << "[GS_search] Error: root node is not Anchor or is null.\n";
        return {};
    }

    // Set initial phase
    double current_phase = static_cast<double>(current_node->children.rbegin()->first);

    // Store candidate anchors and their NetDag results, avoiding redundant computation during DFS
    std::vector<std::tuple<int, double, int>> candidate_anchors_with_results;  // (anchor_id, lb, ged_result)
    // Note: when USE_ND_FILTERS=0, App_test is used and ged_result is exact GED; when =1, ged_result=-1 means no exact result

    // Define variables for tracking the best node (outside loop for later use)
    double min_dist = std::numeric_limits<double>::infinity();
    int min_child_node_id = -1;

    // ========== Special case: root node directly connects to anchor layer ==========
    // When current_phase <= alpha, root node's children are the anchor layer,
    // no hierarchical navigation needed, directly collect candidates from root children
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

    // ========== 2) Loop: phase halves repeatedly until <= alpha ==========

    while (current_phase > net_dag->alpha)
    {
        double child_phase = current_phase / 2.0;
        // Reset per-round minimum distance and node ID
        min_dist = std::numeric_limits<double>::infinity();
        min_child_node_id = -1;

        // Thread-safe access to children map
        // Use shared_lock for concurrent read access
        std::vector<std::pair<int, double>> children_at_phase_copy;
        {
            std::shared_lock<std::shared_mutex> lock(net_dag->children_mutex);

            // Find the corresponding child_phase
            auto it = current_node->children.find(static_cast<int>(child_phase));
            if (it == current_node->children.end())
            {
                // If no corresponding child_phase found, exit
                return {};
            }

            // Copy children_at_phase to avoid holding lock during iteration
            children_at_phase_copy = it->second;
        } // Release lock here

        const auto &children_at_phase = children_at_phase_copy;

        // ========== 3) Iterate children_at_phase, LB check ==========

        for (const auto &[child_node_id, child_node_dist] : children_at_phase)
        {
            // First do a simple child_node_dist check
            // TEMPORARILY DISABLED: This pre-filtering may cause recall loss
            // if (child_node_dist <= 1.5 * current_phase + 2 * tau + 3 * error_tolerance_search)
            if (true)  // Check all children without pre-filtering
            {
                // Get childNode
                auto childNode = net_dag->nodes[child_node_id];
                if (!childNode || !childNode->graph)
                {
                    // Skip invalid node
                    continue;
                }

                // Get the graph for LB check
                Graph *db_graph = childNode->graph.get();
                Graph *query_graph = (query_node && query_node->graph)
                                         ? query_node->graph.get()
                                         : nullptr;
                if (!query_graph)
                {
                    continue;
                }

                // Declare outside for later use
                ui netdag_lb, netdag_ged;
                bool is_last_layer = (child_phase <= net_dag->alpha);

                // NDC statistics: count each visited node once (deduplicated)
                stats.ND_ndc_count++;

                // ========== Three NetDag modes ==========
                if (nd_mode == "filters") {
                    // ========== Mode 1: ND_only_filters (use traditional LB filters only, no AStar) ==========
                    auto t0 = std::chrono::high_resolution_clock::now();

                    // Use unified ged_lower_bound_filter
                    stats.ND_lb_count++;
                    ui threshold = static_cast<ui>(child_phase + tau);
                    netdag_lb = query_graph->ged_lower_bound_filter(
                        db_graph, threshold, vM.size(), eM.size(), max_n);
                    netdag_ged = INF; // filters mode has no exact GED, set to INF

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio)
                    {
                        continue;
                    }
                }
                else if (nd_mode == "astar") {
                    // ========== Mode 2: ND_only_AStar (skip traditional filters, use AStar directly) ==========
                    auto t0 = std::chrono::high_resolution_clock::now();

                    stats.ND_astar_count++; // App_test is full A* search, counted in astar_count
                    Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                    app.init(db_graph, query_graph);
                    netdag_ged = app.App_test(nullptr, nullptr);
                    netdag_lb = app.get_overall_lb();
                    stats.ND_app_test_count++; // Count App_test invocations

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_astar_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio)
                    {
                        continue;
                    }
                }
                else if (nd_mode == "filters_astar") {
                    // ========== Mode 3: ND_filters_AStar (use filters first, only use AStar at last layer when lb<=tau) ==========
                    // Non-last layer: filters only
                    // Last layer: use filters first, if lb<=tau, then call AStar for exact GED

                    auto t0 = std::chrono::high_resolution_clock::now();

                    // Use unified ged_lower_bound_filter
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

                    // Key decision: only call AStar at last layer when lb<=tau
                    if (is_last_layer && netdag_lb <= tau)
                    {
                        auto t2 = std::chrono::high_resolution_clock::now();

                        stats.ND_astar_count++;
                        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
                        app.init(db_graph, query_graph);
                        netdag_ged = app.App_test(nullptr, nullptr);
                        netdag_lb = app.get_overall_lb();  // Update with AStar's lb
                        stats.ND_app_test_count++;

                        auto t3 = std::chrono::high_resolution_clock::now();
                        stats.ND_astar_time += std::chrono::duration<double>(t3 - t2).count();
                    }
                    else
                    {
                        // Not last layer, or last layer but lb>tau, do not call AStar
                        netdag_ged = INF;
                    }
                }
                else {
                    // Unknown mode, default to filters mode
                    std::cerr << "[GS_search] Warning: Unknown nd_mode '" << nd_mode << "', using 'filters' mode\n";

                    auto t0 = std::chrono::high_resolution_clock::now();

                    // Use unified ged_lower_bound_filter
                    stats.ND_lb_count++;
                    ui threshold = static_cast<ui>(child_phase + tau);
                    netdag_lb = query_graph->ged_lower_bound_filter(
                        db_graph, threshold, vM.size(), eM.size(), max_n);
                    netdag_ged = INF;

                    auto t1 = std::chrono::high_resolution_clock::now();
                    stats.ND_lb_time += std::chrono::duration<double>(t1 - t0).count();

                    if (netdag_lb > (child_phase + tau) * nd_filter_ratio) continue;
                }

                // ========== 4) Select node based on strategy ==========
                // If last layer, collect all qualifying anchors while tracking best node for continued traversal
                if (child_phase <= net_dag->alpha)
                {
                    // Last layer: collect all anchors satisfying lower bound condition
                    // Use error_tolerance_search parameter to expand candidate range
                    double range_threshold = net_dag->alpha + tau + error_tolerance_search;
                    if (netdag_lb <= range_threshold)
                    {
                        // Store anchor_id, lower bound and exact GED result (currently using App_test, so exact result available)
                        candidate_anchors_with_results.emplace_back(child_node_id, netdag_lb, netdag_ged);
                    }

                    // Also track best node for continuing while loop
                    if (netdag_lb < min_dist)
                    {
                        min_dist = netdag_lb;
                        min_child_node_id = child_node_id;
                    }
                }
                else
                {
                    // Non-last layer: choose strategy based on fast_down flag
                    if (fast_down)
                    {
                        // Fast-down strategy: immediately select first qualifying node
                        if (netdag_lb <= (child_phase + tau) * nd_filter_ratio)
                        {
                            min_dist = netdag_lb;
                            min_child_node_id = child_node_id;
                            break;  // Immediately break out of for loop, select this node to descend
                        }
                    }
                    else
                    {
                        // Greedy strategy: select node with minimum distance
                        if (netdag_lb < min_dist)
                        {
                            min_dist = netdag_lb;
                            min_child_node_id = child_node_id;
                        }
                    }
                }

            } // end if (child_node_dist <= ...)
        }

        // If no suitable child found, exit
        if (min_child_node_id == -1)
        {
            return {};
        }

        // Move to next childNode
        current_node = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[min_child_node_id]);
        if (!current_node)
        {
            return {};
        }
        current_phase = child_phase;
    }

    // ========== 5) If while exits, then current_phase <= alpha ==========


    return candidate_anchors_with_results;
}

std::vector<int> GismaSearchEngine::SS_search(std::shared_ptr<Node> query_node, int anchor_id, double netdag_lb, int netdag_ged, double tau, SearchStats &stats, const std::string &dfs_mode_override)
{
    int query_id = query_node->node_id;

    // if (tau >= 6) use_ML = true;
    std::vector<int> exact_results_within_tau;

    EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);

    // Count total nodes of used EPTs
    if (ept) {
        stats.EPT_total_nodes_in_used_epts += ept->tree_nodes.size();
    }

    // Directly call traverse_ept_and_search(...), passing NetDag results to avoid redundant DFS root computation
    traverse_ept_and_search(*ept, query_node, exact_results_within_tau, tau, stats, netdag_lb, netdag_ged, dfs_mode_override);
    return exact_results_within_tau;
}




void draw_all_needed_db_graphs(
    const EditPathTree &ept,
    const std::shared_ptr<Node> &query_node,  // for getting query->graph
    const std::string &output_dir
)
{
    namespace fs = std::filesystem;
    fs::create_directories(output_dir);

    if (ept.tree_nodes.empty()) {
        std::cerr << "[draw_all_needed_db_graphs] EPT empty => no nodes.\n";
        return;
    }

    // 1) draw root's db_graph
    size_t root_idx = ept.root_index;
    const TreeNode &root_node = ept.tree_nodes[root_idx];
    if (root_node.db_graph) {
        root_node.db_graph->draw_single_graph(
            *root_node.db_graph, 
            output_dir, 
            "root"
        );
    } else {
        std::cerr << "[draw_all_needed_db_graphs] root has no db_graph\n";
    }

    // 2) draw all level=1 nodes => db_graph
    for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
        const TreeNode &tn = ept.tree_nodes[i];
        if (tn.level == 1 && tn.db_graph) {
            std::string prefix = "lvl1_node_" + std::to_string(i);
            tn.db_graph->draw_single_graph(
                *tn.db_graph,
                output_dir,
                prefix
            );
        }
    }

    // 3) draw query_node's db_graph
    if (query_node && query_node->graph) {
        query_node->graph->draw_single_graph(
            *query_node->graph,
            output_dir,
            "query_" + std::to_string(query_node->node_id)
        );
    } else {
        std::cerr << "[draw_all_needed_db_graphs] query_node->graph is null.\n";
    }
}

void draw_merged_all_in_one_image(
    const EditPathTree & ept,
    const std::vector<bool> & node_has_answer,
    const std::shared_ptr<Node> & query_node,
    double tau,
    const std::string & out_dir
)
{
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // construct filename: merged_tau_{tau}.dot / merged_tau_{tau}.png
    // for decimal output use std::to_string(tau). can also add more format control.
    std::string dot_file = out_dir + "/merged_tau_" + std::to_string(tau) + ".dot";
    std::string png_file = out_dir + "/merged_tau_" + std::to_string(tau) + ".png";

    std::ofstream ofs(dot_file);
    if (!ofs.is_open()) {
        std::cerr << "[draw_merged_all_in_one_image] cannot open " << dot_file << "\n";
        return;
    }

    // write main graph header
    ofs << "graph G {\n";
    ofs << "  rankdir=LR;\n"; 
    ofs << "  node [shape=circle];\n";

    // use subgraph cluster to separate root, query, level=1
    // and write (with answer) or (no answer) in label

    // ========== 1) Root ==========
    size_t root_idx = ept.root_index;
    if (root_idx < ept.tree_nodes.size()) {
        const TreeNode &root_node = ept.tree_nodes[root_idx];
        if (root_node.db_graph) {
            bool ans = node_has_answer[root_idx];
            // cluster_0
            ofs << "  subgraph cluster_root {\n";
            ofs << "    label=\"Root(" << root_idx << ") " 
                << (ans ? "(with answer)" : "(no answer)") 
                << "\";\n";
            ofs << "    style=rounded;\n";

            // output vertices/edges in root_node.db_graph => use special prefix to avoid conflicts
            // e.g. "root0_u"
            Graph *g = root_node.db_graph;
            for (ui u = 0; u < g->n; u++) {
                ui lbl = (u < g->vlabels_vec.size()) ? g->vlabels_vec[u] : 0;
                ofs << "    root0_" << u
                    << " [label=\"" << lbl << "\"];\n";
            }
            // edges (u<v)
            for (ui u = 0; u < g->adjacency_list.size(); u++) {
                for (auto &p : g->adjacency_list[u]) {
                    ui v = p.first;
                    if (v > u) {
                        ofs << "    root0_" << u 
                            << " -- root0_" << v << ";\n";
                    }
                }
            }

            ofs << "  }\n\n"; // subgraph end
        }
    }

    // ========== 2) Query ==========
    // if query_node exists => cluster_query
    if (query_node && query_node->graph) {
        bool ans_query = false;
        // query_node->node_id may not be in ept.tree_nodes? 
        // or we can see if node_id < ept.tree_nodes.size() and node_has_answer[node_id]
        if ((size_t)query_node->node_id < ept.tree_nodes.size()) {
            ans_query = node_has_answer[query_node->node_id];
        }

        ofs << "  subgraph cluster_query {\n";
        ofs << "    label=\"Query(" << query_node->node_id << ") "
            << (ans_query ? "(with answer)" : "(no answer)")
            << "\";\n";
        ofs << "    style=rounded;\n";

        Graph *gq = query_node->graph.get();
        for (ui u = 0; u < gq->n; u++) {
            ui lbl = (u < gq->vlabels_vec.size()) ? gq->vlabels_vec[u] : 0;
            ofs << "    query_" << u
                << " [label=\"" << lbl << "\"];\n";
        }
        for (ui u = 0; u < gq->adjacency_list.size(); u++) {
            for (auto &p : gq->adjacency_list[u]) {
                ui v = p.first;
                if (v > u) {
                    ofs << "    query_" << u 
                        << " -- query_" << v << ";\n";
                }
            }
        }

        ofs << "  }\n\n"; 
    }

    // ========== 3) level=1 ==========
    // assign cluster_l1_{i} sequentially, i represents ept node index
    // label="L1 i (with answer/no)"
    for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
        const TreeNode &tn = ept.tree_nodes[i];
        if (tn.level == 1 && tn.db_graph) {
            bool ans = node_has_answer[i];

            ofs << "  subgraph cluster_l1_" << i << " {\n";
            ofs << "    label=\"L1(" << i << ") " 
                << (ans ? "(with answer)" : "(no answer)")
                << "\";\n";
            ofs << "    style=rounded;\n";

            Graph *g = tn.db_graph;
            for (ui u = 0; u < g->n; u++) {
                ui lbl = (u < g->vlabels_vec.size()) ? g->vlabels_vec[u] : 0;
                ofs << "    l1_" << i << "_" << u 
                    << " [label=\"" << lbl << "\"];\n";
            }
            for (ui u = 0; u < g->adjacency_list.size(); u++) {
                for (auto &p : g->adjacency_list[u]) {
                    ui v = p.first;
                    if (v > u) {
                        ofs << "    l1_" << i << "_" << u
                            << " -- l1_" << i << "_" << v 
                            << ";\n";
                    }
                }
            }

            ofs << "  }\n\n"; 
        }
    }

    // end of main graph
    ofs << "}\n";
    ofs.close();

    // call dot => png
    std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file;
    int ret = std::system(cmd.c_str());
    if (ret == 0) {
        std::cout << "[draw_merged_all_in_one_image] => " << png_file << " generated.\n";
    } else {
        std::cerr << "[WARN] dot command failed with code=" << ret << "\n";
    }
}

void draw_merged_root_query_bfs1(
    const EditPathTree & ept,
    const std::vector<bool> & node_has_answer,
    const std::shared_ptr<Node> & query_node,
    double tau,
    const std::string & out_dir
)
{
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // 1) filename
    int tau_int = static_cast<int>(tau);
    std::string dot_file = out_dir + "/merged_tau_" + std::to_string(tau_int) + ".dot";
    std::string png_file = out_dir + "/merged_tau_" + std::to_string(tau_int) + ".png";

    std::ofstream ofs(dot_file);
    if (!ofs.is_open()) {
        std::cerr << "[draw_merged_root_query_bfs1] cannot open " << dot_file << "\n";
        return;
    }

    // 2) write graph header
    ofs << "graph G {\n";
    ofs << "  rankdir=LR;\n"; 
    ofs << "  node [shape=circle];\n";

    // ========== A) root ==========
    size_t root_idx = ept.root_index;
    if (root_idx < ept.tree_nodes.size()) {
        const TreeNode &root_node = ept.tree_nodes[root_idx];
        if (root_node.db_graph) {
            // root with/without answer
            bool ansRoot = node_has_answer[root_idx];
            std::string lbl_root = std::string(ansRoot ? "(with answer) " : "(no answer) ")
                                   + "Root(" + std::to_string(root_idx) + ")";

            ofs << "  subgraph cluster_root {\n";
            ofs << "    label=\"" << lbl_root << "\";\n";
            ofs << "    style=rounded;\n";

            Graph *g = root_node.db_graph;
            // node
            for (ui u = 0; u < g->n; u++) {
                ui lbl = (u < g->vlabels_vec.size()) ? g->vlabels_vec[u] : 0;
                ofs << "    root_" << u 
                    << " [label=\"" << lbl << "\"];\n";
            }
            // edges
            for (ui u = 0; u < g->adjacency_list.size(); u++) {
                for (auto &p : g->adjacency_list[u]) {
                    ui v = p.first;
                    if (v > u) {
                        ofs << "    root_" << u << " -- root_" << v << ";\n";
                    }
                }
            }
            ofs << "  }\n\n";
        }
    }

    // ========== B) query ==========
    // do not write answer
    if (query_node && query_node->graph) {
        ofs << "  subgraph cluster_query {\n";
        ofs << "    label=\"Query(" << query_node->node_id << ")\";\n";
        ofs << "    style=rounded;\n";

        Graph *gq = query_node->graph.get();
        for (ui u = 0; u < gq->n; u++) {
            ui lbl = (u < gq->vlabels_vec.size()) ? gq->vlabels_vec[u] : 0;
            ofs << "    query_" << u 
                << " [label=\"" << lbl << "\"];\n";
        }
        for (ui u = 0; u < gq->adjacency_list.size(); u++) {
            for (auto &p : gq->adjacency_list[u]) {
                ui v = p.first;
                if (v > u) {
                    ofs << "    query_" << u << " -- query_" << v << ";\n";
                }
            }
        }

        ofs << "  }\n\n";
    }

    // ========== C) BFS depth=1 ==========
    // first compute BFS depth
    std::vector<int> bfs_depth(ept.tree_nodes.size(), -1);
    {
        bfs_depth[root_idx] = 0;
        std::queue<size_t>Q;
        Q.push(root_idx);
        while(!Q.empty()){
            auto u = Q.front(); Q.pop();
            int d = bfs_depth[u];
            for (auto c : ept.tree_nodes[u].children_indices) {
                if (bfs_depth[c] == -1) {
                    bfs_depth[c] = d + 1;
                    Q.push(c);
                }
            }
        }
    }

    // split into with/no
    std::vector<size_t> bfs1_with;
    std::vector<size_t> bfs1_no;
    for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
        if (bfs_depth[i] == 1) {
            bool ans = node_has_answer[i];
            if (ans) {
                bfs1_with.push_back(i);
            } else {
                bfs1_no.push_back(i);
            }
        }
    }

    // (C.1) draw BFS=1 with answer first
    for (auto i : bfs1_with) {
        const TreeNode &tn = ept.tree_nodes[i];
        std::string lbl_bfs1 = "(with answer) BFS(" + std::to_string(i) + ") depth=1";

        ofs << "  subgraph cluster_bfs1_" << i << " {\n";
        ofs << "    label=\"" << lbl_bfs1 << "\";\n";
        ofs << "    style=rounded;\n";

        if (tn.db_graph) {
            Graph *g = tn.db_graph;
            for (ui u = 0; u < g->n; u++){
                ui lbl = (u < g->vlabels_vec.size())? g->vlabels_vec[u]:0;
                ofs << "    bfs1_"<< i << "_" << u << " [label=\"" << lbl << "\"];\n";
            }
            for (ui u = 0; u < g->adjacency_list.size(); u++){
                for (auto &p : g->adjacency_list[u]) {
                    ui v = p.first;
                    if (v > u) {
                        ofs << "    bfs1_"<< i<<"_"<< u << " -- bfs1_"<< i<<"_"<< v << ";\n";
                    }
                }
            }
        }
        ofs << "  }\n\n";
    }

    // (C.2) then draw BFS=1 no answer
    for (auto i : bfs1_no) {
        const TreeNode &tn = ept.tree_nodes[i];
        std::string lbl_bfs1 = "(no answer) BFS(" + std::to_string(i) + ") depth=1";

        ofs << "  subgraph cluster_bfs1_" << i << " {\n";
        ofs << "    label=\"" << lbl_bfs1 << "\";\n";
        ofs << "    style=rounded;\n";

        if (tn.db_graph) {
            Graph *g = tn.db_graph;
            for (ui u = 0; u < g->n; u++){
                ui lbl = (u < g->vlabels_vec.size())? g->vlabels_vec[u]:0;
                ofs << "    bfs1_"<< i << "_" << u << " [label=\"" << lbl << "\"];\n";
            }
            for (ui u = 0; u < g->adjacency_list.size(); u++){
                for (auto &p : g->adjacency_list[u]) {
                    ui v = p.first;
                    if (v > u) {
                        ofs << "    bfs1_"<< i<<"_"<< u << " -- bfs1_"<< i<<"_"<< v << ";\n";
                    }
                }
            }
        }
        ofs << "  }\n\n";
    }

    ofs << "}\n";
    ofs.close();

    // dot => png
    std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file;
    int ret = std::system(cmd.c_str());
    if (ret == 0) {
        std::cout << "[draw_merged_root_query_bfs1] => " << png_file << " generated.\n";
    } else {
        std::cerr << "[WARN] dot command failed with code=" << ret << "\n";
    }
}

void draw_all_exact_results_in_one_image_plus_query(
    const std::vector<int> & exact_results,
    const std::vector<std::shared_ptr<Node>> & db_node_list,
    const std::shared_ptr<Node> & query_node,
    double tau,
    const std::string & out_dir
)
{
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    int tau_int = static_cast<int>(tau);
    std::string dot_file = out_dir + "/allAnswers_and_query_tau_" + std::to_string(tau_int) + ".dot";
    std::string png_file = out_dir + "/allAnswers_and_query_tau_" + std::to_string(tau_int) + ".png";

    std::ofstream ofs(dot_file);
    if(!ofs.is_open()){
        std::cerr << "[draw_all_exact_results_in_one_image_plus_query] cannot open " << dot_file << "\n";
        return;
    }

    // write dot header
    ofs << "graph G {\n";
    ofs << "  rankdir=LR;\n";
    ofs << "  node [shape=circle];\n";

    // ========== A) draw query subgraph first ==========
    if (query_node && query_node->graph) {
        // ensure query_graph is also initialized
        query_node->graph->initialize_vectors_from_arrays();

        ofs << "  subgraph cluster_query {\n";
        ofs << "    label=\"Query(" << query_node->node_id << ")\";\n";
        ofs << "    style=rounded;\n";

        Graph *gq = query_node->graph.get();
        for (ui u = 0; u < gq->n; u++) {
            ui lbl = (u < gq->vlabels_vec.size()) ? gq->vlabels_vec[u] : 0;
            ofs << "    query_" << u
                << " [label=\"" << lbl << "\"];\n";
        }
        for (ui u = 0; u < gq->adjacency_list.size(); u++) {
            for (auto &p : gq->adjacency_list[u]) {
                ui v = p.first;
                if (v > u) {
                    ofs << "    query_" << u
                        << " -- query_" << v
                        << ";\n";
                }
            }
        }

        ofs << "  }\n\n";
    }

    // ========== B) exact_results => dedup => subgraph ==========

    // 1) dedup
    std::unordered_set<int> unique_ids;
    for (auto id : exact_results) {
        unique_ids.insert(id);
    }

    // 2) for each unique ID => subgraph cluster_ans_id
    bool anyDraw = false;
    for (auto id : unique_ids) {
        // range check
        if (id < 0 || id >= (int)db_node_list.size()) {
            std::cerr << "[WARN] invalid ID=" << id << "\n";
            continue;
        }
        auto nodePtr = db_node_list[id];
        if(!nodePtr || !nodePtr->graph) {
            std::cerr << "[WARN] db_node_list["<<id<<"] is null or no graph\n";
            continue;
        }

        // before visualization, initialize
        nodePtr->graph->initialize_vectors_from_arrays();

        Graph *g = nodePtr->graph.get();
        anyDraw = true;

        // label => "DB ID=xxx"
        ofs << "  subgraph cluster_ans_" << id << " {\n";
        ofs << "    label=\"DB ID=" << id << "\";\n";
        ofs << "    style=rounded;\n";

        // node
        for (ui u = 0; u < g->n; u++) {
            ui lbl = (u < g->vlabels_vec.size()) ? g->vlabels_vec[u] : 0;
            ofs << "    ans_" << id << "_" << u
                << " [label=\"" << lbl << "\"];\n";
        }

        // edges
        for (ui u = 0; u < g->adjacency_list.size(); u++){
            for (auto &p : g->adjacency_list[u]) {
                ui v = p.first;
                if (v > u) {
                    ofs << "    ans_" << id << "_" << u
                        << " -- ans_" << id << "_" << v
                        << ";\n";
                }
            }
        }
        ofs << "  }\n\n";
    }

    if (!anyDraw) {
        ofs << "  // no exact results => none drawn\n";
    }

    ofs << "}\n";
    ofs.close();

    // call dot => png
    std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file;
    int ret = std::system(cmd.c_str());
    if (ret==0){
        std::cout<<"[draw_all_exact_results_in_one_image_plus_query] => "<<png_file<<" generated.\n";
    } else {
        std::cerr<<"[WARN] dot command failed with code="<<ret<<"\n";
    }
}

bool dfs_find_path_no_parent_index(
    const EditPathTree & ept,
    size_t cur,
    size_t target,
    std::vector<size_t> &temp,
    std::vector<size_t> &result_path
)
{
    // 1) first push current node into temp path
    temp.push_back(cur);

    // 2) if it is exactly target, path found => copy to result_path
    if (cur == target) {
        result_path = temp;  // copy
        temp.pop_back();
        return true;
    }

    // 3) otherwise iterate cur's child nodes
    const TreeNode &tn = ept.tree_nodes[cur];
    for (auto child : tn.children_indices) {
        // DFS
        if (dfs_find_path_no_parent_index(ept, child, target, temp, result_path)) {
            // found => return true
            temp.pop_back(); 
            return true;
        }
    }

    // 4) if not found => backtrack
    temp.pop_back();
    return false;
}

void draw_path_to_answer_in_one_image(
    const EditPathTree & ept,
    size_t answer_idx,
    const std::shared_ptr<Node> & query_node,
    double tau,
    const std::string & out_dir
)
{
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    int tau_int = static_cast<int>(tau);
    // filename
    std::string dot_file = out_dir + "/path_to_answer_noparent_" 
                           + std::to_string(answer_idx) + "_tau_" 
                           + std::to_string(tau_int) + ".dot";
    std::string png_file = out_dir + "/path_to_answer_noparent_" 
                           + std::to_string(answer_idx) + "_tau_" 
                           + std::to_string(tau_int) + ".png";

    std::ofstream ofs(dot_file);
    if (!ofs.is_open()) {
        std::cerr << "[draw_path_to_answer_in_one_image_no_parent_plus_query] cannot open " 
                  << dot_file << "\n";
        return;
    }

    // 1) find path to root->answer_idx
    std::vector<size_t> path_nodes;
    {
        std::vector<size_t> temp;
        bool ok = dfs_find_path_no_parent_index(
            ept,
            ept.root_index,  // from root
            answer_idx,      
            temp,
            path_nodes       // out
        );
        if (!ok) {
            std::cerr 
                << "[draw_path_to_answer_in_one_image_no_parent_plus_query] cannot find path from root to answer_idx=" 
                << answer_idx << "\n";
            return;
        }
    }

    ofs << "graph G {\n";
    ofs << "  rankdir=LR;\n";
    ofs << "  node [shape=circle];\n";

    // ============ A) draw root->answer_idx path first ============
    for (size_t i = 0; i < path_nodes.size(); i++) {
        size_t node_idx = path_nodes[i];
        const TreeNode &tn = ept.tree_nodes[node_idx];

        // show level / root / answer
        std::string label_str = "PathStep #" + std::to_string(i) +
                                " (EPT node=" + std::to_string(node_idx) +
                                ", level=" + std::to_string(tn.level) + ")";
        if (node_idx == ept.root_index) {
            label_str += " [root]";
        }
        if (node_idx == answer_idx && node_idx != ept.root_index) {
            label_str += " [answer]";
        }

        ofs << "  subgraph cluster_path_" << i << " {\n";
        ofs << "    label=\"" << label_str << "\";\n";
        ofs << "    style=rounded;\n";

        if (tn.db_graph) {
            tn.db_graph->initialize_vectors_from_arrays();
            Graph *g = tn.db_graph;

            // output vertices
            for (ui u = 0; u < g->n; u++) {
                ui lbl = (u < g->vlabels_vec.size()) ? g->vlabels_vec[u] : 0;
                ofs << "    step" << i << "_" << u 
                    << " [label=\"" << lbl << "\"];\n";
            }

            // output edges
            for (ui u = 0; u < g->adjacency_list.size(); u++){
                for (auto &p : g->adjacency_list[u]) {
                    ui v = p.first;
                    if (v > u) {
                        ofs << "    step" << i << "_" << u
                            << " -- step" << i << "_" << v
                            << ";\n";
                    }
                }
            }
        } else {
            ofs << "    // node_idx="<< node_idx <<" has no db_graph\n";
        }

        ofs << "  }\n\n";
    }

    // ============ B) then draw query subgraph ============
    if (query_node && query_node->graph) {
        ofs << "  subgraph cluster_query {\n";
        ofs << "    label=\"Query(" << query_node->node_id << ")\";\n";
        ofs << "    style=rounded;\n";

        query_node->graph->initialize_vectors_from_arrays();
        Graph *gq = query_node->graph.get();
        // node
        for (ui u = 0; u < gq->n; u++) {
            ui lbl = (u < gq->vlabels_vec.size()) ? gq->vlabels_vec[u] : 0;
            ofs << "    query_" << u 
                << " [label=\"" << lbl << "\"];\n";
        }
        // edges
        for (ui u = 0; u < gq->adjacency_list.size(); u++){
            for (auto &p : gq->adjacency_list[u]) {
                ui v = p.first;
                if (v > u) {
                    ofs << "    query_" << u
                        << " -- query_" << v
                        << ";\n";
                }
            }
        }

        ofs << "  }\n\n";
    }

    ofs << "}\n";
    ofs.close();

    // dot => png
    std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file;
    int ret = std::system(cmd.c_str());
    if (ret == 0) {
        std::cout << "[draw_path_to_answer_in_one_image_no_parent_plus_query] => "
                  << png_file << " generated.\n";
    } else {
        std::cerr << "[WARN] dot command failed with code=" << ret << "\n";
    }
}


/**
 * @brief Print statistics grouped by node.level in EPT
 */
void GismaSearchEngine::print_statistics_by_ept_level(
    const EditPathTree &ept,
    const std::vector<bool> &node_has_answer
)
{
    // Collect each level -> nodes at that level
    std::map<int, std::vector<size_t>> nodes_per_level;
    for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
        int lvl = ept.tree_nodes[i].level; // EPT's own level
        nodes_per_level[lvl].push_back(i);
    }

    std::cout << "\n===== [Per-level EPT Statistics (using node.level)] =====\n";
    for (auto &pair : nodes_per_level) {
        int lvl = pair.first;
        const auto &node_indices = pair.second;
        size_t total_cnt = node_indices.size();

        // split into “with answer” vs “no answer”
        size_t has_answer_count = 0;
        for (auto idx : node_indices) {
            if (node_has_answer[idx]) {
                has_answer_count++;
            }
        }
        size_t no_answer_count = total_cnt - has_answer_count;

        std::cout << "----- [Level = " << lvl << "] -----\n";
        std::cout << "Total nodes   : " << total_cnt << "\n";
        std::cout << "  => with answer   : " << has_answer_count << " nodes\n";
        std::cout << "  => without answer: " << no_answer_count << " nodes\n";
    }
    std::cout << "=========================================================\n\n";
}


/**
 * @brief  Perform actual BFS to compute each node's depth relative to root_index
 *         stored in bfs_depth[node_index]
 */
void GismaSearchEngine::compute_bfs_depth(
    const EditPathTree &ept,
    std::vector<int> &bfs_depth
)
{
    if (ept.tree_nodes.empty()) return;
    std::fill(bfs_depth.begin(), bfs_depth.end(), -1);

    // BFS depth of root_index = 0
    size_t root_idx = ept.root_index;
    bfs_depth[root_idx] = 0;

    // BFS using queue
    std::queue<size_t> Q;
    Q.push(root_idx);

    while (!Q.empty()) {
        auto u = Q.front();
        Q.pop();
        int parent_depth = bfs_depth[u];
        // iterate its child nodes
        for (auto child_idx : ept.tree_nodes[u].children_indices) {
            // if depth not yet marked, set to parent_depth + 1
            if (bfs_depth[child_idx] == -1) {
                bfs_depth[child_idx] = parent_depth + 1;
                Q.push(child_idx);
            }
        }
    }
}


/**
 * @brief  Print statistics grouped by BFS depth using bfs_depth[] from previous step
 */
void GismaSearchEngine::print_statistics_by_bfs_depth(
    const EditPathTree &ept,
    const std::vector<bool> &node_has_answer,
    const std::vector<int> &bfs_depth
)
{
    // Collect BFS depth -> list of node indices
    std::map<int, std::vector<size_t>> level_map;
    for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
        int d = bfs_depth[i];
        if (d >= 0) { // already visited by BFS
            level_map[d].push_back(i);
        }
        else {
            // If EPT has disconnected branches, or nodes unreachable by BFS
            // d will remain -1
            // not handled here; can be classified as "disconnected" if needed 
        }
    }

    std::cout << "\n===== [Per-level EPT Statistics (using BFS depth)] =====\n";
    for (auto &pair : level_map) {
        int lvl = pair.first;
        const auto &node_indices = pair.second;
        size_t total_cnt = node_indices.size();

        // split into “with answer” vs “no answer”
        size_t has_answer_count = 0;
        for (auto idx : node_indices) {
            if (node_has_answer[idx]) {
                has_answer_count++;
            }
        }
        size_t no_answer_count = total_cnt - has_answer_count;

        std::cout << "----- [BFS depth = " << lvl << "] -----\n";
        std::cout << "Total nodes   : " << total_cnt << "\n";
        std::cout << "  => with answer   : " << has_answer_count << " nodes\n";
        std::cout << "  => without answer: " << no_answer_count << " nodes\n";
    }
    std::cout << "=========================================================\n\n";
}














// COMPUTE_ASTAR_ONLY_FOR_DATA_GRAPH macro is now defined in Utility.h

// Helper function to compute EPT subtree size
size_t GismaSearchEngine::calculate_subtree_size(const EditPathTree& ept, size_t node_index) {
    if (node_index >= ept.tree_nodes.size()) {
        return 0;
    }

    const TreeNode& node = ept.tree_nodes[node_index];
    size_t subtree_size = 1; // current node itself

    // Recursively compute subtree size of all children
    for (size_t child_index : node.children_indices) {
        subtree_size += calculate_subtree_size(ept, child_index);
    }

    return subtree_size;
}

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
    
    // Convert tau to ui type to avoid type issues
    ui tau_ui = (ui)tau;
    
#ifdef USE_DYNAMIC_DEPTH_PROBE
    // ========== Dynamic depth probing function (defined only when needed)==========
    auto probe_max_depth = [&ept](size_t node_idx) -> int {
        const TreeNode& current = ept.tree_nodes[node_idx];
        int current_level = current.level;
        
        // Find maximum level in subtree
        std::function<int(size_t)> find_max_level = [&](size_t idx) -> int {
            if (idx >= ept.tree_nodes.size()) {
                return current_level;  // return current level as baseline
            }
            
            const TreeNode& n = ept.tree_nodes[idx];
            int max_level = n.level;  // this node's level
            
            // Recursively search all child nodes
            for (size_t ch : n.children_indices) {
                int child_max = find_max_level(ch);
                max_level = std::max(max_level, child_max);
            }
            
            return max_level;
        };
        
        int max_level_in_subtree = find_max_level(node_idx);
        return max_level_in_subtree - current_level;  // return level difference
    };
#endif

    // ========== Node type statistics ==========
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }
    
    if (!node.completed_db_graph_ids.empty()) {
        stats.EPT_nodes_with_completed_ids++;
    }
    
    // Count edit operation types
    if (node.op.type != EditOperation::NONE) {
        stats.EPT_op_type_count[node.op.type]++;
    }
    
    // ========== estimate_lb for passing to child nodes ==========
    ui new_estimate_lb = estimate_lb;  // default to passed-in value

    // Determine if it is a db graph (root node or node with completed_db_graph_ids)
    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    // ========== Lower Bound check ==========
    bool should_compute_ged = true;

    // NDC counting flag: ensure each node counted only once during actual computation
    bool ndc_counted = false;

    // Count total visited nodes
    stats.EPT_total_nodes_visited++;

    // Debug counters
    static size_t debug_total_nodes = 0;
    static size_t debug_estimate_lb_skip = 0;
    static size_t debug_filter_skip = 0;
    static size_t debug_estimate_lb_skip_at_root = 0;
    static size_t debug_filter_skip_at_root = 0;
    debug_total_nodes++;

    bool is_root = (node_index == ept.root_index);

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    // LB Propagation: if estimate_lb > tau, skip computation
    if (estimate_lb > tau_ui) {
        should_compute_ged = false;
        stats.lb_pruning_count++;  // Count App_test computations skipped due to estimate_lb > tau
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
        // NDC statistics
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        // Flag: entering filter computation, indicating computation was performed
        stats.EPT_nodes_computed++;

        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        // Track filter time separately for db graphs and intermediate graphs
        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }

        new_estimate_lb = std::max(new_estimate_lb, lb);  // update estimate_lb

        if (lb > tau_ui) {
            should_compute_ged = false;
            stats.EPT_filter_pruned_nodes++;  // Count nodes pruned by filter
            debug_filter_skip++;
            if (is_root) {
                debug_filter_skip_at_root++;
            }
        }
    }
    
    // ========== Root node optimization: directly use GED cached by NetDag ==========
    // Note: must be before COMPUTE_GED_ONLY_FOR_COMPLETED check, since root node's GED was already computed in NetDag

    // DEBUG: trace search process of specific anchor (disabled)
    // bool debug_this_anchor = (node_index == ept.root_index &&
    //     (ept.anchor_id == 12735 || ept.anchor_id == 11953 || ept.anchor_id == 5791 ||
    //      ept.anchor_id == 16611 || ept.anchor_id == 11382 || ept.anchor_id == 9934));
    // if (debug_this_anchor) {
    //     printf("[DEBUG_ANCHOR] anchor_id=%d, query_id=%d, anchor_netdag_ged=%d, anchor_netdag_lb=%.1f, tau=%u, completed_db_graph_ids.size=%zu\n",
    //            (int)ept.anchor_id, query_node->node_id, anchor_netdag_ged, anchor_netdag_lb, tau_ui, node.completed_db_graph_ids.size());
    // }

    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF) {
        // Only use cache when netdag_ged is valid (not INF)
        // When nd_mode="filters", netdag_ged=INF, cannot skip computation
        // When nd_mode="astar"/"filters_astar", netdag_ged may be valid

        // If GED computed by NetDag <= tau, directly add to answer
        if (anchor_netdag_ged <= (int)tau_ui) {
            // Root node: ensure anchor itself is added, then add other graphs from completed_db_graph_ids
            int anchor_id = static_cast<int>(node.anchor_id);

            // First check if anchor_id is already in completed_db_graph_ids
            bool anchor_in_completed = false;
            for (int id : node.completed_db_graph_ids) {
                if (id == anchor_id) {
                    anchor_in_completed = true;
                    break;
                }
            }

            // If anchor not in completed_db_graph_ids, add separately
            if (!anchor_in_completed) {
                exact_results_within_tau.push_back(anchor_id);
                stats.EPT_results_from_astar++;
            }

            // Add all graphs from completed_db_graph_ids
            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                           node.completed_db_graph_ids.begin(),
                                           node.completed_db_graph_ids.end());
            stats.EPT_results_from_astar += node.completed_db_graph_ids.size();

            found_here = true;
            stats.root_netdag_ged_reuse_count++;  // Count root node reuse occurrences
        }
        // else: netdag_ged > tau, do not add to answer, but continue traversing children
        // because children may reduce GED to <=tau through edit operations

        // Update estimate_lb to NetDag's overall_lb (for passing to children)
        new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));

        // Root node already used NetDag's GED, no further computation needed
        should_compute_ged = false;
    } else if (node_index == ept.root_index && anchor_netdag_lb >= 0) {
        // When nd_mode="filters": netdag_ged=INF, but netdag_lb is valid
        // Pass netdag_lb to children, but do not skip root node's GED computation
        new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));
    }

    // ========== Check ifNeed to computeGED ==========
    // If EPT filters or only_compute_db_graph is enabled, only compute GED for nodes with completed_db_graph_idsnodes compute GED
    // Root node (anchor itself) even if completed_db_graph_ids is empty stillNeed to computeGED，because anchor itself may be a result
    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== If LB check passes, execute App_test ==========
    if (should_compute_ged) {
        // NDC statistics：If LB skipped but A* computed, still count
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        // Flag: GED computation executed
        stats.EPT_nodes_computed++;

        // Execute App_test computation (no reuse, with overall_lb setting)
        auto t0 = std::chrono::high_resolution_clock::now();
        Application app(tau_ui, "BMao", app_max_iter);
        app.init(db_g, qry_g);

        int ged = app.App_test(nullptr, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double>(t1 - t0).count();

        stats.EPT_astar_count++;
        stats.EPT_astar_time += elapsed_time;

        // Track verification time separately for db graphs and intermediate graphs
        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        // Process results
        if (ged <= (int)tau_ui) {
            // For root node, ensure anchor itself is added to results
            if (node_index == ept.root_index) {
                int anchor_id = static_cast<int>(ept.anchor_id);

                // Check if anchor is already in completed_db_graph_ids
                bool anchor_in_completed = std::find(node.completed_db_graph_ids.begin(),
                                                     node.completed_db_graph_ids.end(),
                                                     anchor_id) != node.completed_db_graph_ids.end();
                if (!anchor_in_completed) {
                    exact_results_within_tau.push_back(anchor_id);
                    stats.EPT_results_from_astar++;
                }
            }

            // Add all graphs from completed_db_graph_ids
            size_t num_results = node.completed_db_graph_ids.size();
            stats.EPT_results_from_astar += num_results;

            exact_results_within_tau.insert(exact_results_within_tau.end(),
                                          node.completed_db_graph_ids.begin(),
                                          node.completed_db_graph_ids.end());
            found_here = true;
        }

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
        // LB Propagation: Update estimate_lb
        new_estimate_lb = app.get_overall_lb();
#endif

        // ========== Subtree Pruning check ==========
#if USE_SUBTREE_PRUNING
        // Get current graph pair's overall_lb (local variable)
        ui overall_lb = app.get_overall_lb();

        // Subtree Pruning: Leaf nodeskip (no child nodes to prune)
        if (!node.children_indices.empty() && overall_lb > 0 && overall_lb < 10000) {  // assuming reasonable GED does not exceed 10000
            // Compute maximum steps remaining from current node
            int max_step_more;
            
#ifdef USE_DYNAMIC_DEPTH_PROBE
            // Using dynamic depth probing
            max_step_more = probe_max_depth(node_index);
#else
            // Using fixed maximum depth
            max_step_more = MAX_EPT_DEPTH - node.level;
#endif
            
            // Debug output (controlled by DEBUG_PRUNING macro)
#ifdef DEBUG_PRUNING
            printf("Subtree pruning check: node.level=%d, max_step_more=%d, tau=%u, overall_lb=%u\n",
                   node.level, max_step_more, tau_ui, overall_lb);
            printf("  -> Condition: %d + %u < %u ? %s\n",
                   max_step_more, tau_ui, overall_lb,
                   (max_step_more + (int)tau_ui < (int)overall_lb) ? "YES (prune)" : "NO (continue)");
#endif
            
            // Pruning condition: even if each subsequent step is optimal edit (reducing by 1),
            // the final GED is at least overall_lb - max_step_more
            // if this value > tau, no qualifying solution can be found
            if (max_step_more + (int)tau_ui < (int)overall_lb) {
#ifdef DEBUG_PRUNING
                printf("[SUBTREE_PRUNE] Node %zu (level=%d): max_steps=%d, tau=%u, overall_lb=%u, overall_ub=%u, is_leaf=%s -> PRUNED\n",
                       node_index, node.level, max_step_more, tau_ui, overall_lb, app.get_overall_ub(),
                       node.children_indices.empty() ? "yes" : "no");
#endif

                // Count subtree pruning effectiveness
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // minus current node itself
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                // statistics:how many triggered by leaf nodes
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
        // Skipped App_test computation (possibly due to estimate_lb or filter), but still check if prunable
#if USE_SUBTREE_PRUNING
        // Subtree Pruning: Leaf nodeskip (no child nodes to prune)
        if (!node.children_indices.empty() && new_estimate_lb > 0 && new_estimate_lb < 10000) {
            int max_step_more;
            
#ifdef USE_DYNAMIC_DEPTH_PROBE
            max_step_more = probe_max_depth(node_index);
#else
            max_step_more = MAX_EPT_DEPTH - node.level;
#endif
            
            if (max_step_more + (int)tau_ui < (int)new_estimate_lb) {
                // Pruning: do not recurseProcess child nodes

                // Count subtree pruning effectiveness（estimate_lb branch）
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // minus current node itself
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                return found_here;
            }
        }
#endif
    }
#endif


    // RecursivelyProcess child nodes
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }
        
        // Compute child node's estimate_lb
        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        ui child_estimate_lb = (level_diff >= (int)new_estimate_lb) ? 0 : new_estimate_lb - level_diff;

        // Count times estimate_lb is propagated to child nodes
        stats.lb_propagation_count++;

        if (dfs_traverse_no_reuse(ch, ept, query_node,
                               exact_results_within_tau, tau, stats,
                               child_estimate_lb, anchor_netdag_lb, anchor_netdag_ged)) {
            found_child = true;
        }
    }

    // Return whether current node or subtree found an answer
    return found_here || found_child;
}

// Pure DFS traversal - using App_baseline, no LB propagation, no lookahead pruning
// This is the simplest traversal, for baseline comparison
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

    // NDC counting flag: ensure each node counted only once during actual computation
    bool ndc_counted = false;

    // Node type statistics
    if (node.children_indices.empty()) {
        stats.EPT_leaf_nodes_processed++;
    } else {
        stats.EPT_internal_nodes_processed++;
    }

    // Check if node is associated with a database graph (not intermediate graph)
    // root node is anchor itself (db graph), or nodes with completed_db_graph_ids are also db graphs
    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    // Decide whether toNeed to computeGED
    bool should_compute_ged = true;

    // ========== Root node special handling: use GED value cached by NetDag ==========
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF) {
        // Only use cache when netdag_ged is valid (not INF)
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
        // Root node already used NetDag's GED, no further computation needed
        should_compute_ged = false;
    }

    // ========== LB Filter (consistent with simple mode) ==========
    if (use_ept_filters && should_compute_ged) {
        // NDC statistics：only count when actually computing LB
        if (!ndc_counted && is_db_graph) {
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

        // Track filter time separately for db graphs and intermediate graphs
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

    // ========== Check if GED computation is needed (controlled by use_ept_filters or only_compute_db_graph)==========
    // If use_ept_filters or only_compute_db_graph is enabled, only compute GED for nodes with completed_db_graph_ids
    // Root node even if completed_db_graph_ids is empty stillNeed to computeGED
    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== If GED computation needed, execute App_baseline ==========
    if (should_compute_ged) {
        // NDC statistics：If LB skipped but A* computed, still count
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        stats.EPT_nodes_computed++;

        auto t0 = std::chrono::high_resolution_clock::now();
        Application app(tau_ui, "BMao", app_max_iter);
        app.init(db_g, qry_g);

        // Use App_baseline - the most basic computation method
        int ged = app.App_baseline(nullptr, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double>(t1 - t0).count();

        stats.EPT_astar_count++;
        stats.EPT_astar_time += elapsed_time;

        // Track verification time separately for db graphs and intermediate graphs
        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        // Process results
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
        // Note: no LB propagation, no lookahead pruning
    }

    // RecursivelyProcess child nodes（Pure DFS, do not propagate estimate_lb）
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
    int                             anchor_netdag_ged)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;
    ui new_estimate_lb = estimate_lb;

    
    std::shared_ptr<SearchSnapshot> current_snapshot = nullptr;
    
    // Dynamic depth probing function
#if USE_SUBTREE_PRUNING && defined(USE_DYNAMIC_DEPTH_PROBE)
    auto probe_max_depth = [&ept](size_t node_idx) -> int {
        const TreeNode& current = ept.tree_nodes[node_idx];
        int current_level = current.level;

        std::function<int(size_t)> find_max_level = [&](size_t idx) -> int {
            if (idx >= ept.tree_nodes.size()) {
                return current_level;
            }

            const TreeNode& n = ept.tree_nodes[idx];
            int max_level = n.level;

            for (size_t ch : n.children_indices) {
                int child_max = find_max_level(ch);
                max_level = std::max(max_level, child_max);
            }

            return max_level;
        };

        int max_level_in_subtree = find_max_level(node_idx);
        return max_level_in_subtree - current_level;
    };
#endif

    // NDC counting flag: ensure each node counted only once during actual computation
    bool ndc_counted = false;

    // Node type statistics
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

    // Determine if it is a db graph (root node or node with completed_db_graph_ids)
    bool is_db_graph = (node_index == ept.root_index) || !node.completed_db_graph_ids.empty();

    // ========== Priority check dummy-only fast path ==========
    bool used_dummy_fast_path = false;
    int dummy_fast_path_ged = -1;

    // Check if dummy-only fast path can be used (prioritized over LB check)
    if (parent_snapshot && parent_snapshot->v.size() == 1 &&
        (parent_snapshot->v[0].level < 0 || parent_snapshot->v[0].image < 0)) {
        // This is a dummy-only snapshot
        int ged_gap = (node.parent_index < ept.tree_nodes.size()) ?
                     (node.level - ept.tree_nodes[node.parent_index].level) : 1;

        if (ged_gap > 0 && ged_gap <= parent_snapshot->margin) {
            // Can use fast path: GED is certainly > tau
            used_dummy_fast_path = true;
            dummy_fast_path_ged = tau_ui + 1;
            stats.EPT_reuse_attempt++;
            stats.EPT_reuse_count++;
            stats.EPT_reuse_fail_parent_snapshot_size_one++;
            stats.EPT_reuse_fail_parent_snapshot_size_one_dummy++;
        }
    }

    // Lower Bound check (only executed for non-fast path)
    bool should_compute_ged = !used_dummy_fast_path;

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    // LB Propagation: if estimate_lb > tau, skip computation
    if (!used_dummy_fast_path && estimate_lb > tau_ui) {
        should_compute_ged = false;
        stats.lb_pruning_count++;  // Count App_test computations skipped due to estimate_lb > tau（unified version）
#ifdef DEBUG_PRUNING
        printf("[STATS] LB propagation skip (unified): count=%zu, estimate_lb=%u, tau=%u\n",
               stats.lb_pruning_count, estimate_lb, tau_ui);
#endif
    }
#endif

    if (use_ept_filters && should_compute_ged) {
        // NDC statistics：only count when actually computing LB
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }

        auto lb_start = std::chrono::high_resolution_clock::now();

        stats.EPT_lb_count++;
        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);
        if (lb > tau_ui) {
            should_compute_ged = false;
            // Even if GED computation skipped, use LB to update estimate_lb for child nodes
            new_estimate_lb = std::max(new_estimate_lb, lb);
        }

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        // Track filter time separately for db graphs and intermediate graphs
        if (is_db_graph) {
            stats.EPT_db_graph_lb_count++;
            stats.EPT_db_graph_lb_time += lb_duration;
        } else {
            stats.EPT_intermediate_graph_lb_count++;
            stats.EPT_intermediate_graph_lb_time += lb_duration;
        }
    }

    // If EPT filters or only_compute_db_graph is enabled, only compute GED for nodes with completed_db_graph_idsnodes compute GED
    // Root node (anchor itself) even if completed_db_graph_ids is empty stillNeed to computeGED，because anchor itself may be a result
    if ((use_ept_filters || only_compute_db_graph) && should_compute_ged && node.completed_db_graph_ids.empty() && node_index != ept.root_index) {
        should_compute_ged = false;
    }

    // ========== Root node optimization: use GED cached by NetDag,to avoid duplicateA* computation ==========
    // GS_search phase already computed exact GED for anchor; if no snapshot needed for children, A* can be skipped
    bool root_used_netdag_cache = false;
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF && should_compute_ged) {
        // Check if close child nodes need snapshot
        bool needs_snapshot = false;
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            int distance = ept.tree_nodes[ch].level - node.level;
            if (distance > 0 && distance <= MAX_GED_GAP) {
                needs_snapshot = true;
                break;
            }
        }
#endif

        if (!needs_snapshot) {
            // No snapshot needed, directly use GED cached by NetDag
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
        // else: Need snapshot for child node reuse, let subsequent A* run normally
    }

    // GED computation
    if (should_compute_ged) {
        // NDC statistics：If LB skipped but A* computed, still count
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }
        bool use_baseline = false;
        int optimal_margin = 0;
        bool has_close_child = false;  // moved to outer scope for later reuse
        
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        // Use level field instead of accumulated_ops.size()
        int current_level = node.level;

        // Compute optimal_margin and set has_close_child
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;

            const TreeNode &child_node = ept.tree_nodes[ch];
            int child_level = child_node.level;
            int distance = child_level - current_level;

            // If a close child node found (distance must be >0 to count as real child)
            if (distance > 0 && distance <= MAX_GED_GAP) {
                has_close_child = true;
                optimal_margin = std::max(optimal_margin, distance);
            }
        }
        
        // Limit margin to not exceed MAX_MARGIN
        optimal_margin = std::min(optimal_margin, MAX_MARGIN);

        // If no close child nodes, use baseline (except for root node)
        if (!has_close_child && !node.children_indices.empty() && node_index != ept.root_index) {
            use_baseline = true;
            optimal_margin = 0;  // baseline is equivalent to margin=0
        }
#endif
        
        // Determine if reuse is possible
        bool can_reuse = false;
        
        enum ReuseFailReason {
            NO_FAILURE = 0,
            NO_PARENT_SNAPSHOT,           // Parent node did not compute GED with AStar (or AStar reuse)
            PARENT_SNAPSHOT_EMPTY,        // Parent node computed but extract is empty (v.size()==0)
            PARENT_SNAPSHOT_SIZE_ONE,     // Parent node computed but only 1 node (v.size()==1)
            ROOT_NODE,
            NO_OP,
            MULTI_OPS,
            VERTEX_COUNT_CHANGED,
            MO_INCOMPATIBLE,
            USE_BASELINE_DISTANT
        } fail_reason = NO_FAILURE;
        
        if (use_baseline) {
            fail_reason = USE_BASELINE_DISTANT;
        } else {
            // Modified reuse condition check: try reuse even if parent_snapshot is empty
            if (node_index == ept.root_index) {
                stats.EPT_reuse_fail_root_node++;
                fail_reason = ROOT_NODE;
            } else if (node.op.type == EditOperation::NONE) {
                stats.EPT_reuse_fail_no_op++;
                fail_reason = NO_OP;
            } else if (parent_snapshot == nullptr || parent_db_vertex_count < 0) {
                // Truly no parent node (root node or parent node did not compute GED)
                stats.EPT_reuse_fail_no_parent_snapshot++;
                fail_reason = NO_PARENT_SNAPSHOT;
            } else if (node.accumulated_ops.size() > MAX_GED_GAP) {
                stats.EPT_reuse_fail_multi_ops++;
                fail_reason = MULTI_OPS;
            } else {
                // Further check reuse conditions (relaxed parent_snapshot->v size restriction)
                if (db_g->n != parent_db_vertex_count) {
                    stats.EPT_reuse_fail_vertex_count_changed++;
                    fail_reason = VERTEX_COUNT_CHANGED;
                } else {
                    // Check mo array compatibility
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
                        stats.EPT_reuse_fail_mo_incompatible++;
                        fail_reason = MO_INCOMPATIBLE;
                    } else if (parent_snapshot->v.empty()) {
                        stats.EPT_reuse_fail_parent_snapshot_empty++;
                        fail_reason = PARENT_SNAPSHOT_EMPTY;
                    } else if (parent_snapshot->v.size() == 1) {
                        stats.EPT_reuse_fail_parent_snapshot_size_one++;
                        // dummy-only already handled above, only process real nodes herenode
                        if (!(parent_snapshot->v[0].level < 0 || parent_snapshot->v[0].image < 0)) {
                            // Realnodecan reuse!
                            stats.EPT_reuse_fail_parent_snapshot_size_one_real++;
                            can_reuse = true;
                            stats.EPT_reuse_attempt++;
                            fail_reason = NO_FAILURE;
                        } else {
                            // Dummynode（already checked above, should not satisfy fast path condition）
                            stats.EPT_reuse_fail_parent_snapshot_size_one_dummy++;
                            fail_reason = PARENT_SNAPSHOT_SIZE_ONE;
                        }
                    } else {  // v.size() >= 2
                        can_reuse = true;
                        stats.EPT_reuse_attempt++;
                        fail_reason = NO_FAILURE;
                    }
                }
            }
        }
        
        Application app(tau_ui, "BMao", app_max_iter);
        app.init(db_g, qry_g);
        app.set_disable_reuse_lsa(disable_reuse_lsa);

        int ged = -1;
        bool used_reuse = false;
        double elapsed_time = 0.0;

        if (can_reuse) {
            // When chain_reuse=false, reuse nodes do not save snapshot, so margin is not needed
            if (chain_reuse) {
                app.set_margin(optimal_margin);  // chain reuse needs to save snapshot for subsequent use
            } else {
                app.set_margin(0);  // No need to save snapshot for subsequent use, use margin=0 to reduce computation
            }
            // Normal reuse path (dummy-only already handled above)
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.app_reuse(*parent_snapshot, tau_ui, node.accumulated_ops.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            used_reuse = (ged >= 0);
        }

        if (!used_reuse) {
            // AStar needs to set margin based on children distance (may need to save snapshot)
            app.set_margin(optimal_margin);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            stats.EPT_astar_count++;
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.App();  // using already set optimal_margin
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
        }

        if (used_reuse) {
            stats.EPT_reuse_count++;
            if (node.op.type != EditOperation::NONE) {
                stats.EPT_reuse_by_op_type[node.op.type]++;
            }
            
            // verify reuse effectiveness (controlled by command line parameter)
            if (verify_reuse_baseline) {
                // Use AppForComputation as baseline (with same tau upper bound)
                Application app_verify(tau_ui, "BMao", app_max_iter);
                app_verify.init(db_g, qry_g);

                auto t_verify_start = std::chrono::high_resolution_clock::now();
                int ged_standard = app_verify.AppForComputation(nullptr, nullptr);
                auto t_verify_end = std::chrono::high_resolution_clock::now();

                double verify_time = std::chrono::duration<double>(t_verify_end - t_verify_start).count();
                stats.EPT_verification_astar_time += verify_time;
                stats.EPT_reuse_verifications++;

                // AppForComputation baseline statistics
                stats.EPT_baseline_app_count++;
                stats.EPT_baseline_app_time += verify_time;
                stats.EPT_baseline_reuse_time += elapsed_time;  // record corresponding reuse time

                // Collectsample pairs (up to MAX_BASELINE_SAMPLES )
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
            }
        }
        
        stats.EPT_astar_time += elapsed_time;

        // Track verification time separately for db graphs and intermediate graphs
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
                case NO_OP:
                    stats.EPT_astar_time_no_op += elapsed_time;
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
        // LB Propagation: Update estimate_lb
        new_estimate_lb = app.get_overall_lb();
        // Root node: combine NetDag's lb, take tighter value for child propagation
        if (node_index == ept.root_index && anchor_netdag_lb >= 0) {
            new_estimate_lb = std::max(new_estimate_lb, static_cast<ui>(anchor_netdag_lb));
        }
#endif

#if USE_SUBTREE_PRUNING
        ui overall_lb = app.get_overall_lb();
        // Root node: combine NetDag's lb, take tighter value for Subtree Pruning
        if (node_index == ept.root_index && anchor_netdag_lb >= 0) {
            overall_lb = std::max(overall_lb, static_cast<ui>(anchor_netdag_lb));
        }

        // Subtree Pruning: Leaf nodeskip (no child nodes to prune)
        if (!node.children_indices.empty() && overall_lb > 0 && overall_lb < 10000) {
            int max_step_more;

#ifdef USE_DYNAMIC_DEPTH_PROBE
            max_step_more = probe_max_depth(node_index);
#else
            max_step_more = MAX_EPT_DEPTH - node.level;
#endif

            if (max_step_more + (int)tau_ui < (int)overall_lb) {
                // Count subtree pruning effectiveness（unified version）
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // minus current node itself
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                return found_here;
            }
        }
#endif

        if (ged <= (int)tau_ui) {
            // For root node, ensure anchor itself is added to results (old EPT files may not include anchor_id)
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

            // Add all graphs from completed_db_graph_ids
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
        } else {
            // GED > tau, rejected
        }
        
        // ========== Extract snapshot (for child node reuse)==========
        // Only generate snapshot when close children exist
        // When chain_reuse=true, reuse nodes also save snapshot for subsequent node reuse
        bool should_save_snapshot = (ged >= 0 && ged < INT_MAX && has_close_child);
        if (!chain_reuse) {
            should_save_snapshot = should_save_snapshot && !used_reuse;
        }
        if (should_save_snapshot) {
            current_snapshot = std::make_shared<SearchSnapshot>();
            
            try {
                app.extract_snapshot(*current_snapshot);
                current_snapshot->ub = ged;
                current_snapshot->margin = optimal_margin;  // save margin info

                if (current_snapshot->v.empty()) {
                    current_snapshot = nullptr;
                }
            } catch (...) {
                current_snapshot = nullptr;
            }
        }
    }
#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    else if (used_dummy_fast_path) {
        // Used dummy-only fast path, directly use results
        // dummy fast path GED is always > tau, so there are no results
        // Note: this is not true search tree reuse, does notNeed to computebaseline

        // No snapshot needed, since fast path was used
        current_snapshot = nullptr;
    }
    else {
        // Computation skipped (LB Propagation etc.), but still check if prunable
#if USE_SUBTREE_PRUNING
        // Subtree Pruning: Leaf nodeskip (no child nodes to prune)
        if (!node.children_indices.empty() && new_estimate_lb > 0 && new_estimate_lb < 10000) {
            int max_step_more;

#ifdef USE_DYNAMIC_DEPTH_PROBE
            max_step_more = probe_max_depth(node_index);
#else
            max_step_more = MAX_EPT_DEPTH - node.level;
#endif

            if (max_step_more + (int)tau_ui < (int)new_estimate_lb) {
                // Count subtree pruning effectiveness（unified version，estimate_lb branch）
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1; // minus current node itself
                stats.subtree_pruning_avoided_nodes += avoided_nodes;

                return found_here;
            }
        }
#endif
    }
#endif
    
    // RecursivelyProcess child nodes
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }
        
        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        ui child_estimate_lb = (level_diff >= (int)new_estimate_lb) ?
                              0 : new_estimate_lb - level_diff;

        // Count times estimate_lb is propagated to child nodes
        stats.lb_propagation_count++;

        if (dfs_traverse(ch, ept, query_node,
                                exact_results_within_tau, tau, stats,
                                current_snapshot,
                                db_g->n,
                                child_estimate_lb,
                                anchor_netdag_lb,
                                anchor_netdag_ged)) {
            found_child = true;
        }
    }

    return found_here || found_child;
}

// ========== dfs_traverse_no_SP: Same as unified but with Subtree Pruning disabled ==========
// Keep Reuse + Distance Propagation, but do not use Subtree Pruning
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
    int                             anchor_netdag_ged)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;
    ui new_estimate_lb = estimate_lb;

    std::shared_ptr<SearchSnapshot> current_snapshot = nullptr;

    // NDC counting flag: ensure each node counted only once during actual computation
    bool ndc_counted = false;

    // Node type statistics
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

    // dummy-only fast path
    bool used_dummy_fast_path = false;
    int dummy_fast_path_ged = -1;

    if (parent_snapshot && parent_snapshot->v.size() == 1 &&
        (parent_snapshot->v[0].level < 0 || parent_snapshot->v[0].image < 0)) {
        int ged_gap = (node.parent_index < ept.tree_nodes.size()) ?
                     (node.level - ept.tree_nodes[node.parent_index].level) : 1;

        if (ged_gap > 0 && ged_gap <= parent_snapshot->margin) {
            used_dummy_fast_path = true;
            dummy_fast_path_ged = tau_ui + 1;
            stats.EPT_reuse_attempt++;
            stats.EPT_reuse_count++;
            stats.EPT_reuse_fail_parent_snapshot_size_one++;
            stats.EPT_reuse_fail_parent_snapshot_size_one_dummy++;
        }
    }

    bool should_compute_ged = !used_dummy_fast_path;

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    // LB Propagation: if estimate_lb > tau, skip computation
    if (!used_dummy_fast_path && estimate_lb > tau_ui) {
        should_compute_ged = false;
        stats.lb_pruning_count++;
    }
#endif

    if (use_ept_filters && should_compute_ged) {
        // NDC statistics：only count when actually computing LB
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }

        auto lb_start = std::chrono::high_resolution_clock::now();

        stats.EPT_lb_count++;
        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);
        if (lb > tau_ui) {
            should_compute_ged = false;
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

    // ========== Root node optimization: use GED cached by NetDag,to avoid duplicateA* computation ==========
    bool root_used_netdag_cache = false;
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF && should_compute_ged) {
        bool needs_snapshot = false;
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            int distance = ept.tree_nodes[ch].level - node.level;
            if (distance > 0 && distance <= MAX_GED_GAP) {
                needs_snapshot = true;
                break;
            }
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

    // GED computation (Same as unified, but without Subtree Pruning)
    if (should_compute_ged) {
        // NDC statistics：If LB skipped but A* computed, still count
        if (!ndc_counted && is_db_graph) {
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
            if (distance > 0 && distance <= MAX_GED_GAP) {
                has_close_child = true;
                optimal_margin = std::max(optimal_margin, distance);
            }
        }
        optimal_margin = std::min(optimal_margin, MAX_MARGIN);
        if (!has_close_child && !node.children_indices.empty() && node_index != ept.root_index) {
            use_baseline = true;
            optimal_margin = 0;
        }
#endif

        bool can_reuse = false;

        if (!use_baseline) {
            if (node_index == ept.root_index) {
                stats.EPT_reuse_fail_root_node++;
            } else if (node.op.type == EditOperation::NONE) {
                stats.EPT_reuse_fail_no_op++;
            } else if (parent_snapshot == nullptr || parent_db_vertex_count < 0) {
                stats.EPT_reuse_fail_no_parent_snapshot++;
            } else if (node.accumulated_ops.size() > MAX_GED_GAP) {
                stats.EPT_reuse_fail_multi_ops++;
            } else {
                if (db_g->n != parent_db_vertex_count) {
                    stats.EPT_reuse_fail_vertex_count_changed++;
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
                        stats.EPT_reuse_fail_mo_incompatible++;
                    } else if (parent_snapshot->v.empty()) {
                        stats.EPT_reuse_fail_parent_snapshot_empty++;
                    } else if (parent_snapshot->v.size() == 1) {
                        stats.EPT_reuse_fail_parent_snapshot_size_one++;
                        if (!(parent_snapshot->v[0].level < 0 || parent_snapshot->v[0].image < 0)) {
                            stats.EPT_reuse_fail_parent_snapshot_size_one_real++;
                            can_reuse = true;
                            stats.EPT_reuse_attempt++;
                        } else {
                            stats.EPT_reuse_fail_parent_snapshot_size_one_dummy++;
                        }
                    } else {
                        can_reuse = true;
                        stats.EPT_reuse_attempt++;
                    }
                }
            }
        }

        Application app(tau_ui, "BMao", app_max_iter);
        app.init(db_g, qry_g);
        app.set_disable_reuse_lsa(disable_reuse_lsa);

        int ged = -1;
        bool used_reuse = false;
        double elapsed_time = 0.0;

        if (can_reuse) {
            // When chain_reuse=false, reuse nodes do not save snapshot, so margin is not needed
            if (chain_reuse) {
                app.set_margin(optimal_margin);  // chain reuse needs to save snapshot for subsequent use
            } else {
                app.set_margin(0);  // No need to save snapshot for subsequent use, use margin=0 to reduce computation
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.app_reuse(*parent_snapshot, tau_ui, node.accumulated_ops.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            used_reuse = (ged >= 0);
        }

        if (!used_reuse) {
            app.set_margin(optimal_margin);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            stats.EPT_astar_count++;
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.App();
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
        }

        if (used_reuse) {
            stats.EPT_reuse_count++;
            if (node.op.type != EditOperation::NONE) {
                stats.EPT_reuse_by_op_type[node.op.type]++;
            }
        }

        stats.EPT_astar_time += elapsed_time;

        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

#ifdef USE_ESTIMATE_LB_OPTIMIZATION
        // LB Propagation: Update estimate_lb
        new_estimate_lb = app.get_overall_lb();
#endif

        // Note: intentionally skipping Subtree Pruning check here

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

        // Extract snapshot (for child node reuse)
        // When chain_reuse=true, reuse nodes also save snapshot for subsequent node reuse
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
                if (current_snapshot->v.empty()) {
                    current_snapshot = nullptr;
                }
            } catch (...) {
                current_snapshot = nullptr;
            }
        }
    }
#ifdef USE_ESTIMATE_LB_OPTIMIZATION
    else if (used_dummy_fast_path) {
        current_snapshot = nullptr;
    }
    // Note: also skip Subtree Pruning check here (else branch)
#endif

    // RecursivelyProcess child nodes
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
                              anchor_netdag_lb, anchor_netdag_ged)) {
            found_child = true;
        }
    }

    return found_here || found_child;
}

// ========== dfs_traverse_no_LP: Same as unified but with Distance Propagation disabled ==========
// Keep Reuse + Subtree Pruning, but do not use Distance Propagation (LB Propagation)
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
    int                             anchor_netdag_ged)
{
    const TreeNode &node = ept.tree_nodes[node_index];
    bool found_here = false;

    Graph *db_g = node.db_graph;
    Graph *qry_g = query_node->graph.get();

    ui tau_ui = (ui)tau;
    ui new_estimate_lb = estimate_lb;  // will not be updated (DP disabled)

    std::shared_ptr<SearchSnapshot> current_snapshot = nullptr;

#ifdef USE_DYNAMIC_DEPTH_PROBE
    // Subtree Pruning needs dynamic depth probing
    auto probe_max_depth = [&ept](size_t node_idx) -> int {
        const TreeNode& current = ept.tree_nodes[node_idx];
        int current_level = current.level;

        std::function<int(size_t)> find_max_level = [&](size_t idx) -> int {
            if (idx >= ept.tree_nodes.size()) {
                return current_level;
            }
            const TreeNode& n = ept.tree_nodes[idx];
            int max_level = n.level;
            for (size_t ch : n.children_indices) {
                int child_max = find_max_level(ch);
                max_level = std::max(max_level, child_max);
            }
            return max_level;
        };

        int max_level_in_subtree = find_max_level(node_idx);
        return max_level_in_subtree - current_level;
    };
#endif

    // NDC counting flag: ensure each node counted only once during actual computation
    bool ndc_counted = false;

    // Node type statistics
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

    // dummy-only fast path
    bool used_dummy_fast_path = false;
    int dummy_fast_path_ged = -1;

    if (parent_snapshot && parent_snapshot->v.size() == 1 &&
        (parent_snapshot->v[0].level < 0 || parent_snapshot->v[0].image < 0)) {
        int ged_gap = (node.parent_index < ept.tree_nodes.size()) ?
                     (node.level - ept.tree_nodes[node.parent_index].level) : 1;

        if (ged_gap > 0 && ged_gap <= parent_snapshot->margin) {
            used_dummy_fast_path = true;
            dummy_fast_path_ged = tau_ui + 1;
            stats.EPT_reuse_attempt++;
            stats.EPT_reuse_count++;
            stats.EPT_reuse_fail_parent_snapshot_size_one++;
            stats.EPT_reuse_fail_parent_snapshot_size_one_dummy++;
        }
    }

    bool should_compute_ged = !used_dummy_fast_path;

    // Note: intentionally skipping LB Propagation estimate_lb > tau check

    ui ept_filter_lb = 0;  // Save EPT filter lb for current node SP use (do not pass to child nodes)

    if (use_ept_filters && should_compute_ged) {
        // NDC statistics：only count when actually computing LB
        if (!ndc_counted && is_db_graph) {
            stats.EPT_ndc_count++;
            ndc_counted = true;
        }

        auto lb_start = std::chrono::high_resolution_clock::now();

        stats.EPT_lb_count++;
        ui lb = qry_g->ged_lower_bound_filter(db_g, tau_ui, vM.size(), eM.size(), max_n);
        ept_filter_lb = lb;  // save lb for SP use
        if (lb > tau_ui) {
            should_compute_ged = false;
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

    // ========== Root node optimization: use GED cached by NetDag,to avoid duplicateA* computation ==========
    bool root_used_netdag_cache = false;
    if (node_index == ept.root_index && anchor_netdag_ged >= 0 && anchor_netdag_ged < (int)INF && should_compute_ged) {
        bool needs_snapshot = false;
#if USE_BASELINE_FOR_DISTANT_CHILDREN
        for (size_t ch : node.children_indices) {
            if (ch >= ept.tree_nodes.size()) continue;
            int distance = ept.tree_nodes[ch].level - node.level;
            if (distance > 0 && distance <= MAX_GED_GAP) {
                needs_snapshot = true;
                break;
            }
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
            // Note: intentionally not updating new_estimate_lb (LP disabled)
            should_compute_ged = false;
            root_used_netdag_cache = true;
            stats.root_netdag_ged_reuse_count++;
        }
    }

    // GED computation (Same as unified, but without LB Propagation update)
    if (should_compute_ged) {
        // NDC statistics：If LB skipped but A* computed, still count
        if (!ndc_counted && is_db_graph) {
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
            if (distance > 0 && distance <= MAX_GED_GAP) {
                has_close_child = true;
                optimal_margin = std::max(optimal_margin, distance);
            }
        }
        optimal_margin = std::min(optimal_margin, MAX_MARGIN);
        if (!has_close_child && !node.children_indices.empty() && node_index != ept.root_index) {
            use_baseline = true;
            optimal_margin = 0;
        }
#endif

        bool can_reuse = false;

        if (!use_baseline) {
            if (node_index == ept.root_index) {
                stats.EPT_reuse_fail_root_node++;
            } else if (node.op.type == EditOperation::NONE) {
                stats.EPT_reuse_fail_no_op++;
            } else if (parent_snapshot == nullptr || parent_db_vertex_count < 0) {
                stats.EPT_reuse_fail_no_parent_snapshot++;
            } else if (node.accumulated_ops.size() > MAX_GED_GAP) {
                stats.EPT_reuse_fail_multi_ops++;
            } else {
                if (db_g->n != parent_db_vertex_count) {
                    stats.EPT_reuse_fail_vertex_count_changed++;
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
                        stats.EPT_reuse_fail_mo_incompatible++;
                    } else if (parent_snapshot->v.empty()) {
                        stats.EPT_reuse_fail_parent_snapshot_empty++;
                    } else if (parent_snapshot->v.size() == 1) {
                        stats.EPT_reuse_fail_parent_snapshot_size_one++;
                        if (!(parent_snapshot->v[0].level < 0 || parent_snapshot->v[0].image < 0)) {
                            stats.EPT_reuse_fail_parent_snapshot_size_one_real++;
                            can_reuse = true;
                            stats.EPT_reuse_attempt++;
                        } else {
                            stats.EPT_reuse_fail_parent_snapshot_size_one_dummy++;
                        }
                    } else {
                        can_reuse = true;
                        stats.EPT_reuse_attempt++;
                    }
                }
            }
        }

        Application app(tau_ui, "BMao", app_max_iter);
        app.init(db_g, qry_g);
        app.set_disable_reuse_lsa(disable_reuse_lsa);

        int ged = -1;
        bool used_reuse = false;
        double elapsed_time = 0.0;

        if (can_reuse) {
            // When chain_reuse=false, reuse nodes do not save snapshot, so margin is not needed
            if (chain_reuse) {
                app.set_margin(optimal_margin);  // chain reuse needs to save snapshot for subsequent use
            } else {
                app.set_margin(0);  // No need to save snapshot for subsequent use, use margin=0 to reduce computation
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.app_reuse(*parent_snapshot, tau_ui, node.accumulated_ops.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
            used_reuse = (ged >= 0);
        }

        if (!used_reuse) {
            app.set_margin(optimal_margin);
            app.set_disable_lsa_pruning(disable_lsa_pruning);
            stats.EPT_astar_count++;
            auto t0 = std::chrono::high_resolution_clock::now();
            ged = app.App();
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration<double>(t1 - t0).count();
        }

        if (used_reuse) {
            stats.EPT_reuse_count++;
            if (node.op.type != EditOperation::NONE) {
                stats.EPT_reuse_by_op_type[node.op.type]++;
            }
        }

        stats.EPT_astar_time += elapsed_time;

        if (is_db_graph) {
            stats.EPT_db_graph_astar_count++;
            stats.EPT_db_graph_astar_time += elapsed_time;
        } else {
            stats.EPT_intermediate_graph_astar_count++;
            stats.EPT_intermediate_graph_astar_time += elapsed_time;
        }

        // Note: intentionally not updating new_estimate_lb (DP disabled)

        // Subtree Pruning check (retained)
#if USE_SUBTREE_PRUNING
        ui overall_lb = app.get_overall_lb();

        if (!node.children_indices.empty() && overall_lb > 0 && overall_lb < 10000) {
            int max_step_more;

#ifdef USE_DYNAMIC_DEPTH_PROBE
            max_step_more = probe_max_depth(node_index);
#else
            max_step_more = MAX_EPT_DEPTH - node.level;
#endif

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

        // Extract snapshot (for child node reuse)
        // When chain_reuse=true, reuse nodes also save snapshot for subsequent node reuse
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
                if (current_snapshot->v.empty()) {
                    current_snapshot = nullptr;
                }
            } catch (...) {
                current_snapshot = nullptr;
            }
        }
    }
    else if (used_dummy_fast_path) {
        current_snapshot = nullptr;
    }
    else {
        // Computation skipped, but still check Subtree Pruning (retained)
        // Use current node's EPT filter computed lb (not propagated from parent node's estimate_lb)
#if USE_SUBTREE_PRUNING
        if (!node.children_indices.empty() && ept_filter_lb > 0 && ept_filter_lb < 10000) {
            int max_step_more;

#ifdef USE_DYNAMIC_DEPTH_PROBE
            max_step_more = probe_max_depth(node_index);
#else
            max_step_more = MAX_EPT_DEPTH - node.level;
#endif

            if (max_step_more + (int)tau_ui < (int)ept_filter_lb) {
                stats.subtree_pruning_decisions++;
                size_t avoided_nodes = calculate_subtree_size(ept, node_index) - 1;
                stats.subtree_pruning_avoided_nodes += avoided_nodes;
                return found_here;
            }
        }
#endif
    }

    // RecursivelyProcess child nodes
    bool found_child = false;
    for (size_t ch : node.children_indices) {
        if (ch >= ept.tree_nodes.size()) {
            continue;
        }

        const TreeNode &child_node = ept.tree_nodes[ch];
        int level_diff = child_node.level - node.level;
        // Note: since DP is disabled, new_estimate_lb will not be updated, passing 0 here
        ui child_estimate_lb = 0;

        stats.lb_propagation_count++;

        if (dfs_traverse_no_LP(ch, ept, query_node,
                              exact_results_within_tau, tau, stats,
                              current_snapshot, db_g->n, child_estimate_lb,
                              anchor_netdag_lb, anchor_netdag_ged)) {
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

    // Use override parameter (if any), otherwise use member variable, avoid data race in parallel mode
    const std::string &effective_dfs_mode = dfs_mode_override.empty() ? this->dfs_mode : dfs_mode_override;

    // Select DFS traversal method based on dfs_mode parameter (default: dfs_traverse with all optimizations)
    if (effective_dfs_mode == "no_reuse") {
        // Use no_reuse version (SP + DP optimizations, but no search tree reuse)
        ui initial_estimate_lb = 0;

        dfs_traverse_no_reuse(ept.root_index, ept, query_node,
                           exact_results_within_tau, tau, stats,
                           initial_estimate_lb, anchor_netdag_lb, anchor_netdag_ged);
    } else if (effective_dfs_mode == "only_dfs") {
        // Use pure DFS traversal (App_baseline, no optimizations)
        dfs_traverse_only_dfs(ept.root_index, ept, query_node,
                             exact_results_within_tau, tau, stats,
                             anchor_netdag_lb, anchor_netdag_ged);
    } else if (effective_dfs_mode == "no_SP") {
        // Use no_SP version (reuse + DP optimizations, but no Subtree Pruning)
        ui initial_estimate_lb = 0;
        dfs_traverse_no_SP(ept.root_index, ept, query_node,
                          exact_results_within_tau, tau, stats,
                          nullptr, -1, initial_estimate_lb,
                          anchor_netdag_lb, anchor_netdag_ged);
    } else if (effective_dfs_mode == "no_LP") {
        // Use no_LP version (reuse + SP optimizations, but no Distance Propagation)
        ui initial_estimate_lb = 0;
        dfs_traverse_no_LP(ept.root_index, ept, query_node,
                          exact_results_within_tau, tau, stats,
                          nullptr, -1, initial_estimate_lb,
                          anchor_netdag_lb, anchor_netdag_ged);
    } else {
        // Default: use dfs_traverse (all optimizations: Reuse + SP + LP)
        ui initial_estimate_lb = 0;
        dfs_traverse(ept.root_index, ept, query_node,
                            exact_results_within_tau, tau, stats,
                            nullptr, -1, initial_estimate_lb,
                            anchor_netdag_lb, anchor_netdag_ged);
    }
}


std::vector<int> GismaSearchEngine::EPT_forbidden_cluster_search(std::shared_ptr<Node> query_node,
                                                            int anchor_id,
                                                            double tau,
                                                            SearchStats &stats)
{
    std::vector<int> results;

    // 1) Get corresponding Anchor
    auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
    if (!anchor_ptr)
    {
        // If not obtainable or not Anchor, returnempty
        return results;
    }

    Graph *query_graph = query_node->graph.get();
    if (!query_graph)
    {
        return results;
    }

    // Record time
    double extra_astar_time = 0.0;
    double extra_lb_time = 0.0;

    // ========== A) First compute LB between anchor->graph and query_graph ==========
    stats.EPT_ndc_count++;  // NDC: checking anchor itself
    auto anchor_lb_start = std::chrono::high_resolution_clock::now();
    stats.EPT_lb_count++;

    ui anchor_lb = query_graph->ged_lower_bound_filter(
        anchor_ptr->graph.get(), static_cast<ui>(tau), vM.size(), eM.size(), max_n);

    auto anchor_lb_end = std::chrono::high_resolution_clock::now();
    extra_lb_time += std::chrono::duration<double>(anchor_lb_end - anchor_lb_start).count();

    if (anchor_lb <= tau)
    {
        // LB passed, call A* for exact GED
        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;

        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.init(anchor_ptr->graph.get(), query_graph);
        int ged_res = app.AppForComputation(nullptr, nullptr);

        auto astar_end = std::chrono::high_resolution_clock::now();
        extra_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        // If A* result <= tau => add to results
        if (ged_res <= (int)tau)
        {
            results.push_back(anchor_id);
        }
    }

    // ========== B) Iterate anchor_ptr->nodes_in_exact_cluster_vec with triangle inequality check + lower bound check + A* ==========

    const auto &clusterCopy = anchor_ptr->nodes_in_exact_cluster_vec;

    

    for (auto &pair_item : clusterCopy)
    {
        double dist_anchor_to_node = pair_item.first;
        int cluster_node_id = pair_item.second;

        // (B.1) simple triangle inequality filtering:
        // std::cout << "anchorDist: " << anchorDist << ", dist_anchor_to_node: " << dist_anchor_to_node << std::endl;

        // if (anchorDist - dist_anchor_to_node > tau)
        // {
        //     // means (node->query) is at least > tau, can skip
        //     continue;
        // }

        // ========== LB computation ==========
        stats.EPT_ndc_count++;  // NDC: clusternode check
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        ui lb = query_graph->ged_lower_bound_filter(
            db[cluster_node_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        extra_lb_time += std::chrono::duration<double>(lb_end - lb_start).count();

        if (lb > tau)
        {
            continue;
        }

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;

        Application app(static_cast<ui>(tau), "BMao", app_max_iter);
        app.init(db[cluster_node_id], query_graph);
        app.set_disable_lsa_pruning(disable_lsa_pruning);
        int ged_res = app.App();

        auto astar_end = std::chrono::high_resolution_clock::now();
        extra_astar_time += std::chrono::duration<double>(astar_end - astar_start).count();

        // ========== 7) If A* result <= tau => add to results ==========
        if (ged_res <= (int)tau)
        {
            results.push_back(cluster_node_id);
        }
    }

    // Collect timing - using precisely accumulated time
    stats.EPT_lb_time += extra_lb_time;
    stats.EPT_astar_time += extra_astar_time;

    return results;
}

std::vector<int> GismaSearchEngine::extra_cluster_search(std::shared_ptr<Node> query_node,
                                                    int anchor_id,
                                                    double tau,
                                                    SearchStats &stats)
{
    std::vector<int> results;

    // Get Anchor object by anchor_id
    auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
    if (!anchor_ptr)
    {
        // If not obtainable or not Anchor, returnemptyResults
        return results;
    }

    Graph *query_graph = query_node->graph.get();
    if (!query_graph)
    {
        return results;
    }

    // 1) Directly use anchor_ptr->nodes_in_cluster_vec
    const auto &clusterCopy = anchor_ptr->nodes_in_cluster_vec;

    // 2) Iterate clusterCopy
    for (auto &pair_item : clusterCopy)
    {
        double dist = pair_item.first;
        int cluster_node_id = pair_item.second;

        // ========== LB computation ==========
        stats.EPT_ndc_count++;  // NDC: clusternode check
        auto lb_start = std::chrono::high_resolution_clock::now();
        stats.EPT_lb_count++;

        // Use same comprehensive filter as Base+GS (fix recall issue)
        ui lb = query_graph->ged_lower_bound_filter(
            db[cluster_node_id], static_cast<ui>(tau), vM.size(), eM.size(), max_n);

        auto lb_end = std::chrono::high_resolution_clock::now();
        double lb_duration = std::chrono::duration<double>(lb_end - lb_start).count();
        stats.EPT_lb_time += lb_duration;

        // graphs in extra_cluster are all db graphs
        stats.EPT_db_graph_lb_count++;
        stats.EPT_db_graph_lb_time += lb_duration;

        if (lb > tau)
        {
            continue;
        }

        // ========== A* ==========
        auto astar_start = std::chrono::high_resolution_clock::now();
        stats.EPT_astar_count++;

        Application app((ui)tau, "BMao", app_max_iter);
        app.init(db[cluster_node_id], query_graph);
        app.set_disable_lsa_pruning(disable_lsa_pruning);
        int ged_res = app.App();

        auto astar_end = std::chrono::high_resolution_clock::now();
        double astar_duration = std::chrono::duration<double>(astar_end - astar_start).count();
        stats.EPT_astar_time += astar_duration;

        // graphs in extra_cluster are all db graphs
        stats.EPT_db_graph_astar_count++;
        stats.EPT_db_graph_astar_time += astar_duration;

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

    // 1) GS => obtain anchor IDs with NetDag LB distances
    auto nd_start = std::chrono::high_resolution_clock::now();
    std::vector<std::tuple<int, double, int>> candidate_anchors_with_results = GS_search(query_node, tau, stats);

    if (candidate_anchors_with_results.empty())
    {
        // ND_total_time is now computed as ND_lb_time + ND_astar_time in print_summary()
        // auto nd_end = std::chrono::high_resolution_clock::now();
        // stats.ND_total_time += std::chrono::duration<double>(nd_end - nd_start).count();
        return {};
    }
    // ND_total_time is now computed as ND_lb_time + ND_astar_time in print_summary()
    // auto nd_end = std::chrono::high_resolution_clock::now();
    // stats.ND_total_time += std::chrono::duration<double>(nd_end - nd_start).count();

    // 2) Perform SS_search for each anchor ID
    auto EPT_start = std::chrono::high_resolution_clock::now();

    // For summarizing all SS_search durations and corresponding anchor_ids
    std::vector<std::pair<int, double>> ss_time_records;
    ss_time_records.reserve(candidate_anchors_with_results.size());

    std::vector<int> results;
    std::vector<int> ept_results_all;      // DEBUG: Collect all EPTResults
    std::vector<int> extra_results_all;    // DEBUG: Collect all extra_clusterResults
    std::unordered_set<int> ept_coverage;    // DEBUG: all graph IDs covered by EPT
    std::unordered_set<int> extra_coverage;  // DEBUG: all graph IDs covered by extra cluster

    for (const auto& [anchor_id, netdag_lb, netdag_ged] : candidate_anchors_with_results)
    {
        // DEBUG: Collect EPT and Extra coverage
        auto anchor_ptr = std::dynamic_pointer_cast<Anchor>(net_dag->nodes[anchor_id]);
        if (anchor_ptr) {
            // EPT coverage: get all graph IDs from EPT tree itself
            // EPT results only come from: ept.anchor_id and node.completed_db_graph_ids
            EditPathTree *ept = ept_manager->get_ept_no_lock(anchor_id);
            if (ept) {
                ept_coverage.insert(ept->anchor_id);  // anchor itself
                for (const auto& tree_node : ept->tree_nodes) {
                    for (int id : tree_node.completed_db_graph_ids) {
                        ept_coverage.insert(id);
                    }
                }
            }
            // Extra coverage: nodes_in_cluster
            for (const auto& p : anchor_ptr->nodes_in_cluster_vec) {
                extra_coverage.insert(p.second);
            }
        }

        // （a）SS_search statistics
        auto ss_start = std::chrono::high_resolution_clock::now();
        auto anchor_results = SS_search(query_node, anchor_id, netdag_lb, netdag_ged, tau, stats, dfs_mode_override);
        auto ss_end = std::chrono::high_resolution_clock::now();

        double ss_duration = std::chrono::duration<double>(ss_end - ss_start).count();
        // Record (anchor_id, SS_search duration)
        ss_time_records.emplace_back(anchor_id, ss_duration);

        // （b）MergeResults
        results.insert(results.end(), anchor_results.begin(), anchor_results.end());
        ept_results_all.insert(ept_results_all.end(), anchor_results.begin(), anchor_results.end());  // DEBUG

        // ========== (optional) Supplementary search: BMao-scan-like on anchor->nodes_in_cluster ==========
        auto cluster_results = extra_cluster_search(query_node, anchor_id, tau, stats);
        results.insert(results.end(), cluster_results.begin(), cluster_results.end());
        extra_results_all.insert(extra_results_all.end(), cluster_results.begin(), cluster_results.end());  // DEBUG
    }

    // ========== DEBUG: Accumulated EPT and extra_cluster recall statistics ==========
#if 0  // set to 0 to disable debug output
    {
        // Static variables for accumulated statistics
        static std::mutex debug_mutex;
        static int debug_query_count = 0;
        static long long debug_total_gt = 0;
        static long long debug_total_ept_gt = 0;      // GT count within EPT coverage
        static long long debug_total_extra_gt = 0;    // GT count within extra coverage
        static long long debug_total_ept_hit = 0;
        static long long debug_total_extra_hit = 0;
        static long long debug_total_all_hit = 0;
        static bool registered_atexit = false;

        // Register atexit callback, output summary at program end
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
            // Collect all ground truth with ged <= tau (consistent with compute_recall)
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

                // dedup
                std::unordered_set<int> ept_set(ept_results_all.begin(), ept_results_all.end());
                std::unordered_set<int> extra_set(extra_results_all.begin(), extra_results_all.end());
                std::unordered_set<int> all_set(results.begin(), results.end());

                // Compute GT count within each coverage range and hit count
                // Use global coverage sets (entire database coverage), not query-visited coverage
                int ept_gt = 0, extra_gt = 0;
                int ept_hit = 0, extra_hit = 0, all_hit = 0;

                // DEBUG: Collect anchor set visited by current query
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

                    // DEBUG: If GT in EPT coverage but not found, output reason
                    if (in_ept && !ept_set.count(id)) {
                        // Find which anchor this GT belongs to
                        int owner_anchor = -1;
                        bool anchor_was_visited = false;

                        for (const auto& anchor : net_dag->anchors) {
                            if (!anchor) continue;
                            // Check if it is anchor itself
                            if (anchor->node_id == id) {
                                owner_anchor = anchor->node_id;
                                break;
                            }
                            // Check if in anchor's EPT
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

                // Accumulated statistics
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

    // EPT_total_time is now computed as EPT_lb_time + EPT_astar_time in print_summary()
    return results;
}

// Gisma with reuse disabled (using no_reuse mode)
std::vector<int> GismaSearchEngine::Gisma_no_reuse_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "no_reuse");
}

// Gisma pure DFS traversal (using App_baseline, no optimizations)
std::vector<int> GismaSearchEngine::Gisma_only_dfs_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "only_dfs");
}

// Gisma with Subtree Pruning disabled (ablation experiment)
std::vector<int> GismaSearchEngine::Gisma_no_SP_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "no_SP");
}

// Gisma with Distance Propagation/LB Propagation disabled (ablation experiment)
std::vector<int> GismaSearchEngine::Gisma_no_LP_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats)
{
    return Gisma_search(query_node, tau, stats, "no_LP");
}


std::tuple<bool, double, int, int> GismaSearchEngine::compute_recall(int query_id, const std::vector<int> &exact_results_within_tau, double tau)
{
    std::lock_guard<std::mutex> lock(ground_truth_mutex);
    auto it = ground_truth.find(query_id);
    if (it != ground_truth.end())
    {
        const std::map<double, std::vector<int>> &distances = it->second;
        std::vector<int> ids_within_tau;

        for (const auto &distance_pair : distances)
        {
            double ged = distance_pair.first;
            const std::vector<int> &graph_ids = distance_pair.second;
            if (ged <= tau)
            {
                ids_within_tau.insert(ids_within_tau.end(), graph_ids.begin(), graph_ids.end());
            }
        }

        int ground_truth_size = ids_within_tau.size();

        if (ground_truth_size > 0)
        {
            std::unordered_set<int> exact_result_set(exact_results_within_tau.begin(), exact_results_within_tau.end());
            std::unordered_set<int> ground_truth_set(ids_within_tau.begin(), ids_within_tau.end());

            int intersection_count = 0;
            for (const int &id : ground_truth_set)
            {
                if (exact_result_set.find(id) != exact_result_set.end())
                {
                    intersection_count++;
                }
            }

            double recall = static_cast<double>(intersection_count) / ground_truth_size;
            return std::make_tuple(true, recall, intersection_count, ground_truth_size);
        }
        else
        {
            return std::make_tuple(false, 0.0, 0, 0);
        }
    }
    else
    {
        return std::make_tuple(false, 0.0, 0, 0);
    }
}

std::tuple<bool, double, double, double, int, int> GismaSearchEngine::compute_recall_precision_IoU(int query_id, const std::vector<int> &exact_results_within_tau, double tau)
{
    std::lock_guard<std::mutex> lock(ground_truth_mutex);
    auto it = ground_truth.find(query_id);
    if (it != ground_truth.end())
    {
        const std::map<double, std::vector<int>> &distances = it->second;
        std::vector<int> ids_within_tau;

        // 1) Find all nodes in ground truth with actual ged <= tau
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
            // 2) build set for intersection
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

            // 3) compute recall, precision, iou
            double recall = static_cast<double>(intersection_count) / ground_truth_size;

            int exact_size = static_cast<int>(exact_result_set.size());  // using deduplicated size
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
            // ground_truth_size == 0 => indicates no matches for this tau
            // still return false
            return std::make_tuple(false, 0.0, 0.0, 0.0, 0, 0);
        }
    }
    else
    {
        // no ground_truth found
        return std::make_tuple(false, 0.0, 0.0, 0.0, 0, 0);
    }
}


void GismaSearchEngine::perform_search(double tau)
{
    // ========== 0) Global statistics variables (Recall/Precision/IoU related) ==========
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

    // ========== 1) Global statistics (ND/EPT) ==========
    SearchStats global_stats;

    // ========== 2) Record each query's duration ==========
    std::vector<double> all_query_times;
    all_query_times.reserve(query_node_list.size());

    // New: record detailed info for each query
    std::vector<QueryDetails> query_details_list;
    query_details_list.reserve(query_node_list.size());

    size_t total_queries = q_end - q_start + 1;

    // ========== 3) Iterate query nodes  ==========
    for (size_t idx = 0; idx <= q_end - q_start; ++idx)
    {
        const auto &query_node = query_node_list[idx];

        auto total_query_start = std::chrono::high_resolution_clock::now();

        SearchStats local_stats;
        std::vector<int> results;

        // (a) Call corresponding search method
        if (this->search_method == "Gisma")
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

        // (b) Accumulate local_stats to global
        global_stats.add(local_stats);

        auto total_query_end = std::chrono::high_resolution_clock::now();
        double total_query_time = std::chrono::duration<double>(total_query_end - total_query_start).count();

        // (c) Add to statistics
        total_query_time_sum += total_query_time;
        all_query_times.push_back(total_query_time);

        // New: store (query_id, total_query_time)
        int query_id = query_node->node_id;

        // (d) Collect results statistics
        int num_results = static_cast<int>(results.size());
        total_results_found += num_results;
        if (num_results > 0)
        {
            queries_with_non_empty_results++;
        }

        // (e) recall / precision / iou
        auto [has_ground_truth, recall, precision, iou, intersection_count, ground_truth_size]
            = this->compute_recall_precision_IoU(query_id, results, tau);

        // Save query details (including recall etc.)
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

        // Display progress bar
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
    
    // ========== 4) Call unified output function ==========
    // Compute total time (for non-parallel version, use sum of query times)
    double total_time = total_query_time_sum;
    
    // For compatibility, generate query_time_pairs from query_details_list
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

    // Save experiment results to file (only when save_logs is enabled)
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
    // ========== 0) Define parallel statistics variables ==========
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

    std::mutex recall_mutex;  // protect total_recall, total_precision, total_iou

    // ========== 1) Global statistics (ND/EPT) ==========
    SearchStats global_stats;

    // ========== 2) Record each query's duration ==========
    std::vector<double> all_query_times;
    all_query_times.reserve(query_node_list.size());

    // New: record detailed info for each query
    std::vector<QueryDetails> query_details_list;
    query_details_list.reserve(query_node_list.size());

    // Mutex for protecting float or multi-field simultaneous update operations
    std::mutex stats_mutex;

    // ========== 3) Start parallel timing ==========
    auto total_start = std::chrono::high_resolution_clock::now();

    // ========== 4) Determine thread count & evenly distribute queries to threads ==========
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = (num_threads <= 200) ? num_threads : 200; // maximum 200 threads
    std::cout << "[perform_search_parallel] Using " << num_threads << " threads.\n";

    size_t queries_per_thread = (query_node_list.size() + num_threads - 1) / num_threads;

    // ========== 5) Create multi-threaded tasks (std::async) ==========
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (unsigned int i = 0; i < num_threads; ++i)
    {
        size_t start_index = i * queries_per_thread;
        size_t end_index = std::min(start_index + queries_per_thread, query_node_list.size());

        // Capture all needed references/values
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
                // Process queries assigned to this thread [start_index, end_index)
                for (size_t idx = start_index; idx < end_index; ++idx)
                {

                    auto &query_node = query_node_list[idx];
                    int query_id = query_node->node_id;

                    // Entire query timing
                    auto total_query_start = std::chrono::high_resolution_clock::now();

                    SearchStats local_stats; // Each query has local statistics
                    std::vector<int> results;

                    // Call different functions based on search_method
                    if (this->search_method == "Gisma")
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

                    // ========== Compute Recall/Precision/IoU ==========
                    auto [has_ground_truth, recall, precision, iou, intersection_count, ground_truth_size]
                        = this->compute_recall_precision_IoU(query_id, results, tau);
                    // printf("results.size() = %d, ground_truth_size = %d\n", results.size(), ground_truth_size);

                    // ========== Create QueryDetails object ==========
                    QueryDetails query_detail(query_id, total_query_time, recall, precision, iou, has_ground_truth);
                    query_detail.query_nodes = query_node->graph->n;
                    query_detail.query_edges = query_node->graph->m;
                    query_detail.result_count = (int)results.size();
                    query_detail.lb_time = local_stats.total_lb_time();
                    query_detail.astar_time = local_stats.total_astar_time();
                    // EPT statistics need to be extracted from local_stats
                    query_detail.total_ept_nodes = local_stats.EPT_total_nodes_visited;
                    query_detail.nodes_computed = local_stats.EPT_nodes_computed;
                    query_detail.nodes_pruned = local_stats.EPT_filter_pruned_nodes;
                    query_detail.lb_pruning_count = local_stats.lb_pruning_count;
                    query_detail.subtree_pruned = local_stats.subtree_pruning_avoided_nodes;

                    // ========== Real-time JSON save (outside lock, avoid blocking) ==========
                    if (save_logs && !experiment_base_dir.empty()) {
                        this->save_single_query_json(query_detail, tau, experiment_base_dir);
                    }

                    // ========== Update statistics (lock required) ==========
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);

                        // Add query duration
                        all_query_times.push_back(total_query_time);
                        query_details_list.push_back(query_detail);

                        // Accumulate various times
                        total_query_time_sum += total_query_time;

                        // Accumulate local_stats to global_stats
                        global_stats.add(local_stats);

                        // If ground truth exists, accumulate recall/precision/IoU
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

                        // Always accumulate intersection and ground_truth counts
                        total_intersection_count += intersection_count;
                        total_ground_truth_count += ground_truth_size;
                    }

                    // Atomic update counts
                    total_queries_processed++;
                    total_results_found += (int)results.size();
                    if (!results.empty())
                    {
                        queries_with_non_empty_results++;
                    }
                }
            }));
    }

    // ========== 6) Wait for all threads to complete ==========
    for (auto &f : futures)
    {
        f.get();
    }

    // ========== 7) End parallel timing ==========
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    // ========== 8) Call unified output function ==========
    // For compatibility, generate query_time_pairs from query_details_list
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
        total_time,  // Parallel version total time
        total_recall,
        total_precision,
        total_iou,
        valid_query_count.load(),
        total_intersection_count.load(),
        total_ground_truth_count.load(),
        query_time_pairs,
        true  // is_parallel = true
    );

    // Save experiment results to file (only when save_logs is enabled)
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
    double total_time,  // Parallel version total time
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
    printf("Total queries processed: %d\n", total_queries_processed);
    printf("Total results found: %d / %d\n", total_results_found, total_ground_truth_count);
    printf("Queries with non-empty results: %d\n", queries_with_non_empty_results);

    double sum_of_query_times = global_stats.total_lb_time() + global_stats.total_astar_time();
    printf("Total query time: %f seconds.\n", sum_of_query_times);

    int valid_q = valid_query_count;
    if (valid_q > 0) {
        double avg_recall = total_recall / valid_q;
        printf("Average recall (over %d queries):    %f\n", valid_q, avg_recall);
    }
    double recall_overall = (total_ground_truth_count > 0) ?
        (double)total_intersection_count / total_ground_truth_count : 0.0;
    printf("Overall recall (all queries combined): %.6f\n", recall_overall);

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
    // Create results directory
    std::string exp_dir;
    std::string exp_name;
    char timestamp[64] = "N/A";

    // If experiment_base_dir is set, use it directly as output directory
    if (!experiment_base_dir.empty()) {
        exp_dir = experiment_base_dir;
        std::filesystem::create_directories(exp_dir);
        printf("\n[Saving Results] Using experiment base directory: %s\n", exp_dir.c_str());

        // Use directory name as experiment name
        exp_name = std::filesystem::path(exp_dir).filename().string();

        // Generate timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        LOCALTIME_SAFE(&tm_now, &time_t_now);
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_now);
    } else {
        // Otherwise use default directory naming
        std::string base_dir = "./experiment_results";
        std::filesystem::create_directories(base_dir);

        // Generate timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        LOCALTIME_SAFE(&tm_now, &time_t_now);
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_now);

        // Generate experiment name：dataset_method_tau_timestamp
        std::ostringstream exp_name_stream;
        exp_name_stream << dataset_name << "_" << search_method
                        << "_tau" << static_cast<int>(tau)
                        << "_" << timestamp;
        exp_name = exp_name_stream.str();

        // Create experiment-specific directory
        exp_dir = base_dir + "/" + exp_name;
        std::filesystem::create_directories(exp_dir);

        printf("\n[Saving Results] Experiment directory: %s\n", exp_dir.c_str());
    }

    // ========== 1. Save overall statistics (summary.txt) ==========
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

    // Determine if two-layer index structure (Gisma) is used
    bool uses_two_layer_index = (search_method == "Gisma" || search_method == "Gisma-no-reuse");

    summary_file << "--- Time Breakdown ---\n";
    if (uses_two_layer_index) {
        // Gisma: show ND + EPT decomposition
        double nd_total = global_stats.ND_lb_time + global_stats.ND_astar_time;
        double ept_total = global_stats.EPT_lb_time + global_stats.EPT_astar_time;
        summary_file << "ND Total Time: " << nd_total << " seconds\n";
        summary_file << "  - ND LB Time: " << global_stats.ND_lb_time << " seconds\n";
        summary_file << "  - ND AStar Time: " << global_stats.ND_astar_time << " seconds\n";
        summary_file << "EPT Total Time: " << ept_total << " seconds\n";
        summary_file << "  - EPT LB Time: " << global_stats.EPT_lb_time << " seconds\n";
        summary_file << "  - EPT AStar Time: " << global_stats.EPT_astar_time << " seconds\n\n";
    } else {
        // App-BMao, AStar-BMao etc.: only show total LB + AStar
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
        // Gisma: show ND and EPT separately
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
        // App-BMao, AStar-BMao: show total
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

    // ========== 2. Save detailed info for each query (query_details.csv) ==========
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

    // ========== 3. Save configuration parameters (config.txt) ==========
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

// Save single query JSON results (real-time save)
void GismaSearchEngine::save_single_query_json(
    const QueryDetails& query_detail,
    double tau,
    const std::string& output_dir
) {
    // Generate filename：query_<id>.json
    std::ostringstream filename;
    filename << "query_" << query_detail.query_id << ".json";
    std::string filepath = output_dir + "/" + filename.str();

    // Call QueryDetails save_to_json method
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

// Single query search implementation
QueryDetails GismaSearchEngine::search_single_query(
    int query_id,
    std::shared_ptr<Node> query_node,
    double tau_value,
    SearchStats& local_stats
) {
    auto query_start = std::chrono::high_resolution_clock::now();

    // Execute full Gisma search, directly passing tau parameter
    std::vector<int> results = Gisma_search(query_node, tau_value, local_stats);

    auto query_end = std::chrono::high_resolution_clock::now();
    double total_query_time = std::chrono::duration<double>(query_end - query_start).count();

    // Compute precision
    auto [has_ground_truth, recall, precision, iou, intersection_count, ground_truth_size]
        = compute_recall_precision_IoU(query_id, results, tau_value);

    // Create QueryDetails
    QueryDetails qd(query_id, total_query_time, recall, precision, iou, has_ground_truth);

    // Fill in extra info
    qd.query_nodes = query_node->graph->n;
    qd.query_edges = query_node->graph->m;
    qd.result_count = (int)results.size();
    qd.lb_time = local_stats.total_lb_time();
    qd.astar_time = local_stats.total_astar_time();
    qd.total_ept_nodes = local_stats.EPT_total_nodes_in_used_epts;
    qd.nodes_computed = local_stats.EPT_nodes_computed;
    qd.nodes_pruned = local_stats.subtree_pruning_avoided_nodes + local_stats.lb_pruning_count;
    qd.lb_pruning_count = local_stats.lb_pruning_count;
    qd.subtree_pruned = local_stats.subtree_pruning_avoided_nodes;

    return qd;
}
