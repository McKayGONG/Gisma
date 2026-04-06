#ifndef GISMASEARCHENGINE_H
#define GISMASEARCHENGINE_H

#include "NetDag.h"
#include "EditPathTree.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <cstddef>
#include <mutex>
#include "ReuseSearchTree.h"

struct SearchStats {
    // ========== 1) NetDag (ND) Related Statistics ==========

    // --- Counts ---
    int ND_ML_count = 0;        // NetDag: ML distance estimation count
    int ND_lb_count = 0;        // NetDag: lower bound filter count
    int ND_astar_count = 0;     // NetDag: A* search count
    int ND_ndc_count = 0;       // NetDag: unique node count (deduplicated, for NDC statistics)

    int ND_size_lb_count = 0;
    int ND_vertex_lb_count = 0;
    int ND_edge_lb_degree_count = 0;
    int ND_edge_lb_label_count = 0;
    int ND_app_test_count = 0;  // NetDag: App_test exact lower bound computation count

    // --- Timing ---
    double ND_ML_time = 0.0;    // NetDag: ML distance estimation time
    double ND_lb_time = 0.0;    // NetDag: lower bound filter total time
    double ND_astar_time = 0.0; // NetDag: A* search total time

    // ========== 2) EPT Related Statistics ==========

    // --- Counts ---
    int EPT_ML_count = 0;
    int EPT_lb_count = 0;
    int EPT_astar_count = 0;
    int EPT_ndc_count = 0;      // EPT: unique node count (deduplicated, for NDC statistics)

    int EPT_size_lb_count = 0;
    int EPT_vertex_lb_count = 0;
    int EPT_edge_lb_degree_count = 0;
    int EPT_edge_lb_label_count = 0;

    // --- Timing ---
    double EPT_ML_time = 0.0;
    double EPT_lb_time = 0.0;
    double EPT_astar_time = 0.0;

    // ========== 3) NetDag/EPT Total Time ==========
    double ND_total_time  = 0.0;   // NetDag overall time
    double EPT_total_time = 0.0;   // EPT overall time

    // ========== 4) Total Time & Index Overhead ==========

    // total_time represents the global time for one or multiple searches (assigned externally)
    double total_time = 0.0;

    // Index Overhead for NetDag and EPT (can be filled externally or computed internally)
    double ND_index_overhead = 0.0;  
    double EPT_index_overhead = 0.0;

    size_t EPT_reuse_attempt = 0;
    size_t EPT_reuse_count = 0;
    size_t EPT_reuse_verifications = 0;    // verification count
    size_t EPT_reuse_correct = 0;          // correct count
    size_t EPT_reuse_incorrect = 0;        // incorrect count
    std::map<int, int> EPT_ged_diff_distribution;  // GED difference distribution

    // ========== Added: Verification AStar Time Statistics ==========
    double EPT_verification_astar_time = 0.0;  // total time for verification AStar

    // ========== Reuse Statistics Breakdown ==========
    // Reuse failure reason categories
    size_t EPT_reuse_fail_no_parent_snapshot = 0;     // parent did not use AStar to compute GED
    size_t EPT_reuse_fail_parent_snapshot_empty = 0;  // parent computed but extract is empty (v.size()==0)
    size_t EPT_reuse_fail_parent_snapshot_size_one = 0; // parent computed but only 1 node (v.size()==1)
    size_t EPT_reuse_fail_parent_snapshot_size_one_dummy = 0; // above case with dummy node
    size_t EPT_reuse_fail_parent_snapshot_size_one_real = 0;  // above case with real node
    size_t EPT_reuse_fail_root_node = 0;              // is root node
    size_t EPT_reuse_fail_no_op = 0;                  // op.type == NONE
    size_t EPT_reuse_fail_multi_ops = 0;              // accumulated_ops.size() != 1
    size_t EPT_reuse_fail_vertex_count_changed = 0;   // db_g->n != parent_db_vertex_count
    size_t EPT_reuse_fail_mo_incompatible = 0;        // MO array incompatible

    // AStar time statistics categorized by failure reason
    double EPT_astar_time_no_parent_snapshot = 0.0;
    double EPT_astar_time_parent_snapshot_empty = 0.0;
    double EPT_astar_time_parent_snapshot_size_one = 0.0;
    double EPT_astar_time_root_node = 0.0;
    double EPT_astar_time_no_op = 0.0;
    double EPT_astar_time_multi_ops = 0.0;
    double EPT_astar_time_vertex_count_changed = 0.0;
    double EPT_astar_time_mo_incompatible = 0.0;
    double EPT_astar_time_lb_pruned = 0.0;            // caused by LB pruning

    // Reuse success time statistics
    double EPT_reuse_success_time = 0.0;              // total time for successful reuse

    // Result source statistics
    size_t EPT_results_from_reuse = 0;                // number of results found via reuse
    size_t EPT_results_from_astar = 0;                // number of results found via AStar

    // Statistics by edit operation type
    std::map<int, size_t> EPT_op_type_count;          // count of each edit operation type
    std::map<int, size_t> EPT_reuse_by_op_type;       // reuse success count by operation type

    // Node type statistics
    size_t EPT_leaf_nodes_processed = 0;              // number of leaf nodes processed
    size_t EPT_internal_nodes_processed = 0;          // number of internal nodes processed
    size_t EPT_nodes_with_completed_ids = 0;          // number of nodes with completed_db_graph_ids

    size_t EPT_reuse_found_ged_le_tau = 0;    // times reuse method found GED <= tau

    // ========== Debug Statistics: DB Graph vs Intermediate Graph ==========
    // DB graph: root node (anchor) or nodes with completed_db_graph_ids
    // Intermediate graph: intermediate nodes generated via edit operations
    int EPT_db_graph_lb_count = 0;           // DB graph filter count
    double EPT_db_graph_lb_time = 0.0;       // DB graph filter time
    int EPT_db_graph_astar_count = 0;        // DB graph verification count
    double EPT_db_graph_astar_time = 0.0;    // DB graph verification time

    int EPT_intermediate_graph_lb_count = 0;      // intermediate graph filter count
    double EPT_intermediate_graph_lb_time = 0.0;  // intermediate graph filter time
    int EPT_intermediate_graph_astar_count = 0;   // intermediate graph verification count
    double EPT_intermediate_graph_astar_time = 0.0; // intermediate graph verification time
    size_t EPT_astar_found_ged_le_tau = 0;    // times standard AStar found GED <= tau (during verification)

    // AppForComputation baseline statistics (for paper EXP-5)
    size_t EPT_baseline_app_count = 0;           // AppForComputation baseline call count
    double EPT_baseline_app_time = 0.0;          // AppForComputation baseline total time
    double EPT_baseline_reuse_time = 0.0;        // reuse time corresponding to baseline (astar_reuse only)

    // EXP-5: sample pairs (reuse_time_ms, baseline_time_ms), store at most 10
    std::vector<std::pair<double, double>> baseline_samples;
    static const size_t MAX_BASELINE_SAMPLES = 10;

    // ========== 6) Optimization Effect Statistics ==========

    // Subtree pruning statistics
    size_t subtree_pruning_decisions = 0;       // number of subtree pruning decisions made
    size_t subtree_pruning_avoided_nodes = 0;   // descendant nodes avoided due to subtree pruning (excluding the decision node itself)
    size_t subtree_pruning_on_leaf_nodes = 0;   // subtree pruning triggered on leaf nodes

    // LB Pruning & Propagation statistics
    size_t lb_pruning_count = 0;           // LB Pruning: times GED computation skipped because estimate_lb > tau
    size_t lb_propagation_count = 0;       // LB Propagation: times estimate_lb propagated to child nodes

    // NetDag cache reuse statistics
    size_t root_netdag_ged_reuse_count = 0;        // times root node directly reused NetDag cached GED

    // Node visit statistics
    size_t EPT_total_nodes_visited = 0;            // total nodes visited in EPT
    size_t EPT_nodes_computed = 0;                 // nodes that were computed (filter + GED)
    size_t EPT_filter_pruned_nodes = 0;            // nodes pruned by EPT filter
    size_t EPT_total_nodes_in_used_epts = 0;       // total nodes in used EPTs (for computing subtree pruning rate)

    // ========== 5) Methods to Get Totals ==========

    // --- Counts ---
    int total_ML_count() const {
        return ND_ML_count + EPT_ML_count;
    }
    int total_lb_count() const {
        return ND_lb_count + EPT_lb_count;
    }
    int total_astar_count() const {
        return ND_astar_count + EPT_astar_count;
    }

    int total_size_lb_count() const {
        return ND_size_lb_count + EPT_size_lb_count;
    }
    int total_vertex_lb_count() const {
        return ND_vertex_lb_count + EPT_vertex_lb_count;
    }
    int total_edge_lb_degree_count() const {
        return ND_edge_lb_degree_count + EPT_edge_lb_degree_count;
    }
    int total_edge_lb_label_count() const {
        return ND_edge_lb_label_count + EPT_edge_lb_label_count;
    }

    // --- Timing ---
    double total_ML_time() const {
        return ND_ML_time + EPT_ML_time;
    }
    double total_lb_time() const {
        return ND_lb_time + EPT_lb_time;
    }
    double total_astar_time() const {
        return ND_astar_time + EPT_astar_time;
    }

    // ========== 6) Accumulation (e.g., merging statistics from multiple queries) ==========

    void add(const SearchStats& other) {
        // ========== ND Section ==========
        ND_ML_count    += other.ND_ML_count;
        ND_lb_count    += other.ND_lb_count;
        ND_astar_count += other.ND_astar_count;
        ND_ndc_count   += other.ND_ndc_count;

        ND_size_lb_count       += other.ND_size_lb_count;
        ND_vertex_lb_count     += other.ND_vertex_lb_count;
        ND_edge_lb_degree_count+= other.ND_edge_lb_degree_count;
        ND_edge_lb_label_count += other.ND_edge_lb_label_count;
        ND_app_test_count      += other.ND_app_test_count;

        ND_ML_time    += other.ND_ML_time;
        ND_lb_time    += other.ND_lb_time;
        ND_astar_time += other.ND_astar_time;
        ND_total_time += other.ND_total_time;

        // ========== EPT Section ==========
        EPT_ML_count    += other.EPT_ML_count;
        EPT_lb_count    += other.EPT_lb_count;
        EPT_astar_count += other.EPT_astar_count;
        EPT_ndc_count   += other.EPT_ndc_count;

        EPT_size_lb_count       += other.EPT_size_lb_count;
        EPT_vertex_lb_count     += other.EPT_vertex_lb_count;
        EPT_edge_lb_degree_count+= other.EPT_edge_lb_degree_count;
        EPT_edge_lb_label_count += other.EPT_edge_lb_label_count;

        EPT_ML_time    += other.EPT_ML_time;
        EPT_lb_time    += other.EPT_lb_time;
        EPT_astar_time += other.EPT_astar_time;
        EPT_total_time += other.EPT_total_time;

        // ========== Index Overhead & total time ==========
        ND_index_overhead  += other.ND_index_overhead;
        EPT_index_overhead += other.EPT_index_overhead;
        total_time         += other.total_time;

        EPT_reuse_attempt += other.EPT_reuse_attempt;
        EPT_reuse_count += other.EPT_reuse_count;

        EPT_reuse_verifications += other.EPT_reuse_verifications;
        EPT_reuse_correct += other.EPT_reuse_correct;
        EPT_reuse_incorrect += other.EPT_reuse_incorrect;
        
        // Merge GED difference distribution
        for (const auto& pair : other.EPT_ged_diff_distribution) {
            EPT_ged_diff_distribution[pair.first] += pair.second;
        }
        
        // Merge verification AStar time
        EPT_verification_astar_time += other.EPT_verification_astar_time;

        // Merge new reuse failure statistics
        EPT_reuse_fail_no_parent_snapshot += other.EPT_reuse_fail_no_parent_snapshot;
        EPT_reuse_fail_parent_snapshot_empty += other.EPT_reuse_fail_parent_snapshot_empty;
        EPT_reuse_fail_parent_snapshot_size_one += other.EPT_reuse_fail_parent_snapshot_size_one;
        EPT_reuse_fail_parent_snapshot_size_one_dummy += other.EPT_reuse_fail_parent_snapshot_size_one_dummy;
        EPT_reuse_fail_parent_snapshot_size_one_real += other.EPT_reuse_fail_parent_snapshot_size_one_real;
        EPT_reuse_fail_root_node += other.EPT_reuse_fail_root_node;
        EPT_reuse_fail_no_op += other.EPT_reuse_fail_no_op;
        EPT_reuse_fail_multi_ops += other.EPT_reuse_fail_multi_ops;
        EPT_reuse_fail_vertex_count_changed += other.EPT_reuse_fail_vertex_count_changed;
        EPT_reuse_fail_mo_incompatible += other.EPT_reuse_fail_mo_incompatible;

        // Merge time statistics
        EPT_astar_time_no_parent_snapshot += other.EPT_astar_time_no_parent_snapshot;
        EPT_astar_time_parent_snapshot_empty += other.EPT_astar_time_parent_snapshot_empty;
        EPT_astar_time_parent_snapshot_size_one += other.EPT_astar_time_parent_snapshot_size_one;
        EPT_astar_time_root_node += other.EPT_astar_time_root_node;
        EPT_astar_time_no_op += other.EPT_astar_time_no_op;
        EPT_astar_time_multi_ops += other.EPT_astar_time_multi_ops;
        EPT_astar_time_vertex_count_changed += other.EPT_astar_time_vertex_count_changed;
        EPT_astar_time_mo_incompatible += other.EPT_astar_time_mo_incompatible;
        EPT_astar_time_lb_pruned += other.EPT_astar_time_lb_pruned;
        EPT_reuse_success_time += other.EPT_reuse_success_time;
        
        // Merge result source statistics
        EPT_results_from_reuse += other.EPT_results_from_reuse;
        EPT_results_from_astar += other.EPT_results_from_astar;
        
        // Merge node type statistics
        EPT_leaf_nodes_processed += other.EPT_leaf_nodes_processed;
        EPT_internal_nodes_processed += other.EPT_internal_nodes_processed;
        EPT_nodes_with_completed_ids += other.EPT_nodes_with_completed_ids;
        
        EPT_reuse_found_ged_le_tau += other.EPT_reuse_found_ged_le_tau;
        EPT_astar_found_ged_le_tau += other.EPT_astar_found_ged_le_tau;

        // Merge AppForComputation baseline statistics
        EPT_baseline_app_count += other.EPT_baseline_app_count;
        EPT_baseline_app_time += other.EPT_baseline_app_time;
        EPT_baseline_reuse_time += other.EPT_baseline_reuse_time;

        // Merge DB graph vs intermediate graph statistics
        EPT_db_graph_lb_count += other.EPT_db_graph_lb_count;
        EPT_db_graph_lb_time += other.EPT_db_graph_lb_time;
        EPT_db_graph_astar_count += other.EPT_db_graph_astar_count;
        EPT_db_graph_astar_time += other.EPT_db_graph_astar_time;
        EPT_intermediate_graph_lb_count += other.EPT_intermediate_graph_lb_count;
        EPT_intermediate_graph_lb_time += other.EPT_intermediate_graph_lb_time;
        EPT_intermediate_graph_astar_count += other.EPT_intermediate_graph_astar_count;
        EPT_intermediate_graph_astar_time += other.EPT_intermediate_graph_astar_time;

        // ========== Merge optimization effect statistics ==========
        subtree_pruning_decisions += other.subtree_pruning_decisions;
        subtree_pruning_avoided_nodes += other.subtree_pruning_avoided_nodes;
        subtree_pruning_on_leaf_nodes += other.subtree_pruning_on_leaf_nodes;
        lb_pruning_count += other.lb_pruning_count;
        lb_propagation_count += other.lb_propagation_count;
        root_netdag_ged_reuse_count += other.root_netdag_ged_reuse_count;
        EPT_total_nodes_visited += other.EPT_total_nodes_visited;
        EPT_nodes_computed += other.EPT_nodes_computed;
        EPT_filter_pruned_nodes += other.EPT_filter_pruned_nodes;
        EPT_total_nodes_in_used_epts += other.EPT_total_nodes_in_used_epts;

        // Merge operation type statistics
        for (const auto& [k, v] : other.EPT_op_type_count) {
            EPT_op_type_count[k] += v;
        }
        for (const auto& [k, v] : other.EPT_reuse_by_op_type) {
            EPT_reuse_by_op_type[k] += v;
        }
    }

    // ========== 7) Print Summary Function ==========

    void print_summary(std::ostream& os = std::cout) const {
        // Calculate ND_total_time and EPT_total_time as sum of LB + AStar times
        double computed_ND_total_time = ND_lb_time + ND_astar_time;
        double computed_EPT_total_time = EPT_lb_time + EPT_astar_time;

        os << "============== NetDag Stats ==============\n";
        os << "ML_count: " << ND_ML_count << " (time=" << ND_ML_time << "s)\n"
           << "lb_count: " << ND_lb_count << " (time=" << ND_lb_time << "s)\n"
           << "astar_count: " << ND_astar_count << " (time=" << ND_astar_time << "s)\n"
           << "  size_lb_count: " << ND_size_lb_count << "\n"
           << "  vertex_lb_count: " << ND_vertex_lb_count << "\n"
           << "  edge_lb_degree_count: " << ND_edge_lb_degree_count << "\n"
           << "  edge_lb_label_count:  " << ND_edge_lb_label_count << "\n"
           << "ND_total_time: " << computed_ND_total_time << "s\n"
           << "Index Overhead (ND): " << ND_index_overhead << "s\n\n";

        os << "============== EPT Stats ==============\n";
        os << "ML_count: " << EPT_ML_count << " (time=" << EPT_ML_time << "s)\n"
           << "lb_count: " << EPT_lb_count << " (time=" << EPT_lb_time << "s)\n"
           << "astar_count: " << EPT_astar_count << " (time=" << EPT_astar_time << "s)\n"
           << "  size_lb_count: " << EPT_size_lb_count << "\n"
           << "  vertex_lb_count: " << EPT_vertex_lb_count << "\n"
           << "  edge_lb_degree_count: " << EPT_edge_lb_degree_count << "\n"
           << "  edge_lb_label_count:  " << EPT_edge_lb_label_count << "\n"
           << "EPT_total_time: " << computed_EPT_total_time << "s\n"
           << "Index Overhead (EPT): " << EPT_index_overhead << "s\n\n";

        // DB graph vs intermediate graph separate statistics
        os << "============== DB Graph vs Intermediate Graph Stats ==============\n";
        double db_lb_avg = (EPT_db_graph_lb_count > 0) ? (EPT_db_graph_lb_time * 1e9 / EPT_db_graph_lb_count) : 0;
        double inter_lb_avg = (EPT_intermediate_graph_lb_count > 0) ? (EPT_intermediate_graph_lb_time * 1e9 / EPT_intermediate_graph_lb_count) : 0;
        double db_astar_avg = (EPT_db_graph_astar_count > 0) ? (EPT_db_graph_astar_time * 1e6 / EPT_db_graph_astar_count) : 0;
        double inter_astar_avg = (EPT_intermediate_graph_astar_count > 0) ? (EPT_intermediate_graph_astar_time * 1e6 / EPT_intermediate_graph_astar_count) : 0;

        os << "DB Graph (anchor/completed):\n"
           << "  filter: " << EPT_db_graph_lb_count << " calls, " << EPT_db_graph_lb_time << "s, avg=" << db_lb_avg << "ns\n"
           << "  verify: " << EPT_db_graph_astar_count << " calls, " << EPT_db_graph_astar_time << "s, avg=" << db_astar_avg << "us\n";
        os << "Intermediate Graph (edited):\n"
           << "  filter: " << EPT_intermediate_graph_lb_count << " calls, " << EPT_intermediate_graph_lb_time << "s, avg=" << inter_lb_avg << "ns\n"
           << "  verify: " << EPT_intermediate_graph_astar_count << " calls, " << EPT_intermediate_graph_astar_time << "s, avg=" << inter_astar_avg << "us\n\n";

        os << "============== Total Stats ==============\n";
        os << "Total ML_count: " << total_ML_count() << " (time=" << total_ML_time() << "s)\n"
           << "Total lb_count: " << total_lb_count() << " (time=" << total_lb_time() << "s)\n"
           << "Total astar_count: " << total_astar_count() << " (time=" << total_astar_time() << "s)\n\n"
           << "  - Size-based LB: " << total_size_lb_count() << "\n"
           << "  - Vertex LB: " << total_vertex_lb_count() << "\n"
           << "  - Edge LB (degree): " << total_edge_lb_degree_count() << "\n"
           << "  - Edge LB (label): " << total_edge_lb_label_count() << "\n"
           << "Global total_time: " << total_time << "s\n";
    }
};

// Detailed information for a single query
struct QueryDetails {
    int query_id;
    double time;
    double recall;
    double precision;
    double iou;
    bool has_ground_truth;
    bool has_results;       // found > 0, controls whether precision participates in averaging

    // Added fields for experiment mode
    int query_nodes = 0;
    int query_edges = 0;
    int result_count = 0;
    double lb_time = 0.0;
    double astar_time = 0.0;
    // ND and EPT separate timing
    double nd_lb_time = 0.0;
    double nd_astar_time = 0.0;
    double ept_lb_time = 0.0;
    double ept_astar_time = 0.0;
    // ND and EPT separate counts
    int nd_lb_count = 0;
    int nd_astar_count = 0;
    int nd_ndc_count = 0;    // NDC statistics: unique node count (deduplicated)
    int ept_lb_count = 0;
    int ept_astar_count = 0;
    int ept_ndc_count = 0;   // NDC statistics: unique node count (deduplicated)
    size_t total_ept_nodes = 0;
    size_t nodes_computed = 0;
    size_t nodes_pruned = 0;
    size_t lb_propagation_count = 0;   // LB Propagation: propagation count
    size_t lb_pruning_count = 0;       // LB Pruning: times GED computation was skipped
    size_t subtree_pruned = 0;              // Subtree Pruning: nodes avoided
    size_t subtree_pruning_decisions = 0;   // Subtree Pruning: decision count
    size_t reuse_count = 0;             // Search Tree Reuse: success count
    size_t reuse_attempt = 0;           // Search Tree Reuse: attempt count
    double reuse_success_time = 0.0;    // Search Tree Reuse: total time for successful reuse

    // EXP-5: Baseline statistics
    size_t baseline_app_count = 0;      // Baseline AppForComputation call count
    double baseline_app_time = 0.0;     // Baseline AppForComputation total time
    double baseline_reuse_time = 0.0;   // reuse time corresponding to baseline
    size_t baseline_correct = 0;        // times Reuse and Baseline results match
    size_t baseline_incorrect = 0;      // times Reuse and Baseline results differ
    std::vector<std::pair<double, double>> baseline_samples;  // (reuse_time_ms, baseline_time_ms) sample pairs

    // DB graph vs intermediate graph statistics
    int db_graph_lb_count = 0;
    double db_graph_lb_time = 0.0;
    int db_graph_astar_count = 0;
    double db_graph_astar_time = 0.0;
    int intermediate_graph_lb_count = 0;
    double intermediate_graph_lb_time = 0.0;
    int intermediate_graph_astar_count = 0;
    double intermediate_graph_astar_time = 0.0;

    QueryDetails(int id, double t, double r = -1.0, double p = -1.0, double i = -1.0, bool has_gt = false, bool has_res = false)
        : query_id(id), time(t), recall(r), precision(p), iou(i), has_ground_truth(has_gt), has_results(has_res) {}

    // Save to JSON file
    void save_to_json(const std::string& filepath, double tau) const;
};

class GismaSearchEngine {
public:
    std::shared_ptr<NetDag> net_dag;
    double tau_index;
    // double tau_search;  // Removed member variable, changed to function parameter
    double error_tolerance_search;
    int q_start;
    int q_end;
    bool has_ged_matrix;
    std::vector<double> ged_matrix;  // One-dimensional matrix

    std::string search_method;
    std::string dataset_name;  // dataset name, used for saving experiment results
    std::string experiment_base_dir;  // experiment mode base directory; if set, use as root for result storage
    bool save_logs;  // whether to save detailed logs to file
    std::string nd_mode;   // NetDag mode: "filters" / "astar" / "filters_astar"
    std::string dfs_mode;  // DFS traversal mode: "unified" / "no_reuse" / "no_SP" / "no_LP" / "only_dfs" / "reuse"
    bool use_ept_filters;  // whether to use EPT lower bound filtering
    bool only_compute_db_graph;   // only compute GED for EPT nodes with completed_db_graph_ids (i.e., DB graph nodes)
    bool fast_down;        // whether to use fast-down strategy (descend on first match in non-leaf levels)
    int app_max_iter;      // A* max iterations (controls APP_CNT/ASTAR_CNT/REUSE_CNT)
    int exact_max_iter;    // exact computation iteration limit (for training data generation)
    bool disable_lsa_pruning;  // disable LSa pruning (still save lsa_lb for reuse)
    bool disable_reuse_lsa;    // disable LSa recomputation in reuse
    bool verify_reuse_baseline;  // verify reuse effectiveness: compute baseline time with AppForComputation on each reuse
    bool chain_reuse;            // chain reuse: nodes using reuse also save snapshots for subsequent nodes
    bool disable_subtree_pruning;  // [deprecated] now use dfs_mode="no_SP" instead
    bool disable_lb_propagation;   // [deprecated] now use dfs_mode="no_LP" instead
    double nd_filter_ratio;  // NetDag filter tightening coefficient, condition changed to lb <= (alpha + tau) * ratio
    std::vector<std::shared_ptr<Node>> db_node_list;
    std::vector<std::shared_ptr<Node>> query_node_list;
    int size_of_DB;
    int N;  // For storing the size of the GED matrix

    // Added members
    std::vector<Graph*> db;  // Database graphs

    std::map<int, std::map<double, std::vector<int>>> ground_truth;  // Ground truth data
    std::map<std::string, ui> vM, eM;
    ui max_n;

    // Helper arrays
    int* vlabel_cnt;
    int* elabel_cnt;
    int* degree_q;
    int* degree_g;
    int* tmp;

    // Added member variable: for storing all EPTs
    std::unordered_map<ui, TreeNode*> epts;  // mapping from anchor ID to EPT root node
    // EPT manager pointer
    EditPathTreeManager* ept_manager;

    // EPT total node count statistics
    size_t total_ept_nodes = 0;  // total nodes across all EPTs

    struct ReuseStats {
        size_t total_astar_calls = 0;      // total AStar call count
        size_t reuse_attempts = 0;         // reuse attempt count
        size_t reuse_success = 0;          // successful reuse count
        double total_astar_time = 0.0;     // total computation time
        double reuse_time = 0.0;           // reuse computation time
        double baseline_time = 0.0;        // non-reuse computation time
        
        void print_summary() const {
            std::cout << "\n===== Reuse Statistics =====\n";
            std::cout << "Total AStar calls: " << total_astar_calls << "\n";
            std::cout << "Reuse attempts: " << reuse_attempts << "\n";
            std::cout << "Reuse success: " << reuse_success << "\n";
            if (reuse_attempts > 0) {
                std::cout << "Reuse success rate: " 
                          << (100.0 * reuse_success / reuse_attempts) << "%\n";
            }
            if (reuse_success > 0 && (total_astar_calls - reuse_success) > 0) {
                double avg_reuse_time = reuse_time / reuse_success;
                double avg_baseline_time = baseline_time / (total_astar_calls - reuse_success);
                std::cout << "Average reuse time: " << avg_reuse_time << "s\n";
                std::cout << "Average baseline time: " << avg_baseline_time << "s\n";
                std::cout << "Speedup factor: " << (avg_baseline_time / avg_reuse_time) << "x\n";
            }
            std::cout << "==========================\n";
        }
    };

    // Constructor
    GismaSearchEngine(
        std::shared_ptr<NetDag> net_dag,
        double tau_index,
        // double tau_search,  // Removed: changed to function parameter
        double error_tolerance_search,
        int q_start,
        int q_end,
        bool has_ged_matrix,
        const std::vector<double>& ged_matrix,  // Changed to a one-dimensional matrix

        const std::string& search_method,
        const std::string& dataset_name,  // added dataset_name parameter

        const std::vector<std::shared_ptr<Node>>& db_node_list,
        const std::vector<std::shared_ptr<Node>>& query_node_list,
        int N,  // Pass N
        std::vector<Graph*> db,  // Database graphs
        const std::map<int, std::map<double, std::vector<int>>>& ground_truth,  // Ground truth data
        const std::map<std::string, ui>& vM,
        const std::map<std::string, ui>& eM,
        ui max_n,
        EditPathTreeManager* ept_manager,
        const std::string& nd_mode = "filters",  // NetDag mode: "filters"/"astar"/"filters_astar"
        const std::string& dfs_mode = "unified",  // DFS traversal mode: "unified"/"simple"/"reuse"
        bool use_ept_filters = false,  // whether to use EPT lower bound filtering
        bool only_compute_db_graph = false,  // only compute GED for EPT nodes with completed_db_graph_ids
        int _app_max_iter = 2300,  // A* max iterations
        bool _fast_down = false,  // whether to use fast-down strategy
        int _exact_max_iter = 1000000,  // exact computation iteration limit
        double _nd_filter_ratio = 1.0,  // NetDag filter tightening coefficient
        bool _disable_lsa_pruning = false,  // disable LSa pruning
        bool _disable_reuse_lsa = false,   // disable LSa recomputation in reuse
        bool _verify_reuse_baseline = false,  // verify reuse effectiveness
        bool _chain_reuse = false  // chain reuse: nodes using reuse also save snapshots
    );

    ~GismaSearchEngine();  // Destructor

    // Get GED value from matrix
    double get_ged_from_matrix(int i, int j);

    // Compute GED online (placeholder function)
    double compute_ged_online_by_pyg(int i, int query_index);
    

    std::vector<std::tuple<int, double, int>> GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> SS_search(std::shared_ptr<Node> query_node, int anchor_id, double netdag_lb, int netdag_ged, double tau, SearchStats &stats, const std::string &dfs_mode_override = "");
    std::vector<int> SS_search_with_ept(
        std::shared_ptr<Node> query_node,
        const EditPathTree &ept,
        double tau,
        SearchStats &stats
    );

    // print_statistics_by_ept_level
    void print_statistics_by_ept_level(
        const EditPathTree &ept,
        const std::vector<bool> &node_has_answer
    );

    // compute_bfs_depth
    void compute_bfs_depth(
        const EditPathTree &ept,
        std::vector<int> &bfs_depth
    );

    // print_statistics_by_bfs_depth
    void print_statistics_by_bfs_depth(
        const EditPathTree &ept,
        const std::vector<bool> &node_has_answer,
        const std::vector<int> &bfs_depth
    );

    std::vector<int> BMao_scan_on_subset(std::shared_ptr<Node> query_node,
                                         const std::vector<int> &subset,
                                         double tau,
                                         SearchStats &stats);

    std::vector<int> EPT_forbidden_cluster_search(std::shared_ptr<Node> query_node,
        int anchor_id,
        double tau,
        SearchStats &stats);
    std::vector<int> extra_cluster_search(std::shared_ptr<Node> query_node,
                                                    int anchor_id,
                                                    double tau,
                                                    SearchStats &stats);

    std::vector<int> Gisma_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats, const std::string &dfs_mode_override = "");
    std::vector<int> Gisma_no_reuse_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> Gisma_only_dfs_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> Gisma_no_SP_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> Gisma_no_LP_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> BMao_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> App_BMao_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> AStar_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // pure Base, using AStar()
    std::vector<int> AStar_scan_no_lsa_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // pure Base, using AStar() with LSa pruning disabled
    std::vector<int> AStar_BMao_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> Base_GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // Base+GS: only use GS to select clusters, verify with App-BMao
    std::vector<int> Base_SS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // Base+SS: only use SS to select clusters, verify with App-BMao
    std::vector<int> Base_All_EPT_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // Base_All_EPT: traverse all anchors without filtering
    std::vector<int> BMao_export_candidates(std::shared_ptr<Node> query_node, double tau, const std::string& output_file);
    std::vector<int> answer_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    // Helper function to calculate EPT subtree size
    size_t calculate_subtree_size(const EditPathTree& ept, size_t node_index);

    bool dfs_traverse_no_reuse(
        size_t                node_index,
        const EditPathTree   &ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>     &exact_results_within_tau,
        double                tau,
        SearchStats          &stats,
        ui                    estimate_lb,
        double                anchor_netdag_lb = -1.0,
        int                   anchor_netdag_ged = -1);
    
    bool dfs_traverse(
        size_t node_index,
        const EditPathTree& ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>& exact_results_within_tau,
        double tau,
        SearchStats& stats,
        std::shared_ptr<SearchSnapshot> parent_snapshot,
        int parent_db_vertex_count,
        ui estimate_lb,
        double anchor_netdag_lb,
        int anchor_netdag_ged
    );

    // dfs_traverse_no_SP: Reuse + DP, but with Subtree Pruning disabled
    bool dfs_traverse_no_SP(
        size_t node_index,
        const EditPathTree& ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>& exact_results_within_tau,
        double tau,
        SearchStats& stats,
        std::shared_ptr<SearchSnapshot> parent_snapshot,
        int parent_db_vertex_count,
        ui estimate_lb,
        double anchor_netdag_lb,
        int anchor_netdag_ged
    );

    // dfs_traverse_no_LP: Reuse + SP, but with LB Propagation disabled
    bool dfs_traverse_no_LP(
        size_t node_index,
        const EditPathTree& ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>& exact_results_within_tau,
        double tau,
        SearchStats& stats,
        std::shared_ptr<SearchSnapshot> parent_snapshot,
        int parent_db_vertex_count,
        ui estimate_lb,
        double anchor_netdag_lb,
        int anchor_netdag_ged
    );

    // Pure DFS traversal (using App_baseline, no LB propagation, no Subtree Pruning)
    bool dfs_traverse_only_dfs(
        size_t                node_index,
        const EditPathTree   &ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>     &exact_results_within_tau,
        double                tau,
        SearchStats          &stats,
        double                anchor_netdag_lb = -1.0,
        int                   anchor_netdag_ged = -1);
    




    void traverse_ept_and_search(
        const EditPathTree &ept,
        std::shared_ptr<Node> query_node,
        std::vector<int> &exact_results_within_tau,
        double tau,
        SearchStats &stats,
        double anchor_netdag_lb = -1.0,
        int anchor_netdag_ged = -1,
        const std::string &dfs_mode_override = ""
    );
    // void traverse_ept_and_search(
    //     const EditPathTree& ept,
    //     std::shared_ptr<Node> query_node,
    //     std::vector<int>& exact_results_within_tau,
    //     SearchStats &stats);

    std::tuple<bool, double, int, int> compute_recall(int query_id, const std::vector<int>& exact_results_within_tau, double tau);
    std::tuple<bool, double, double, double, int, int> compute_recall_precision_IoU(int query_id, const std::vector<int>& exact_results_within_tau, double tau);
    void perform_search(double tau);

    void perform_search_parallel(double tau);

    // Single query search interface (for experiment mode)
    QueryDetails search_single_query(
        int query_id,
        std::shared_ptr<Node> query_node,
        double tau_value,
        SearchStats& local_stats
    );

    void print_search_statistics(
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
        const std::vector<std::pair<int, double>>& query_time_pairs,
        bool is_parallel = false
    );

    // Save single query JSON result (for real-time saving)
    void save_single_query_json(
        const QueryDetails& query_detail,
        double tau,
        const std::string& output_dir
    );

    void save_experiment_results(
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
        bool is_parallel = false
    );

private:
    // Mutex for protecting ground_truth access
    mutable std::mutex ground_truth_mutex;

    // Get the EPT root node for the specified anchor ID from epts
    TreeNode* get_ept_root(ui anchor_id);

    // Get target graph ID list from EPT
    // void collect_graph_ids(TreeNode* node, std::vector<int>& ids);
};

#endif // GISMASEARCHENGINE_H
