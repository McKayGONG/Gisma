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
    size_t EPT_reuse_fail_no_parent_snapshot = 0;     // no parent snapshot available
    size_t EPT_reuse_fail_root_node = 0;              // is root node
    size_t EPT_reuse_fail_no_op = 0;                  // op.type == NONE
    size_t EPT_reuse_fail_multi_ops = 0;              // accumulated_ops.size() != 1
    size_t EPT_reuse_fail_parent_too_small = 0;       // parent snapshot node count <= 1
    size_t EPT_reuse_fail_vertex_count_changed = 0;   // db_g->n != parent_db_vertex_count
    size_t EPT_reuse_fail_mo_incompatible = 0;        // MO array incompatible

    // AStar time statistics categorized by failure reason
    double EPT_astar_time_no_parent_snapshot = 0.0;
    double EPT_astar_time_root_node = 0.0;
    double EPT_astar_time_no_op = 0.0;
    double EPT_astar_time_multi_ops = 0.0;
    double EPT_astar_time_parent_too_small = 0.0;
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
    size_t EPT_astar_found_ged_le_tau = 0;    // times standard AStar found GED <= tau (during verification)

    // AppForComputation baseline statistics (for paper EXP-5)
    size_t EPT_baseline_app_count = 0;           // AppForComputation baseline call count
    double EPT_baseline_app_time = 0.0;          // AppForComputation baseline total time

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

        ND_size_lb_count       += other.ND_size_lb_count;
        ND_vertex_lb_count     += other.ND_vertex_lb_count;
        ND_edge_lb_degree_count+= other.ND_edge_lb_degree_count;
        ND_edge_lb_label_count += other.ND_edge_lb_label_count;

        ND_ML_time    += other.ND_ML_time;
        ND_lb_time    += other.ND_lb_time;
        ND_astar_time += other.ND_astar_time;
        ND_total_time += other.ND_total_time;

        // ========== EPT Section ==========
        EPT_ML_count    += other.EPT_ML_count;
        EPT_lb_count    += other.EPT_lb_count;
        EPT_astar_count += other.EPT_astar_count;

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
        EPT_reuse_fail_root_node += other.EPT_reuse_fail_root_node;
        EPT_reuse_fail_no_op += other.EPT_reuse_fail_no_op;
        EPT_reuse_fail_multi_ops += other.EPT_reuse_fail_multi_ops;
        EPT_reuse_fail_parent_too_small += other.EPT_reuse_fail_parent_too_small;
        EPT_reuse_fail_vertex_count_changed += other.EPT_reuse_fail_vertex_count_changed;
        EPT_reuse_fail_mo_incompatible += other.EPT_reuse_fail_mo_incompatible;
        
        // Merge time statistics
        EPT_astar_time_no_parent_snapshot += other.EPT_astar_time_no_parent_snapshot;
        EPT_astar_time_root_node += other.EPT_astar_time_root_node;
        EPT_astar_time_no_op += other.EPT_astar_time_no_op;
        EPT_astar_time_multi_ops += other.EPT_astar_time_multi_ops;
        EPT_astar_time_parent_too_small += other.EPT_astar_time_parent_too_small;
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
        os << "============== NetDag Stats ==============\n";
        os << "ML_count: " << ND_ML_count << " (time=" << ND_ML_time << "s)\n"
           << "lb_count: " << ND_lb_count << " (time=" << ND_lb_time << "s)\n"
           << "astar_count: " << ND_astar_count << " (time=" << ND_astar_time << "s)\n"
           << "  size_lb_count: " << ND_size_lb_count << "\n"
           << "  vertex_lb_count: " << ND_vertex_lb_count << "\n"
           << "  edge_lb_degree_count: " << ND_edge_lb_degree_count << "\n"
           << "  edge_lb_label_count:  " << ND_edge_lb_label_count << "\n"
           << "ND_total_time: " << ND_total_time << "s\n"
           << "Index Overhead (ND): " << ND_index_overhead << "s\n\n";

        os << "============== EPT Stats ==============\n";
        os << "ML_count: " << EPT_ML_count << " (time=" << EPT_ML_time << "s)\n"
           << "lb_count: " << EPT_lb_count << " (time=" << EPT_lb_time << "s)\n"
           << "astar_count: " << EPT_astar_count << " (time=" << EPT_astar_time << "s)\n"
           << "  size_lb_count: " << EPT_size_lb_count << "\n"
           << "  vertex_lb_count: " << EPT_vertex_lb_count << "\n"
           << "  edge_lb_degree_count: " << EPT_edge_lb_degree_count << "\n"
           << "  edge_lb_label_count:  " << EPT_edge_lb_label_count << "\n"
           << "EPT_total_time: " << EPT_total_time << "s\n"
           << "Index Overhead (EPT): " << EPT_index_overhead << "s\n\n";

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
class GismaSearchEngine {
public:
    std::shared_ptr<NetDag> net_dag;
    double tau_index;
    double tau_search;
    double error_tolerance_search;
    int q_start;
    int q_end;
    bool has_ged_matrix;
    std::vector<double> ged_matrix;  // One-dimensional matrix
    
    std::string search_method;
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
        double tau_search,
        double error_tolerance_search,
        int q_start,
        int q_end,
        bool has_ged_matrix,
        const std::vector<double>& ged_matrix,  // Changed to a one-dimensional matrix
        
        const std::string& search_method,
       
        const std::vector<std::shared_ptr<Node>>& db_node_list,
        const std::vector<std::shared_ptr<Node>>& query_node_list,
        int N,  // Pass N
        std::vector<Graph*> db,  // Database graphs
        const std::map<int, std::map<double, std::vector<int>>>& ground_truth,  // Ground truth data
        const std::map<std::string, ui>& vM,
        const std::map<std::string, ui>& eM,
        ui max_n,
        EditPathTreeManager* ept_manager
    );

    ~GismaSearchEngine();  // Destructor

    // Get GED value from matrix
    double get_ged_from_matrix(int i, int j);

    // Compute GED online (placeholder function)
    double compute_ged_online_by_pyg(int i, int query_index);
    
    
    std::vector<int> GS_search(std::shared_ptr<Node> query_node, SearchStats &stats);
    std::vector<int> SS_search(std::shared_ptr<Node> query_node, int anchor_id, SearchStats &stats);

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
                                         SearchStats &stats);
    
    std::vector<int> EPT_forbidden_cluster_search(std::shared_ptr<Node> query_node,
        int anchor_id,
        SearchStats &stats);
    std::vector<int> extra_cluster_search(std::shared_ptr<Node> query_node,
                                                    int anchor_id,
                                                    SearchStats &stats);

    std::vector<int> Gisma_search(std::shared_ptr<Node> query_node, SearchStats &stats);
    std::vector<int> BMao_scan_search(std::shared_ptr<Node> query_node, SearchStats &stats);
    std::vector<int> BMao_export_candidates(std::shared_ptr<Node> query_node, const std::string& output_file);
    std::vector<int> answer_search(std::shared_ptr<Node> query_node, SearchStats &stats);
    bool dfs_traverse_simple(
        size_t                node_index,
        const EditPathTree   &ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>     &exact_results_within_tau,
        SearchStats          &stats,
        ui                    estimate_lb);
    
    bool dfs_traverse(
        size_t node_index,
        const EditPathTree& ept,
        std::shared_ptr<Node> query_node,
        std::vector<int>& exact_results_within_tau,
        SearchStats& stats,
        std::shared_ptr<SearchSnapshot> parent_snapshot,
        int parent_db_vertex_count,
        ui estimate_lb
    );
    




    void traverse_ept_and_search(
        const EditPathTree &ept,
        std::shared_ptr<Node> query_node,
        std::vector<int> &exact_results_within_tau,
        SearchStats &stats
    );
    // void traverse_ept_and_search(
    //     const EditPathTree& ept,
    //     std::shared_ptr<Node> query_node,
    //     std::vector<int>& exact_results_within_tau,
    //     SearchStats &stats);

    std::tuple<bool, double, int, int> compute_recall(int query_id, const std::vector<int>& exact_results_within_tau);
    std::tuple<bool, double, double, double, int, int> compute_recall_precision_IoU(int query_id, const std::vector<int>& exact_results_within_tau);
    void perform_search();

    void perform_search_parallel();

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
private:
    // Mutex for protecting ground_truth access
    mutable std::mutex ground_truth_mutex;

    // Get the EPT root node for the specified anchor ID from epts
    TreeNode* get_ept_root(ui anchor_id);

    // Get target graph ID list from EPT
    // void collect_graph_ids(TreeNode* node, std::vector<int>& ids);
};

#endif // GISMASEARCHENGINE_H
