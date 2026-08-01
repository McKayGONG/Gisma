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
    // ========== 1) NetDag (ND) 相关统计 ==========

    // --- 计数 ---
    int ND_ML_count = 0;        // NetDag: ML 距离估计次数
    int ND_lb_count = 0;        // NetDag: 下界过滤计数
    int ND_astar_count = 0;     // NetDag: A* 搜索计数
    int ND_ndc_count = 0;       // NetDag: 唯一节点计数（去重，用于NDC统计）

    int ND_size_lb_count = 0;
    int ND_vertex_lb_count = 0;
    int ND_edge_lb_degree_count = 0;
    int ND_edge_lb_label_count = 0;
    int ND_app_test_count = 0;  // NetDag: App_test 精确下界计算次数

    // --- 耗时 ---
    double ND_ML_time = 0.0;    // NetDag: ML 距离耗时
    double ND_lb_time = 0.0;    // NetDag: 下界过滤总耗时
    double ND_astar_time = 0.0; // NetDag: A* 搜索总耗时

    // ========== 2) EPT 相关统计 ==========

    // --- 计数 ---
    int EPT_ML_count = 0;
    int EPT_lb_count = 0;
    int EPT_astar_count = 0;
    int EPT_ndc_count = 0;      // EPT: 唯一节点计数（去重，用于NDC统计）

    int EPT_size_lb_count = 0;
    int EPT_vertex_lb_count = 0;
    int EPT_edge_lb_degree_count = 0;
    int EPT_edge_lb_label_count = 0;

    // --- 耗时 ---
    double EPT_ML_time = 0.0;
    double EPT_lb_time = 0.0;
    double EPT_astar_time = 0.0;

    // ========== 3) NetDag/EPT 部分的总时间 ==========
    double ND_total_time  = 0.0;   // NetDag 整体耗时
    double EPT_total_time = 0.0;   // EPT  整体耗时

    // ========== E7: gated search-time stats (only filled when g_e7_stats==true) ==========
    long long e7_ept_trees = 0;          // # EPT trees entered (descended into) this query
    long long e7_answer_depth_sum = 0;   // sum of EPT level (root=0) over found answers
    long long e7_answer_count = 0;       // # found answers (denominator for avg depth)

    // ========== 4) 总时间 & Index Overhead ==========

    // total_time 可以理解为一次搜索/多次搜索全局的时间(可在外部逻辑里赋值)
    double total_time = 0.0;

    // NetDag 和 EPT 的 Index Overhead (可外部填充或自行计算)
    double ND_index_overhead = 0.0;  
    double EPT_index_overhead = 0.0;

    size_t EPT_reuse_attempt = 0;
    size_t EPT_reuse_count = 0;
    size_t EPT_reuse_verifications = 0;    // 验证次数
    size_t EPT_reuse_correct = 0;          // 正确次数
    size_t EPT_reuse_incorrect = 0;        // 错误次数
    std::map<int, int> EPT_ged_diff_distribution;  // GED差异分布
    
    // ========== 新增：验证AStar的时间统计 ==========
    double EPT_verification_astar_time = 0.0;  // 验证用AStar的总时间

    // ========== 复用统计细分 ==========
    // 复用失败的原因分类
    size_t EPT_reuse_fail_no_parent_snapshot = 0;     // 父节点没有snapshot（总计）
    size_t EPT_reuse_fail_no_parent_lp_skipped = 0; // 父节点被LP跳过
    size_t EPT_reuse_fail_no_parent_filter_skipped = 0; // 父节点被EPT filter跳过
    size_t EPT_reuse_fail_no_parent_reuse_no_chain = 0; // 父节点reuse成功但chain_reuse=false
    size_t EPT_reuse_fail_no_parent_other = 0;        // 其他原因(ged<0/INF, snapshot空, 无近距离child等)
    // LP-skip × reuse-able 交叉统计：被 LP 跳过的 node 里，结构上本可 reuse 的占比
    size_t lp_skip_reuseable = 0;      // LP-skipped 且结构上 reuse-able (非root+同vertex数+ops<=max_ged_gap)
    size_t lp_skip_not_reuseable = 0;  // LP-skipped 但结构上不可 reuse
    size_t EPT_reuse_fail_root_node = 0;              // 是根节点
    size_t EPT_reuse_fail_multi_ops = 0;              // accumulated_ops.size() != 1
    size_t EPT_reuse_fail_vertex_count_changed = 0;   // db_g->n != parent_db_vertex_count → 直接 fallback baseline

    // 按失败原因分类的AStar时间统计
    double EPT_astar_time_no_parent_snapshot = 0.0;
    double EPT_astar_time_parent_snapshot_empty = 0.0;
    double EPT_astar_time_parent_snapshot_size_one = 0.0;
    double EPT_astar_time_root_node = 0.0;
    double EPT_astar_time_no_op = 0.0;
    double EPT_astar_time_multi_ops = 0.0;
    double EPT_astar_time_vertex_count_changed = 0.0;
    double EPT_astar_time_mo_incompatible = 0.0;
    double EPT_astar_time_lb_pruned = 0.0;            // LB剪枝导致的
    
    // 复用成功的时间统计
    double EPT_reuse_success_time = 0.0;              // 复用成功的总时间
    
    // 结果来源统计
    size_t EPT_results_from_reuse = 0;                // 通过复用找到的结果数
    size_t EPT_results_from_astar = 0;                // 通过AStar找到的结果数
    
    // 按编辑操作类型统计
    std::map<int, size_t> EPT_op_type_count;          // 各种编辑操作类型的计数
    std::map<int, size_t> EPT_reuse_by_op_type;       // 按操作类型分的复用成功次数
    
    // 节点类型统计
    size_t EPT_leaf_nodes_processed = 0;              // 处理的叶子节点数
    size_t EPT_internal_nodes_processed = 0;          // 处理的内部节点数
    size_t EPT_nodes_with_completed_ids = 0;          // 有completed_db_graph_ids的节点数

    size_t EPT_reuse_found_ged_le_tau = 0;    // reuse方法找到的GED<=tau的次数

    // ========== 调试统计：db图 vs 中间图 ==========
    // db图：根节点(anchor)或有completed_db_graph_ids的节点
    // 中间图：通过编辑操作生成的中间节点
    int EPT_db_graph_lb_count = 0;           // db图的filter次数
    double EPT_db_graph_lb_time = 0.0;       // db图的filter时间
    int EPT_db_graph_astar_count = 0;        // db图的verification次数
    double EPT_db_graph_astar_time = 0.0;    // db图的verification时间

    int EPT_intermediate_graph_lb_count = 0;      // 中间图的filter次数
    double EPT_intermediate_graph_lb_time = 0.0;  // 中间图的filter时间
    int EPT_intermediate_graph_astar_count = 0;   // 中间图的verification次数
    double EPT_intermediate_graph_astar_time = 0.0; // 中间图的verification时间

    // Extra cluster (not in EPT) stats
    int extra_lb_count = 0;
    double extra_lb_time = 0.0;
    int extra_astar_count = 0;
    double extra_astar_time = 0.0;
    int extra_ndc_count = 0;
    size_t EPT_astar_found_ged_le_tau = 0;    // 标准AStar找到的GED<=tau的次数（验证时）

    // AppForComputation baseline 统计（用于论文 EXP-5）
    size_t EPT_baseline_app_count = 0;           // AppForComputation baseline 调用次数
    double EPT_baseline_app_time = 0.0;          // AppForComputation baseline 总时间
    double EPT_baseline_reuse_time = 0.0;        // 与 baseline 对应的 reuse 时间（只有 astar_reuse）

    // Margin overhead: App(margin) vs AppForComputation(no margin)
    size_t margin_overhead_count = 0;
    double margin_overhead_with_margin_time = 0.0;
    double margin_overhead_without_margin_time = 0.0;
    size_t margin_overhead_correct = 0;
    size_t margin_overhead_incorrect = 0;

    // Reuse speedup distribution buckets
    size_t reuse_speedup_gt3x = 0;      // baseline/reuse > 3x
    size_t reuse_speedup_2x_3x = 0;     // 2x - 3x
    size_t reuse_speedup_1x_2x = 0;     // 1x - 2x
    size_t reuse_speedup_05x_1x = 0;    // 0.5x - 1x (reuse slower)
    size_t reuse_speedup_lt05x = 0;     // < 0.5x (reuse much slower)
    // Snapshot size when reuse happens
    size_t reuse_snapshot_size_1 = 0;    // dummy only
    size_t reuse_snapshot_size_2_10 = 0;
    size_t reuse_snapshot_size_gt10 = 0;
    // Cross-tab: [size_bucket][speedup_bucket]
    //   size: 0=dummy(1), 1=small(2-10), 2=large(>10)
    //   speedup: 0=>3x, 1=2-3x, 2=1-2x, 3=0.5-1x, 4=<0.5x
    size_t reuse_xtab[3][5] = {};
    // Total time per snapshot size bucket (for overall speedup calculation)
    double reuse_xtime[3] = {};       // reuse elapsed_time per size bucket
    double baseline_xtime[3] = {};    // baseline verify_time per size bucket

    // EXP-5: 样本对 (reuse_time_ms, baseline_time_ms)，最多存10个
    std::vector<std::pair<double, double>> baseline_samples;
    static const size_t MAX_BASELINE_SAMPLES = 10;

    // Chain reuse depth statistics: depth 0 = non-chain (parent fresh A*), N = N reuses above
    static const int MAX_CHAIN_DEPTH = 16;
    size_t chain_depth_count[MAX_CHAIN_DEPTH]   = {};  // # reuses at this depth
    size_t chain_depth_correct[MAX_CHAIN_DEPTH] = {};  // verify_reuse: agreement with baseline
    size_t chain_depth_pos[MAX_CHAIN_DEPTH]     = {};  // baseline ged <= tau
    size_t chain_depth_tp[MAX_CHAIN_DEPTH]      = {};  // reuse <= tau AND baseline <= tau
    double chain_depth_reuse_time[MAX_CHAIN_DEPTH]    = {}; // reuse time per depth (s)
    double chain_depth_baseline_time[MAX_CHAIN_DEPTH] = {}; // baseline time per depth (s)

    // ========== 6) 优化效果统计 ==========

    // Subtree pruning 统计
    size_t subtree_pruning_decisions = 0;       // 做出子树剪枝决策的次数
    size_t subtree_pruning_avoided_nodes = 0;   // 因子树剪枝而避免访问的子孙节点数（不包括决策节点本身）
    size_t subtree_pruning_on_leaf_nodes = 0;   // 在叶子节点触发的子树剪枝次数

    // LB Pruning & Propagation 统计
    size_t lb_pruning_count = 0;           // LB Pruning: 因 estimate_lb > tau 跳过GED计算的次数
    size_t lb_propagation_count = 0;       // LB Propagation: 传递 estimate_lb 到子节点的次数

    // NetDag缓存复用统计
    size_t root_netdag_ged_reuse_count = 0;        // 根节点直接使用NetDag缓存GED的次数

    // 节点访问统计
    size_t EPT_total_nodes_visited = 0;            // EPT中总共访问的节点数
    size_t EPT_nodes_computed = 0;                 // 做了计算的节点数（filter + GED）
    size_t EPT_filter_pruned_nodes = 0;            // 被EPT filter剪枝的节点数
    size_t EPT_total_nodes_in_used_epts = 0;       // 使用的EPT的总节点数（用于计算subtree剪枝率）

    // ========== 5) 获取总计的方法 ==========

    // --- 计数 ---
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

    // --- 耗时 ---
    double total_ML_time() const {
        return ND_ML_time + EPT_ML_time;
    }
    double total_lb_time() const {
        return ND_lb_time + EPT_lb_time;
    }
    double total_astar_time() const {
        return ND_astar_time + EPT_astar_time;
    }

    // ========== 6) 累加（例如合并多个查询的统计） ==========

    void add(const SearchStats& other) {
        // ========== ND 部分 ==========
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

        // ========== EPT 部分 ==========
        EPT_ML_count    += other.EPT_ML_count;
        EPT_lb_count    += other.EPT_lb_count;
        EPT_astar_count += other.EPT_astar_count;
        EPT_ndc_count   += other.EPT_ndc_count;
        e7_ept_trees        += other.e7_ept_trees;
        e7_answer_depth_sum += other.e7_answer_depth_sum;
        e7_answer_count     += other.e7_answer_count;

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
        
        // 合并GED差异分布
        for (const auto& pair : other.EPT_ged_diff_distribution) {
            EPT_ged_diff_distribution[pair.first] += pair.second;
        }
        
        // 合并验证AStar时间
        EPT_verification_astar_time += other.EPT_verification_astar_time;

        // 合并新的复用失败统计
        EPT_reuse_fail_no_parent_snapshot += other.EPT_reuse_fail_no_parent_snapshot;
        EPT_reuse_fail_no_parent_lp_skipped += other.EPT_reuse_fail_no_parent_lp_skipped;
        EPT_reuse_fail_no_parent_filter_skipped += other.EPT_reuse_fail_no_parent_filter_skipped;
        EPT_reuse_fail_no_parent_reuse_no_chain += other.EPT_reuse_fail_no_parent_reuse_no_chain;
        EPT_reuse_fail_no_parent_other += other.EPT_reuse_fail_no_parent_other;
        lp_skip_reuseable += other.lp_skip_reuseable;
        lp_skip_not_reuseable += other.lp_skip_not_reuseable;
        EPT_reuse_fail_root_node += other.EPT_reuse_fail_root_node;
        EPT_reuse_fail_multi_ops += other.EPT_reuse_fail_multi_ops;
        EPT_reuse_fail_vertex_count_changed += other.EPT_reuse_fail_vertex_count_changed;

        // 合并时间统计
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
        
        // 合并结果来源统计
        EPT_results_from_reuse += other.EPT_results_from_reuse;
        EPT_results_from_astar += other.EPT_results_from_astar;
        
        // 合并节点类型统计
        EPT_leaf_nodes_processed += other.EPT_leaf_nodes_processed;
        EPT_internal_nodes_processed += other.EPT_internal_nodes_processed;
        EPT_nodes_with_completed_ids += other.EPT_nodes_with_completed_ids;
        
        EPT_reuse_found_ged_le_tau += other.EPT_reuse_found_ged_le_tau;
        EPT_astar_found_ged_le_tau += other.EPT_astar_found_ged_le_tau;

        // 合并 AppForComputation baseline 统计
        EPT_baseline_app_count += other.EPT_baseline_app_count;
        EPT_baseline_app_time += other.EPT_baseline_app_time;
        EPT_baseline_reuse_time += other.EPT_baseline_reuse_time;
        margin_overhead_count += other.margin_overhead_count;
        margin_overhead_with_margin_time += other.margin_overhead_with_margin_time;
        margin_overhead_without_margin_time += other.margin_overhead_without_margin_time;
        margin_overhead_correct += other.margin_overhead_correct;
        margin_overhead_incorrect += other.margin_overhead_incorrect;
        reuse_speedup_gt3x += other.reuse_speedup_gt3x;
        reuse_speedup_2x_3x += other.reuse_speedup_2x_3x;
        reuse_speedup_1x_2x += other.reuse_speedup_1x_2x;
        reuse_speedup_05x_1x += other.reuse_speedup_05x_1x;
        reuse_speedup_lt05x += other.reuse_speedup_lt05x;
        reuse_snapshot_size_1 += other.reuse_snapshot_size_1;
        reuse_snapshot_size_2_10 += other.reuse_snapshot_size_2_10;
        reuse_snapshot_size_gt10 += other.reuse_snapshot_size_gt10;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 5; ++j)
                reuse_xtab[i][j] += other.reuse_xtab[i][j];
        for (int i = 0; i < MAX_CHAIN_DEPTH; ++i) {
            chain_depth_count[i]         += other.chain_depth_count[i];
            chain_depth_correct[i]       += other.chain_depth_correct[i];
            chain_depth_pos[i]           += other.chain_depth_pos[i];
            chain_depth_tp[i]            += other.chain_depth_tp[i];
            chain_depth_reuse_time[i]    += other.chain_depth_reuse_time[i];
            chain_depth_baseline_time[i] += other.chain_depth_baseline_time[i];
        }
        for (int i = 0; i < 3; ++i) {
            reuse_xtime[i] += other.reuse_xtime[i];
            baseline_xtime[i] += other.baseline_xtime[i];
        }

        // 合并db图 vs 中间图统计
        EPT_db_graph_lb_count += other.EPT_db_graph_lb_count;
        EPT_db_graph_lb_time += other.EPT_db_graph_lb_time;
        EPT_db_graph_astar_count += other.EPT_db_graph_astar_count;
        EPT_db_graph_astar_time += other.EPT_db_graph_astar_time;
        EPT_intermediate_graph_lb_count += other.EPT_intermediate_graph_lb_count;
        EPT_intermediate_graph_lb_time += other.EPT_intermediate_graph_lb_time;
        EPT_intermediate_graph_astar_count += other.EPT_intermediate_graph_astar_count;
        EPT_intermediate_graph_astar_time += other.EPT_intermediate_graph_astar_time;
        extra_lb_count += other.extra_lb_count;
        extra_lb_time += other.extra_lb_time;
        extra_astar_count += other.extra_astar_count;
        extra_astar_time += other.extra_astar_time;
        extra_ndc_count += other.extra_ndc_count;

        // ========== 合并优化效果统计 ==========
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

        // 合并操作类型统计
        for (const auto& [k, v] : other.EPT_op_type_count) {
            EPT_op_type_count[k] += v;
        }
        for (const auto& [k, v] : other.EPT_reuse_by_op_type) {
            EPT_reuse_by_op_type[k] += v;
        }
    }

    // ========== 7) 打印汇总函数 ==========

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

        // db图 vs 中间图分开统计
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

// 单个query的详细信息
struct QueryDetails {
    int query_id;
    double time;
    double recall;
    double precision;
    double iou;
    bool has_ground_truth;
    bool has_results;       // found > 0, 用于控制 precision 是否参与平均

    // 新增字段用于experiment模式
    int query_nodes = 0;
    int query_edges = 0;
    int result_count = 0;
    double lb_time = 0.0;
    double astar_time = 0.0;
    // ND和EPT分别的时间
    double nd_lb_time = 0.0;
    double nd_astar_time = 0.0;
    double ept_lb_time = 0.0;
    double ept_astar_time = 0.0;
    // ND和EPT分别的count
    int nd_lb_count = 0;
    int nd_astar_count = 0;
    int nd_ndc_count = 0;    // NDC统计：唯一节点计数（去重）
    int ept_lb_count = 0;
    int ept_astar_count = 0;
    int ept_ndc_count = 0;   // NDC统计：唯一节点计数（去重）
    long long e7_ept_trees = 0;          // E7: # EPT trees entered this query
    long long e7_answer_depth_sum = 0;   // E7: sum of answer EPT levels (root=0)
    long long e7_answer_count = 0;       // E7: # answers (denominator for avg depth)
    size_t total_ept_nodes = 0;
    size_t nodes_computed = 0;
    size_t nodes_pruned = 0;
    size_t lb_propagation_count = 0;   // LB Propagation: 传递次数
    size_t lb_pruning_count = 0;       // LB Pruning: 跳过GED计算的次数
    size_t subtree_pruned = 0;              // Subtree Pruning 避免的节点数
    size_t subtree_pruning_decisions = 0;   // Subtree Pruning 决策次数
    size_t reuse_count = 0;             // Search Tree Reuse 成功次数
    size_t reuse_attempt = 0;           // Search Tree Reuse 尝试次数
    double reuse_success_time = 0.0;    // Search Tree Reuse 成功时的总时间

    // EXP-5: Baseline 统计
    size_t baseline_app_count = 0;      // Baseline AppForComputation 调用次数
    double baseline_app_time = 0.0;     // Baseline AppForComputation 总时间
    double baseline_reuse_time = 0.0;   // 与 baseline 对应的 reuse 时间
    size_t baseline_correct = 0;        // Reuse 和 Baseline 结果一致的次数
    size_t baseline_incorrect = 0;      // Reuse 和 Baseline 结果不一致的次数
    std::vector<std::pair<double, double>> baseline_samples;  // (reuse_time_ms, baseline_time_ms) 样本对

    // Reuse fail reason breakdown
    size_t reuse_fail_no_parent_snapshot = 0;
    size_t reuse_fail_no_parent_lp_skipped = 0;
    size_t reuse_fail_no_parent_filter_skipped = 0;
    size_t reuse_fail_no_parent_reuse_no_chain = 0;
    size_t reuse_fail_no_parent_other = 0;
    size_t lp_skip_reuseable = 0;
    size_t lp_skip_not_reuseable = 0;
    size_t reuse_fail_root_node = 0;
    size_t reuse_fail_multi_ops = 0;
    size_t reuse_fail_vertex_count_changed = 0;

    // db图 vs 中间图统计
    int db_graph_lb_count = 0;
    double db_graph_lb_time = 0.0;
    int db_graph_astar_count = 0;
    double db_graph_astar_time = 0.0;
    int intermediate_graph_lb_count = 0;
    double intermediate_graph_lb_time = 0.0;
    int intermediate_graph_astar_count = 0;
    double intermediate_graph_astar_time = 0.0;

    // Extra cluster stats
    int extra_lb_count_qd = 0;
    double extra_lb_time_qd = 0.0;
    int extra_astar_count_qd = 0;
    double extra_astar_time_qd = 0.0;
    int extra_ndc_count_qd = 0;

    // Margin overhead
    size_t margin_overhead_count_qd = 0;
    double margin_overhead_with_margin_time_qd = 0.0;
    double margin_overhead_without_margin_time_qd = 0.0;
    size_t margin_overhead_correct_qd = 0;
    size_t margin_overhead_incorrect_qd = 0;

    size_t reuse_speedup_gt3x_qd = 0;
    size_t reuse_speedup_2x_3x_qd = 0;
    size_t reuse_speedup_1x_2x_qd = 0;
    size_t reuse_speedup_05x_1x_qd = 0;
    size_t reuse_speedup_lt05x_qd = 0;
    size_t reuse_snapshot_size_1_qd = 0;
    size_t reuse_snapshot_size_2_10_qd = 0;
    size_t reuse_snapshot_size_gt10_qd = 0;
    size_t reuse_xtab_qd[3][5] = {};
    double reuse_xtime_qd[3] = {};
    double baseline_xtime_qd[3] = {};

    // Chain depth stats (per query)
    static const int QD_MAX_CHAIN_DEPTH = 16;
    size_t chain_depth_count_qd[QD_MAX_CHAIN_DEPTH]   = {};
    size_t chain_depth_correct_qd[QD_MAX_CHAIN_DEPTH] = {};
    size_t chain_depth_pos_qd[QD_MAX_CHAIN_DEPTH]     = {};
    size_t chain_depth_tp_qd[QD_MAX_CHAIN_DEPTH]      = {};
    double chain_depth_reuse_time_qd[QD_MAX_CHAIN_DEPTH]    = {};
    double chain_depth_baseline_time_qd[QD_MAX_CHAIN_DEPTH] = {};

    QueryDetails(int id, double t, double r = -1.0, double p = -1.0, double i = -1.0, bool has_gt = false, bool has_res = false)
        : query_id(id), time(t), recall(r), precision(p), iou(i), has_ground_truth(has_gt), has_results(has_res) {}

    // 保存为JSON文件
    void save_to_json(const std::string& filepath, double tau) const;
};

class GismaSearchEngine {
public:
    std::shared_ptr<NetDag> net_dag;
    double tau_index;
    // double tau_search;  // 移除成员变量，改为函数参数
    double error_tolerance_search;
    int q_start;
    int q_end;
    bool has_ged_matrix;
    std::vector<double> ged_matrix;  // One-dimensional matrix

    std::string search_method;
    std::string dataset_name;  // 数据集名称，用于保存实验结果
    std::string experiment_base_dir;  // 实验模式基础目录，如果设置则使用此目录作为结果保存根目录
    bool save_logs;  // 是否保存详细日志到文件
    std::string nd_mode;   // NetDag模式: "filters" / "astar" / "filters_astar"
    std::string dfs_mode;  // DFS遍历模式: "unified" / "no_reuse" / "no_SP" / "no_LP" / "only_dfs" / "reuse"
    bool use_ept_filters;  // 是否使用EPT下界过滤
    bool only_compute_db_graph;   // 只对有completed_db_graph_ids的EPT节点（即对应db图的节点）计算GED
    bool e7_stats = false;        // E7: gated search-time stats (working-set RSS / #EPT trees / answer depth). Default off = zero overhead.
    bool fast_down;        // 是否使用快速下降策略（非最后一层找到第一个满足条件的就往下走）
    int app_max_iter;      // A*最大迭代次数(控制APP_CNT/ASTAR_CNT/REUSE_CNT)
    int exact_max_iter;    // 精确计算迭代限制（训练数据生成）
    bool disable_lsa_pruning;  // 禁用LSa剪枝（仍保存lsa_lb供reuse用）
    bool disable_reuse_lsa;    // 禁用reuse中的LSa重计算（true=triangle 三角扣减）
    bool exact_value_mode = false;  // App() / app_reuse() 算精確 GED（停用 ≤tau early-exit）
    bool early_stop_at_tau = false; // parent sibling loop break at lb>tau + child reuse 跳 intersection
    bool verify_reuse;  // 验证reuse效果：每次reuse时用AppForComputation计算baseline时间
    bool chain_reuse;            // 链式复用：使用reuse的节点也保存snapshot供后续节点复用
    std::string ged_algorithm = "App";  // E8 verifier 切换：search 小步验证器。"App"=默认App-BMao(reuse/App)；"AStar"=精确A*(绕过reuse)。从 config 覆盖
    // 干净实验开关(--orig_verifier): index 路径改用原作者引擎(origbmao, 48B State)做验证器,
    // 与 app-lsa 逐位同验证器 => 剩余差异纯粹是 Gisma 框架。static: 启动赋值一次, 搜索期只读。
    static bool use_orig_verifier;
    int max_ged_gap = 3;         // 最大GED gap（默认3）
    int max_margin = 3;          // 最大margin（默认3）
    bool all_edge_labels_same = true;  // 跳過 edge label 比較；默认开（Gisma GED 模型无边标签）。从 config 覆盖
    bool disable_subtree_pruning;  // [已废弃] 现在使用 dfs_mode="no_SP" 代替
    bool disable_lb_propagation;   // [已废弃] 现在使用 dfs_mode="no_LP" 代替
    double nd_filter_ratio;  // NetDag筛选收紧系数，条件改为 lb ≤ (alpha + tau) * ratio
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

    // 新增的成员变量：用于存储所有的 EPTs
    std::unordered_map<ui, TreeNode*> epts;  // 锚点ID到EPT根节点的映射
    // 添加 EPT 管理器指针
    EditPathTreeManager* ept_manager;

    // EPT总节点数统计
    size_t total_ept_nodes = 0;  // 所有EPT的节点总数

    struct ReuseStats {
        size_t total_astar_calls = 0;      // 总的AStar调用次数
        size_t reuse_attempts = 0;         // 尝试复用的次数
        size_t reuse_success = 0;          // 成功复用的次数
        double total_astar_time = 0.0;     // 总计算时间
        double reuse_time = 0.0;           // 复用计算时间
        double baseline_time = 0.0;        // 非复用计算时间
        
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
        // double tau_search,  // 移除：改为函数参数
        double error_tolerance_search,
        int q_start,
        int q_end,
        bool has_ged_matrix,
        const std::vector<double>& ged_matrix,  // Changed to a one-dimensional matrix

        const std::string& search_method,
        const std::string& dataset_name,  // 添加 dataset_name 参数

        const std::vector<std::shared_ptr<Node>>& db_node_list,
        const std::vector<std::shared_ptr<Node>>& query_node_list,
        int N,  // Pass N
        std::vector<Graph*> db,  // Database graphs
        const std::map<int, std::map<double, std::vector<int>>>& ground_truth,  // Ground truth data
        const std::map<std::string, ui>& vM,
        const std::map<std::string, ui>& eM,
        ui max_n,
        EditPathTreeManager* ept_manager,
        const std::string& nd_mode = "filters",  // NetDag模式: "filters"/"astar"/"filters_astar"
        const std::string& dfs_mode = "unified",  // DFS遍历模式: "unified"/"simple"/"reuse"
        bool use_ept_filters = false,  // 是否使用EPT下界过滤
        bool only_compute_db_graph = false,  // 只对有completed_db_graph_ids的EPT节点计算GED
        int _app_max_iter = 2300,  // A*最大迭代次数
        bool _fast_down = false,  // 是否使用快速下降策略
        int _exact_max_iter = 1000000,  // 精确计算迭代限制
        double _nd_filter_ratio = 1.0,  // NetDag筛选收紧系数
        bool _disable_lsa_pruning = false,  // 禁用LSa剪枝
        bool _disable_reuse_lsa = false,   // 禁用reuse中的LSa重计算（true=triangle）
        bool _verify_reuse = false,  // 验证reuse效果
        bool _chain_reuse = false  // 链式复用：使用reuse的节点也保存snapshot
    );

    ~GismaSearchEngine();  // Destructor

    // Get GED value from matrix
    double get_ged_from_matrix(int i, int j);

    // Compute GED online (placeholder function)
    

    std::vector<std::tuple<int, double, int>> GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> SS_search(std::shared_ptr<Node> query_node, int anchor_id, double netdag_lb, int netdag_ged, double tau, SearchStats &stats, const std::string &dfs_mode_override = "");
    std::vector<int> SS_search_with_ept(
        std::shared_ptr<Node> query_node,
        const EditPathTree &ept,
        double tau,
        SearchStats &stats
    );



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
    std::vector<int> App_BMao_orig_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats); // 原版 App-BMao (origbmao 引擎)
    std::vector<int> AStar_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // 纯Base，用AStar()
    std::vector<int> AStar_scan_no_lsa_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // 纯Base，用AStar()但禁用LSa剪枝
    std::vector<int> AStar_BMao_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    std::vector<int> App_LSa_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);   // E8 app-lsa: full scan + LSa (approx)
    std::vector<int> AStar_LSa_scan_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats); // E8 astar-lsa: full scan + LSa (exact)
    std::vector<int> Base_GS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // Base+GS: 只用GS选cluster，App-BMao验证
    std::vector<int> Base_SS_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // Base+SS: 只用SS选cluster，App-BMao验证
    std::vector<int> Base_All_EPT_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);  // Base_All_EPT: 遍历所有anchor，不过滤
    std::vector<int> answer_search(std::shared_ptr<Node> query_node, double tau, SearchStats &stats);
    // 计算EPT子树大小的辅助函数
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
        int anchor_netdag_ged,
        int parent_no_snapshot_reason = 0,
        size_t parent_node_index = SIZE_MAX  // 由 caller 傳實際 parent index 供 reuse_log 用
    );

    // dfs_traverse_no_SP: Reuse + DP，但禁用 Subtree Pruning
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
        int anchor_netdag_ged,
        int parent_no_snapshot_reason = 0,
        size_t parent_node_index = SIZE_MAX
    );

    // dfs_traverse_no_LP: Reuse + SP，但禁用 LB Propagation
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
        int anchor_netdag_ged,
        int parent_no_snapshot_reason = 0,
        size_t parent_node_index = SIZE_MAX
    );

    // 纯DFS遍历（使用App_baseline，无LB传递，无Subtree Pruning）
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

    std::tuple<bool, double, double, double, int, int> compute_recall_precision_IoU(int query_id, const std::vector<int>& exact_results_within_tau, double tau);
    void perform_search(double tau);

    void perform_search_parallel(double tau);

    // 单个query搜索接口（用于experiment模式）

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

    // 保存单个query的JSON结果（用于实时保存）
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
    // 互斥锁用于保护 ground_truth 访问
    mutable std::mutex ground_truth_mutex;

    // 从 epts 中获取指定锚点 ID 的 EPT 根节点
    TreeNode* get_ept_root(ui anchor_id);

    // 从 EPT 中获取目标图 ID 列表
    // void collect_graph_ids(TreeNode* node, std::vector<int>& ids);
};

#endif // GISMASEARCHENGINE_H
