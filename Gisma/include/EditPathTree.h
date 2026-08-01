// EditPathTree.h

#ifndef EDIT_PATH_TREE_H
#define EDIT_PATH_TREE_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <climits> // For INT_MAX
#include <queue>
#include <fstream>
#include <sstream>
#include <mutex>
#include <shared_mutex>
#include "Graph.h"

// 类型定义
typedef unsigned int ui;

// 前向声明
class EditPathTree;

struct TreeStatistics {
    // ========== 1) 深度相关 ==========
    int max_depth = 0;
    int min_depth = std::numeric_limits<int>::max();

    // 每层节点数
    std::map<int, int> nodes_per_level;

    // “层级分布” => [深度 => (原始层级 => 个数)]
    std::map<int, std::map<int, int>> level_breakdown;

    // ========== 2) Fanout 统计 ==========
    int max_fanout = 0;
    int min_fanout = 0; // 若要表示未赋值，可配合判断
    int total_fanout = 0;
    int internal_node_count = 0;
    long long total_graph_count = 0;
    // ---- 每层 fanout 分布 (FanoutLevelStats) ----
    struct FanoutLevelStats {
        int count = 0;                                  // 有子节点的数量(本层的内节点数)
        int max_fanout = 0;                             // fanout最大值
        int min_fanout = std::numeric_limits<int>::max();
        long long sum_fanout = 0;                       // fanout求和
        long long sum_squares = 0;                      // fanout^2 求和(用于方差)
    };
    // 用来统计：每个 EPT-level => 这层所有“叶子”的 BFS-dist 的分布
    struct LeafDistStats {
        long long sum_dist = 0;       // dist 求和
        long long sum_squares = 0;    // dist^2 求和 (用于标准差)
        int count = 0;               // 叶子数量
    };

    // EPT-level => LeafDistStats
    std::map<int, LeafDistStats> leaf_dist_per_ept_level;

    // 每层 -> FanoutLevelStats
    std::map<int, FanoutLevelStats> fanout_per_level;

    // ========== 3) 新增字段：总节点数 ==========
    // (在遍历时，每访问一个节点都 +1)
    int total_nodes = 0;

    // ========== 4) 新增字段：每层的叶子节点数 ==========
    // (在遍历时，若 children_indices.empty() 则 leaf_count_per_level[level]++)
    std::map<int, int> leaf_count_per_level;

    // ========== 5) 再新增: 叶子节点总数 (若需要) ==========
    // (在遍历时，只要是叶子 children_indices.empty() 就 total_leaf_nodes++)
    int total_leaf_nodes = 0;

    // ========== 6) 若你要 BFS 统计 “距离(步数)->叶子数” 分布 ==========
    // (在 BFS 时, 如果 dist 步到达一个叶子 => leaf_distance_count[dist]++)
    std::map<int,int> leaf_distance_count;
};



// TreeNode 类
class TreeNode {
public:
    // 编辑操作
    EditOperation op;

    // 子节点索引
    std::vector<size_t> children_indices;

    // 多个完成的 db_graph_id 列表
    std::vector<int> completed_db_graph_ids;

    size_t parent_index;

    int level;              // 原始层级
    int simplified_level;   // 简化后的层级
    int max_subtree_depth = -1;  // max level in subtree (precomputed, -1 = not computed)
    int ept_node_id;        // EPT节点ID
    ui anchor_id;           // 锚点ID
    int tree_node_graph_id;
    int db_graph_n;
    int db_graph_m;
    ui* db_vlabels;
    Graph* db_graph;
    std::vector<EditOperation> accumulated_ops;
    PseudoGraph pseudo_graph;
    std::shared_ptr<Graph> graph_ptr;

    // ML_graph removed: using embedding vectors instead  
    // ML_graph ml_graph;
    std::vector<float> embedding; // GREED embedding vector for this TreeNode

    TreeNode(const EditOperation& operation = EditOperation(),
             int level = 0, int ept_node_id = 0, ui anchor_id = 0,
             int tree_node_graph_id = -1, size_t parent_idx = SIZE_MAX, 
             const PseudoGraph& pg = PseudoGraph(), const std::shared_ptr<Graph>& gp = nullptr,
             const std::vector<float>& emb = std::vector<float>());

    ~TreeNode();

    // 保存到文件的函数
    void save_to_file(const std::string& filename) const;

    // 保存到流的函数
    void save_to_stream(std::ostream& os) const;

    // 从流中加载节点的函数
    void load_from_stream(std::istream& is);

    // 收集数据库图 ID 的成员函数
    void collect_graph_ids(const std::vector<TreeNode>& nodes, std::vector<int>& ids, int depth_limit, int current_depth) const;

    // 静态函数用于从文件加载单个 EPT
    static TreeNode* load_from_file(const std::string& filename);
};


// EditPathTree 类
class EditPathTree {
public:
    std::vector<TreeNode> tree_nodes;
    
    // std::unordered_map<int, std::shared_ptr<Graph>> tree_node_graph_map;
    
    size_t root_index;           // 根节点的索引
    TreeStatistics stats;
    ui anchor_id; // 添加 anchor_id 成员变量
    
    // 构造函数和析构函数
    EditPathTree(ui anchor_id = 0);
    ~EditPathTree() = default;

    // 计算统计信息
    void compute_statistics();

    // Precompute max_subtree_depth for all nodes
    void precompute_max_subtree_depth();
 
    // 收集目标图 ID
    void collect_graph_ids(std::vector<int>& ids, int depth_limit) const;

    // E11 真增量插入(merge): 把 db 图 g 合并进本 EPT。full_ops = root(anchor)->g 的 edit 操作集(可乱序)。
    // 从 root 沿"整条压缩边的 op 集 ⊆ 剩余 op"的子节点贪心下走(共享前缀, 不拆已有节点)，
    // 走不动了就建【一个压缩节点】(accumulated_ops = 剩余全部, db_graph 指向真实 g) 挂上, g 落其中。
    // 若剩余恰好为空(g 与某现有节点重合)→ 直接把 gid 加入该节点。调用后需 precompute_max_subtree_depth()。
    void insert_graph_merge(const std::vector<EditOperation>& full_ops, Graph* g_graph, int db_graph_id);

    // E11: 移除指定 db 图 id(从所有节点的 completed_db_graph_ids 删除)，返回实际被删的(去重)id。
    std::vector<int> remove_db_graph_ids(const std::unordered_set<int>& ids_to_remove);

    // 保存和加载函数
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);
    void reorder() {
        std::vector<int> dfs_ord;
        auto dfs = [&](auto &&self, int u)->void {
        dfs_ord.push_back(u);
        for (auto v : tree_nodes[u].children_indices) 
            self(self, v);
        };
        dfs(dfs, root_index);
        std::map<int, int> new_id;
        for (int i = 0; i < dfs_ord.size(); i++) new_id[dfs_ord[i]] = i;
        std::vector<TreeNode> new_tree_nodes;
        for (int i = 0; i < dfs_ord.size(); i++) {
            new_tree_nodes.push_back(tree_nodes[dfs_ord[i]]);
            auto &u = new_tree_nodes.back();
            for (auto &v : u.children_indices) v = new_id[v];
            if (new_id.count(u.parent_index)) u.parent_index = new_id[u.parent_index];
            if (new_id.count(u.ept_node_id)) u.ept_node_id = new_id[u.ept_node_id];
            u.db_graph_m=u.graph_ptr->m;
            u.db_graph_n=u.graph_ptr->n;
        }
        swap(tree_nodes, new_tree_nodes);
        root_index = 0;
    }
    
};

// EditPathTreeManager 类
class EditPathTreeManager {
public:
    // 加载所有 EPT
    void load_all_epts_from_directory_parallel(const std::string& directory_path);
    // 根据锚点 ID 获取 EPT
    EditPathTree* get_ept(ui anchor_id);

    // 无锁版本：仅在确保 EPT 加载完成后的只读阶段使用
    // 用于查询阶段，避免 shared_lock 的原子操作开销
    EditPathTree* get_ept_no_lock(ui anchor_id) const;

    // 获取已加载的 EPT 数量
    size_t get_ept_count() const;

    // 获取所有EPT的总节点数
    size_t get_total_nodes() const;

    // E11: 收集所有 EPT 叶子节点(children_indices 为空、非 root/anchor)对应的 db 图 id。
    // 用于「去掉 100 个叶子图再插回」的 update 实验(叶子无子树→remove/insert 不重构层级)。
    std::vector<int> collect_leaf_db_ids() const;

    // E11 真增量插入: 列出所有已加载的 anchor id(便于遍历 EPT)。
    std::vector<ui> all_anchor_ids() const;

    // 清理所有 EPTs
    void clear_all_epts();

    // 析构函数，清理 EPTs
    ~EditPathTreeManager() = default;

private:
    // 锚点 ID 到 EPT 的映射，使用智能指针管理内存
    std::unordered_map<ui, std::unique_ptr<EditPathTree>> ept_map;

    // 读写锁用于保护 ept_map，允许多线程并发读取
    // 使用 shared_mutex 替代 mutex，解决并行查询时的锁竞争问题
    mutable std::shared_mutex map_mutex;
};


// 构建和打印编辑路径树的函数声明
void build_edit_path_tree(
    EditPathTree& ept,
    const std::vector<std::vector<EditOperation>>& edit_operations_list,
    const std::vector<ui>& db_graph_ids,
    ui anchor_id,
    const Graph& anchor_graph);
void build_edit_path_tree_recursive(
    EditPathTree& ept,
    const std::vector<std::unordered_set<EditOperation, EditOperationHash, EditOperationEqual>>& undone_ops_lists,
    const std::vector<ui>& db_graph_ids,
    ui anchor_id,
    size_t node_idx,
    const std::vector<ui>& curr_indices,
    int current_level,
    int& ept_node_counter);

void simplifyNode(EditPathTree& ept, size_t node_index);
void simplify_EPT(EditPathTree& ept);


void update_simplified_levels(EditPathTree& ept, size_t node_index, int current_level);

std::string get_wl_hash(const TreeNode &tn);
// 压缩 completed_db_graph_ids（实现见 EditPathTree.cpp）

#endif // EDIT_PATH_TREE_H
