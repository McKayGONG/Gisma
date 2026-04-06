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

// Type definition
typedef unsigned int ui;

// Forward declaration
class EditPathTree;

struct TreeStatistics {
    // ========== 1) Depth Related ==========
    int max_depth = 0;
    int min_depth = std::numeric_limits<int>::max();

    // Nodes per level
    std::map<int, int> nodes_per_level;

    // “Level breakdown” => [depth => (original level => count)]
    std::map<int, std::map<int, int>> level_breakdown;

    // ========== 2) Fanout Statistics ==========
    int max_fanout = 0;
    int min_fanout = 0; // can be combined with a check to indicate unassigned
    int total_fanout = 0;
    int internal_node_count = 0;
    long long total_graph_count = 0;
    // ---- Per-level fanout distribution (FanoutLevelStats) ----
    struct FanoutLevelStats {
        int count = 0;                                  // number of nodes with children (internal nodes at this level)
        int max_fanout = 0;                             // maximum fanout
        int min_fanout = std::numeric_limits<int>::max();
        long long sum_fanout = 0;                       // sum of fanouts
        long long sum_squares = 0;                      // sum of fanout^2 (for variance)
    };
    // Statistics: for each EPT-level => distribution of BFS-dist of all leaves at this level
    struct LeafDistStats {
        long long sum_dist = 0;       // sum of dist
        long long sum_squares = 0;    // sum of dist^2 (for standard deviation)
        int count = 0;               // leaf count
    };

    // EPT-level => LeafDistStats
    std::map<int, LeafDistStats> leaf_dist_per_ept_level;

    // Level -> FanoutLevelStats
    std::map<int, FanoutLevelStats> fanout_per_level;

    // ========== 3) Added field: total node count ==========
    // (during traversal, increment for each visited node)
    int total_nodes = 0;

    // ========== 4) Added field: leaf node count per level ==========
    // (during traversal, if children_indices.empty() then leaf_count_per_level[level]++)
    std::map<int, int> leaf_count_per_level;

    // ========== 5) Added: total leaf node count ==========
    // (during traversal, if children_indices.empty() then total_leaf_nodes++)
    int total_leaf_nodes = 0;

    // ========== 6) BFS statistics: “distance (steps) -> leaf count” distribution ==========
    // (during BFS, if dist steps reach a leaf => leaf_distance_count[dist]++)
    std::map<int,int> leaf_distance_count;
};



// TreeNode class
class TreeNode {
public:
    // Edit operation
    EditOperation op;

    // Child node indices
    std::vector<size_t> children_indices;

    // List of completed db_graph_ids
    std::vector<int> completed_db_graph_ids;

    size_t parent_index;

    int level;              // original level
    int simplified_level;   // simplified level
    int ept_node_id;        // EPT node ID
    ui anchor_id;           // anchor ID
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

    // Save to file function
    void save_to_file(const std::string& filename) const;

    // Save to stream function
    void save_to_stream(std::ostream& os) const;

    // Load node from stream function
    void load_from_stream(std::istream& is);

    // Member function to collect database graph IDs
    void collect_graph_ids(const std::vector<TreeNode>& nodes, std::vector<int>& ids, int depth_limit, int current_depth) const;

    // Static function to load a single EPT from file
    static TreeNode* load_from_file(const std::string& filename);
};


// EditPathTree class
class EditPathTree {
public:
    std::vector<TreeNode> tree_nodes;
    
    // std::unordered_map<int, std::shared_ptr<Graph>> tree_node_graph_map;
    
    size_t root_index;           // root node index
    TreeStatistics stats;
    ui anchor_id; // anchor_id member variable

    // Constructor and destructor
    EditPathTree(ui anchor_id = 0);
    ~EditPathTree() = default;

    // Compute statistics
    void compute_statistics();
 
    // Collect target graph IDs
    void collect_graph_ids(std::vector<int>& ids, int depth_limit) const;

    // Save and load functions
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

// EditPathTreeManager class
class EditPathTreeManager {
public:
    // Load all EPTs
    void load_all_epts_from_directory(const std::string& directory_path);
    void load_all_epts_from_directory_parallel(const std::string& directory_path);
    // Get EPT by anchor ID
    EditPathTree* get_ept(ui anchor_id);

    // Lock-free version: only use in read-only phase after EPT loading is complete
    // Used in query phase to avoid atomic operation overhead of shared_lock
    EditPathTree* get_ept_no_lock(ui anchor_id) const;

    // Get the number of loaded EPTs
    size_t get_ept_count() const;

    // Get the total node count across all EPTs
    size_t get_total_nodes() const;

    // Clear all EPTs
    void clear_all_epts();

    // Destructor, clears EPTs
    ~EditPathTreeManager() = default;

private:
    // Mapping from anchor ID to EPT, using smart pointers for memory management
    std::unordered_map<ui, std::unique_ptr<EditPathTree>> ept_map;

    // Read-write lock for protecting ept_map, allowing concurrent multi-thread reads
    // Using shared_mutex instead of mutex to resolve lock contention during parallel queries
    mutable std::shared_mutex map_mutex;
};


// Function declarations for building and printing edit path trees
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

void rebuild_and_reindex_ept_dfs(EditPathTree& ept);
void simplifyNode(EditPathTree& ept, size_t node_index);
void dfs_reindex(EditPathTree& ept, size_t old_idx, std::vector<bool>& visited,
                  std::vector<size_t>& old2new, std::vector<TreeNode>& new_nodes, size_t& nextIndex);
void remove_parent_inf_nodes(EditPathTree& ept);
void simplify_EPT(EditPathTree& ept);

void print_EPT(const EditPathTree& ept, bool simplified = false);
void print_EPT_with_actual_steps(const EditPathTree& ept);
void test_EPT_structure(const EditPathTree &ept);

void update_simplified_levels(EditPathTree& ept, size_t node_index, int current_level);

double compute_distance(const Graph& g1, const Graph& g2);
std::string get_wl_hash(const TreeNode &tn);
void merge_children_by_WLhash_bfs(EditPathTree & ept);
// Shrink completed_db_graph_ids (implementation in EditPathTree.cpp)
void shrink_completed_ids_to_first(EditPathTree& ept);

#endif // EDIT_PATH_TREE_H
