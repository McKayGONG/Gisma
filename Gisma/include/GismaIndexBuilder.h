#ifndef GISMAINDEXBUILDER_H
#define GISMAINDEXBUILDER_H

#include "Utility.h"
#include "NetDag.h"
#include "Graph.h"
#include "Anchor.h"  // Ensure Anchor header is included
#include "Node.h"    // Ensure Node header is included
#include "EditPathTree.h"
#include "Config.h"
#include "NetDagConstructor.h"
#include "EditPathForestConstructor.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <functional>

// Forward declarations
class NetDagConstructor;
class EditPathForestConstructor;

class GismaIndexBuilder {
public:
    // Callback function type for statistics collection during NetDag construction
    // Parameters: r_k, anchors, phase
    using ClusterStatsCallback = std::function<void(
        double,                                         // r_k value
        const std::vector<std::shared_ptr<Anchor>>&,   // current anchors
        int                                             // current phase
    )>;

    // Constructor
    GismaIndexBuilder(const std::string& index_name,
                const std::vector<std::shared_ptr<Graph>>& graphs,
                const std::vector<std::string>& nameList,
                int root_ind = 0,
                double alpha = 9.0,
                double tau = 13.0,
                double error_tolerance_index = 0.0,
                std::shared_ptr<NetDag> net_dag = nullptr,
                bool use_parallel = false,
                bool has_ged_matrix = false,
                const std::vector<double>& ged_matrix = {},
                int N = 0, // matrix size
                double max_exact_ged_for_EPT = 17.0,
                const std::string& folder_name = "AIDS_divided",
                const std::map<std::string, ui> &vM = {},
                const std::map<std::string, ui> &eM = {},
                ui max_n = 0,
                const std::vector<std::vector<float>>& embeddings = {},
                bool enable_friends_reassign = false);
   
    ~GismaIndexBuilder();
    // Main coordination methods
    void overall_NetDag_const();
    void net_tree_const();
    void construct_EPT();
    void construct_EPT_parallel();
    
    // Delegated NetDag construction methods
    void compute_ged_to_csv();  // Only compute GED and save to csv, no NetDag modification
    void reassign_nodes_in_cluster();
    void reassign_nodes_in_cluster_with_csv();
    double compute_ged_online(const std::shared_ptr<Node> &node1, const std::shared_ptr<Node> &node2);
    void initialize();
    std::vector<double> getStateList(double maxRadius, double alpha);
    void update_phase();
    void find_max_dist();
    void update_friends();
    void update_child_and_parent_by_phase_dict();
    void link_parent_and_child();
    void update_one_anchor();
    void get_nodes_in_cover_range();
    void make_friends(std::shared_ptr<Anchor> anchor1, std::shared_ptr<Anchor> anchor2, double distance);
    void add_edge(std::shared_ptr<Anchor> parent, std::shared_ptr<Node> child, int child_phase, double child_dist);
    
    // Delegated EPT construction methods
    void compute_edit_path_and_save_to_csv();
    void find_within_tau_and_save_csv(const std::string &csv_filename);
    void build_one_ball_EPT_from_csv(int EPT_root_ind,
                                     double alpha,
                                     const std::string &csv_path,
                                     const std::string &ept_filename);
    void update_one_exact_anchor();
    void update_one_exact_anchor_parallel();
    int select_queries_from_csv(
        int root_ind,
        int need_count,
        const std::string &csv_path,
        const std::string &out_queries_txt,
        std::string &out_filtered_csv
    );
    int select_queries_by_ged_ranges(
        int root_ind,
        const std::string &csv_path,
        const std::string &output_dir
    );
    
    // Utility methods
    double get_ged_from_matrix(int i, int j);


    const std::vector<std::shared_ptr<Anchor>>& get_anchors() const;
    const std::vector<std::shared_ptr<Node>>& get_nodes() const;
    void setNetDag(std::shared_ptr<NetDag> nd);
    void setAnchors(const std::vector<std::shared_ptr<Anchor>>& anchors);
    void setNodes(const std::vector<std::shared_ptr<Node>>& nodes);

    // Set callback for cluster statistics collection
    void set_cluster_stats_callback(ClusterStatsCallback callback);

    // Get the current callback (for NetDagConstructor to use)
    const ClusterStatsCallback& get_cluster_stats_callback() const;
public:
    // Public member access for constructors
    std::string index_name;
    std::string dataset;
    std::vector<std::shared_ptr<Graph>> graphs;
    std::vector<std::string> nameList;
    int root_ind;
    double alpha;
    double tau;
    double error_tolerance_index;
    std::shared_ptr<NetDag> net_dag;
    bool use_parallel;
    bool has_ged_matrix;
    std::vector<double> ged_matrix;
    int N; // matrix size
    double max_exact_ged_for_EPT;
    std::string folder_name;
    int NDC;
    const std::map<std::string, ui>& vM;
    const std::map<std::string, ui>& eM;
    ui max_n;
    std::vector<std::shared_ptr<Anchor>> anchors;
    std::vector<std::shared_ptr<Node>> nodes;
    std::unordered_map<double, std::unordered_map<int, int>> child_and_parent_by_phase_dict;
    std::vector<double> r_list;
    std::vector<int> stateList;
    double brute_range;
    int phase;
    double r_k;
    double child_range;
    std::map<int, std::vector<int>> parent_by_phase_dict;
    std::vector<double> phaseList;  // Changed from std::vector<int> to std::vector<double> to match .cpp file
    double max_dist;
    std::shared_ptr<Node> max_node;
    int max_anchor_node_id;
    std::map<int, double> upper_bound_book;
    std::vector<double> ged_range;
    std::unordered_map<int, TreeNode*> anchor_ept_map;
    std::vector<std::vector<float>> embeddings;
    bool enable_friends_reassign;

    // Callback for statistics collection (for select-alpha mode)
    ClusterStatsCallback cluster_stats_callback_;

private:
    // Constructor instances
    std::unique_ptr<NetDagConstructor> netdag_constructor_;
    std::unique_ptr<EditPathForestConstructor> ept_constructor_;
};

#endif // NETDAGCONST_H