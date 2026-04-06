#ifndef NETDAGCONSTRUCTOR_H
#define NETDAGCONSTRUCTOR_H

#include "Utility.h"
#include "NetDag.h"
#include "Graph.h"
#include "Anchor.h"
#include "Node.h"
#include <string>
class GismaIndexBuilder;
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>

// Forward declaration
class NetDagConst;

class NetDagConstructor {
public:
    // Constructor
    NetDagConstructor(GismaIndexBuilder* parent);
    
    // NetDag construction methods
    void overall_NetDag_const();
    void net_tree_const();
    void initialize();
    void update_phase();
    
    // Phase and structure updates
    void find_max_dist();
    void update_friends();
    void update_child_and_parent_by_phase_dict();
    void link_parent_and_child();
    void update_one_anchor();
    void get_nodes_in_cover_range();
    
    // Graph distance and connections
    double compute_ged_online(const std::shared_ptr<Node> &node1, const std::shared_ptr<Node> &node2);
    void make_friends(std::shared_ptr<Anchor> anchor1, std::shared_ptr<Anchor> anchor2, double distance);
    void add_edge(std::shared_ptr<Anchor> parent, std::shared_ptr<Node> child, int child_phase, double child_dist);
    
    // Node reassignment and clustering
    void compute_ged_to_csv();  // Only compute GED and save to csv, no NetDag modification
    void reassign_nodes_in_cluster();
    void reassign_nodes_in_cluster_with_csv();
    
    // Utility methods
    std::vector<double> getStateList(double maxRadius, double alpha);
    void filter_friends_before_saving();

    // Debug methods for coverage verification
    int verify_coverage(const std::string& checkpoint_name);

private:
    GismaIndexBuilder* parent_;  // Reference to parent GismaIndexBuilder object
};

#endif // NETDAGCONSTRUCTOR_H