#ifndef NETDAG_H
#define NETDAG_H

#include "Utility.h"
#include "Node.h"
#include "Anchor.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <shared_mutex>

class NetDag {
public:
    std::string file_signature;
    std::shared_ptr<Node> root;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Anchor>> anchors;
    double alpha;
    double tau;
    std::map<int, std::vector<int>> parent_by_phase_dict;

    // Thread-safe mutex for protecting Anchor children map access
    // Use mutable to allow locking in const methods
    mutable std::shared_mutex children_mutex;

    // Default constructor
    NetDag();

    // Parameterized constructor
    NetDag(const std::string &file_signature,
           std::shared_ptr<Node> root,
           double alpha = 0.0,
           double tau = 0.0);

    void add_node(const std::shared_ptr<Node> &node);
    void add_anchor(const std::shared_ptr<Anchor> &anchor);

    // Function to print NetDag information
    void print_netdag_info();

    // Save to file
    void save_to_file(const std::string& filename) const;
    static void load_from_file(NetDag& netdag, const std::string& filename);

    // Minimal loading: only read anchor IDs + nodes_in_exact_cluster (skip Nodes section, graph data, children, hierarchy)
    // Used for construct_EPF mode, avoiding loading the full 47GB NetDag
    static void load_anchors_only(NetDag& netdag, const std::string& filename);
};

#endif // NETDAG_H
