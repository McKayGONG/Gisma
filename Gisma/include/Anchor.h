#ifndef ANCHOR_H
#define ANCHOR_H

#include "Node.h"
#include <vector>
#include <map>
#include <utility>
#include <memory>
#include <queue>

class Anchor : public Node {
public:
    int anchor_id;
    std::vector<std::pair<double, int>> friends;
    std::map<int, std::vector<std::pair<int, double>>> children; // (node_id, child_dist)
    std::priority_queue<std::pair<double, int>> nodes_in_cluster;

    // Struct for storing node distance, node ID, and mapping information
    struct ExactClusterNode {
        double dist;
        int node_id;
        std::vector<std::pair<ui, ui>> mapping;

        // Comparison function for priority_queue, sorting by distance ascending
        bool operator<(const ExactClusterNode& other) const {
            // priority_queue defaults to max-heap; to achieve min-heap, use greater comparison
            return dist > other.dist;
        }
    };

    // Use the new struct to store nodes_in_exact_cluster
    std::priority_queue<ExactClusterNode> nodes_in_exact_cluster;
    std::vector<std::pair<double, int>> nodes_in_cluster_vec;
    std::vector<std::pair<double, int>> nodes_in_exact_cluster_vec;
    std::vector<std::pair<int, double>> nodes_in_cover_range;
    double r_a;

    // Default constructor
    Anchor()
        : Node(),  // call Node's default constructor
          anchor_id(-1),
          r_a(0.0) {
        is_anchor = true;
    }
    
    Anchor(int node_id,
           const std::shared_ptr<Graph>& graph = nullptr,
           const std::string& file_name = "",
           int nearest_anchor = -1,
           double nearest_anchor_dist = -1.0,
           const std::vector<float> &emb = std::vector<float>(),
           int anchor_id = -1)
        : Node(node_id, graph, file_name, true, nearest_anchor, nearest_anchor_dist, emb),
          anchor_id(anchor_id),
          r_a(0.0) {}

    void update_cover_range(double new_range) {
        r_a = new_range;
    }
    void fill_vectors_from_queues() {
        // 1) Clear original vectors
        nodes_in_cluster_vec.clear();
        nodes_in_exact_cluster_vec.clear();

        // 2) Copy nodes_in_cluster => nodes_in_cluster_vec
        {
            auto tmp = nodes_in_cluster; // copy priority queue
            while(!tmp.empty()) {
                nodes_in_cluster_vec.push_back(tmp.top());
                tmp.pop();
            }
        }

        // 3) Copy nodes_in_exact_cluster => nodes_in_exact_cluster_vec
        //    Note: copy dist, node_id from ExactClusterNode
        {
            auto tmp = nodes_in_exact_cluster; // copy
            while(!tmp.empty()) {
                // IMPORTANT: Copy by value BEFORE pop() to avoid undefined behavior
                // Previously: const auto &ecn = tmp.top(); pop(); - UB!
                const auto ecn = tmp.top();  // Copy by value
                tmp.pop();
                // Only keep (dist, node_id); mapping can be considered if needed
                nodes_in_exact_cluster_vec.push_back({ecn.dist, ecn.node_id});
            }
        }
    }
    
};

#endif // ANCHOR_H
