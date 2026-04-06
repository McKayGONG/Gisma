#ifndef NODE_H
#define NODE_H

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include "Graph.h"

class Node {
public:
    int node_id;
    std::shared_ptr<Graph> graph;
    std::string file_name;
    bool is_anchor;
    int nearest_anchor;
    double nearest_anchor_dist;

    // GREED embedding vector
    std::vector<float> embedding;

    Node()
        : node_id(-1),
          graph(nullptr),
          file_name(""),
          is_anchor(false),
          nearest_anchor(-1),
          nearest_anchor_dist(-1.0) {}

    Node(int node_id,
         const std::shared_ptr<Graph>& graph = nullptr,
         const std::string& file_name = "",
         bool is_anchor = false,
         int nearest_anchor = -1,
         double nearest_anchor_dist = -1.0,
         const std::vector<float> &emb = std::vector<float>())
        : node_id(node_id),
          graph(graph),
          file_name(file_name),
          is_anchor(is_anchor),
          nearest_anchor(nearest_anchor),
          nearest_anchor_dist(nearest_anchor_dist),
          embedding(emb)
    {
        
    }

    virtual ~Node() = default;

    bool operator<(const Node& other) const {
        if (nearest_anchor_dist == other.nearest_anchor_dist) {
            return node_id < other.node_id;
        }
        return nearest_anchor_dist > other.nearest_anchor_dist;
    }
};

#endif // NODE_H
