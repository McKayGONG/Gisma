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

    // 定义一个结构体，用于存储节点距离、节点 ID 和映射信息
    struct ExactClusterNode {
        double dist;
        int node_id;
        std::vector<std::pair<ui, ui>> mapping;

        // 为 priority_queue 实现比较函数，使其按距离从小到大排序
        bool operator<(const ExactClusterNode& other) const {
            // priority_queue 默认是大顶堆，为实现小顶堆，使用 greater 比较
            return dist > other.dist;
        }
    };

    // 使用新的结构体来存储 nodes_in_exact_cluster
    std::priority_queue<ExactClusterNode> nodes_in_exact_cluster;
    std::vector<std::pair<double, int>> nodes_in_cluster_vec;
    std::vector<std::pair<double, int>> nodes_in_exact_cluster_vec;
    std::vector<std::pair<int, double>> nodes_in_cover_range;
    double r_a;

    // 默认构造函数
    Anchor()
        : Node(),  // 调用 Node 的默认构造函数
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
        // 1) 清空原 vector
        nodes_in_cluster_vec.clear();
        nodes_in_exact_cluster_vec.clear();

        // 2) 复制 nodes_in_cluster => nodes_in_cluster_vec
        {
            auto tmp = nodes_in_cluster; // 拷贝优先队列
            while(!tmp.empty()) {
                nodes_in_cluster_vec.push_back(tmp.top());
                tmp.pop();
            }
        }

        // 3) 复制 nodes_in_exact_cluster => nodes_in_exact_cluster_vec
        //    注意要把 ExactClusterNode 的 dist、node_id 拷过去
        {
            auto tmp = nodes_in_exact_cluster; // 拷贝
            while(!tmp.empty()) {
                // IMPORTANT: Copy by value BEFORE pop() to avoid undefined behavior
                // Previously: const auto &ecn = tmp.top(); pop(); - UB!
                const auto ecn = tmp.top();  // Copy by value
                tmp.pop();
                // 只保留 (dist, node_id), 若需要 mapping也可再考虑
                nodes_in_exact_cluster_vec.push_back({ecn.dist, ecn.node_id});
            }
        }
    }
    
};

#endif // ANCHOR_H
