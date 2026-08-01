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

    // 默认构造函数
    NetDag();

    // 带参数的构造函数
    NetDag(const std::string &file_signature,
           std::shared_ptr<Node> root,
           double alpha = 0.0,
           double tau = 0.0);

    void add_anchor(const std::shared_ptr<Anchor> &anchor);

    // 打印 NetDag 信息的函数
    void print_netdag_info();

    // 保存到文件
    void save_to_file(const std::string& filename) const;
    static void load_from_file(NetDag& netdag, const std::string& filename);

    // 极简加载：只读取 anchor 的 ID + nodes_in_exact_cluster（跳过 Nodes 段、graph 数据、children、hierarchy）
    // 用于 construct_EPF 模式，避免加载完整 47GB NetDag
    static void load_anchors_only(NetDag& netdag, const std::string& filename);
};

#endif // NETDAG_H
