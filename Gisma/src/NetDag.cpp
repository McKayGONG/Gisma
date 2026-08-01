#include "NetDag.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <cmath>
#include <climits>

NetDag::NetDag()
    : file_signature(""),
      root(nullptr),
      alpha(0.0),
      tau(0.0) {
    // 初始化其他成员变量
    nodes.clear();
    anchors.clear();
    parent_by_phase_dict.clear();
}

NetDag::NetDag(const std::string& file_signature,
               std::shared_ptr<Node> root,
               double alpha,
               double tau)
    : file_signature(file_signature),
      root(root),
      alpha(alpha),
      tau(tau) {
    if (root) {
        if (root->node_id >= nodes.size()) {
            nodes.resize(root->node_id + 1);  // 重新调整大小
        }
        nodes[root->node_id] = root;  // 现在可以安全地分配
    }
}


void NetDag::add_anchor(const std::shared_ptr<Anchor>& anchor) {
    anchors.push_back(anchor);
}

void NetDag::save_to_file(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
        return;
    }

    // Write basic NetDag information
    ofs << "NetDag File Signature: " << file_signature << "\n";
    ofs << "alpha: " << alpha << "\n";
    ofs << "Tau: " << tau << "\n";

    // Count non-null nodes
    size_t actual_node_count = 0;
    for (const auto& node : nodes) {
        if (node) {
            ++actual_node_count;
        }
    }

    // Write actual node count and anchor count
    ofs << "Number of Nodes: " << actual_node_count << "\n";
    ofs << "Number of Anchors: " << anchors.size() << "\n";

    // Write node information
    ofs << "\n# Nodes\n";
    for (const auto& node : nodes) {
        if (node) {
            ofs << "NodeID: " << node->node_id << "\n";
            ofs << "FileName: " << node->file_name << "\n";
            ofs << "IsAnchor: " << (node->is_anchor ? "true" : "false") << "\n";
            ofs << "NearestAnchorID: " << node->nearest_anchor << "\n";
            ofs << "NearestAnchorDist: " << node->nearest_anchor_dist << "\n";

            // Save graph data
            if (node->graph) {
                ofs << "HasGraph: true\n";
                node->graph->save_to_stream(ofs);
            } else {
                ofs << "HasGraph: false\n";
            }

            ofs << "EndNode\n";
        }
    }

    // Write anchor information
    ofs << "\n# Anchors\n";
    for (const auto& anchor : anchors) {
        ofs << "AnchorID: " << anchor->anchor_id << "\n";
        // Include data inherited from Node
        ofs << "NodeID: " << anchor->node_id << "\n";
        ofs << "FileName: " << anchor->file_name << "\n";
        ofs << "IsAnchor: " << (anchor->is_anchor ? "true" : "false") << "\n";
        ofs << "NearestAnchorID: " << anchor->nearest_anchor << "\n";
        ofs << "NearestAnchorDist: " << anchor->nearest_anchor_dist << "\n";

        // Save graph data
        if (anchor->graph) {
            ofs << "HasGraph: true\n";
            anchor->graph->save_to_stream(ofs);
        } else {
            ofs << "HasGraph: false\n";
        }

        // Write children
        ofs << "ChildrenCount: " << anchor->children.size() << "\n";
        for (const auto& child_entry : anchor->children) {
            int phase = child_entry.first;
            const auto& child_list = child_entry.second;
            ofs << "Phase: " << phase << " ChildListSize: " << child_list.size() << "\n";
            for (const auto& child_pair : child_list) {
                ofs << child_pair.first << " " << child_pair.second << "\n";
            }
        }

        // Write nodes_in_cluster
        ofs << "NodesInClusterCount: " << anchor->nodes_in_cluster.size() << "\n";
        // Since priority_queue doesn't support iterators, we need to copy it
        std::priority_queue<std::pair<double, int>> temp_cluster = anchor->nodes_in_cluster;
        while (!temp_cluster.empty()) {
            auto pair = temp_cluster.top();
            temp_cluster.pop();
            ofs << pair.first << " " << pair.second << "\n";
        }

        // Write nodes_in_exact_cluster
        ofs << "NodesInExactClusterCount: " << anchor->nodes_in_exact_cluster.size() << "\n";
        // Copy the priority_queue
        std::priority_queue<Anchor::ExactClusterNode> temp_exact_cluster = anchor->nodes_in_exact_cluster;
        while (!temp_exact_cluster.empty()) {
            // IMPORTANT: Copy the value BEFORE pop() to avoid undefined behavior
            // Previously: const auto& cluster_node = temp_exact_cluster.top(); pop();
            // This was UB because the reference becomes invalid after pop()
            Anchor::ExactClusterNode cluster_node = temp_exact_cluster.top();  // Copy by value
            temp_exact_cluster.pop();
            // Write dist and node_id
            ofs << "Dist: " << cluster_node.dist << "\n";
            ofs << "NodeID: " << cluster_node.node_id << "\n";
            // Write mapping
            ofs << "MappingSize: " << cluster_node.mapping.size() << "\n";
            for (const auto& mapping_pair : cluster_node.mapping) {
                ofs << mapping_pair.first << " " << mapping_pair.second << "\n";
            }
        }

        // Write r_a
        ofs << "RA: " << anchor->r_a << "\n";
        ofs << "EndAnchor\n";
    }

    // Write parent_by_phase_dict
    ofs << "\n# ParentByPhaseDict\n";
    ofs << "DictSize: " << parent_by_phase_dict.size() << "\n";
    for (const auto& entry : parent_by_phase_dict) {
        int phase = entry.first;
        const auto& node_ids = entry.second;
        ofs << "Phase: " << phase << " NodeCount: " << node_ids.size() << "\n";
        for (int node_id : node_ids) {
            ofs << node_id << " ";
        }
        ofs << "\n";
    }

    // Save root node ID
    ofs << "RootNodeID: " << (root ? root->node_id : -1) << "\n";

    ofs.close();
    std::cout << "NetDag data has been saved to " << filename << std::endl;
}


// 需要包含的头文件
#include <omp.h>
#include <thread>
#include <atomic>
#include <chrono>

void NetDag::load_from_file(NetDag& netdag, const std::string& filename) {
    // 使用更大的缓冲区提高I/O性能
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return;
    }
    
    // 设置更大的缓冲区
    const size_t buffer_size = 1024 * 1024; // 1MB buffer
    std::vector<char> buffer(buffer_size);
    ifs.rdbuf()->pubsetbuf(buffer.data(), buffer_size);

    // 兼容Windows/Linux换行符：getline后去掉尾部\r
    auto getline_trim = [&](std::ifstream& s, std::string& l) -> std::ifstream& {
        std::getline(s, l);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        return s;
    };
    
    std::string line;
    line.reserve(1024); // 预分配空间减少内存分配

    // Read basic NetDag information
    getline_trim(ifs, line); // NetDag File Signature
    netdag.file_signature = line.substr(line.find(":") + 2);

    getline_trim(ifs, line); // alpha
    netdag.alpha = std::stod(line.substr(line.find(":") + 2));

    getline_trim(ifs, line); // Tau
    netdag.tau = std::stod(line.substr(line.find(":") + 2));

    // Read node and anchor counts
    getline_trim(ifs, line); // Number of Nodes
    int actual_node_count = std::stoi(line.substr(line.find(":") + 2));

    getline_trim(ifs, line); // Number of Anchors
    int num_anchors = std::stoi(line.substr(line.find(":") + 2));

    // 预分配空间
    netdag.nodes.clear();
    netdag.nodes.resize(actual_node_count);
    netdag.anchors.reserve(num_anchors);

    // 用于存储需要并行初始化的节点
    std::vector<std::shared_ptr<Node>> nodes_to_initialize;
    nodes_to_initialize.reserve(actual_node_count);

    // Skip empty line and # Nodes
    getline_trim(ifs, line); // Empty line
    getline_trim(ifs, line); // # Nodes

    printf("[NetDag::load_from_file] Loading %d nodes...\n", actual_node_count);
    fflush(stdout);

    // Phase 1: 串行读取所有节点数据（但延迟Graph初始化）
    for (int i = 0; i < actual_node_count; ) {
        if (i % 100000 == 0) {
            printf("[NetDag::load_from_file] Progress: %d/%d nodes (%.1f%%)\n",
                   i, actual_node_count, 100.0 * i / actual_node_count);
            fflush(stdout);
        }

        getline_trim(ifs, line);
        if (line == "EndNode") {
            i++;
            continue;
        }

        auto node = std::make_shared<Node>();
        bool has_graph_data = false;

        do {
            if (line.find("NodeID:") == 0) {
                node->node_id = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("FileName:") == 0) {
                node->file_name = line.substr(line.find(":") + 2);
            } else if (line.find("IsAnchor:") == 0) {
                std::string value = line.substr(line.find(":") + 2);
                node->is_anchor = (value == "true");
            } else if (line.find("NearestAnchorID:") == 0) {
                node->nearest_anchor = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("NearestAnchorDist:") == 0) {
                node->nearest_anchor_dist = std::stod(line.substr(line.find(":") + 2));
            } else if (line.find("HasGraph:") == 0) {
                std::string has_graph_str = line.substr(line.find(":") + 2);
                bool has_graph = (has_graph_str == "true");
                if (has_graph) {
                    node->graph = std::make_shared<Graph>();
                    node->graph->load_from_stream(ifs);
                    has_graph_data = true;
                    // 注意：这里不进行初始化，延迟到并行阶段
                } else {
                    node->graph = nullptr;
                }
            }

            getline_trim(ifs, line);
        } while (line != "EndNode");

        netdag.nodes[node->node_id] = node;

        // 记录需要初始化的节点
        if (has_graph_data) {
            nodes_to_initialize.push_back(node);
        }
        
        i++;
    }

    printf("[NetDag::load_from_file] Finished loading %d nodes\n", actual_node_count);
    fflush(stdout);

    // Skip empty line and # Anchors
    getline_trim(ifs, line); // Empty line
    getline_trim(ifs, line); // # Anchors

    printf("[NetDag::load_from_file] Loading %d anchors...\n", num_anchors);
    fflush(stdout);

    // Read anchor information (同样延迟Graph初始化)
    for (int i = 0; i < num_anchors; i++) {
        if (i % 1000 == 0) {
            printf("[NetDag::load_from_file] Anchor progress: %d/%d (%.1f%%)\n",
                   i, num_anchors, 100.0 * i / num_anchors);
            fflush(stdout);
        }
        auto anchor = std::make_shared<Anchor>();
        bool has_graph_data = false;
        
        while (getline_trim(ifs, line) && line != "EndAnchor") {
            if (line.find("AnchorID:") == 0) {
                anchor->anchor_id = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("NodeID:") == 0) {
                anchor->node_id = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("FileName:") == 0) {
                anchor->file_name = line.substr(line.find(":") + 2);
            } else if (line.find("IsAnchor:") == 0) {
                std::string value = line.substr(line.find(":") + 2);
                anchor->is_anchor = (value == "true");
            } else if (line.find("NearestAnchorID:") == 0) {
                anchor->nearest_anchor = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("NearestAnchorDist:") == 0) {
                anchor->nearest_anchor_dist = std::stod(line.substr(line.find(":") + 2));
            } else if (line.find("HasGraph:") == 0) {
                std::string has_graph_str = line.substr(line.find(":") + 2);
                bool has_graph = (has_graph_str == "true");
                if (has_graph) {
                    anchor->graph = std::make_shared<Graph>();
                    anchor->graph->load_from_stream(ifs);
                    has_graph_data = true;
                    // 延迟初始化
                } else {
                    anchor->graph = nullptr;
                }
            } else if (line.find("ChildrenCount:") == 0) {
                int children_count = std::stoi(line.substr(line.find(":") + 2));
                
                for (int j = 0; j < children_count; ++j) {
                    getline_trim(ifs, line);
                    int phase;
                    int child_list_size;
                    size_t pos_phase = line.find("Phase: ");
                    size_t pos_child_list_size = line.find("ChildListSize: ");
                    if (pos_phase != std::string::npos && pos_child_list_size != std::string::npos) {
                        phase = std::stoi(line.substr(pos_phase + 7, pos_child_list_size - (pos_phase + 7)));
                        child_list_size = std::stoi(line.substr(pos_child_list_size + 14));
                    } else {
                        std::cerr << "Error parsing Phase and ChildListSize, line content: " << line << std::endl;
                        continue;
                    }

                    std::vector<std::pair<int, double>> child_list;
                    child_list.reserve(child_list_size);
                    
                    for (int k = 0; k < child_list_size; ++k) {
                        getline_trim(ifs, line);
                        std::stringstream ss(line);
                        int first;
                        double second;
                        ss >> first >> second;
                        child_list.emplace_back(first, second);
                    }
                    anchor->children[phase] = std::move(child_list);
                }
            } else if (line.find("NodesInClusterCount:") == 0) {
                int cluster_count = std::stoi(line.substr(line.find(":") + 2));
                
                for (int j = 0; j < cluster_count; ++j) {
                    getline_trim(ifs, line);
                    std::stringstream ss(line);
                    double first;
                    int second;
                    ss >> first >> second;
                    anchor->nodes_in_cluster.emplace(first, second);
                }
            } else if (line.find("NodesInExactClusterCount:") == 0) {
                int exact_cluster_count = std::stoi(line.substr(line.find(":") + 2));
                
                for (int j = 0; j < exact_cluster_count; ++j) {
                    Anchor::ExactClusterNode cluster_node;
                    
                    getline_trim(ifs, line);
                    if (line.find("Dist:") == 0) {
                        cluster_node.dist = std::stod(line.substr(line.find(":") + 2));
                    }
                    
                    getline_trim(ifs, line);
                    if (line.find("NodeID:") == 0) {
                        cluster_node.node_id = std::stoi(line.substr(line.find(":") + 2));
                    }
                    
                    getline_trim(ifs, line);
                    size_t mapping_size = 0;
                    if (line.find("MappingSize:") == 0) {
                        mapping_size = std::stoul(line.substr(line.find(":") + 2));
                    }
                    
                    cluster_node.mapping.reserve(mapping_size);
                    
                    for (size_t k = 0; k < mapping_size; ++k) {
                        getline_trim(ifs, line);
                        std::stringstream ss(line);
                        ui first, second;
                        ss >> first >> second;
                        cluster_node.mapping.emplace_back(first, second);
                    }
                    
                    anchor->nodes_in_exact_cluster.emplace(std::move(cluster_node));
                }
            } else if (line.find("RA:") == 0) {
                anchor->r_a = std::stod(line.substr(line.find(":") + 2));
            }
        }
        
        netdag.anchors.push_back(anchor);
        netdag.nodes[anchor->node_id] = anchor;
        
        if (has_graph_data) {
            nodes_to_initialize.push_back(anchor);
        }
    }

    printf("[NetDag::load_from_file] Finished loading %d anchors\n", num_anchors);
    fflush(stdout);

    // 读取剩余的数据（parent_by_phase_dict等）
    // 检查文件是否已到达末尾
    if (!ifs.eof() && ifs.good()) {
        getline_trim(ifs, line); // Empty line

        // 检查是否成功读取且不是EOF
        if (!ifs.eof() && ifs.good() && getline_trim(ifs, line)) {
            // 检查是否是ParentByPhaseDict部分
            if (line.find("# ParentByPhaseDict") != std::string::npos) {
                printf("[NetDag::load_from_file] Loading parent_by_phase_dict...\n");
                fflush(stdout);

                getline_trim(ifs, line); // DictSize: X

                if (line.find("DictSize:") != std::string::npos) {
                    int dict_size = 0;
                    try {
                        dict_size = std::stoi(line.substr(line.find(":") + 2));
                        printf("[NetDag::load_from_file] Dict size: %d\n", dict_size);
                        fflush(stdout);

                        for (int i = 0; i < dict_size; ++i) {
                            getline_trim(ifs, line);
                            int phase;
                            int node_count;
                            size_t pos_phase = line.find("Phase: ");
                            size_t pos_node_count = line.find("NodeCount: ");
                            if (pos_phase != std::string::npos && pos_node_count != std::string::npos) {
                                phase = std::stoi(line.substr(pos_phase + 7, pos_node_count - (pos_phase + 7)));
                                node_count = std::stoi(line.substr(pos_node_count + 11));
                            }

                            getline_trim(ifs, line);
                            std::stringstream ss_nodes(line);
                            std::vector<int> node_ids;
                            node_ids.reserve(node_count);

                            for (int j = 0; j < node_count; ++j) {
                                int node_id;
                                ss_nodes >> node_id;
                                node_ids.push_back(node_id);
                            }
                            netdag.parent_by_phase_dict[phase] = std::move(node_ids);
                        }
                    } catch (const std::exception& e) {
                        printf("[WARNING] Failed to parse parent_by_phase_dict: %s\n", e.what());
                        fflush(stdout);
                    }
                }

                // 读取RootNodeID
                while (getline_trim(ifs, line) && line.empty());

                if (line.find("RootNodeID:") == 0) {
                    int root_node_id = std::stoi(line.substr(line.find(":") + 1));
                    if (root_node_id >= 0 && static_cast<size_t>(root_node_id) < netdag.nodes.size()) {
                        netdag.root = netdag.nodes[root_node_id];
                    }
                }
            } else {
                printf("[WARNING] ParentByPhaseDict section not found, file may be truncated\n");
                fflush(stdout);
            }
        } else {
            printf("[WARNING] File ends after anchors, parent_by_phase_dict section missing\n");
            fflush(stdout);
        }
    } else {
        printf("[WARNING] File ends after anchors (EOF reached), parent_by_phase_dict section missing\n");
        fflush(stdout);
    }

    ifs.close();

    printf("[NetDag::load_from_file] File reading completed. Nodes to initialize: %zu\n", nodes_to_initialize.size());
    fflush(stdout);

    // Phase 2: 并行初始化所有Graph
    const int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()),
                                    static_cast<int>(nodes_to_initialize.size()));

    printf("[NetDag::load_from_file] Starting graph initialization (skipped, commented out)\n");
    fflush(stdout);
    
    if (num_threads > 1 && nodes_to_initialize.size() > 100) {
        // 使用OpenMP并行处理
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 10)
        for (size_t i = 0; i < nodes_to_initialize.size(); i++) {
            auto& node = nodes_to_initialize[i];
            // if (node && node->graph) {
                // node->graph->initialize_vectors_from_arrays();
                // node->ml_graph = node->graph->to_ML_graph();
            // }
        }
    } else {
        // 节点太少，使用串行处理
        for (size_t i = 0; i < nodes_to_initialize.size(); i++) {
            auto& node = nodes_to_initialize[i];
            if (node && node->graph) {
                // node->graph->initialize_vectors_from_arrays();
                // node->ml_graph = node->graph->to_ML_graph();
            }
        }
    }

    printf("[NetDag::load_from_file] NetDag loading completed successfully!\n");
    fflush(stdout);
}


// 极简加载：只读取 anchor 的 ID + nodes_in_exact_cluster
// 跳过整个 Nodes 段、anchor 的 graph 数据、children、parent_by_phase_dict
// 用于 construct_EPF 模式，避免加载完整大 NetDag（如 Chemical1M 47GB）
void NetDag::load_anchors_only(NetDag& netdag, const std::string& filename) {
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return;
    }

    // 设置更大的缓冲区
    const size_t buffer_size = 1024 * 1024; // 1MB buffer
    std::vector<char> buffer(buffer_size);
    ifs.rdbuf()->pubsetbuf(buffer.data(), buffer_size);

    auto getline_trim = [&](std::ifstream& s, std::string& l) -> std::ifstream& {
        std::getline(s, l);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        return s;
    };

    std::string line;
    line.reserve(1024);

    // Read header
    getline_trim(ifs, line); // NetDag File Signature
    netdag.file_signature = line.substr(line.find(":") + 2);

    getline_trim(ifs, line); // alpha
    netdag.alpha = std::stod(line.substr(line.find(":") + 2));

    getline_trim(ifs, line); // Tau
    netdag.tau = std::stod(line.substr(line.find(":") + 2));

    getline_trim(ifs, line); // Number of Nodes
    int actual_node_count = std::stoi(line.substr(line.find(":") + 2));

    getline_trim(ifs, line); // Number of Anchors
    int num_anchors = std::stoi(line.substr(line.find(":") + 2));

    printf("[load_anchors_only] Header: %d nodes, %d anchors, alpha=%.1f, tau=%.1f\n",
           actual_node_count, num_anchors, netdag.alpha, netdag.tau);
    printf("[load_anchors_only] Skipping entire Nodes section...\n");
    fflush(stdout);

    // 快速跳过整个 Nodes 段：只扫描 EndNode 标记
    int skipped_nodes = 0;
    while (getline_trim(ifs, line)) {
        if (line == "# Anchors") break;  // 找到 Anchors 段开始
        if (line == "EndNode") {
            skipped_nodes++;
            if (skipped_nodes % 100000 == 0) {
                printf("[load_anchors_only] Skipped %d/%d nodes (%.1f%%)\n",
                       skipped_nodes, actual_node_count, 100.0 * skipped_nodes / actual_node_count);
                fflush(stdout);
            }
        }
    }
    printf("[load_anchors_only] Skipped %d nodes. Now loading anchors...\n", skipped_nodes);
    fflush(stdout);

    // 预分配 anchors
    netdag.anchors.reserve(num_anchors);

    // 读取 anchor 信息：只保留 AnchorID, NodeID, NodesInExactCluster
    for (int i = 0; i < num_anchors; i++) {
        if (i % 1000 == 0) {
            printf("[load_anchors_only] Anchor progress: %d/%d (%.1f%%)\n",
                   i, num_anchors, 100.0 * i / num_anchors);
            fflush(stdout);
        }

        auto anchor = std::make_shared<Anchor>();
        int lines_in_anchor = 0;

        while (getline_trim(ifs, line) && line != "EndAnchor") {
            lines_in_anchor++;
            if (line.find("AnchorID:") == 0) {
                anchor->anchor_id = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("NodeID:") == 0) {
                anchor->node_id = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("FileName:") == 0) {
                anchor->file_name = line.substr(line.find(":") + 2);
            } else if (line.find("IsAnchor:") == 0) {
                anchor->is_anchor = true;
            } else if (line.find("NearestAnchorID:") == 0) {
                anchor->nearest_anchor = std::stoi(line.substr(line.find(":") + 2));
            } else if (line.find("NearestAnchorDist:") == 0) {
                anchor->nearest_anchor_dist = std::stod(line.substr(line.find(":") + 2));
            } else if (line.find("HasGraph:") == 0) {
                // 跳过 graph 数据（从 db 加载）
                std::string has_graph_str = line.substr(line.find(":") + 2);
                if (has_graph_str == "true") {
                    // Graph::save_to_stream 格式：GraphDataStart ... GraphDataEnd
                    // 直接扫描到 GraphDataEnd 标记
                    std::string gline;
                    while (getline_trim(ifs, gline)) {
                        if (gline == "GraphDataEnd") break;
                    }
                }
            } else if (line.find("ChildrenCount:") == 0) {
                // 跳过 children 数据
                int children_count = std::stoi(line.substr(line.find(":") + 2));
                for (int j = 0; j < children_count; ++j) {
                    getline_trim(ifs, line); // Phase: X ChildListSize: Y
                    int child_list_size = 0;
                    size_t pos = line.find("ChildListSize: ");
                    if (pos != std::string::npos) {
                        child_list_size = std::stoi(line.substr(pos + 14));
                    }
                    for (int k = 0; k < child_list_size; ++k) {
                        getline_trim(ifs, line); // child_node_id child_dist
                    }
                }
            } else if (line.find("NodesInClusterCount:") == 0) {
                // 读取 nodes_in_cluster
                int cluster_count = std::stoi(line.substr(line.find(":") + 2));
                for (int j = 0; j < cluster_count; ++j) {
                    getline_trim(ifs, line);
                    std::stringstream ss(line);
                    double first;
                    int second;
                    ss >> first >> second;
                    anchor->nodes_in_cluster.emplace(first, second);
                }
            } else if (line.find("NodesInExactClusterCount:") == 0) {
                // 读取 nodes_in_exact_cluster（核心数据）
                int exact_cluster_count = std::stoi(line.substr(line.find(":") + 2));
                for (int j = 0; j < exact_cluster_count; ++j) {
                    Anchor::ExactClusterNode cluster_node;

                    getline_trim(ifs, line);
                    if (line.find("Dist:") == 0) {
                        cluster_node.dist = std::stod(line.substr(line.find(":") + 2));
                    }

                    getline_trim(ifs, line);
                    if (line.find("NodeID:") == 0) {
                        cluster_node.node_id = std::stoi(line.substr(line.find(":") + 2));
                    }

                    getline_trim(ifs, line);
                    size_t mapping_size = 0;
                    if (line.find("MappingSize:") == 0) {
                        mapping_size = std::stoul(line.substr(line.find(":") + 2));
                    }

                    cluster_node.mapping.reserve(mapping_size);
                    for (size_t k = 0; k < mapping_size; ++k) {
                        getline_trim(ifs, line);
                        std::stringstream ss(line);
                        ui first, second;
                        ss >> first >> second;
                        cluster_node.mapping.emplace_back(first, second);
                    }

                    anchor->nodes_in_exact_cluster.emplace(std::move(cluster_node));
                }
            } else if (line.find("RA:") == 0) {
                anchor->r_a = std::stod(line.substr(line.find(":") + 2));
            }
        }

        // 诊断：在出错位置附近打印详情
        if (i >= 41325 && i <= 41340) {
            printf("[DIAG] anchor[%d]: anchor_id=%d, node_id=%d, lines=%d, exact_cluster=%zu\n",
                   i, anchor->anchor_id, anchor->node_id, lines_in_anchor,
                   anchor->nodes_in_exact_cluster.size());
            fflush(stdout);
        }
        if (anchor->anchor_id == -1 && i > 0 && i <= 41340) {
            printf("[DIAG] FIRST BAD anchor at index %d, lines_in_anchor=%d\n", i, lines_in_anchor);
            fflush(stdout);
        }

        netdag.anchors.push_back(anchor);
    }

    printf("[load_anchors_only] Finished loading %d anchors (graphs not loaded, use db instead)\n", num_anchors);
    fflush(stdout);

    ifs.close();
}


void NetDag::print_netdag_info() {
    int num_nodes = nodes.size();
    std::cout << "Total number of nodes: " << num_nodes << std::endl;
    std::cout << "Number of nodes at each level:" << std::endl;

    // Reverse iterate over each level and count number of anchors and average number of children
    for (auto it = parent_by_phase_dict.rbegin(); it != parent_by_phase_dict.rend(); ++it) {
        int level = it->first;
        const auto& nodes_at_level = it->second;

        std::vector<int> num_children_list;
        int anchor_count = 0;

        for (int node_id : nodes_at_level) {
            auto node = nodes[node_id];

            // Cast Node to Anchor
            auto anchor = std::dynamic_pointer_cast<Anchor>(node);
            if (!anchor) {
                std::cerr << "Error: Node with ID " << node_id << " is not an Anchor." << std::endl;
                continue;
            }

            anchor_count++;

            auto& children_at_level = anchor->children[level / 2];

            std::set<int> unique_children;
            for (const auto& child : children_at_level) {
                unique_children.insert(child.first);
            }

            num_children_list.push_back(unique_children.size());
        }

        double avg_num_children = 0.0;
        if (!num_children_list.empty()) {
            avg_num_children = std::accumulate(num_children_list.begin(), num_children_list.end(), 0.0) / num_children_list.size();
        }

        std::cout << "Level " << level << ": " << anchor_count << " anchors" << std::endl;
        std::cout << "Average number of children per node at Level " << level << ": " << avg_num_children << std::endl;
    }

    // Statistics for all anchors' nodes_in_cluster and nodes_in_exact_cluster
    std::vector<int> nodes_in_cluster_counts;
    std::vector<int> nodes_in_exact_cluster_counts;

    for (const auto& anchor : anchors) {
        // Count nodes_in_cluster
        nodes_in_cluster_counts.push_back(anchor->nodes_in_cluster.size());

        // Count nodes_in_exact_cluster
        nodes_in_exact_cluster_counts.push_back(anchor->nodes_in_exact_cluster.size());
    }

    // Compute statistics for nodes_in_cluster
    double mean_cluster_count = 0.0;
    int max_cluster_count = 0;
    int min_cluster_count = std::numeric_limits<int>::max();

    if (!nodes_in_cluster_counts.empty()) {
        mean_cluster_count = std::accumulate(nodes_in_cluster_counts.begin(), nodes_in_cluster_counts.end(), 0.0) / nodes_in_cluster_counts.size();
        max_cluster_count = *std::max_element(nodes_in_cluster_counts.begin(), nodes_in_cluster_counts.end());
        min_cluster_count = *std::min_element(nodes_in_cluster_counts.begin(), nodes_in_cluster_counts.end());
    }

    // Compute statistics for nodes_in_exact_cluster
    double mean_exact_cluster_count = 0.0;
    int max_exact_cluster_count = 0;
    int min_exact_cluster_count = std::numeric_limits<int>::max();

    if (!nodes_in_exact_cluster_counts.empty()) {
        mean_exact_cluster_count = std::accumulate(nodes_in_exact_cluster_counts.begin(), nodes_in_exact_cluster_counts.end(), 0.0) / nodes_in_exact_cluster_counts.size();
        max_exact_cluster_count = *std::max_element(nodes_in_exact_cluster_counts.begin(), nodes_in_exact_cluster_counts.end());
        min_exact_cluster_count = *std::min_element(nodes_in_exact_cluster_counts.begin(), nodes_in_exact_cluster_counts.end());
    }

    // Output statistics for nodes_in_cluster
    std::cout << "Statistics for nodes_in_cluster:" << std::endl;
    std::cout << "Average number of nodes in cluster: " << mean_cluster_count << std::endl;
    std::cout << "Maximum number of nodes in cluster: " << max_cluster_count << std::endl;
    std::cout << "Minimum number of nodes in cluster: " << min_cluster_count << std::endl;

    // Output statistics for nodes_in_exact_cluster
    std::cout << "Statistics for nodes_in_exact_cluster:" << std::endl;
    std::cout << "Average number of nodes in exact cluster: " << mean_exact_cluster_count << std::endl;
    std::cout << "Maximum number of nodes in exact cluster: " << max_exact_cluster_count << std::endl;
    std::cout << "Minimum number of nodes in exact cluster: " << min_exact_cluster_count << std::endl;

    // Output total number of anchors
    std::cout << "Total number of anchors: " << anchors.size() << std::endl;
}
