// EditPathTree.cpp

#include "EditPathTree.h"
#include "Graph.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <stack>
#include <filesystem>
#include "GismaSearchEngine.h"
#include <filesystem>
#include <future>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <atomic>
#include <condition_variable>
namespace fs = std::filesystem;

// TreeNode.cpp

// TreeNode 构造函数
TreeNode::TreeNode(const EditOperation& operation,
                   int level,
                   int ept_node_id,
                   ui anchor_id,
                   int tree_node_graph_id,
                   size_t parent_idx,
                   const PseudoGraph& pg,
                   const std::shared_ptr<Graph>& gp,
                   const std::vector<float>& emb)
    : op(operation),
      level(level),
      simplified_level(0),
      ept_node_id(ept_node_id),
      anchor_id(anchor_id),
      tree_node_graph_id(tree_node_graph_id),
      parent_index(parent_idx),
      pseudo_graph(pg),
      graph_ptr(gp),
      embedding(emb)
      // ml_graph() // Removed: using embedding vectors instead
{
    if (operation.type != static_cast<EditOperation::OperationType>(-1)) {
        accumulated_ops = std::vector<EditOperation>{operation};
    } else {
        accumulated_ops.clear();
    }

    // 如果需要在构造时由graph_ptr直接生成ml_graph（当graph_ptr存在且有效时）：
    // if (graph_ptr) {
    //     graph_ptr->initialize_vectors_from_arrays();
    //     ml_graph = graph_ptr->to_ML_graph();
    // }
    // 若不需要立刻转换为ML_graph，可在加载EPT时或其他合适的时机调用
}

// 析构函数实现
TreeNode::~TreeNode() {
    // 如果使用指针存储子节点，需要手动释放内存
    // 但如果使用索引或其他方式，不需要手动删除
    // for (auto child : children) {
    //     delete child;
    // }
}

// TreeNode::collect_graph_ids 函数
void TreeNode::collect_graph_ids(const std::vector<TreeNode>& nodes, std::vector<int>& ids, int depth_limit, int current_depth) const {
    // 如果当前深度超过限制，则不继续递归
    if (current_depth > depth_limit) {
        return;
    }

    // 将当前节点已完成的全部数据库图 ID 加入到 ids 中
    for (int cid : completed_db_graph_ids) {
        ids.push_back(cid);
    }

    // 递归遍历子节点
    for (size_t child_idx : children_indices) {
        const TreeNode& child = nodes[child_idx];
        // 计算新的深度：当前深度 + 子节点的 accumulated_ops.size()
        int new_depth = current_depth + static_cast<int>(child.accumulated_ops.size());
        child.collect_graph_ids(nodes, ids, depth_limit, new_depth);
    }
}


// 构造函数的实现
EditPathTree::EditPathTree(ui anchor_id)
    : root_index(0), anchor_id(anchor_id) {
    tree_nodes.clear();
}

void TreeNode::save_to_file(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Error: Unable to open file for writing TreeNode: " << filename << std::endl;
        return;
    }

    try {
        // 写入文件开始标记
        ofs << "BEGIN_TREE_NODE\n";

        // 写入节点开始标记
        ofs << "NodeStart\n";

        // 保存节点到流
        save_to_stream(ofs);

        // 写入节点结束标记
        ofs << "NodeEnd\n";

        // 写入文件结束标记
        ofs << "END_TREE_NODE\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: Exception occurred while writing TreeNode to file: " << e.what() << std::endl;
    }

    ofs.close();
}

// 保存到流的函数
void TreeNode::save_to_stream(std::ostream& os) const {
    // 写入节点基本信息
    os << "EPTNodeID: " << ept_node_id << "\n";
    os << "AnchorID: " << anchor_id << "\n";
    os << "TreeNodeGraphID: " << tree_node_graph_id << "\n";
    os << "Level: " << level << "\n";
    os << "SimplifiedLevel: " << simplified_level << "\n";

    // 写入编辑操作信息
    os << "EditOperation: "
       << static_cast<int>(op.type) << " "
       << op.u << " "
       << op.v << " "
       << op.old_label << " "
       << op.new_label << "\n";

    // 写入累积的编辑操作
    os << "AccumulatedOpsCount: " << accumulated_ops.size() << "\n";
    for (const auto& edit_op : accumulated_ops) {
        os << "AccumulatedOp: "
           << static_cast<int>(edit_op.type) << " "
           << edit_op.u << " "
           << edit_op.v << " "
           << edit_op.old_label << " "
           << edit_op.new_label << "\n";
    }

    // 写入子节点信息
    os << "ChildrenCount: " << children_indices.size() << "\n";
    os << "ChildIndices:";
    for (size_t child_idx : children_indices) {
        os << " " << child_idx;
    }
    os << "\n";

    // 写入已完成的 db_graph_ids
    os << "CompletedDBGraphIDsCount: " << completed_db_graph_ids.size() << "\n";
    for (int cid : completed_db_graph_ids) {
        os << "CompletedDBGraphID: " << cid << "\n";
    }

    // 写入 Graph 数据
    os << "GraphStart\n";
    if (graph_ptr) {
        graph_ptr->save_to_stream(os);
    } else {
        os << "NullGraph\n";
    }
    os << "GraphEnd\n";
}


void TreeNode::load_from_stream(std::istream& is) {
    auto preprocess_line = [](std::string& line) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
        }
    };

    std::string line;

    while (std::getline(is, line)) {
        preprocess_line(line);
        if (line.empty()) continue;
        if (line == "NodeEnd") {
            return;
        }

        if (line == "NodeStart") {
            std::cerr << "Error: Unexpected NodeStart inside a node.\n";
            return;
        }

        if (line == "GraphEnd") {
            continue;
        }

        try {
            if (line.find("EPTNodeID:") == 0) {
                ept_node_id = std::stoi(line.substr(std::string("EPTNodeID:").length()));
            } else if (line.find("AnchorID:") == 0) {
                anchor_id = std::stoul(line.substr(std::string("AnchorID:").length()));
            } else if (line.find("TreeNodeGraphID:") == 0) {
                tree_node_graph_id = std::stoi(line.substr(std::string("TreeNodeGraphID:").length()));
            } else if (line.find("Level:") == 0) {
                level = std::stoi(line.substr(std::string("Level:").length()));
            } else if (line.find("SimplifiedLevel:") == 0) {
                simplified_level = std::stoi(line.substr(std::string("SimplifiedLevel:").length()));
            } else if (line.find("EditOperation:") == 0) {
                std::istringstream iss(line.substr(std::string("EditOperation:").length()));
                int type_int;
                iss >> type_int >> op.u >> op.v >> op.old_label >> op.new_label;
                op.type = static_cast<EditOperation::OperationType>(type_int);
            } else if (line.find("AccumulatedOpsCount:") == 0) {
                int ops_count = std::stoi(line.substr(std::string("AccumulatedOpsCount:").length()));
                accumulated_ops.clear();
                for (int i = 0; i < ops_count; i++) {
                    if (std::getline(is, line)) {
                        preprocess_line(line);
                        if (line.find("AccumulatedOp:") == 0) {
                            std::istringstream iss(line.substr(std::string("AccumulatedOp:").length()));
                            int type_int;
                            EditOperation acc_op;
                            iss >> type_int >> acc_op.u >> acc_op.v >> acc_op.old_label >> acc_op.new_label;
                            acc_op.type = static_cast<EditOperation::OperationType>(type_int);
                            accumulated_ops.push_back(acc_op);
                        } else {
                            std::cerr << "Warning: Expected 'AccumulatedOp:' but got: " << line << "\n";
                        }
                    } else {
                        std::cerr << "Error: Unexpected end of file while reading accumulated operations.\n";
                        return;
                    }
                }
            } else if (line.find("ChildrenCount:") == 0) {
                int children_count = std::stoi(line.substr(std::string("ChildrenCount:").length()));
                children_indices.clear();
                if (std::getline(is, line)) {
                    preprocess_line(line);
                    if (line.find("ChildIndices:") == 0) {
                        std::istringstream iss(line.substr(std::string("ChildIndices:").length()));
                        size_t child_idx;
                        while (iss >> child_idx) {
                            children_indices.push_back(child_idx);
                        }
                        if (children_indices.size() != (size_t)children_count) {
                            std::cerr << "Warning: ChildrenCount (" << children_count << ") does not match number of ChildIndices (" << children_indices.size() << ")\n";
                        }
                    } else {
                        if (children_count == 0) {
                            for (auto it = line.rbegin(); it != line.rend(); ++it) {
                                is.putback(*it);
                            }
                            is.putback('\n');
                        } else {
                            std::cerr << "Error: Expected 'ChildIndices:' but got: " << line << "\n";
                            return;
                        }
                    }
                } else {
                    if (children_count != 0) {
                        std::cerr << "Error: Unexpected end of file while reading child indices.\n";
                        return;
                    }
                }
            } else if (line.find("CompletedDBGraphIDsCount:") == 0) {
                int ccount = std::stoi(line.substr(std::string("CompletedDBGraphIDsCount:").length()));
                completed_db_graph_ids.clear();
                for (int i = 0; i < ccount; i++) {
                    if (std::getline(is, line)) {
                        preprocess_line(line);
                        if (line.find("CompletedDBGraphID:") == 0) {
                            int cid_val = std::stoi(line.substr(std::string("CompletedDBGraphID:").length()));
                            completed_db_graph_ids.push_back(cid_val);
                        } else {
                            std::cerr << "Warning: Expected 'CompletedDBGraphID:' but got: " << line << "\n";
                        }
                    } else {
                        std::cerr << "Error: Unexpected end of file while reading CompletedDBGraphIDs.\n";
                        return;
                    }
                }
            } else if (line == "GraphStart") {
                if (std::getline(is, line)) {
                    preprocess_line(line);
                    if (line == "NullGraph") {
                        graph_ptr = nullptr;
                    } else {
                        graph_ptr = std::make_shared<Graph>();
                        std::ostringstream graph_data_stream;
                        graph_data_stream << line << "\n";
                        while (std::getline(is, line)) {
                            preprocess_line(line);
                            if (line == "GraphEnd") {
                                break;
                            }
                            graph_data_stream << line << "\n";
                        }
                        std::istringstream graph_stream(graph_data_stream.str());
                        graph_ptr->load_from_stream(graph_stream);

                        // 在成功加载graph_ptr后，初始化并转换为ML_graph
                        if (graph_ptr) {
                            graph_ptr->initialize_vectors_from_arrays();
                            // ml_graph = graph_ptr->to_ML_graph(); // Removed: using embedding vectors instead
                            db_graph_n = graph_ptr->n;
                            db_graph_m = graph_ptr->m;
                            db_vlabels = graph_ptr->vlabels;
                            db_graph = graph_ptr.get();
                        }
                    }
                } else {
                    std::cerr << "Error: Unexpected end of file after 'GraphStart'.\n";
                    return;
                }
            } else {
                std::cerr << "Warning: Unrecognized line: " << line << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: '" << line << "'. Exception: " << e.what() << "\n";
            return;
        }
    }

    std::cerr << "Error: Unexpected end of file while parsing TreeNode.\n";
}

// 从文件加载单个 TreeNode 的静态函数
TreeNode* TreeNode::load_from_file(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: Unable to open TreeNode file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    std::getline(ifs, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "BEGIN_TREE_NODE") {
        std::cerr << "Error: TreeNode file does not start with BEGIN_TREE_NODE." << std::endl;
        return nullptr;
    }

    // 检查 NodeStart 标记
    std::getline(ifs, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "NodeStart") {
        std::cerr << "Error: Expected NodeStart after BEGIN_TREE_NODE." << std::endl;
        return nullptr;
    }

    // 创建节点并加载
    TreeNode* node = new TreeNode();
    node->load_from_stream(ifs);

    // 检查 END_TREE_NODE 标记
    if (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line != "END_TREE_NODE") {
            std::cerr << "Error: Expected END_TREE_NODE at the end of TreeNode file." << std::endl;
            delete node;
            return nullptr;
        }
    } else {
        std::cerr << "Error: Unexpected end of file when reading END_TREE_NODE." << std::endl;
        delete node;
        return nullptr;
    }

    return node;
}




void EditPathTree::save_to_file(const std::string& filename) const {
    if (tree_nodes.empty()) {
        std::cerr << "Error: Tree is empty. Cannot save to file." << std::endl;
        return;
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Unable to open file for writing EPT: " << filename << std::endl;
        return;
    }

    // Write file header
    ofs << "BEGIN_EPT\n";

    // Write AnchorID
    ofs << "AnchorID: " << anchor_id << "  # ID of the anchor node\n";

    // Write RootIndex
    ofs << "RootIndex: " << root_index << "  # Index of the root node in tree_nodes\n";

    // Write NodeCount
    ofs << "NodeCount: " << tree_nodes.size() << "  # Total number of nodes in the EPT\n";

    // Save each TreeNode
    for (const auto& node : tree_nodes) {
        ofs << "NodeStart\n";
        node.save_to_stream(ofs);
        ofs << "NodeEnd\n";
    }

    // Write file footer
    ofs << "END_EPT\n";
    ofs.close();
}

// 从文件加载 EPT 的函数
void EditPathTree::load_from_file(const std::string& filename) {
    tree_nodes.clear();

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: Unable to open file for reading EPT: " << filename << std::endl;
        return;
    }

    std::string line;
    // 通用的行预处理函数
    auto preprocess_line = [](std::string& line) {
        // 去除行首尾空白字符
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        // 忽略注释
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
            // 去除因删除注释后产生的多余空白字符
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
        }
    };

    std::getline(ifs, line);
    preprocess_line(line);
    if (line != "BEGIN_EPT") {
        std::cerr << "Error: Invalid EPT file format. Expected 'BEGIN_EPT'." << std::endl;
        return;
    }

    // Read AnchorID
    std::getline(ifs, line);
    preprocess_line(line);
    if (line.find("AnchorID:") == 0) {
        std::string value_str = line.substr(9);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        anchor_id = std::stoul(value_str);
    } else {
        std::cerr << "Error: Missing 'AnchorID' in EPT file." << std::endl;
        return;
    }

    // Read RootIndex
    std::getline(ifs, line);
    preprocess_line(line);
    if (line.find("RootIndex:") == 0) {
        std::string value_str = line.substr(10);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        root_index = std::stoul(value_str);
    } else {
        std::cerr << "Error: Missing 'RootIndex' in EPT file." << std::endl;
        return;
    }

    // Read NodeCount
    std::getline(ifs, line);
    preprocess_line(line);
    size_t node_count = 0;
    if (line.find("NodeCount:") == 0) {
        std::string value_str = line.substr(10);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        node_count = std::stoul(value_str);
    } else {
        std::cerr << "Error: Missing 'NodeCount' in EPT file." << std::endl;
        return;
    }

    // Read TreeNodes
    for (size_t i = 0; i < node_count; ++i) {
        std::getline(ifs, line);
        preprocess_line(line);

        if (line != "NodeStart") {
            std::cerr << "Error: Expected 'NodeStart', but got: " << line << std::endl;
            return;
        }

        TreeNode node;
        node.load_from_stream(ifs);
        tree_nodes.push_back(node);
    }

    std::getline(ifs, line);
    preprocess_line(line);
    if (line != "END_EPT") {
        std::cerr << "Error: Expected 'END_EPT', but got: " << line << std::endl;
        return;
    }

    ifs.close();
}



void EditPathTreeManager::load_all_epts_from_directory_parallel(const std::string& directory_path) {
    namespace fs = std::filesystem;

    // 检查目录是否存在
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Error: Directory does not exist: " << directory_path << std::endl;
        return;
    }

    // 收集所有文件路径
    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            file_paths.push_back(entry.path().string());
        }
    }

    size_t num_files = file_paths.size();
    if (num_files == 0) {
        std::cout << "No files found in directory: " << directory_path << std::endl;
        return;
    } else {
        std::cout << "Found " << num_files << " files in directory: " << directory_path << std::endl;
    }

    // 用于跟踪进度
    std::atomic<size_t> files_processed(0);

    // 设置线程数量，可以根据需要调整
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // 如果无法获取线程数，默认使用4个线程

    // 创建线程池
    std::vector<std::thread> thread_pool;

    // 分割文件列表，分配给不同的线程
    std::vector<std::vector<std::string>> file_chunks(num_threads);
    for (size_t i = 0; i < file_paths.size(); ++i) {
        file_chunks[i % num_threads].push_back(file_paths[i]);
    }

    // 为每个线程创建任务
    for (size_t t = 0; t < num_threads; ++t) {
        thread_pool.emplace_back([&, t]() {
            for (const auto& file_path : file_chunks[t]) {
                // 加载单个 EPT
                auto ept = std::make_unique<EditPathTree>();
                ept->load_from_file(file_path);

                // 检查是否成功加载
                if (ept->tree_nodes.empty()) {
                    std::cerr << "Warning: Failed to load EPT from file: " << file_path << std::endl;
                    continue;
                }

                ui anchor_id = ept->anchor_id;

                ept->precompute_max_subtree_depth();
                {
                    std::unique_lock<std::shared_mutex> lock(map_mutex);
                    ept_map[anchor_id] = std::move(ept);
                }

                // 更新进度
                size_t processed = ++files_processed;

                // 显示进度（可选）
                if (processed % 100 == 0 || processed == num_files) {
                    double progress = static_cast<double>(processed) / num_files;
                    int bar_width = 50;
                    std::cout << "\r[";
                    int pos = static_cast<int>(bar_width * progress);
                    for (int i = 0; i < bar_width; ++i) {
                        if (i < pos) std::cout << "=";
                        else if (i == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << int(progress * 100.0) << "% (" << processed << "/" << num_files << ")";
                    std::cout.flush();
                }
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : thread_pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // 最后确保进度条显示为100%
    std::cout << std::endl;
    std::cout << "Successfully loaded " << ept_map.size() << " EPTs from directory: " << directory_path << std::endl;
}





// EditPathTree::collect_graph_ids 函数
void EditPathTree::collect_graph_ids(std::vector<int>& ids, int depth_limit) const {
    if (!tree_nodes.empty()) {
        tree_nodes[root_index].collect_graph_ids(tree_nodes, ids, depth_limit, 0); // 初始深度为 0
    }
}

void EditPathTree::precompute_max_subtree_depth() {
    if (tree_nodes.empty()) return;
    // Post-order: process from last to first (children always have higher indices)
    for (int i = (int)tree_nodes.size() - 1; i >= 0; --i) {
        TreeNode& node = tree_nodes[i];
        int max_depth = node.level;
        for (size_t ch : node.children_indices) {
            if (ch < tree_nodes.size() && tree_nodes[ch].max_subtree_depth > max_depth) {
                max_depth = tree_nodes[ch].max_subtree_depth;
            }
        }
        node.max_subtree_depth = max_depth;
    }
}

// 计算两个图之间的距离的函数

// 根据锚点 ID 获取 EPT
// 使用 shared_lock 允许多线程并发读取，解决并行查询时的锁竞争问题
EditPathTree* EditPathTreeManager::get_ept(ui anchor_id) {
    std::shared_lock<std::shared_mutex> lock(map_mutex);
    auto it = ept_map.find(anchor_id);
    if (it != ept_map.end()) {
        return it->second.get();
    } else {
        std::cerr << "EPT not found for Anchor ID: " << anchor_id << std::endl;
        return nullptr;
    }
}

// 无锁版本：仅在确保 EPT 加载完成后的只读阶段使用
// 用于查询阶段，完全消除锁开销（包括 shared_lock 的原子操作）
EditPathTree* EditPathTreeManager::get_ept_no_lock(ui anchor_id) const {
    auto it = ept_map.find(anchor_id);
    if (it != ept_map.end()) {
        return it->second.get();
    }
    return nullptr;  // 不打印错误信息，避免并发输出问题
}

// 获取已加载的 EPT 数量
// 使用 shared_lock 允许多线程并发读取
size_t EditPathTreeManager::get_ept_count() const {
    std::shared_lock<std::shared_mutex> lock(map_mutex);
    return ept_map.size();
}

void EditPathTreeManager::clear_all_epts() {
    std::unique_lock<std::shared_mutex> lock(map_mutex);  // 写操作需要独占锁
    ept_map.clear();  // 智能指针会自动清理EPT对象
}

std::vector<int> EditPathTreeManager::collect_leaf_db_ids() const {
    // E11: 遍历所有 EPT，收集叶子节点(无 children、非 root)对应的 db 图 id。
    std::vector<int> ids;
    std::shared_lock<std::shared_mutex> lock(map_mutex);
    for (const auto& kv : ept_map) {
        const EditPathTree* ept = kv.second.get();
        if (!ept) continue;
        for (size_t i = 0; i < ept->tree_nodes.size(); ++i) {
            if (i == ept->root_index) continue;            // 跳过 root(=anchor)
            const TreeNode& node = ept->tree_nodes[i];
            if (!node.children_indices.empty()) continue;   // 仅叶子
            for (int gid : node.completed_db_graph_ids) ids.push_back(gid);
        }
    }
    return ids;
}

std::vector<ui> EditPathTreeManager::all_anchor_ids() const {
    std::vector<ui> ids;
    std::shared_lock<std::shared_mutex> lock(map_mutex);
    ids.reserve(ept_map.size());
    for (const auto& kv : ept_map) ids.push_back(kv.first);
    return ids;
}

// E11: 从本 EPT 所有节点的 completed_db_graph_ids 删除指定 id，返回实际被删的(去重)id 列表。
std::vector<int> EditPathTree::remove_db_graph_ids(const std::unordered_set<int>& ids_to_remove) {
    std::unordered_set<int> removed_set;
    for (auto& node : tree_nodes) {
        auto& v = node.completed_db_graph_ids;
        if (v.empty()) continue;
        std::vector<int> kept;
        kept.reserve(v.size());
        for (int id : v) {
            if (ids_to_remove.count(id)) removed_set.insert(id);
            else kept.push_back(id);
        }
        v.swap(kept);
    }
    return std::vector<int>(removed_set.begin(), removed_set.end());
}

// E11 真增量插入(merge): 把 db 图 g 合并进本 EPT。详见头文件注释。
void EditPathTree::insert_graph_merge(const std::vector<EditOperation>& full_ops, Graph* g_graph, int db_graph_id) {
    if (tree_nodes.empty() || !g_graph) return;
    // 剩余待消化的 op 集合(可乱序)
    std::unordered_set<EditOperation, EditOperationHash, EditOperationEqual> remaining(full_ops.begin(), full_ops.end());

    // 从 root 贪心下走: 每步选"整条压缩边 op 集 ⊆ 剩余"且共享最多的子节点
    size_t cur = root_index;
    while (!remaining.empty()) {
        long long best_child = -1; size_t best_share = 0;
        for (size_t c : tree_nodes[cur].children_indices) {
            const auto& cops = tree_nodes[c].accumulated_ops;
            if (cops.empty() || cops.size() > remaining.size()) continue;
            bool all_in = true;
            for (const auto& op : cops) if (!remaining.count(op)) { all_in = false; break; }
            if (all_in && cops.size() > best_share) { best_share = cops.size(); best_child = (long long)c; }
        }
        if (best_child < 0) break;                       // 没有可继续共享的子节点 → 另起分支
        for (const auto& op : tree_nodes[best_child].accumulated_ops) remaining.erase(op);
        cur = (size_t)best_child;
    }

    if (remaining.empty()) {
        // g 与现有节点 cur 重合 → 直接登记
        tree_nodes[cur].completed_db_graph_ids.push_back(db_graph_id);
        return;
    }

    // 另起一个压缩节点: accumulated_ops = 剩余全部, db_graph 指向真实 g(非拥有别名)
    std::vector<EditOperation> rem_ops(remaining.begin(), remaining.end());
    EditOperation none_op;  // type=NONE(-1)
    size_t new_idx = tree_nodes.size();
    int new_level = tree_nodes[cur].level + (int)rem_ops.size();
    PseudoGraph pg(*g_graph);
    std::shared_ptr<Graph> gp(g_graph, [](Graph*){});
    TreeNode node(none_op, new_level, (int)new_idx, anchor_id, (int)new_idx, cur, pg, gp);
    node.accumulated_ops = rem_ops;
    node.db_graph_n = g_graph->n;
    node.db_graph_m = g_graph->m;
    node.db_vlabels = g_graph->vlabels;
    node.db_graph = g_graph;
    node.completed_db_graph_ids.clear();
    node.completed_db_graph_ids.push_back(db_graph_id);
    node.max_subtree_depth = new_level;
    tree_nodes.push_back(std::move(node));
    tree_nodes[cur].children_indices.push_back(new_idx);
}



void build_edit_path_tree_recursive(
    EditPathTree& ept,
    const std::vector<std::unordered_set<EditOperation, EditOperationHash, EditOperationEqual>>& undone_ops_lists,
    const std::vector<ui>& db_graph_ids,
    ui anchor_id,
    size_t node_idx,
    const std::vector<ui>& curr_indices,
    int current_level,
    int& ept_node_counter)
{
    // [DEBUG] 添加递归调试信息
    // if (current_level == 0) {
    //     std::cout << "[DEBUG EPT] Starting EPT build for anchor " << anchor_id
    //               << " with " << curr_indices.size() << " graphs" << std::endl;
    //
    //     // 检查每个图的初始操作数量
    //     for (size_t i = 0; i < undone_ops_lists.size() && i < 5; i++) {
    //         std::cout << "[DEBUG EPT] Graph " << i << " has " << undone_ops_lists[i].size()
    //                   << " undone operations" << std::endl;
    //     }
    // }
    //
    // std::cout << "[DEBUG EPT] Level " << current_level << ": Processing "
    //           << curr_indices.size() << " graphs" << std::endl;

    // 如果没有剩余的图，直接返回
    if (curr_indices.empty()) {
        // std::cout << "[DEBUG EPT] Level " << current_level << ": No remaining graphs, returning" << std::endl;
        return;
    }

    // 将进行中的图和已完成的图分开
    std::vector<ui> done_indices;
    std::vector<ui> ongoing_indices;
    for (ui idx : curr_indices) {
        if (undone_ops_lists[idx].empty()) {
            // 如果没有未完成的操作，表示图已经完成
            done_indices.push_back(idx);
        } else {
            // 如果图有未完成的操作，将其放入进行中的图中
            ongoing_indices.push_back(idx);
        }
    }

    // std::cout << "[DEBUG EPT] Level " << current_level << ": " << done_indices.size()
    //           << " completed, " << ongoing_indices.size() << " ongoing" << std::endl;

    // 将已完成图的 db_graph_id 加入当前节点的 completed_db_graph_ids
    if (!done_indices.empty()) {
        for (ui idx : done_indices) {
            int db_graph_id = static_cast<int>(db_graph_ids[idx]);
            ept.tree_nodes[node_idx].completed_db_graph_ids.push_back(db_graph_id);
        }
    }

    // [DEBUG] 如果所有图都完成了，不继续展开
    if (ongoing_indices.empty()) {
        // std::cout << "[DEBUG EPT] Level " << current_level
        //           << ": All graphs completed, stopping expansion" << std::endl;
        return;
    }

    // 收集所有进行中的图的操作并统计出现频率
    std::unordered_map<EditOperation, std::vector<ui>, EditOperationHash, EditOperationEqual> op_to_indices;
    // std::cout << "node_idx: " << node_idx << "  ongoing_indices.size(): " << ongoing_indices.size() << std::endl;
    for (ui idx : ongoing_indices) {
        for (const auto& op : undone_ops_lists[idx]) {
            op_to_indices[op].push_back(idx);  // 记录每个操作出现的图的索引
        }
    }
   

    // 根据操作出现的频率排序，优先处理出现次数最多的操作
    std::vector<std::pair<EditOperation, std::vector<ui>>> sorted_ops(op_to_indices.begin(), op_to_indices.end());
    std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto& a, const auto& b) {
        return a.second.size() > b.second.size();  // 按操作出现频率降序排序
    });
    // for (const auto& [op, indices] : sorted_ops) {
    //     if (indices.size() > 1) std::cout << "Sorted Operation: " << op.to_string() << "  Count: " << indices.size() << std::endl;
    // }
    std::unordered_set<ui> processed_indices;  // 记录已经处理过的图
    bool any_operation_applied = false;  // 标记是否有操作被成功应用

    // std::cout << "[DEBUG EPT] Level " << current_level << ": Trying to apply "
    //           << sorted_ops.size() << " different operations" << std::endl;

    // 遍历所有的操作，尝试应用于进行中的图
    for (const auto& op_pair : sorted_ops) {
        const EditOperation& op = op_pair.first;  // 当前操作
        const std::vector<ui>& indices_with_op = op_pair.second;  // 需要应用该操作的图的索引

        std::vector<ui> applicable_indices;  // 存储能应用当前操作的图的索引

        bool can_apply = ept.tree_nodes[node_idx].pseudo_graph.can_apply_operation(op);

        // std::cout << "[DEBUG EPT] Level " << current_level << ": Operation " << op.to_string()
        //           << " can_apply=" << (can_apply ? "YES" : "NO")
        //           << " affects " << indices_with_op.size() << " graphs" << std::endl;

        if (!can_apply) continue;
        for (ui idx : indices_with_op) {
            // 如果当前图未被处理过且能够应用此操作
            if (processed_indices.count(idx) == 0) {
                applicable_indices.push_back(idx);
            }
        }

        // 如果没有图能应用该操作，则跳过当前操作
        if (applicable_indices.empty()) {
            continue;
        }

        // 应用操作，生成子节点
        PseudoGraph child_pseudo_graph = ept.tree_nodes[node_idx].pseudo_graph;
        child_pseudo_graph.apply_edit_operation(op);  // 在当前图的基础上应用操作

        // 创建子节点
        // 增加节点计数器
        ept_node_counter++;
        size_t child_idx = ept.tree_nodes.size();
        TreeNode child_node(op, current_level + 1,
                            child_idx, anchor_id,
                            child_idx,  // tree_node_graph_id 使用 ept_node_counter
                            node_idx,          // 当前节点的索引作为父节点索引
                            child_pseudo_graph);

        // 记录已应用的操作
        child_node.accumulated_ops.clear();
        child_node.accumulated_ops.push_back(op);

        // 将 pseudo_graph 转换为 Graph，并存储到子节点的 graph_ptr
        child_node.graph_ptr = std::make_shared<Graph>(child_node.pseudo_graph.to_graph());

        child_node.graph_ptr->id = std::to_string(child_node.tree_node_graph_id);

        

        // 将子节点添加到树中
        ept.tree_nodes.push_back(child_node);
        
        // size_t child_idx = ept_node_counter;
        
        // 将子节点的索引加入父节点的 children_indices
        ept.tree_nodes[node_idx].children_indices.push_back(child_idx);
        auto updated_undone_ops_lists = undone_ops_lists;
        for (ui idx : applicable_indices) {
            updated_undone_ops_lists[idx].erase(op);  // 删除已完成的操作
            processed_indices.insert(idx);  // 标记为已处理
        }

        any_operation_applied = true;  // 标记本次递归中至少应用了一次操作

        // 递归地处理子节点
        build_edit_path_tree_recursive(ept, updated_undone_ops_lists, db_graph_ids, anchor_id,
                                       child_idx, applicable_indices, current_level + 1, ept_node_counter);
    }

    // 如果没有成功应用任何操作，但仍有未处理的图，表示此分支无法完成
    if (!any_operation_applied) {
        std::vector<ui> final_remaining_indices;
        for (ui idx : ongoing_indices) {
            // 找到没有成功处理的图
            if (processed_indices.count(idx) == 0) {
                final_remaining_indices.push_back(idx);
            }
        }

        // 如果仍有图未能完成，报错并打印相关信息
        if (!final_remaining_indices.empty()) {
            std::cerr << "Error: The following db_graph_ids cannot be completed from this branch:\n";
            for (ui idx : final_remaining_indices) {
                int db_graph_id = (int)db_graph_ids[idx];
                std::cerr << db_graph_id << " ";
            }
            std::cerr << "\n";

            // 打印当前 pseudo_graph 状态
            std::cerr << "Current pseudo graph state:\n";
            ept.tree_nodes[node_idx].pseudo_graph.print_pseudo_graph();

            // 打印剩余操作
            std::cerr << "Remaining operations for these DB graphs:\n";
            for (ui idx : final_remaining_indices) {
                int db_graph_id = (int)db_graph_ids[idx];
                std::cerr << "DB Graph ID: " << db_graph_id << ", Undone ops:\n";
                for (const auto& op : undone_ops_lists[idx]) {
                    std::cerr << op.to_string() << "\n";
                }
            }
        }
    }
}



void print_tree(const EditPathTree& ept) {
    std::string tmp;
    tmp += "Tree begin\n";

    for (size_t i = 0; i < ept.tree_nodes.size(); ++i) {
        const TreeNode& node = ept.tree_nodes[i];

        // 打印节点的 ept_node_id
        tmp += std::to_string(i) + " (ept_node_id = " + std::to_string(node.ept_node_id) + ") ";

        // 打印父节点索引
        tmp += "(parent = ";
        if (node.parent_index == SIZE_MAX) {
            tmp += "None"; // 或者 "ROOT"
        } else {
            tmp += std::to_string(node.parent_index);
        }
        tmp += ") : "; 

        // 打印所有子节点索引
        for (size_t child_idx : node.children_indices) {
            tmp += std::to_string(child_idx) + " ; ";
        }

        tmp += "\n";
    }

    tmp += "Tree end\n";
    std::cout << tmp << std::endl;
}

void build_edit_path_tree(
    EditPathTree& ept,
    const std::vector<std::vector<EditOperation>>& edit_operations_list,
    const std::vector<ui>& db_graph_ids,
    ui anchor_id,
    const Graph& anchor_graph)
{
    int ept_node_counter = 0;
    ept.tree_nodes.clear();

    PseudoGraph root_pseudo_graph(anchor_graph);
    std::shared_ptr<Graph> root_graph_ptr = std::make_shared<Graph>(anchor_graph);

    EditOperation dummy_op;
    dummy_op.type = static_cast<EditOperation::OperationType>(-1);

    // 去掉db_graph_id参数
    TreeNode root_node(dummy_op, 0, ept_node_counter, anchor_id, ept_node_counter, SIZE_MAX, root_pseudo_graph);
    root_node.graph_ptr = root_graph_ptr;
    root_node.graph_ptr->id = std::to_string(root_node.tree_node_graph_id);
    ept_node_counter++;

    root_node.accumulated_ops.clear();
    // 重要：将anchor_id添加到root节点的completed_db_graph_ids中
    // 这样当query与anchor的GED<=tau时，anchor本身会被返回为结果
    root_node.completed_db_graph_ids.push_back((int)anchor_id);

    ept.tree_nodes.push_back(root_node);
    ept.root_index = 0;

    size_t num_graphs = edit_operations_list.size();
    std::vector<std::unordered_set<EditOperation, EditOperationHash, EditOperationEqual>> undone_ops_lists(num_graphs);
    for (size_t i = 0; i < num_graphs; ++i) {
        for (const auto& op : edit_operations_list[i]) {
            undone_ops_lists[i].insert(op);
        }
    }

    std::vector<ui> indices(num_graphs);
    for (size_t i = 0; i < num_graphs; ++i) {
        indices[i] = (ui)i;
    }

    build_edit_path_tree_recursive(ept, undone_ops_lists, db_graph_ids, anchor_id,
                                   ept.root_index, indices, 0, ept_node_counter);
    print_tree(ept);
    simplify_EPT(ept);
    print_tree(ept);
    // update_simplified_levels(ept, ept.root_index, 0);
    // 重新编号并更新 ept_node_id
    // print_tree(ept);

    ept.reorder();
    print_tree(ept);
    
   

    // 覆盖率检查
    std::queue<size_t> q;
    q.push(ept.root_index);

    std::vector<int> all_completed_ids_list;
    while (!q.empty()) {
        size_t nid = q.front(); q.pop();
        const TreeNode& node = ept.tree_nodes[nid];

        for (int cid : node.completed_db_graph_ids) {
            all_completed_ids_list.push_back(cid);
        }

        for (auto child_idx : node.children_indices) {
            q.push(child_idx);
        }
    }

    std::unordered_set<int> all_completed_ids(all_completed_ids_list.begin(), all_completed_ids_list.end());

    std::unordered_set<int> input_db_set;
    for (auto id : db_graph_ids) {
        input_db_set.insert((int)id);
    }

    size_t should_contain_count = input_db_set.size();
    size_t actually_contain_count = 0;

    for (auto cid : input_db_set) {
        if (all_completed_ids.find(cid) != all_completed_ids.end()) {
            actually_contain_count++;
        }
    }

    if (actually_contain_count < should_contain_count) {
        std::cout << "Missing these db_graph_ids:\n";
        std::vector<int> missing_ids;
        std::unordered_map<int, size_t> dbid_to_index;
        for (size_t i = 0; i < db_graph_ids.size(); ++i) {
            dbid_to_index[(int)db_graph_ids[i]] = i;
        }

        for (auto cid : input_db_set) {
            if (all_completed_ids.find(cid) == all_completed_ids.end()) {
                missing_ids.push_back(cid);
                std::cout << cid << " ";
            }
        }
        std::cout << std::endl;

        std::cout << "Printing op_list for missing db_graph_ids:\n";
        for (int mid : missing_ids) {
            if (dbid_to_index.find(mid) != dbid_to_index.end()) {
                size_t idx = dbid_to_index[mid];
                const auto& op_list = edit_operations_list[idx];
                std::cout << "DB Graph ID: " << mid << ", op_list size: " << op_list.size() << "\n";
                for (const auto& op : op_list) {
                    std::cout << op.to_string() << "\n";
                }
            } else {
                std::cout << "DB Graph ID " << mid << " not found in dbid_to_index map.\n";
            }
        }
    }
}

void simplifyNode(EditPathTree& ept, size_t node_idx) {
    // 1. 安全检查
    if (node_idx >= ept.tree_nodes.size()) {
        return;
    }
    TreeNode& node = ept.tree_nodes[node_idx];

    // 2. 先让所有子节点进行递归化简
    for (size_t child_idx : node.children_indices) {
        if (child_idx < ept.tree_nodes.size()) {
            simplifyNode(ept, child_idx);
        }
    }

    // 3. 尝试合并：当 "子节点没有完成图 && 子节点也只带1个孩子"
    bool canMerge = true;
    while (canMerge) {
        // 查找符合合并条件的子节点
        for (size_t child_idx : node.children_indices) {
            if (child_idx >= ept.tree_nodes.size()) continue;

            TreeNode& child = ept.tree_nodes[child_idx];
            // 条件：child没有 completed_db_graph_ids 且它也有且仅有 1 个子节点
            if (child.completed_db_graph_ids.empty() && child.children_indices.size() == 1) {
                // 准备合并 child -> grandchild
                size_t grandchild_idx = child.children_indices[0];
                if (grandchild_idx >= ept.tree_nodes.size()) break;

                TreeNode& grandchild = ept.tree_nodes[grandchild_idx];

                // 把 child 的所有 accumulated_ops 放到 grandchild 前面
                grandchild.accumulated_ops.insert(
                    grandchild.accumulated_ops.begin(),
                    child.accumulated_ops.begin(),
                    child.accumulated_ops.end()
                );

                // 更新 grandchild 的 parent 指向当前 node
                grandchild.parent_index = node_idx;

                // 把 node 的唯一子节点改成 grandchild
                auto it = std::find(node.children_indices.begin(), node.children_indices.end(), child_idx);
                if (it != node.children_indices.end()) {
                    *it = grandchild_idx;
                }

                // 将被合并的子节点的 parent_index 设置为 SIZE_MAX（或 INF）表示已合并
                child.parent_index = INF;

                // 把 child 清空（它被合并掉了）
                child.children_indices.clear();

                // 继续看新的唯一子节点是否还可合并
                continue;
            }
        }
        // 如果没有可以继续合并的节点，退出循环
        canMerge = false;
    }
}


void simplify_EPT(EditPathTree& ept) {
    if (ept.tree_nodes.empty()) return;

    // 先进行节点合并操作
    simplifyNode(ept, ept.root_index);

}


void update_simplified_levels(EditPathTree& ept, size_t node_index, int current_level) {
    if (node_index >= ept.tree_nodes.size()) return;

    TreeNode& node = ept.tree_nodes[node_index];
    node.simplified_level = current_level;

    for (size_t child_idx : node.children_indices) {
        update_simplified_levels(ept, child_idx, current_level + 1);
    }
}








/**
 * @brief 计算某节点 WL-hash
 *        这里你可以直接 tn.db_graph->compute_wl_hash() 或者写你自己的
 */
std::string get_wl_hash(const TreeNode &tn)
{
    if (!tn.db_graph) return "NO-DB";
    return tn.db_graph->compute_wl_hash(); 
}


