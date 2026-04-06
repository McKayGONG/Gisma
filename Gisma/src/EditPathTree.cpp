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

// TreeNode constructor
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

    // If ml_graph needs to be generated from graph_ptr at construction time (when graph_ptr exists and is valid):
    // if (graph_ptr) {
    //     graph_ptr->initialize_vectors_from_arrays();
    //     ml_graph = graph_ptr->to_ML_graph();
    // }
    // If immediate ML_graph conversion is not needed, it can be called when loading EPT or at another appropriate time
}

// Destructor implementation
TreeNode::~TreeNode() {
    // If using pointers to store child nodes, memory needs manual deallocation
    // But if using indices or other means, no manual deletion needed
    // for (auto child : children) {
    //     delete child;
    // }
}

// TreeNode::collect_graph_ids function
void TreeNode::collect_graph_ids(const std::vector<TreeNode>& nodes, std::vector<int>& ids, int depth_limit, int current_depth) const {
    // If current depth exceeds limit, do not continue recursion
    if (current_depth > depth_limit) {
        return;
    }

    // Add all completed database graph IDs of current node to ids
    for (int cid : completed_db_graph_ids) {
        ids.push_back(cid);
    }

    // Recursively traverse child nodes
    for (size_t child_idx : children_indices) {
        const TreeNode& child = nodes[child_idx];
        // Compute new depth: current depth + child node's accumulated_ops.size()
        int new_depth = current_depth + static_cast<int>(child.accumulated_ops.size());
        child.collect_graph_ids(nodes, ids, depth_limit, new_depth);
    }
}


// Constructor implementation
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
        // Write file start marker
        ofs << "BEGIN_TREE_NODE\n";

        // Write node start marker
        ofs << "NodeStart\n";

        // Save node to stream
        save_to_stream(ofs);

        // Write node end marker
        ofs << "NodeEnd\n";

        // Write file end marker
        ofs << "END_TREE_NODE\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: Exception occurred while writing TreeNode to file: " << e.what() << std::endl;
    }

    ofs.close();
}

// Save-to-stream function
void TreeNode::save_to_stream(std::ostream& os) const {
    // Write node basic info
    os << "EPTNodeID: " << ept_node_id << "\n";
    os << "AnchorID: " << anchor_id << "\n";
    os << "TreeNodeGraphID: " << tree_node_graph_id << "\n";
    os << "Level: " << level << "\n";
    os << "SimplifiedLevel: " << simplified_level << "\n";

    // Write edit operation info
    os << "EditOperation: "
       << static_cast<int>(op.type) << " "
       << op.u << " "
       << op.v << " "
       << op.old_label << " "
       << op.new_label << "\n";

    // Write accumulated edit operations
    os << "AccumulatedOpsCount: " << accumulated_ops.size() << "\n";
    for (const auto& edit_op : accumulated_ops) {
        os << "AccumulatedOp: "
           << static_cast<int>(edit_op.type) << " "
           << edit_op.u << " "
           << edit_op.v << " "
           << edit_op.old_label << " "
           << edit_op.new_label << "\n";
    }

    // Write child node info
    os << "ChildrenCount: " << children_indices.size() << "\n";
    os << "ChildIndices:";
    for (size_t child_idx : children_indices) {
        os << " " << child_idx;
    }
    os << "\n";

    // Write completed db_graph_ids
    os << "CompletedDBGraphIDsCount: " << completed_db_graph_ids.size() << "\n";
    for (int cid : completed_db_graph_ids) {
        os << "CompletedDBGraphID: " << cid << "\n";
    }

    // Write Graph data
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

                        // After successfully loading graph_ptr, initialize and convert to ML_graph
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

// Static function to load a single TreeNode from file
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

    // Check NodeStart marker
    std::getline(ifs, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "NodeStart") {
        std::cerr << "Error: Expected NodeStart after BEGIN_TREE_NODE." << std::endl;
        return nullptr;
    }

    // Create node and load
    TreeNode* node = new TreeNode();
    node->load_from_stream(ifs);

    // Check END_TREE_NODE marker
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

// Function to load EPT from file
void EditPathTree::load_from_file(const std::string& filename) {
    tree_nodes.clear();

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: Unable to open file for reading EPT: " << filename << std::endl;
        return;
    }

    std::string line;
    // General line preprocessing function
    auto preprocess_line = [](std::string& line) {
        // Strip leading and trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        // Ignore comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
            // Strip extra whitespace after comment removal
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

    // Check if directory exists
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Error: Directory does not exist: " << directory_path << std::endl;
        return;
    }

    // Collect all file paths
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

    // For tracking progress
    std::atomic<size_t> files_processed(0);

    // Set thread count, adjustable as needed
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // If thread count cannot be obtained, default to 4 threads

    // Create thread pool
    std::vector<std::thread> thread_pool;

    // Split file list and assign to different threads
    std::vector<std::vector<std::string>> file_chunks(num_threads);
    for (size_t i = 0; i < file_paths.size(); ++i) {
        file_chunks[i % num_threads].push_back(file_paths[i]);
    }

    // Create task for each thread
    for (size_t t = 0; t < num_threads; ++t) {
        thread_pool.emplace_back([&, t]() {
            for (const auto& file_path : file_chunks[t]) {
                // Load single EPT
                auto ept = std::make_unique<EditPathTree>();
                ept->load_from_file(file_path);

                // Check if loading succeeded
                if (ept->tree_nodes.empty()) {
                    std::cerr << "Warning: Failed to load EPT from file: " << file_path << std::endl;
                    continue;
                }

                ui anchor_id = ept->anchor_id;

                {
                    // Lock to protect write access to ept_map (using exclusive lock)
                    std::unique_lock<std::shared_mutex> lock(map_mutex);
                    ept_map[anchor_id] = std::move(ept);
                }

                // Update progress
                size_t processed = ++files_processed;

                // Display progress (optional)
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

    // Wait for all threads to complete
    for (auto& thread : thread_pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Ensure progress bar shows 100% at the end
    std::cout << std::endl;
    std::cout << "Successfully loaded " << ept_map.size() << " EPTs from directory: " << directory_path << std::endl;
}


void EditPathTreeManager::load_all_epts_from_directory(const std::string& directory_path) {
    namespace fs = std::filesystem;

    // Check if directory exists
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Error: Directory does not exist: " << directory_path << std::endl;
        return;
    }

    // Collect all file paths
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

    // Iterate all files, load one by one
    size_t files_processed = 0;
    for (const auto& file_path : file_paths) {
        // Use EditPathTree's load_from_file to load single EPT
        auto ept = std::make_unique<EditPathTree>();
        ept->load_from_file(file_path);

        // Check if loading succeeded
        if (ept->tree_nodes.empty()) {
            std::cerr << "Warning: Failed to load EPT from file: " << file_path << std::endl;
            continue;
        }

        ui anchor_id = ept->anchor_id;

        {
            // Lock to protect write access to ept_map (using exclusive lock)
            std::unique_lock<std::shared_mutex> lock(map_mutex);
            ept_map[anchor_id] = std::move(ept);
        }

        files_processed++;

        // Display progress
        double progress = static_cast<double>(files_processed) / num_files;
        int bar_width = 50;
        std::cout << "\r[";  // Use "\r" to show progress bar on same line
        int pos = static_cast<int>(bar_width * progress);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "% (" << files_processed << "/" << num_files << ")";
        std::cout.flush();  // Flush output buffer to ensure progress bar updates immediately

        // Output file path after processing each file
        std::cout << "Load from file: " << file_path << std::endl;
    }

    // Ensure progress bar shows 100% at the end
    std::cout << std::endl;
    std::cout << "Successfully loaded " << ept_map.size() << " EPTs from directory: " << directory_path << std::endl;
}



// EditPathTree::collect_graph_ids function
void EditPathTree::collect_graph_ids(std::vector<int>& ids, int depth_limit) const {
    if (!tree_nodes.empty()) {
        tree_nodes[root_index].collect_graph_ids(tree_nodes, ids, depth_limit, 0); // initial depth is 0
    }
}

// Function to compute distance between two graphs
double compute_distance(const Graph& g1, const Graph& g2) {
    // Implement actual distance computation logic here
    // Simply return a placeholder value, e.g. absolute difference of vertex counts
    return std::abs(static_cast<int>(g1.n) - static_cast<int>(g2.n));
}

// Get EPT by anchor ID
// Use shared_lock to allow concurrent multi-threaded reads, resolving lock contention during parallel queries
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

// Lock-free version: only use during read-only phase after EPT loading is complete
// For query phase, completely eliminate lock overhead (including shared_lock atomic operations)
EditPathTree* EditPathTreeManager::get_ept_no_lock(ui anchor_id) const {
    auto it = ept_map.find(anchor_id);
    if (it != ept_map.end()) {
        return it->second.get();
    }
    return nullptr;  // Do not print error info to avoid concurrent output issues
}

// Get count of loaded EPTs
// Use shared_lock for concurrent multi-threaded reads
size_t EditPathTreeManager::get_ept_count() const {
    std::shared_lock<std::shared_mutex> lock(map_mutex);
    return ept_map.size();
}

void EditPathTreeManager::clear_all_epts() {
    std::unique_lock<std::shared_mutex> lock(map_mutex);  // Write operation requires exclusive lock
    ept_map.clear();  // Smart pointers will automatically clean up EPT objects
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
    // [DEBUG] Add recursive debug info
    // if (current_level == 0) {
    //     std::cout << "[DEBUG EPT] Starting EPT build for anchor " << anchor_id
    //               << " with " << curr_indices.size() << " graphs" << std::endl;
    //
    //     // Check initial operation count of each graph
    //     for (size_t i = 0; i < undone_ops_lists.size() && i < 5; i++) {
    //         std::cout << "[DEBUG EPT] Graph " << i << " has " << undone_ops_lists[i].size()
    //                   << " undone operations" << std::endl;
    //     }
    // }
    //
    // std::cout << "[DEBUG EPT] Level " << current_level << ": Processing "
    //           << curr_indices.size() << " graphs" << std::endl;

    // If no remaining graphs, return directly
    if (curr_indices.empty()) {
        // std::cout << "[DEBUG EPT] Level " << current_level << ": No remaining graphs, returning" << std::endl;
        return;
    }

    // Separate in-progress and completed graphs
    std::vector<ui> done_indices;
    std::vector<ui> ongoing_indices;
    for (ui idx : curr_indices) {
        if (undone_ops_lists[idx].empty()) {
            // If no pending operations, the graph is complete
            done_indices.push_back(idx);
        } else {
            // If graph has pending operations, put it in in-progress graphs
            ongoing_indices.push_back(idx);
        }
    }

    // std::cout << "[DEBUG EPT] Level " << current_level << ": " << done_indices.size()
    //           << " completed, " << ongoing_indices.size() << " ongoing" << std::endl;

    // Add completed graph's db_graph_id to current node's completed_db_graph_ids
    if (!done_indices.empty()) {
        for (ui idx : done_indices) {
            int db_graph_id = static_cast<int>(db_graph_ids[idx]);
            ept.tree_nodes[node_idx].completed_db_graph_ids.push_back(db_graph_id);
        }
    }

    // [DEBUG] If all graphs are complete, do not continue expansion
    if (ongoing_indices.empty()) {
        // std::cout << "[DEBUG EPT] Level " << current_level
        //           << ": All graphs completed, stopping expansion" << std::endl;
        return;
    }

    // Collect operations from all in-progress graphs and count frequencies
    std::unordered_map<EditOperation, std::vector<ui>, EditOperationHash, EditOperationEqual> op_to_indices;
    // std::cout << "node_idx: " << node_idx << "  ongoing_indices.size(): " << ongoing_indices.size() << std::endl;
    for (ui idx : ongoing_indices) {
        for (const auto& op : undone_ops_lists[idx]) {
            op_to_indices[op].push_back(idx);  // Record graph indices where each operation appears
        }
    }
   

    // Sort by operation frequency, prioritize most frequent operations
    std::vector<std::pair<EditOperation, std::vector<ui>>> sorted_ops(op_to_indices.begin(), op_to_indices.end());
    std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto& a, const auto& b) {
        return a.second.size() > b.second.size();  // Sort by operation frequency in descending order
    });
    // for (const auto& [op, indices] : sorted_ops) {
    //     if (indices.size() > 1) std::cout << "Sorted Operation: " << op.to_string() << "  Count: " << indices.size() << std::endl;
    // }
    std::unordered_set<ui> processed_indices;  // Record already processed graphs
    bool any_operation_applied = false;  // Flag whether any operation was successfully applied

    // std::cout << "[DEBUG EPT] Level " << current_level << ": Trying to apply "
    //           << sorted_ops.size() << " different operations" << std::endl;

    // Iterate all operations, try to apply to in-progress graphs
    for (const auto& op_pair : sorted_ops) {
        const EditOperation& op = op_pair.first;  // current operation
        const std::vector<ui>& indices_with_op = op_pair.second;  // indices of graphs needing this operation

        std::vector<ui> applicable_indices;  // Store indices of graphs that can apply current operation

        bool can_apply = ept.tree_nodes[node_idx].pseudo_graph.can_apply_operation(op);

        // std::cout << "[DEBUG EPT] Level " << current_level << ": Operation " << op.to_string()
        //           << " can_apply=" << (can_apply ? "YES" : "NO")
        //           << " affects " << indices_with_op.size() << " graphs" << std::endl;

        if (!can_apply) continue;
        for (ui idx : indices_with_op) {
            // If current graph not yet processed and can apply this operation
            if (processed_indices.count(idx) == 0) {
                applicable_indices.push_back(idx);
            }
        }

        // If no graph can apply this operation, skip it
        if (applicable_indices.empty()) {
            continue;
        }

        // Apply operation, generate child node
        PseudoGraph child_pseudo_graph = ept.tree_nodes[node_idx].pseudo_graph;
        child_pseudo_graph.apply_edit_operation(op);  // Apply operation on current graph

        // Create child node
        // Increment node counter
        ept_node_counter++;
        size_t child_idx = ept.tree_nodes.size();
        TreeNode child_node(op, current_level + 1,
                            child_idx, anchor_id,
                            child_idx,  // tree_node_graph_id uses ept_node_counter
                            node_idx,          // current node's index as parent index
                            child_pseudo_graph);

        // Record applied operations
        child_node.accumulated_ops.clear();
        child_node.accumulated_ops.push_back(op);

        // Convert pseudo_graph to Graph and store in child node's graph_ptr
        child_node.graph_ptr = std::make_shared<Graph>(child_node.pseudo_graph.to_graph());

        child_node.graph_ptr->id = std::to_string(child_node.tree_node_graph_id);

        

        // Add child node to tree
        ept.tree_nodes.push_back(child_node);
        
        // size_t child_idx = ept_node_counter;
        
        // Add child node's index to parent's children_indices
        ept.tree_nodes[node_idx].children_indices.push_back(child_idx);
        // std::cout << "node_idx: " << node_idx << std::endl;
        // std::cout << "children: ";
        // for (auto idx : ept.tree_nodes[node_idx].children_indices) {
        //     std::cout << idx << " " << std::endl;
        // }
        // std::cout << "Add child node: " << child_idx << " to parent node: " << node_idx << std::endl;
        // std::cout << "child_node.parent_index: " << child_node.parent_index << std::endl;
        // Update operation list
        auto updated_undone_ops_lists = undone_ops_lists;
        for (ui idx : applicable_indices) {
            updated_undone_ops_lists[idx].erase(op);  // Delete completed operation
            processed_indices.insert(idx);  // Mark as processed
        }

        any_operation_applied = true;  // Mark that at least one operation was applied in this recursion

        // Recursively process child nodes
        build_edit_path_tree_recursive(ept, updated_undone_ops_lists, db_graph_ids, anchor_id,
                                       child_idx, applicable_indices, current_level + 1, ept_node_counter);
    }

    // If no operation was successfully applied but there are still unprocessed graphs, this branch cannot complete
    if (!any_operation_applied) {
        std::vector<ui> final_remaining_indices;
        for (ui idx : ongoing_indices) {
            // Find graphs that were not successfully processed
            if (processed_indices.count(idx) == 0) {
                final_remaining_indices.push_back(idx);
            }
        }

        // If graphs still cannot complete, report error and print relevant info
        if (!final_remaining_indices.empty()) {
            std::cerr << "Error: The following db_graph_ids cannot be completed from this branch:\n";
            for (ui idx : final_remaining_indices) {
                int db_graph_id = (int)db_graph_ids[idx];
                std::cerr << db_graph_id << " ";
            }
            std::cerr << "\n";

            // Print current pseudo_graph state
            std::cerr << "Current pseudo graph state:\n";
            ept.tree_nodes[node_idx].pseudo_graph.print_pseudo_graph();

            // Print remaining operations
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

        // Print node's ept_node_id
        tmp += std::to_string(i) + " (ept_node_id = " + std::to_string(node.ept_node_id) + ") ";

        // Print parent node index
        tmp += "(parent = ";
        if (node.parent_index == SIZE_MAX) {
            tmp += "None"; // or "ROOT"
        } else {
            tmp += std::to_string(node.parent_index);
        }
        tmp += ") : "; 

        // Print all child node indices
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

    // Remove db_graph_id parameter
    TreeNode root_node(dummy_op, 0, ept_node_counter, anchor_id, ept_node_counter, SIZE_MAX, root_pseudo_graph);
    root_node.graph_ptr = root_graph_ptr;
    root_node.graph_ptr->id = std::to_string(root_node.tree_node_graph_id);
    ept_node_counter++;

    root_node.accumulated_ops.clear();
    // Important: add anchor_id to root node's completed_db_graph_ids
    // So when query's GED<=tau to anchor, the anchor itself will be returned as result
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
    // Renumber and update ept_node_id
    // rebuild_and_reindex_ept_dfs(ept);
    // print_tree(ept);

    // remove_parent_inf_nodes(ept);
    ept.reorder();
    print_tree(ept);
    
   

    // Coverage check
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
    // 1. Safety check
    if (node_idx >= ept.tree_nodes.size()) {
        return;
    }
    TreeNode& node = ept.tree_nodes[node_idx];

    // 2. First let all child nodes perform recursive simplification
    for (size_t child_idx : node.children_indices) {
        if (child_idx < ept.tree_nodes.size()) {
            simplifyNode(ept, child_idx);
        }
    }

    // 3. Try merging: when "child has no completed graphs && child has only 1 child"
    bool canMerge = true;
    while (canMerge) {
        // Find child nodes meeting merge conditions
        for (size_t child_idx : node.children_indices) {
            if (child_idx >= ept.tree_nodes.size()) continue;

            TreeNode& child = ept.tree_nodes[child_idx];
            // Condition: child has no completed_db_graph_ids and has exactly 1 child
            if (child.completed_db_graph_ids.empty() && child.children_indices.size() == 1) {
                // Prepare to merge child -> grandchild
                size_t grandchild_idx = child.children_indices[0];
                if (grandchild_idx >= ept.tree_nodes.size()) break;

                TreeNode& grandchild = ept.tree_nodes[grandchild_idx];

                // Prepend all child's accumulated_ops to grandchild
                grandchild.accumulated_ops.insert(
                    grandchild.accumulated_ops.begin(),
                    child.accumulated_ops.begin(),
                    child.accumulated_ops.end()
                );

                // Update grandchild's parent to point to current node
                grandchild.parent_index = node_idx;

                // Change node's sole child to grandchild
                auto it = std::find(node.children_indices.begin(), node.children_indices.end(), child_idx);
                if (it != node.children_indices.end()) {
                    *it = grandchild_idx;
                }

                // Set merged child node's parent_index to SIZE_MAX (or INF) to indicate merged
                child.parent_index = INF;

                // Clear child (it was merged)
                child.children_indices.clear();

                // Continue checking if new sole child can be further merged
                continue;
            }
        }
        // If no more nodes can be merged, exit loop
        canMerge = false;
    }
}


void simplify_EPT(EditPathTree& ept) {
    if (ept.tree_nodes.empty()) return;

    // First perform node merge operations
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


// Function to simplify edit path tree
// void simplify_EPT(EditPathTree& ept) {
//     if (ept.tree_nodes.empty()) return;

//     // Use stack to simulate recursion
//     std::stack<size_t> stack;
//     stack.push(ept.root_index);

//     while (!stack.empty()) {
//         size_t node_idx = stack.top();
//         stack.pop();

//         TreeNode& node = ept.tree_nodes[node_idx];

//         if (node.children_indices.empty()) continue;

//         for (size_t i = 0; i < node.children_indices.size(); ++i) {
//             size_t child_idx = node.children_indices[i];
//             TreeNode& child = ept.tree_nodes[child_idx];

//             // When child has no associated real target graph (db_graph_id == -1) and has exactly one child, simplify
//             while (child.db_graph_id == -1 && child.children_indices.size() == 1) {
//                 size_t grandchild_idx = child.children_indices[0];
//                 TreeNode& grandchild = ept.tree_nodes[grandchild_idx];

//                 // Update grandchild's accumulated_ops to only contain operations from child
//                 // Since accumulated_ops already only contains operations from parent to current node, no extra handling needed

//                 // Update grandchild's parent_index
//                 grandchild.parent_index = node_idx;

//                 // Replace child with grandchild
//                 node.children_indices[i] = grandchild_idx;

//                 // Clear child's children list to avoid duplicate references
//                 child.children_indices.clear();

//                 // Continue checking the new child node
//                 child_idx = grandchild_idx;
//                 child = ept.tree_nodes[child_idx];
//             }

//             // Push child index to stack, continue processing
//             stack.push(child_idx);
//         }
//     }
// }


// Update levels after simplification
// void update_simplified_levels(EditPathTree& ept, size_t node_index, int current_level) {
//     if (node_index >= ept.tree_nodes.size()) return;

//     TreeNode& node = ept.tree_nodes[node_index];
//     node.simplified_level = current_level;

//     for (size_t child_idx : node.children_indices) {
//         update_simplified_levels(ept, child_idx, current_level + 1);
//     }
// }





// Function to print tree
void print_EPT(const EditPathTree& ept, bool simplified) {
    if (ept.tree_nodes.empty()) {
        std::cout << "[print_EPT] ept.tree_nodes is empty.\n";
        return;
    }

    const std::vector<TreeNode>& nodes = ept.tree_nodes;
    TreeStatistics stats;

    // Use queue for BFS traversal
    std::queue<size_t> node_queue;
    node_queue.push(ept.root_index);

    while (!node_queue.empty()) {
        size_t node_idx = node_queue.front();
        node_queue.pop();

        const TreeNode& node = nodes[node_idx];

        // ========== 1) Record total node count ========== 
        stats.total_nodes++;

        // ========== 2) Based on simplified flag, choose simplified_level or original level ==========
        int depth = simplified ? node.simplified_level : node.level;
        int original_level = node.level;  // only for recording distribution

        // ========== 3) Count graphs contained in this node ==========
        //  node.completed_db_graph_ids may record multiple completed graph IDs,
        //  accumulate their count to total_graph_count
        stats.total_graph_count += node.completed_db_graph_ids.size();

        // ========== 4) Record depth range (currently determined by leaf nodes for max_depth/min_depth) ==========
        if (node.children_indices.empty()) {
            // Leaf node
            if (depth > stats.max_depth) stats.max_depth = depth;
            if (depth < stats.min_depth) stats.min_depth = depth;
        }

        // ========== 5) Per-layer node count + level distribution ========== 
        stats.nodes_per_level[depth]++;
        stats.level_breakdown[depth][original_level]++;

        // ========== 6) If has children => fanout statistics ==========
        if (!node.children_indices.empty()) {
            int fanout = (int)node.children_indices.size();
            if (fanout > stats.max_fanout) stats.max_fanout = fanout;
            if (stats.min_fanout == 0 || fanout < stats.min_fanout) {
                stats.min_fanout = fanout;
            }
            stats.total_fanout += fanout;
            stats.internal_node_count++;

            // Add child nodes to queue
            for (size_t child_idx : node.children_indices) {
                node_queue.push(child_idx);
            }

            // Record to per-layer fanout statistics
            auto &level_stats = stats.fanout_per_level[depth];
            level_stats.count++;
            level_stats.sum_fanout += fanout;
            level_stats.sum_squares += (long long)fanout * fanout;
            if (fanout > level_stats.max_fanout) {
                level_stats.max_fanout = fanout;
            }
            if (fanout < level_stats.min_fanout) {
                level_stats.min_fanout = fanout;
            }
        }
    }

    // If tree has no leaves => guard against initial values
    if (stats.min_depth == std::numeric_limits<int>::max()) {
        stats.min_depth = 0;
    }

    // ========== Output statistics ==========
    std::cout << "\nTree Statistics:\n";

    // 1) Depth
    std::cout << "Maximum Depth: " << stats.max_depth << std::endl;
    std::cout << "Minimum Depth: " << stats.min_depth << std::endl;

    // 2) Per-layer node count
    std::cout << "Nodes per Level:\n";
    std::vector<int> levels;
    levels.reserve(stats.nodes_per_level.size());
    for (const auto& pair : stats.nodes_per_level) {
        levels.push_back(pair.first);
    }
    std::sort(levels.begin(), levels.end());

    for (int level : levels) {
        int count_nodes = stats.nodes_per_level[level];
        std::cout << "  Level " << level << ": " << count_nodes << " nodes\n";
    }

    // 3) Fanout global info
    std::cout << "Maximum Fanout (global): " << stats.max_fanout << std::endl;
    std::cout << "Minimum Fanout (global): "
              << (stats.internal_node_count > 0 ? stats.min_fanout : 0)
              << std::endl;
    double average_fanout = (stats.internal_node_count > 0)
                            ? (double)stats.total_fanout / stats.internal_node_count
                            : 0.0;
    std::cout << "Average Fanout (global): " << average_fanout << std::endl;

    // 4) Per-layer fanout statistics
    std::cout << "\nFanout statistics per level:\n";
    for (int level : levels) {
        auto it = stats.fanout_per_level.find(level);
        if (it == stats.fanout_per_level.end()) {
            // This layer contains only leaf nodes, no internal nodes => no fanout data
            std::cout << "  Level " << level
                      << ": (no internal nodes, fanout=0)\n";
            continue;
        }

        const auto &fs = it->second;
        if (fs.count == 0) {
            // Also indicates this layer has no internal nodes
            std::cout << "  Level " << level
                      << ": (count=0, no internal nodes)\n";
            continue;
        }

        double avg = (double)fs.sum_fanout / fs.count;
        double mean_of_squares = (double)fs.sum_squares / fs.count;
        double variance = mean_of_squares - avg * avg;
        double stddev = (variance > 1e-12) ? std::sqrt(variance) : 0.0;

        std::cout << "  Level " << level << ": \n"
                  << "    count = " << fs.count << " (internal nodes)\n"
                  << "    max   = " << fs.max_fanout << "\n"
                  << "    min   = " << fs.min_fanout << "\n"
                  << "    avg   = " << avg << "\n"
                  << "    std   = " << stddev << "\n";
    }

    // 5) Print total_graph_count
    std::cout << "\nTotal Graph Count: " << stats.total_graph_count << std::endl;

    // 6) New: print total_nodes
    std::cout << "Total Nodes (BFS counted): " << stats.total_nodes << std::endl;

    std::cout << std::endl;
}



void print_EPT_with_actual_steps(const EditPathTree& ept) {
    if(ept.tree_nodes.empty()) {
        std::cout << "[WARN] ept is empty\n";
        return;
    }

    TreeStatistics stats;

    // Prepare BFS queue: (node_idx, dist)
    std::queue<std::pair<size_t,int>> q;
    q.push({ ept.root_index, 0 });

    while(!q.empty()) {
        auto [curr_idx, dist] = q.front();
        q.pop();

        stats.total_nodes++;

        const TreeNode &node = ept.tree_nodes[curr_idx];
        int ept_level = node.level;  // this is the EPT level

        // if it is a leaf
        if(node.children_indices.empty()) {
            // BFS-dist = dist
            // EPT-level = ept_level

            // 1) leaf count/layer statistics
            stats.leaf_count_per_level[ept_level]++;

            // 2) add BFS-dist to leaf_dist_per_ept_level[ept_level]
            auto &lds = stats.leaf_dist_per_ept_level[ept_level];
            lds.count++;
            lds.sum_dist += dist;
            lds.sum_squares += (long long)dist * dist;
        } 
        else {
            // has children => fanout
            stats.internal_node_count++;
            int fanout = (int)node.children_indices.size();
            stats.total_fanout += fanout;
            if(fanout>stats.max_fanout) stats.max_fanout=fanout;
            if(stats.min_fanout==0 || fanout<stats.min_fanout){
                stats.min_fanout = fanout;
            }

            // BFS push children
            for(auto child_idx : node.children_indices) {
                q.push({ child_idx, dist+1 });
            }
        }
    }

    // ========== Print statistics ==========

    // 1) total node count
    std::cout << "Total Node Count: " << stats.total_nodes << "\n";

    // 2) Per-layer leaf node BFS-dist distribution
    std::cout << "\n=== BFS-dist distribution grouped by EPT-level ===\n";
    // Collect all ept_levels
    std::vector<int> ept_levels;
    ept_levels.reserve(stats.leaf_dist_per_ept_level.size());
    for(const auto &kv : stats.leaf_dist_per_ept_level) {
        ept_levels.push_back(kv.first);
    }
    std::sort(ept_levels.begin(), ept_levels.end());

    for(int lvl : ept_levels){
        auto &lds = stats.leaf_dist_per_ept_level[lvl];
        if(lds.count==0) {
            std::cout<<"  EPT-level="<<lvl<<": (no leaf??)\n";
            continue;
        }
        double avg = (double)lds.sum_dist/ lds.count;
        double mean_sq = (double)lds.sum_squares/ lds.count;
        double var = mean_sq - avg*avg;
        double stddev = (var>1e-12) ? std::sqrt(var) : 0.0;
        std::cout<<"  EPT-level="<<lvl<<" => leaf_count="<<lds.count
                 <<", BFS-dist avg="<<avg<<", std="<<stddev<<"\n";
    }
}

void test_EPT_structure(const EditPathTree &ept)
{
    // 1) basic info
    std::cout << "\n===== [Test EPT Structure] =====\n";
    std::cout << "Tree node count: " << ept.tree_nodes.size() << "\n";
    std::cout << "root_index: " << ept.root_index << "\n";
    std::cout << "anchor_id:  " << ept.anchor_id << "\n";

    if (ept.tree_nodes.empty()) {
        std::cout << "[INFO] EPT is empty => no further tests.\n";
        return;
    }

    // 2) check node basic properties
    int max_level = std::numeric_limits<int>::lowest();
    int min_level = std::numeric_limits<int>::max();
    // count levels => vector<nodeIndex>
    std::map<int, std::vector<size_t>> level_map;

    // assuming during EPT construction, node parent_index is set to SIZE_MAX for root
    // if not, modify as needed
    size_t root_cnt = 0; // count how many nodes parent_index=SIZE_MAX

    std::cout << "\n[Node Basic Info]\n";
    for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
        const TreeNode &node = ept.tree_nodes[i];
        int lvl = node.level;

        // update min/max
        if (lvl > max_level) max_level = lvl;
        if (lvl < min_level) min_level = lvl;

        // add to level_map
        level_map[lvl].push_back(i);

        // check parent_index
        bool is_root_like = (node.parent_index == SIZE_MAX);
        if (is_root_like) {
            root_cnt++;
        }

        // std::cout << "Node #" << i 
        //           << ": parent_index=" << (is_root_like ? -1 : (int)node.parent_index)
        //           << ", level=" << node.level
        //           << ", children=" << node.children_indices.size()
        //           << ", db_graph_n=" << node.db_graph_n
        //           << ", db_graph_m=" << node.db_graph_m
        //           << ", completed_ids=" << node.completed_db_graph_ids.size()
        //           << "\n";
    }

    std::cout << "max_level=" << max_level << ", min_level=" << min_level << "\n";
    std::cout << "nodes that have parent_index=SIZE_MAX: " << root_cnt << "\n";

    // 3) list node count per layer
    std::cout << "\n[Level Distribution by node.level]\n";
    for (auto &pair : level_map) {
        int lvl = pair.first;
        size_t count_nodes = pair.second.size();
        std::cout << "  Level=" << lvl << ", " << count_nodes << " nodes\n";
    }

    // 4) BFS connectivity check (starting from root_index)
    //    if root_index is not necessarily the unique “true root”, you can iterate all parent_index=SIZE_MAX nodes and do BFS for each
    std::vector<bool> visited(ept.tree_nodes.size(), false);

    std::queue<size_t> Q;
    Q.push(ept.root_index);
    visited[ept.root_index] = true;
    int bfs_max_depth = 0;

    // for recording BFS depth
    std::vector<int> bfs_depth(ept.tree_nodes.size(), -1);
    bfs_depth[ept.root_index] = 0;

    while (!Q.empty()) {
        size_t u = Q.front();
        Q.pop();
        int d = bfs_depth[u];

        // update max depth
        if (d > bfs_max_depth) {
            bfs_max_depth = d;
        }

        // iterate child nodes
        for (auto child_idx : ept.tree_nodes[u].children_indices) {
            if (!visited[child_idx]) {
                visited[child_idx] = true;
                bfs_depth[child_idx] = d + 1;
                Q.push(child_idx);
            }
        }
    }

    // count nodes visited by BFS
    int visit_count = 0;
    for (auto b : visited) {
        if (b) visit_count++;
    }

    std::cout << "\n[BFS from root_index=" << ept.root_index << "]\n"
              << "  visited " << visit_count << " / " << ept.tree_nodes.size() << " nodes\n"
              << "  BFS max depth = " << bfs_max_depth << "\n";

    // list unvisited nodes (if any)
    if (visit_count < (int)ept.tree_nodes.size()) {
        std::cout << "  The following nodes were NOT visited from root:\n";
        for (size_t i = 0; i < ept.tree_nodes.size(); i++) {
            if (!visited[i]) {
                std::cout << "    Node #" << i 
                          << " (level=" << ept.tree_nodes[i].level
                          << ", parent=" << ept.tree_nodes[i].parent_index
                          << ")\n";
            }
        }
    }

    // 5) (optional) list BFS depth distribution
    std::map<int,int> bfs_depth_count;
    for (size_t i=0; i<bfs_depth.size(); i++) {
        if (bfs_depth[i] >= 0) {
            bfs_depth_count[bfs_depth[i]]++;
        }
    }
    std::cout << "\n[BFS Depth Distribution]\n";
    for (auto &dc : bfs_depth_count) {
        std::cout << "  depth=" << dc.first 
                  << ", count=" << dc.second << "\n";
    }

    // 6) end
    std::cout << "\n===== [Test EPT Structure DONE] =====\n";
}

void merge_children_by_WLhash_bfs(EditPathTree & ept)
{
    // flag for whether merged/cleared
    std::vector<bool> merged_flags(ept.tree_nodes.size(), false);

    // 1) BFS queue
    std::queue<size_t> Q;
    Q.push(ept.root_index);

    // 2) BFS
    while (!Q.empty()) {
        size_t u = Q.front();
        Q.pop();

        // if this node was merged (flagged), skip it
        if (merged_flags[u]) {
            continue;
        }

        TreeNode &nodeU = ept.tree_nodes[u];

        // Collect "WL-hash -> vector<childIdx>" for children_indices
        std::unordered_map<std::string, std::vector<size_t>> hash_groups;

        // 2.1) build hash_groups
        for (auto childIdx : nodeU.children_indices) {
            // if child has been marked as merged, maybe skip or treat as“empty”node
            if (merged_flags[childIdx]) {
                continue;
            }

            TreeNode &childNode = ept.tree_nodes[childIdx];
            // compute WL-hash
            std::string hval = get_wl_hash(childNode); 
            hash_groups[hval].push_back(childIdx);
        }

        // 2.2) for each hash-value group => merge
        //      keep only the first occurrence  => representative node rep
        //      merge the rest into rep and mark merged_flags=true
        //      and remove from nodeU.children_indices
        std::set<size_t> toRemove; // children that get merged
        for (auto &grpPair : hash_groups) {
            auto &sameHashChildIdxs = grpPair.second;
            if (sameHashChildIdxs.size() <= 1) {
                // only 1 => no merge needed
                continue;
            }

            // representative
            size_t rep = sameHashChildIdxs[0];
            TreeNode &repNode = ept.tree_nodes[rep];

            // merge childrenIndices & completed_ids
            std::set<size_t> unionChildren(repNode.children_indices.begin(), repNode.children_indices.end());
            std::set<int> unionCompleted(repNode.completed_db_graph_ids.begin(), repNode.completed_db_graph_ids.end());

            // the rest
            for (size_t k = 1; k < sameHashChildIdxs.size(); k++){
                size_t otherIdx = sameHashChildIdxs[k];
                TreeNode &otherNode = ept.tree_nodes[otherIdx];

                // union children
                for (auto c : otherNode.children_indices) {
                    unionChildren.insert(c);
                }
                // union completed
                for (auto cid : otherNode.completed_db_graph_ids) {
                    unionCompleted.insert(cid);
                }

                // set merged_flags
                merged_flags[otherIdx] = true;
                // record nodes to remove from nodeU.children_indices otherIdx
                toRemove.insert(otherIdx);
            }

            // write back
            repNode.children_indices.assign(unionChildren.begin(), unionChildren.end());
            repNode.completed_db_graph_ids.assign(unionCompleted.begin(), unionCompleted.end());
        }

        // 2.3) delete toRemove from nodeU.children_indices
        if (!toRemove.empty()) {
            std::vector<size_t> newChildren;
            newChildren.reserve(nodeU.children_indices.size());
            for (auto c : nodeU.children_indices) {
                if (toRemove.find(c) == toRemove.end()) {
                    newChildren.push_back(c);
                }
            }
            nodeU.children_indices.swap(newChildren);
        }

        // 2.4) push all still-valid children into queue
        for (auto c : nodeU.children_indices) {
            if (!merged_flags[c]) {
                Q.push(c);
            }
        }
    }

    // 3) optional: clear merged nodes(making it childless, no completed)
    for (size_t i = 0; i < ept.tree_nodes.size(); i++){
        if (merged_flags[i]) {
            TreeNode &tn = ept.tree_nodes[i];
            tn.children_indices.clear();
            tn.completed_db_graph_ids.clear();
            // can also tn.db_graph=nullptr, etc
        }
    }

    std::cout << "[merge_children_by_WLhash_bfs] done.\n";
}

/**
 * @brief compute WL-hash for a node
 *        here you can directly use tn.db_graph->compute_wl_hash() or write your own
 */
std::string get_wl_hash(const TreeNode &tn)
{
    if (!tn.db_graph) return "NO-DB";
    return tn.db_graph->compute_wl_hash(); 
}


void shrink_completed_ids_to_first(EditPathTree& ept)
{
    if (ept.tree_nodes.empty()) {
        std::cout << "[shrink_completed_ids_to_first] empty tree, skip.\n";
        return;
    }

    size_t touched = 0;
    for (TreeNode& tn : ept.tree_nodes)
    {
        if (tn.completed_db_graph_ids.size() > 1)
        {
            int first = tn.completed_db_graph_ids.front();
            tn.completed_db_graph_ids.clear();
            tn.completed_db_graph_ids.push_back(first);
            ++touched;
        }
    }

    std::cout << "[shrink_completed_ids_to_first] processed "
              << ept.tree_nodes.size() << " nodes; "
              << "shrunk " << touched  << " node(s).\n";
}
