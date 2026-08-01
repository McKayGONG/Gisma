// Utility.cpp
#include "Utility.h"
#include "Graph.h"
#include "Node.h"
#include "Anchor.h"
#include "NetDag.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

// Iteration count variables removed - now controlled via Application/GismaSearchEngine class members
// set from --app_max_iter command-line parameter (default: 2300)

// 实现打开文件的静态函数
FILE* Utility::open_file(const char* file_name, const char* mode) {
    FILE* f = fopen(file_name, mode);
    if (f == nullptr) {
        printf("Cannot open file: %s\n", file_name);
        exit(1);
    }
    return f;
}

// 实现整数转字符串的静态函数


// 实现从txt文件中加载图数据的静态函数
ui Utility::load_db(const char* file_name, std::vector<Graph*>& graphs, std::map<std::string, ui>& vM, std::map<std::string, ui>& eM) {
    FILE* fin = Utility::open_file(file_name, "r");

    const ui MAX_LINE = 1024;
    char line[MAX_LINE];
    if (fgets(line, MAX_LINE, fin) == NULL) {
        fclose(fin);
        return 0;
    }

    ui max_n = 0;
    ui graph_id = 0;  // 用于从0开始为每个图编号
    ui max_vlabel = 0;
    ui max_elabel = 0;
    while (line[0] == 'I' && line[1] == 'D') {  // 检查"ID"开头
        line[0] = 'x';  // 标记当前行已处理

        std::vector<std::pair<int, ui>> vertices;
        std::vector<std::pair<std::pair<int, int>, ui>> edges;
        while (fgets(line, MAX_LINE, fin) != NULL && !(line[0] == 'I' && line[1] == 'D')) {
            if (line[0] == 'v') {
                int a;
                ui lab_id;  // 改为 ui 类型，因为你已经使用整数标签
                sscanf(line + 2, "%d%d", &a, &lab_id);  // 读取节点id和整数标签
                if (lab_id > max_vlabel) max_vlabel = lab_id;
                // std::cout << "max_vlabel: " << max_vlabel << std::endl;
                vertices.emplace_back(a, lab_id);  // 将节点和标签添加到 vertices
                // std::cout << "Vertex: " << a << " Label: " << lab_id << std::endl;
            
            } else if (line[0] == 'e') {
                int a, b;
                ui edge_label;  // 边的标签也是整数
                sscanf(line + 2, "%d%d%d", &a, &b, &edge_label);  // 读取两个节点和边标签
                if (edge_label > max_elabel) max_elabel = edge_label;
                edges.pb(mp(mp(a, b), edge_label));  // 添加边和标签
                edges.pb(mp(mp(b, a), edge_label));  // 添加反向边
            } else {
                std::cerr << "!!! Unrecognized first letter in a line when loading DB!\n";
            }
            line[0] = 'x';  // 标记当前行已处理
        }

        // 排序并验证顶点
        std::sort(vertices.begin(), vertices.end());
        for (ui i = 0; i < vertices.size(); i++) {
            assert(vertices[i].first == static_cast<int>(i));
        }
        if (vertices.size() > max_n) max_n = vertices.size();

        // 排序并验证边
        std::sort(edges.begin(), edges.end());
        for (ui i = 0; i < edges.size(); i++) {
            assert(edges[i].first.first >= 0 && edges[i].first.first < vertices.size());
            assert(edges[i].first.second >= 0 && edges[i].first.second < vertices.size());
            if (i > 0) {
                assert(edges[i].first != edges[i - 1].first);
            }
            assert(edges[i].second < eM.size());
        }

        // 使用从0开始的graph_id代替从文件读取的id
        std::string id = std::to_string(graph_id);
        graph_id++;  // 自增，确保下一个图的id是递增的

        // 创建新的Graph对象并添加到graphs向量中
        graphs.pb(new Graph(id, vertices, edges));
    }
    // std::cout << "max_vlabel: " << max_vlabel << " max_elabel: " << max_elabel << std::endl;
    for (ui i=0; i < max_vlabel+1; i++) {
        vM[std::to_string(i)] = i;
    }
    for (ui i=0; i < max_elabel+1; i++) {
        eM[std::to_string(i)] = i;
    }

    fclose(fin);
    return max_n;
}



// 实现生成包含short_name和参数的index_name，并保留一位小数的函数
std::string Utility::get_index_name(const std::string& dataset, double alpha, double tau, double error_tolerance_index, size_t graph_size) {
    ;
    
    // 使用 std::ostringstream 来保留一位小数
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << dataset << "_" << graph_size << "_"
        << alpha << "_" << tau << "_" << error_tolerance_index;
    
    return oss.str();  // 返回格式化后的 index_name
}

size_t Utility::get_upper_tri_index(int N, int i, int j) {
    if (i > j) std::swap(i, j);
    return static_cast<size_t>(i * N - (i * (i - 1)) / 2 + (j - i));
}

bool Utility::load_exact_ground_truth(const std::string& file_path,
                                      std::map<int, std::map<double, std::vector<int>>>& ground_truth) {
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int query_id;
        double distance;
        char bracket;
        
        // 读取 query_id 和 distance
        if (!(iss >> query_id >> distance >> bracket) || bracket != '[') {
            std::cerr << "Error: Invalid line format: " << line << std::endl;
            continue;
        }

        std::vector<int> graph_ids;
        int graph_id;

        // 读取 graph_ids
        while (iss >> graph_id) {
            graph_ids.push_back(graph_id);
            if (iss.peek() == ',') { // 跳过逗号
                iss.ignore();
            }
        }

        ground_truth[query_id][distance] = graph_ids;
    }

    infile.close();
    return true;
}


// Vector distance calculation implementation
double Utility::euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vector dimensions don't match (" << v1.size() << " vs " << v2.size() << ")" << std::endl;
        return -1.0;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Load embedding vectors from binary file
bool Utility::load_embeddings(const std::string& embedding_file, std::vector<std::vector<float>>& embeddings) {
    std::ifstream file(embedding_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open embedding file " << embedding_file << std::endl;
        return false;
    }
    
    // Read number of graphs and embedding dimension
    int num_graphs, embedding_dim;
    file.read(reinterpret_cast<char*>(&num_graphs), sizeof(int));
    file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int));
    
    if (!file) {
        std::cerr << "Error: Failed to read embedding file header" << std::endl;
        return false;
    }
    
    std::cout << "Loading embeddings: " << num_graphs << " graphs, " << embedding_dim << " dimensions" << std::endl;
    
    // Resize and read embeddings
    embeddings.resize(num_graphs);
    for (int i = 0; i < num_graphs; ++i) {
        // Skip the graph ID (4 bytes) before reading embedding
        int graph_id;
        file.read(reinterpret_cast<char*>(&graph_id), sizeof(int));
        if (!file) {
            std::cerr << "Error: Failed to read graph ID for graph " << i << std::endl;
            return false;
        }

        embeddings[i].resize(embedding_dim);
        file.read(reinterpret_cast<char*>(embeddings[i].data()), embedding_dim * sizeof(float));
        if (!file) {
            std::cerr << "Error: Failed to read embedding for graph " << i << std::endl;
            return false;
        }
    }
    
    std::cout << "Successfully loaded " << num_graphs << " embeddings" << std::endl;
    return true;
}

// E1: construction 计时输出文件路径（空=不统计，默认）。由 main() 从 --timing_log 设置。
std::string Utility::e1_timing_path = "";

// E1: append one structured timing record to e1_timing_path (raw numbers, not logs).
// 默认 e1_timing_path 为空 → 直接返回，不创建任何文件（正常用户零影响）。
void Utility::append_e1_timing(const std::string& record) {
    if (e1_timing_path.empty()) return;
    std::error_code ec;
    std::filesystem::path p(e1_timing_path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream ofs(e1_timing_path, std::ios::app);
    if (ofs.is_open()) ofs << record << "\n";
}

// Predict GED using embedding vectors with simple distance-based mapping