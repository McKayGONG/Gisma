#include "Graph.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <limits>
#include <mutex>
#include <filesystem>

size_t Graph::FEATURE_DIM = 62;
// 构造函数
Graph::Graph() : id(""), n(0), m(0), pstarts(nullptr), edges(nullptr), vlabels(nullptr), elabels(nullptr) {}

Graph::Graph(const std::string& _id, const std::vector<std::pair<int, ui> >& _vertices, const std::vector<std::pair<std::pair<int, int>, ui> >& _edges) {
    id = _id;
    n = _vertices.size();
    m = _edges.size();

    pstarts = new ui[n + 1];
    vlabels = new ui[n];
    edges = new ui[m];
    elabels = new ui[m];

    for (ui i = 0; i < n; i++) vlabels[i] = _vertices[i].second;

    for (ui i = 0; i < m; i++) {
        edges[i] = _edges[i].first.second;
        elabels[i] = _edges[i].second;
        //assert(elabels[i] >= 0 && elabels[i] < 3);
    }

    ui idx = 0;
    pstarts[0] = idx;
    for (ui i = 0; i < n; i++) {
        while (idx < m && _edges[idx].first.first == i) ++idx;
        pstarts[i + 1] = idx;
    }
    assert(pstarts[n] == m);
}

Graph::Graph(const Graph& other) {
    id = other.id;
    n = other.n;
    m = other.m;

    // 深拷贝 pstarts
    pstarts = new ui[n + 1];
    memcpy(pstarts, other.pstarts, (n + 1) * sizeof(ui));

    // 深拷贝 edges
    ui edge_count = pstarts[n];
    edges = new ui[edge_count];
    memcpy(edges, other.edges, edge_count * sizeof(ui));

    // 深拷贝 vlabels
    vlabels = new ui[n];
    memcpy(vlabels, other.vlabels, n * sizeof(ui));

    // 深拷贝 elabels
    elabels = new ui[edge_count];
    memcpy(elabels, other.elabels, edge_count * sizeof(ui));
}

Graph& Graph::operator=(const Graph& other) {
    if (this != &other) {
        // 释放现有内存
        delete[] pstarts;
        delete[] edges;
        delete[] vlabels;
        delete[] elabels;

        id = other.id;
        n = other.n;
        m = other.m;

        // 深拷贝 pstarts
        pstarts = new ui[n + 1];
        memcpy(pstarts, other.pstarts, (n + 1) * sizeof(ui));

        // 深拷贝 edges
        ui edge_count = pstarts[n];
        edges = new ui[edge_count];
        memcpy(edges, other.edges, edge_count * sizeof(ui));

        // 深拷贝 vlabels
        vlabels = new ui[n];
        memcpy(vlabels, other.vlabels, n * sizeof(ui));

        // 深拷贝 elabels
        elabels = new ui[edge_count];
        memcpy(elabels, other.elabels, edge_count * sizeof(ui));
    }
    return *this;
}


// 析构函数
Graph::~Graph() {
    if (pstarts != nullptr) {
        delete[] pstarts;
        pstarts = nullptr;
    }
    if (edges != nullptr) {
        delete[] edges;
        edges = nullptr;
    }
    if (vlabels != nullptr) {
        delete[] vlabels;
        vlabels = nullptr;
    }
    if (elabels != nullptr) {
        delete[] elabels;
        elabels = nullptr;
    }
}



int Graph::size_based_bound(Graph* g) {
    ui r1 = n > g->n ? n - g->n : g->n - n;
    ui r2 = m > g->m ? m - g->m : g->m - m;
    assert(r2 % 1 == 0);
    return r1 + r2 / 2;
}





int Graph::ged_lower_bound_filter(Graph *g, ui verify_upper_bound, int *vlabel_cnt, int *elabel_cnt, int *degree_q, int *degree_g, int *tmp) {
    ui lb = size_based_bound(g);
    if(lb > verify_upper_bound) return lb;

    lb = (n > g->n ? n : g->n);
    for(ui i = 0;i < n;i ++) ++ vlabel_cnt[vlabels[i]];
    for(ui i = 0;i < g->n;i ++) {
        ui vl = g->vlabels[i];
        if(vlabel_cnt[vl] > 0) {
            -- vlabel_cnt[vl];
            -- lb;
        }
    }
    for(ui i = 0;i < n;i ++) vlabel_cnt[vlabels[i]] = 0;
    if(lb > verify_upper_bound) return lb;

    assert(m == pstarts[n]&&g->m == g->pstarts[g->n]);
    for(ui i = 0;i < n;i ++) degree_q[i] = pstarts[i+1] - pstarts[i];
    for(ui i = 0;i < g->n;i ++) degree_g[i] = g->pstarts[i+1] - g->pstarts[i];
    int *degrees_cnt_q = tmp;
    int max_degree_q = 0, max_degree_g = 0;
    memset(degrees_cnt_q, 0, sizeof(int)*n);
    for(ui i = 0;i < n;i ++) {
        int td = degree_q[i];
        ++ degrees_cnt_q[td];
        if(td > max_degree_q) max_degree_q = td;
    }
    int *degrees_cnt_g = degree_q;
    memset(degrees_cnt_g, 0, sizeof(int)*g->n);
    for(ui i = 0;i < g->n;i ++) {
        ui td = degree_g[i];
        ++ degrees_cnt_g[td];
        if(td > max_degree_g) max_degree_g = td;
    }
    ui de = 0, ie = 0;
    while(max_degree_q > 0&& max_degree_g > 0) {
        if(degrees_cnt_q[max_degree_q] == 0) {
            -- max_degree_q;
            continue;
        }
        if(degrees_cnt_g[max_degree_g] == 0) {
            -- max_degree_g;
            continue;
        }

        ui td = mmin(degrees_cnt_q[max_degree_q], degrees_cnt_g[max_degree_g]);
        if(max_degree_q > max_degree_g) de += td*(max_degree_q - max_degree_g);
        else ie += td*(max_degree_g - max_degree_q);
        degrees_cnt_q[max_degree_q] -= td;
        degrees_cnt_g[max_degree_g] -= td;
    }
    while(max_degree_q > 0) {
        de += max_degree_q*degrees_cnt_q[max_degree_q];
        -- max_degree_q;
    }
    while(max_degree_g > 0) {
        ie += max_degree_g*degrees_cnt_g[max_degree_g];
        -- max_degree_g;
    }
    de = (de+1)/2; ie = (ie+1)/2;

    ui edge_lb = de+ie;
    if(de*2 + g->m/2 > m/2&&de*2 + g->m/2 - m/2 > edge_lb) edge_lb = de*2 + g->m/2 - m/2;
    if(ie*2 + m/2 > g->m/2&&ie*2 + m/2 - g->m/2 > edge_lb) edge_lb = ie*2 + m/2 - g->m/2;
    if(lb + edge_lb > verify_upper_bound) return lb + edge_lb;

    ui common_elabel_cnt = 0;
    for(ui i = 0;i < pstarts[n];i ++) ++ elabel_cnt[elabels[i]];
    for(ui i = 0;i < g->pstarts[g->n];i ++) {
        ui el = g->elabels[i];
        if(elabel_cnt[el] > 0) {
            -- elabel_cnt[el];
            ++ common_elabel_cnt;
        }
    }
    for(ui i = 0;i < pstarts[n];i ++) elabel_cnt[elabels[i]] = 0;
    common_elabel_cnt /= 2;
    if(de + g->m/2 - common_elabel_cnt > edge_lb) edge_lb = de + g->m/2 - common_elabel_cnt;
    if(ie + m/2 - common_elabel_cnt > edge_lb) edge_lb = de + m/2 - common_elabel_cnt; // NOTE: 'de' should be 'ie' here (original author's typo); no impact when edge labels are uniform
    ui e_cnt = m;
    if(g->m > e_cnt) e_cnt = g->m;
    e_cnt /= 2;
    if(e_cnt - common_elabel_cnt > edge_lb) edge_lb = e_cnt - common_elabel_cnt;

    return lb + edge_lb;
}

int Graph::vertex_label_bound(Graph* g, size_t vlabel_count_size) {
    ui lb = (n > g->n) ? n : g->n;

    std::vector<int> vlabel_cnt(vlabel_count_size, 0);

    for (ui i = 0; i < n; i++) ++vlabel_cnt[vlabels[i]];
    for (ui i = 0; i < g->n; i++) {
        ui vl = g->vlabels[i];
        if (vlabel_cnt[vl] > 0) {
            --vlabel_cnt[vl];
            --lb;
        }
    }

    return lb;
}


int Graph::degree_difference_bound(Graph* g, size_t max_n) {
    // 初始化度数数组
    std::vector<int> degree_q(n, 0);
    std::vector<int> degree_g(g->n, 0);

    // 计算两个图中每个节点的度数
    for (ui i = 0; i < n; i++) degree_q[i] = pstarts[i + 1] - pstarts[i];
    for (ui i = 0; i < g->n; i++) degree_g[i] = g->pstarts[i + 1] - g->pstarts[i];

    // 统计度数频次
    std::vector<int> degrees_cnt_q(max_n, 0);
    std::vector<int> degrees_cnt_g(max_n, 0);
    int max_degree_q = 0, max_degree_g = 0;

    for (ui i = 0; i < n; i++) {
        int td = degree_q[i];
        ++degrees_cnt_q[td];
        if (td > max_degree_q) max_degree_q = td;
    }

    for (ui i = 0; i < g->n; i++) {
        int td = degree_g[i];
        ++degrees_cnt_g[td];
        if (td > max_degree_g) max_degree_g = td;
    }

    ui de = 0, ie = 0;
    int mdq = max_degree_q;
    int mdg = max_degree_g;

    while (mdq > 0 && mdg > 0) {
        if (degrees_cnt_q[mdq] == 0) {
            --mdq;
            continue;
        }
        if (degrees_cnt_g[mdg] == 0) {
            --mdg;
            continue;
        }

        ui td = std::min(degrees_cnt_q[mdq], degrees_cnt_g[mdg]);
        if (mdq > mdg)
            de += td * (mdq - mdg);
        else
            ie += td * (mdg - mdq);

        degrees_cnt_q[mdq] -= td;
        degrees_cnt_g[mdg] -= td;
    }

    while (mdq > 0) {
        de += mdq * degrees_cnt_q[mdq];
        --mdq;
    }
    while (mdg > 0) {
        ie += mdg * degrees_cnt_g[mdg];
        --mdg;
    }

    de = (de + 1) / 2;
    ie = (ie + 1) / 2;

    ui edge_lb = de + ie;

    // 调整 edge_lb
    if (de * 2 + g->m / 2 > m / 2 && de * 2 + g->m / 2 - m / 2 > edge_lb)
        edge_lb = de * 2 + g->m / 2 - m / 2;
    if (ie * 2 + m / 2 > g->m / 2 && ie * 2 + m / 2 - g->m / 2 > edge_lb)
        edge_lb = ie * 2 + m / 2 - g->m / 2;

    return edge_lb;
}


int Graph::edge_label_bound(Graph* g, size_t elabel_count_size) {
    std::vector<int> elabel_cnt(elabel_count_size, 0);
    ui common_elabel_cnt = 0;

    // 统计当前图的边标签计数
    for (ui i = 0; i < m; i++) ++elabel_cnt[elabels[i]];

    // 与另一个图的边标签进行匹配
    for (ui i = 0; i < g->m; i++) {
        ui el = g->elabels[i];
        if (elabel_cnt[el] > 0) {
            --elabel_cnt[el];
            ++common_elabel_cnt;
        }
    }

    common_elabel_cnt /= 2; // 因为每条边被计算了两次

    ui e_cnt = (m > g->m) ? m : g->m;
    e_cnt /= 2;

    ui edge_lb = e_cnt - common_elabel_cnt;

    return edge_lb;
}

int Graph::ged_lower_bound_filter(Graph* g, ui verify_upper_bound, size_t vlabel_count_size, size_t elabel_count_size, size_t max_n) {
    ui lb = size_based_bound(g);
    if (lb > verify_upper_bound) return lb;

    // 计算基于顶点标签的下界
    ui vertex_lb = vertex_label_bound(g, vlabel_count_size);
    lb = vertex_lb;
    if (lb > verify_upper_bound) return lb;

    // 计算基于度数差异的边编辑操作下界
    ui edge_lb_degree = degree_difference_bound(g, max_n);
    if (lb + edge_lb_degree > verify_upper_bound) return lb + edge_lb_degree;

    // 计算基于边标签的边编辑操作下界
    ui edge_lb_label = edge_label_bound(g, elabel_count_size);
    if (lb + edge_lb_label > verify_upper_bound) return lb + edge_lb_label;

    // 取较大的边编辑操作下界
    ui edge_lb = std::max(edge_lb_degree, edge_lb_label);

    return lb + edge_lb;
}

void Graph::print_graph() const {
    std::cout << "Graph ID: " << id << std::endl;
    std::cout << "Number of nodes: " << n << std::endl;
    std::cout << "Number of edges: " << m << std::endl;

    // 打印节点及其标签，尽量在一行内
    std::cout << "Nodes and their labels: ";
    for (ui i = 0; i < n; ++i) {
        std::cout << "(" << i << ": " << vlabels[i] << ") ";
    }
    std::cout << std::endl;

    // 打印边及其标签，尽量在一行内
    std::cout << "Edges and their labels: ";
    std::unordered_set<std::string> printed_edges; // 用于避免重复打印无向图的边
    for (ui i = 0; i < n; ++i) {
        for (ui j = pstarts[i]; j < pstarts[i + 1]; ++j) {
            ui u = i;
            ui v = edges[j];
            ui label = elabels[j];

            // 为了避免在无向图中重复打印边，可以按照节点编号排序
            ui min_node = std::min(u, v);
            ui max_node = std::max(u, v);
            std::string edge_str = "(" + std::to_string(min_node) + " -- " + std::to_string(max_node) + ": " + std::to_string(label) + ")";

            if (printed_edges.find(edge_str) == printed_edges.end()) {
                std::cout << edge_str << " ";
                printed_edges.insert(edge_str);
            }
        }
    }
    std::cout << std::endl;
}


void Graph::save_to_stream(std::ostream& os) const {
    os << "GraphDataStart\n";
    os << "ID: " << id << "\n";
    // std::cout << "ID: " << id << "\n";
    os << "NumNodes: " << n << "\n";
    os << "NumEdges: " << m << "\n";

    // 保存节点标签
    os << "NodeLabels\n";
    for (ui i = 0; i < n; ++i) {
        os << vlabels[i] << " ";
    }
    os << "\n";

    // 保存边信息
    os << "Edges\n";
    for (ui i = 0; i < n; ++i) {
        ui start = pstarts[i];
        ui end = pstarts[i + 1];
        for (ui j = start; j < end; ++j) {
            os << i << " " << edges[j] << " " << elabels[j] << "\n";
        }
    }

    os << "GraphDataEnd\n";
}


void Graph::load_from_stream(std::istream& is) {
    std::string line;
    auto getline_trim = [](std::istream& s, std::string& l) -> std::istream& {
        std::getline(s, l);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        return s;
    };
    while (getline_trim(is, line)) {
        if (line == "GraphDataStart") {
            break;
        }
    }

    // 读取 ID
    getline_trim(is, line);
    if (line.find("ID:") == 0) {
        id = line.substr(line.find(":") + 2);
    }
    getline_trim(is, line);
    if (line.find("NumNodes:") == 0) {
        n = std::stoul(line.substr(line.find(":") + 2));
    }
    getline_trim(is, line);
    if (line.find("NumEdges:") == 0) {
        m = std::stoul(line.substr(line.find(":") + 2));
    }

    // 分配内存
    pstarts = new ui[n + 1];
    vlabels = new ui[n];
    edges = new ui[m];
    elabels = new ui[m];

    // 读取节点标签
    getline_trim(is, line); // NodeLabels
    getline_trim(is, line);
    std::stringstream ss(line);
    for (ui i = 0; i < n; ++i) {
        ss >> vlabels[i];
    }

    // 初始化 pstarts
    for (ui i = 0; i <= n; ++i) {
        pstarts[i] = 0;
    }

    // 读取边信息
    getline_trim(is, line); // Edges
    ui edge_idx = 0;
    while (getline_trim(is, line) && line != "GraphDataEnd") {
        std::stringstream edge_ss(line);
        ui src, dst, label;
        edge_ss >> src >> dst >> label;

        // 保存边信息
        edges[edge_idx] = dst;
        elabels[edge_idx] = label;

        // 更新 pstarts
        pstarts[src + 1]++;
        edge_idx++;
    }

    // 累积 pstarts
    for (ui i = 1; i <= n; ++i) {
        pstarts[i] += pstarts[i - 1];
    }
}


std::string Graph::compute_wl_hash(ui iterations) const {
    // 初始化节点标签
    std::vector<std::string> labels(n);
    for (ui i = 0; i < n; ++i) {
        labels[i] = std::to_string(vlabels[i]);
    }

    // 迭代地更新节点标签
    for (ui iter = 0; iter < iterations; ++iter) {
        // 收集新的标签
        std::unordered_map<std::string, ui> label_counter;
        std::vector<std::string> new_labels(n);

        for (ui i = 0; i < n; ++i) {
            // 获取邻居的标签
            std::vector<std::string> neighbor_labels;
            for (ui j = pstarts[i]; j < pstarts[i + 1]; ++j) {
                ui neighbor = edges[j];
                neighbor_labels.push_back(labels[neighbor]);
            }

            // 排序邻居标签
            std::sort(neighbor_labels.begin(), neighbor_labels.end());

            // 生成新的标签
            std::stringstream ss;
            ss << labels[i]; // 当前节点的标签
            for (const auto& lbl : neighbor_labels) {
                ss << "_" << lbl;
            }

            std::string new_label = ss.str();
            new_labels[i] = new_label;

            // 统计新标签出现次数
            label_counter[new_label]++;
        }

        // 将新标签映射为紧凑的整数标签（哈希化）
        std::unordered_map<std::string, ui> label_mapping;
        ui label_id = 0;
        for (const auto& pair : label_counter) {
            label_mapping[pair.first] = label_id++;
        }

        // 更新节点标签
        for (ui i = 0; i < n; ++i) {
            labels[i] = std::to_string(label_mapping[new_labels[i]]);
        }
    }

    // 生成图的哈希值
    std::stringstream ss;
    std::vector<std::string> graph_labels = labels;
    std::sort(graph_labels.begin(), graph_labels.end());
    for (const auto& lbl : graph_labels) {
        ss << lbl << "_";
    }
    return ss.str();
}

int Graph::compute_mapping_cost(const Graph& other, const std::vector<std::pair<ui, ui>>& mapping, std::vector<EditOperation>& edit_operations) const {
    int cost = 0;

    // 创建节点映射的查找表
    std::unordered_map<ui, ui> this_to_other;
    std::unordered_map<ui, ui> other_to_this;

    // 记录未映射的节点并为其分配新的编号
    std::unordered_map<ui, ui> new_node_ids;
    ui next_new_node_id = n; // 从锚点图的节点数量开始



    if (n < other.n) {
        for (const auto& pair : mapping) {
            ui u = pair.first;
            ui v = pair.second;
            this_to_other[u] = v;
            other_to_this[v] = u;
        }
    } else {
        for (const auto& pair : mapping) {
            ui u = pair.first;
            ui v = pair.second;
            this_to_other[v] = u;   // 注意这里交换了 u 和 v
            other_to_this[u] = v;
        }
    }

    // 1. 计算节点编辑代价
    int node_substitution_cost = 0;
    int node_deletion_cost = 0;
    int node_insertion_cost = 0;

    // 1.1 节点替换和删除代价
    for (ui u = 0; u < n; ++u) {
        if (this_to_other.count(u)) {
            ui v = this_to_other[u];
            if (vlabels[u] != other.vlabels[v]) {
                node_substitution_cost += 1; // 节点标签不同，替换代价为 1
                // 记录节点替换操作
                edit_operations.push_back({EditOperation::NODE_SUBSTITUTION, static_cast<int>(u), 0, static_cast<int>(vlabels[u]), static_cast<int>(other.vlabels[v])});
            }
        } else {
            // 节点删除代价
            node_deletion_cost += 1;
            // 记录节点删除操作
            edit_operations.push_back({EditOperation::NODE_DELETION, static_cast<int>(u), 0, static_cast<int>(vlabels[u]), 0});
        }
    }

    // 1.2 节点插入代价
    for (ui v = 0; v < other.n; ++v) {
        if (!other_to_this.count(v)) {
            node_insertion_cost += 1;
            // // 直接使用目标图中的节点 ID
            // ui new_node_id = v;
            ui new_node_id = next_new_node_id++;
            new_node_ids[v] = new_node_id;
            // 记录节点插入操作
            edit_operations.push_back({EditOperation::NODE_INSERTION, static_cast<int>(new_node_id), 0, 0, static_cast<int>(other.vlabels[v])});
        }
    }


    cost += node_substitution_cost + node_deletion_cost + node_insertion_cost;

    // 2. 计算边编辑代价
    int edge_substitution_cost = 0;
    int edge_deletion_cost = 0;
    int edge_insertion_cost = 0;

    // 使用集合避免双重计数
    struct pair_hash {
        size_t operator()(const std::pair<ui, ui>& p) const {
            return std::hash<ui>()(p.first) ^ std::hash<ui>()(p.second);
        }
    };

    std::unordered_set<std::pair<ui, ui>, pair_hash> processed_edges;

    // 2.1 处理当前图中的边（边替换和删除代价）
    for (ui u = 0; u < n; ++u) {
        for (ui idx = pstarts[u]; idx < pstarts[u + 1]; ++idx) {
            ui u_neighbor = edges[idx];

            // 避免双重计数
            ui u_min = std::min(u, u_neighbor);
            ui u_max = std::max(u, u_neighbor);
            if (processed_edges.count({u_min, u_max}) > 0) continue;
            processed_edges.insert({u_min, u_max});

            ui edge_label = elabels[idx];

            if (this_to_other.count(u) && this_to_other.count(u_neighbor)) {
                ui v = this_to_other[u];
                ui v_neighbor = this_to_other[u_neighbor];
                // 检查 other 图中是否存在对应的边
                bool edge_found = false;
                ui other_edge_label = 0;
                for (ui idx2 = other.pstarts[v]; idx2 < other.pstarts[v + 1]; ++idx2) {
                    if (other.edges[idx2] == v_neighbor) {
                        edge_found = true;
                        other_edge_label = other.elabels[idx2];
                        break;
                    }
                }
                if (!edge_found) {
                    edge_deletion_cost += 1; // 边删除代价
                    // 记录边删除操作
                    edit_operations.push_back({EditOperation::EDGE_DELETION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(edge_label), 0});
                } else if (edge_label != other_edge_label) {
                    edge_substitution_cost += 1; // 边替换代价
                    // 记录边替换操作
                    edit_operations.push_back({EditOperation::EDGE_SUBSTITUTION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(edge_label), static_cast<int>(other_edge_label)});
                }
                // 如果边存在且标签相同，不需要增加代价
            } else {
                edge_deletion_cost += 1; // 边删除代价
                // 记录边删除操作
                edit_operations.push_back({EditOperation::EDGE_DELETION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(edge_label), 0});
            }
        }
    }

    // 2.2 处理 other 图中的边（边插入代价）
    processed_edges.clear();
    for (ui v = 0; v < other.n; ++v) {
        ui u_mapped = other_to_this.count(v) ? other_to_this[v] : new_node_ids[v];
        for (ui idx = other.pstarts[v]; idx < other.pstarts[v + 1]; ++idx) {
            ui v_neighbor = other.edges[idx];
            ui u_neighbor_mapped = other_to_this.count(v_neighbor) ? other_to_this[v_neighbor] : new_node_ids[v_neighbor];

            // 避免双重计数
            ui u_min = std::min(u_mapped, u_neighbor_mapped);
            ui u_max = std::max(u_mapped, u_neighbor_mapped);
            if (processed_edges.count({u_min, u_max}) > 0) continue;
            processed_edges.insert({u_min, u_max});

            ui edge_label = other.elabels[idx];

            if (u_mapped < n && u_neighbor_mapped < n) {
                // 检查当前图中是否存在对应的边
                bool edge_found = false;
                ui this_edge_label = 0;
                for (ui idx2 = pstarts[u_mapped]; idx2 < pstarts[u_mapped + 1]; ++idx2) {
                    if (edges[idx2] == u_neighbor_mapped) {
                        edge_found = true;
                        this_edge_label = elabels[idx2];
                        break;
                    }
                }
                if (!edge_found) {
                    edge_insertion_cost += 1; // 边插入代价
                    // 记录边插入操作
                    edit_operations.push_back({EditOperation::EDGE_INSERTION, static_cast<int>(u_min), static_cast<int>(u_max), 0, static_cast<int>(edge_label)});
                } else if (this_edge_label != edge_label) {
                    edge_substitution_cost += 1; // 边替换代价
                    // 记录边替换操作
                    edit_operations.push_back({EditOperation::EDGE_SUBSTITUTION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(this_edge_label), static_cast<int>(edge_label)});
                }
                // 如果边存在且标签相同，不需要增加代价
            } else {
                edge_insertion_cost += 1; // 边插入代价
                // 记录边插入操作
                edit_operations.push_back({EditOperation::EDGE_INSERTION, static_cast<int>(u_min), static_cast<int>(u_max), 0, static_cast<int>(edge_label)});
            }
        }
    }

    cost += edge_substitution_cost + edge_deletion_cost + edge_insertion_cost;

    return cost;
}




void Graph::initialize_vectors_from_arrays() {
    // 初始化节点标签向量
    adjacency_list.clear();  
    adjacency_list.resize(n);
    
    vlabels_vec.resize(n);
    for (ui i = 0; i < n; ++i) {
        vlabels_vec[i] = vlabels[i];
    }

    // 初始化邻接列表
    adjacency_list.resize(n);
    for (ui i = 0; i < n; ++i) {
        ui start = pstarts[i];
        ui end = pstarts[i + 1];
        for (ui j = start; j < end; ++j) {
            ui neighbor = edges[j];
            ui edge_label = elabels[j];
            adjacency_list[i].emplace_back(neighbor, edge_label);
        }
    }
}




struct DirectedEdge {
    ui u;
    ui v;

    // 只要 (u,v) 都相同就视为同一条边
    bool operator<(const DirectedEdge &other) const {
        if (u < other.u) return true;
        if (u > other.u) return false;
        // u == other.u
        return v < other.v;
    }
};



PseudoGraph::PseudoGraph() {}

PseudoGraph::PseudoGraph(const Graph& graph) {
    id = graph.id;
    // std::cout << "Constructing PseudoGraph from Graph with n = " << graph.n << std::endl;
    if (graph.n == 0) {
        std::cerr << "Error: Graph has no nodes." << std::endl;
        return;
    }
    if (graph.vlabels == nullptr) {
        std::cerr << "Error: graph.vlabels is null." << std::endl;
        return;
    }
    if (graph.pstarts == nullptr || graph.edges == nullptr || graph.elabels == nullptr) {
        std::cerr << "Error: Graph edge data is null." << std::endl;
        return;
    }
    for (ui i = 0; i < graph.n; ++i) {
        vlabels[i] = graph.vlabels[i];
    }
    for (ui u = 0; u < graph.n; ++u) {
        std::vector<std::pair<ui, ui>> neighbors;
        for (ui idx = graph.pstarts[u]; idx < graph.pstarts[u + 1]; ++idx) {
            ui v = graph.edges[idx];
            ui e_label = graph.elabels[idx];
            neighbors.emplace_back(v, e_label);
        }
        adjacency_list[u] = neighbors;
    }
}


void PseudoGraph::apply_edit_operation(const EditOperation& op) {
    switch (op.type) {
        case EditOperation::NODE_INSERTION: {
            ui node_id = op.u;
            if (vlabels.count(node_id)) {
                std::cerr << "Error: Node Insertion: Node ID " << node_id << " already exists." << std::endl;
                return;
            }
            vlabels[node_id] = op.new_label;
            adjacency_list[node_id] = {}; // 新节点的邻接列表为空
            break;
        }
        case EditOperation::NODE_DELETION: {
            ui node_id = op.u;
            if (!vlabels.count(node_id)) {
                std::cerr << "Error: Node Deletion: Node ID " << node_id << " does not exist." << std::endl;
                return;
            }
            vlabels.erase(node_id);
            adjacency_list.erase(node_id);
            // 删除与该节点相连的边
            for (auto& [u, neighbors] : adjacency_list) {
                neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                               [node_id](const std::pair<ui, ui>& p) { return p.first == node_id; }),
                                neighbors.end());
            }
            break;
        }
        case EditOperation::NODE_SUBSTITUTION: {
            ui node_id = op.u;
            if (!vlabels.count(node_id)) {
                std::cerr << "Error: Node substitution: Node ID " << node_id << " does not exist." << std::endl;
                return;
            }
            vlabels[node_id] = op.new_label;
            break;
        }
        case EditOperation::EDGE_INSERTION: {
            ui u = op.u;
            ui v = op.v;
            if (!vlabels.count(u) || !vlabels.count(v)) {
                std::cerr << "Error: Edge Insertion: Node ID out of range." << std::endl;
                return;
            }
            // 检查是否已经存在边 u - v，避免重复插入
            auto& neighbors_u = adjacency_list[u];
            auto it_u = std::find_if(neighbors_u.begin(), neighbors_u.end(),
                                     [v](const std::pair<ui, ui>& p) { return p.first == v; });
            if (it_u == neighbors_u.end()) {
                neighbors_u.emplace_back(v, op.new_label);
            }

            auto& neighbors_v = adjacency_list[v];
            auto it_v = std::find_if(neighbors_v.begin(), neighbors_v.end(),
                                     [u](const std::pair<ui, ui>& p) { return p.first == u; });
            if (it_v == neighbors_v.end()) {
                neighbors_v.emplace_back(u, op.new_label);
            }
            break;
        }
        case EditOperation::EDGE_DELETION: {
            ui u = op.u;
            ui v = op.v;
            if (!adjacency_list.count(u) || !adjacency_list.count(v)) {
                std::cerr << "Error: Edge Deletion: Node ID out of range." << std::endl;
                return;
            }
            auto& neighbors_u = adjacency_list[u];
            neighbors_u.erase(std::remove_if(neighbors_u.begin(), neighbors_u.end(),
                                             [v](const std::pair<ui, ui>& p) { return p.first == v; }),
                              neighbors_u.end());

            auto& neighbors_v = adjacency_list[v];
            neighbors_v.erase(std::remove_if(neighbors_v.begin(), neighbors_v.end(),
                                             [u](const std::pair<ui, ui>& p) { return p.first == u; }),
                              neighbors_v.end());
            break;
        }
        case EditOperation::EDGE_SUBSTITUTION: {
            ui u = op.u;
            ui v = op.v;
            if (!adjacency_list.count(u) || !adjacency_list.count(v)) {
                std::cerr << "Error: Edge Substitution: Node ID out of range." << std::endl;
                return;
            }
            bool found = false;
            for (auto& p : adjacency_list[u]) {
                if (p.first == v) {
                    p.second = op.new_label;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Error: Edge (" << u << ", " << v << ") not found." << std::endl;
                return;
            }
            // 对于无向图，同样更新 v 的邻接列表
            found = false;
            for (auto& p : adjacency_list[v]) {
                if (p.first == u) {
                    p.second = op.new_label;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Error: Edge (" << v << ", " << u << ") not found." << std::endl;
                return;
            }
            break;
        }
        default:
            std::cerr << "Error: Unknown edit operation type." << std::endl;
            break;
    }
    // 输出当前图的状态
    // std::cout << "After applying operation, PseudoGraph has " << vlabels.size() << " nodes and " << adjacency_list.size() << " adjacency entries." << std::endl;
}

Graph PseudoGraph::to_graph() const {
    Graph g;
    // 如果 id 为空，则设置一个默认值
    if (id.empty()) {
        g.id = "PseudoGraph_" + std::to_string(vlabels.size());
    } else {
        g.id = id;
    }

    // 如果没有节点，返回空图
    if (vlabels.empty()) {
        std::cerr << "Error: PseudoGraph has no nodes." << std::endl;
        g.n = 0;
        g.m = 0;
        g.vlabels = nullptr;
        g.pstarts = nullptr;
        g.edges = nullptr;
        g.elabels = nullptr;
        return g;
    }

    // 收集所有节点 ID
    std::vector<ui> node_ids;
    node_ids.reserve(vlabels.size());
    for (const auto& kv : vlabels) {
        node_ids.push_back(kv.first);
    }
    // 对节点 ID 进行排序
    std::sort(node_ids.begin(), node_ids.end());

    // 建立从原node_id到新索引的映射
    ui n = (ui)vlabels.size();
    std::vector<int> id_to_index(node_ids.back() + 1, -1);
    for (ui i = 0; i < n; i++) {
        ui node_id = node_ids[i];
        id_to_index[node_id] = (int)i;
    }

    g.n = n;
    g.vlabels = new ui[n];

    // 将 vlabels 填入新图
    for (ui i = 0; i < n; i++) {
        ui node_id = node_ids[i];
        g.vlabels[i] = vlabels.at(node_id);
    }

    // 统计总边数（无向图中每条边出现两次，此处暂时不用去重，因为我们只是将PseudoGraph直接映射）
    ui total_edges = 0;
    for (const auto& [node_id, neighbors] : adjacency_list) {
        total_edges += (ui)neighbors.size();
    }
    g.m = total_edges;

    if (g.m == 0) {
        std::cerr << "Warning: PseudoGraph has no edges." << std::endl;
    }

    g.pstarts = new ui[g.n + 1];
    g.edges = new ui[g.m];
    g.elabels = new ui[g.m];

    // 初始化 pstarts
    for (ui i = 0; i <= g.n; i++) {
        g.pstarts[i] = 0;
    }

    // 计算每个节点的边数量以填充 pstarts（这里的边是双向存储的）
    std::vector<ui> deg(g.n, 0);
    for (const auto& [node_id, neighbors] : adjacency_list) {
        int new_id = id_to_index[node_id];
        if (new_id >= 0) {
            deg[new_id] = (ui)neighbors.size();
        }
    }

    // prefix sum
    g.pstarts[0] = 0;
    for (ui i = 1; i <= g.n; i++) {
        g.pstarts[i] = g.pstarts[i-1] + deg[i-1];
    }

    std::vector<ui> current_pos(g.n, 0);
    for (ui i = 0; i < g.n; i++) {
        current_pos[i] = g.pstarts[i];
    }

    // 填充 edges 和 elabels
    for (const auto& [node_id, neighbors] : adjacency_list) {
        int u_new = id_to_index[node_id];
        if (u_new >= 0) {
            for (const auto& [nbr_id, e_label] : neighbors) {
                int v_new = id_to_index[nbr_id];
                if (v_new < 0) {
                    std::cerr << "Error: neighbor id not found in vlabels." << std::endl;
                    continue;
                }
                ui pos = current_pos[u_new]++;
                g.edges[pos] = (ui)v_new;
                g.elabels[pos] = e_label;
            }
        }
    }

    // 检查 edge 数量是否匹配
    if (current_pos.back() != g.m) {
        std::cerr << "Error: Edge count mismatch in to_graph(). Expected " << g.m << " got " << current_pos.back() << std::endl;
    }

    return g;
}


bool PseudoGraph::can_apply_operation(const EditOperation& op) const {
    switch (op.type) {
        case EditOperation::NODE_INSERTION:
            // 检查节点是否已存在
            return vlabels.find(op.u) == vlabels.end();

        case EditOperation::NODE_DELETION:
            // 检查节点是否存在
            if (vlabels.find(op.u) == vlabels.end()) {
                return false;
            }

            // 检查节点是否有相连的边
            {
                auto it = adjacency_list.find(op.u);
                if (it != adjacency_list.end() && !it->second.empty()) {
                    // 节点存在相连的边，不能删除
                    // std::cerr << "Cannot delete node " << op.u << " because it has connected edges." << std::endl;
                    return false;
                }
            }

            // 节点没有相连的边，可以删除
            return true;

        case EditOperation::EDGE_INSERTION:
            // 检查边的两个节点是否存在，且边是否已存在
            return vlabels.find(op.u) != vlabels.end() &&
                   vlabels.find(op.v) != vlabels.end() &&
                   !edge_exists(op.u, op.v);

        case EditOperation::EDGE_DELETION:
            // 检查边是否存在
            return edge_exists(op.u, op.v);

        case EditOperation::NODE_SUBSTITUTION:
            // 检查节点是否存在
            return vlabels.find(op.u) != vlabels.end();

        case EditOperation::EDGE_SUBSTITUTION:
            // 检查边是否存在
            return edge_exists(op.u, op.v);

        default:
            return false;
    }
}


bool PseudoGraph::edge_exists(ui u, ui v) const {
    auto it = adjacency_list.find(u);
    if (it != adjacency_list.end()) {
        for (const auto& neighbor : it->second) {
            if (neighbor.first == v) {
                return true;
            }
        }
    }
    return false;
}

void PseudoGraph::print_pseudo_graph() const {
    // 收集所有节点 ID，并进行排序，以便有序打印
    std::vector<ui> node_ids;
    node_ids.reserve(vlabels.size());
    for (const auto& kv : vlabels) {
        node_ids.push_back(kv.first);
    }
    std::sort(node_ids.begin(), node_ids.end());

    // 统计节点和边数量
    ui n = static_cast<ui>(node_ids.size());

    // 边数统计：对每个节点的邻接列表求和，然后除以2（无向图）
    ui total_edges = 0;
    for (auto node_id : node_ids) {
        auto it = adjacency_list.find(node_id);
        if (it != adjacency_list.end()) {
            total_edges += static_cast<ui>(it->second.size());
        }
    }
    // 每条无向边在两个节点的邻接表中都出现一次，所以除以2
    ui m = total_edges / 2;

    std::cout << "PseudoGraph ID: " << id << std::endl;
    std::cout << "Number of nodes: " << n << std::endl;
    std::cout << "Number of edges: " << m << std::endl;

    // 打印节点及其标签
    std::cout << "Nodes and their labels: ";
    for (auto node_id : node_ids) {
        ui label = vlabels.at(node_id);
        std::cout << "(" << node_id << ": " << label << ") ";
    }
    std::cout << std::endl;

    // 打印边及其标签
    std::cout << "Edges and their labels: ";
    std::unordered_set<std::string> printed_edges; // 用于避免重复打印无向图的边
    for (auto u_id : node_ids) {
        auto it = adjacency_list.find(u_id);
        if (it == adjacency_list.end()) continue; // 没有邻接节点
        for (const auto& nbr : it->second) {
            ui v_id = nbr.first;
            ui label = nbr.second;
            // 为了避免重复打印无向边，将 (u,v) 排序
            ui min_node = std::min(u_id, v_id);
            ui max_node = std::max(u_id, v_id);
            std::string edge_str = "(" + std::to_string(min_node) + " -- " 
                                        + std::to_string(max_node) + ": " 
                                        + std::to_string(label) + ")";

            if (printed_edges.find(edge_str) == printed_edges.end()) {
                std::cout << edge_str << " ";
                printed_edges.insert(edge_str);
            }
        }
    }
    std::cout << std::endl;
}



