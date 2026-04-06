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
// Constructor
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

    // Deep copy pstarts
    pstarts = new ui[n + 1];
    memcpy(pstarts, other.pstarts, (n + 1) * sizeof(ui));

    // Deep copy edges
    ui edge_count = pstarts[n];
    edges = new ui[edge_count];
    memcpy(edges, other.edges, edge_count * sizeof(ui));

    // Deep copy vlabels
    vlabels = new ui[n];
    memcpy(vlabels, other.vlabels, n * sizeof(ui));

    // Deep copy elabels
    elabels = new ui[edge_count];
    memcpy(elabels, other.elabels, edge_count * sizeof(ui));
}

Graph& Graph::operator=(const Graph& other) {
    if (this != &other) {
        // Free existing memory
        delete[] pstarts;
        delete[] edges;
        delete[] vlabels;
        delete[] elabels;

        id = other.id;
        n = other.n;
        m = other.m;

        // Deep copy pstarts
        pstarts = new ui[n + 1];
        memcpy(pstarts, other.pstarts, (n + 1) * sizeof(ui));

        // Deep copy edges
        ui edge_count = pstarts[n];
        edges = new ui[edge_count];
        memcpy(edges, other.edges, edge_count * sizeof(ui));

        // Deep copy vlabels
        vlabels = new ui[n];
        memcpy(vlabels, other.vlabels, n * sizeof(ui));

        // Deep copy elabels
        elabels = new ui[edge_count];
        memcpy(elabels, other.elabels, edge_count * sizeof(ui));
    }
    return *this;
}


// Destructor
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

void Graph::write_graph(FILE* fout, const std::vector<std::string>& _vlabels, const std::vector<std::string>& _elabels, bool bss) {
    assert(fout != NULL);
    if (bss) {
        for (ui i = 0; i < id.length(); i++) if (id[i] < '0' || id[i] > '9') printf("!!! Wrong graph id for bss\n");
        fprintf(fout, "%s\n", id.c_str());
        fprintf(fout, "%d %d\n", n, m / 2);
        for (ui i = 0; i < n; i++) fprintf(fout, "%d\n", vlabels[i]);
        for (ui i = 0; i < n; i++) for (ui j = pstarts[i]; j < pstarts[i + 1]; j++) if (edges[j] > i) {
            fprintf(fout, "%u %u %d\n", i, edges[j], elabels[j]);
        }
    }
    else {
        fprintf(fout, "t # %s\n", id.c_str());
        for (ui i = 0; i < n; i++) fprintf(fout, "v %u %s\n", i, _vlabels[vlabels[i]].c_str());
        for (ui i = 0; i < n; i++) for (ui j = pstarts[i]; j < pstarts[i + 1]; j++) if (edges[j] > i) {
            fprintf(fout, "e %u %u %s\n", i, edges[j], _elabels[elabels[j]].c_str());
        }
    }
}

bool Graph::is_connected() {
    std::vector<ui> Q;
    char* vis = new char[n];
    memset(vis, 0, sizeof(char) * n);
    vis[0] = 1;
    Q.push_back(0);
    for (ui i = 0; i < Q.size(); i++) {
        for (ui j = pstarts[Q[i]]; j < pstarts[Q[i] + 1]; j++) if (!vis[edges[j]]) {
            vis[edges[j]] = 1;
            Q.push_back(edges[j]);
        }
    }

    delete[] vis;

    if (Q.size() == n) return true;
    return false;
}

int Graph::size_based_bound(Graph* g) {
    ui r1 = n > g->n ? n - g->n : g->n - n;
    ui r2 = m > g->m ? m - g->m : g->m - m;
    assert(r2 % 1 == 0);
    return r1 + r2 / 2;
}
// int Graph::size_based_bound(Graph* g) {
//     ui r1 = abs((int)g->n-(int)n); // n > gg.n ? n - g->n : g->n - n;
//     ui r2 = abs((int)g->m-(int)m); //m > g->m ? m - g->m : g->m - m;
//     assert(r2 % 1 == 0);
//     return r1 + r2 / 2;
// }





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
    // Initialize degree arrays
    std::vector<int> degree_q(n, 0);
    std::vector<int> degree_g(g->n, 0);

    // Compute the degree of each vertex in both graphs
    for (ui i = 0; i < n; i++) degree_q[i] = pstarts[i + 1] - pstarts[i];
    for (ui i = 0; i < g->n; i++) degree_g[i] = g->pstarts[i + 1] - g->pstarts[i];

    // Count degree frequencies
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

    // Adjust edge_lb
    if (de * 2 + g->m / 2 > m / 2 && de * 2 + g->m / 2 - m / 2 > edge_lb)
        edge_lb = de * 2 + g->m / 2 - m / 2;
    if (ie * 2 + m / 2 > g->m / 2 && ie * 2 + m / 2 - g->m / 2 > edge_lb)
        edge_lb = ie * 2 + m / 2 - g->m / 2;

    return edge_lb;
}

// int Graph::degree_difference_bound(Graph* g, size_t max_n) {
//     // Initialize degree arrays
//     std::vector<int> degree_q(n, 0);
//     std::vector<int> degree_g(g->n, 0);

//     // Compute the degree of each vertex in both graphs
//     for (ui i = 0; i < n; i++) degree_q[i] = pstarts[i + 1] - pstarts[i];
//     for (ui i = 0; i < g->n; i++) degree_g[i] = g->pstarts[i + 1] - g->pstarts[i];

//     // Count degree frequencies
//     std::vector<int> degrees_cnt_q(max_n, 0);
//     std::vector<int> degrees_cnt_g(max_n, 0);
//     int max_degree_q = 0, max_degree_g = 0;

//     for (ui i = 0; i < n; i++) {
//         int td = degree_q[i];
//         ++degrees_cnt_q[td];
//         if (td > max_degree_q) max_degree_q = td;
//     }

//     for (ui i = 0; i < g->n; i++) {
//         int td = degree_g[i];
//         ++degrees_cnt_g[td];
//         if (td > max_degree_g) max_degree_g = td;
//     }

//     ui de = 0, ie = 0;
//     int mdq = max_degree_q;
//     int mdg = max_degree_g;

//     while (mdq > 0 && mdg > 0) {
//         if (degrees_cnt_q[mdq] == 0) {
//             --mdq;
//             continue;
//         }
//         if (degrees_cnt_g[mdg] == 0) {
//             --mdg;
//             continue;
//         }

//         ui td = std::min(degrees_cnt_q[mdq], degrees_cnt_g[mdg]);
//         if (mdq > mdg)
//             de += td * (mdq - mdg);
//         else
//             ie += td * (mdg - mdq);

//         degrees_cnt_q[mdq] -= td;
//         degrees_cnt_g[mdg] -= td;
//     }

//     while (mdq > 0) {
//         de += mdq * degrees_cnt_q[mdq];
//         --mdq;
//     }
//     while (mdg > 0) {
//         ie += mdg * degrees_cnt_g[mdg];
//         --mdg;
//     }

//     de = (de + 1) / 2;
//     ie = (ie + 1) / 2;

//     ui edge_lb = de + ie;

//     // Adjust edge_lb
//     if (de * 2 + g->m / 2 > m / 2 && de * 2 + g->m / 2 - m / 2 > edge_lb)
//         edge_lb = de * 2 + g->m / 2 - m / 2;
//     if (ie * 2 + m / 2 > g->m / 2 && ie * 2 + m / 2 - g->m / 2 > edge_lb)
//         edge_lb = ie * 2 + m / 2 - g->m / 2;

//     return edge_lb;
// }

int Graph::edge_label_bound(Graph* g, size_t elabel_count_size) {
    std::vector<int> elabel_cnt(elabel_count_size, 0);
    ui common_elabel_cnt = 0;

    // Count edge labels of the current graph
    for (ui i = 0; i < m; i++) ++elabel_cnt[elabels[i]];

    // Match edge labels with the other graph
    for (ui i = 0; i < g->m; i++) {
        ui el = g->elabels[i];
        if (elabel_cnt[el] > 0) {
            --elabel_cnt[el];
            ++common_elabel_cnt;
        }
    }

    common_elabel_cnt /= 2; // each edge was counted twice

    ui e_cnt = (m > g->m) ? m : g->m;
    e_cnt /= 2;

    ui edge_lb = e_cnt - common_elabel_cnt;

    return edge_lb;
}

int Graph::ged_lower_bound_filter(Graph* g, ui verify_upper_bound, size_t vlabel_count_size, size_t elabel_count_size, size_t max_n) {
    ui lb = size_based_bound(g);
    if (lb > verify_upper_bound) return lb;

    // Compute vertex-label-based lower bound
    ui vertex_lb = vertex_label_bound(g, vlabel_count_size);
    lb = vertex_lb;
    if (lb > verify_upper_bound) return lb;

    // Compute edge-edit lower bound based on degree differences
    ui edge_lb_degree = degree_difference_bound(g, max_n);
    if (lb + edge_lb_degree > verify_upper_bound) return lb + edge_lb_degree;

    // Compute edge-edit lower bound based on edge labels
    ui edge_lb_label = edge_label_bound(g, elabel_count_size);
    if (lb + edge_lb_label > verify_upper_bound) return lb + edge_lb_label;

    // Take the larger edge-edit lower bound
    ui edge_lb = std::max(edge_lb_degree, edge_lb_label);

    return lb + edge_lb;
}

int Graph::ged_lower_bound_filter_ori(Graph *g, ui verify_upper_bound, size_t vlabel_count_size, size_t elabel_count_size, size_t max_n) {
    ui lb = size_based_bound(g);
    if(lb > verify_upper_bound) return lb;
    ui final_lb = lb;
    
    lb = (n > g->n ? n : g->n);

    // Create local arrays inside the function
    std::vector<int> vlabel_cnt(vlabel_count_size, 0);
    std::vector<int> elabel_cnt(elabel_count_size, 0);
    std::vector<int> degree_q(max_n, 0);
    std::vector<int> degree_g(max_n, 0);
    std::vector<int> tmp(n, 0);

    // Count vertex labels of the query graph
    for(ui i = 0; i < n; i++) ++vlabel_cnt[vlabels[i]];
    for(ui i = 0; i < g->n; i++) {
        ui vl = g->vlabels[i];
        if(vlabel_cnt[vl] > 0) {
            --vlabel_cnt[vl];
            --lb;
        }
    }
    // No need to reset vlabel_cnt since it is a local variable
    if(lb > verify_upper_bound) return lb;
    
    assert(m == pstarts[n] && g->m == g->pstarts[g->n]);
    for(ui i = 0; i < n; i++) degree_q[i] = pstarts[i+1] - pstarts[i];
    for(ui i = 0; i < g->n; i++) degree_g[i] = g->pstarts[i+1] - g->pstarts[i];

    int *degrees_cnt_q = tmp.data();
    int max_degree_q = 0, max_degree_g = 0;
    memset(degrees_cnt_q, 0, sizeof(int)*n);
    for(ui i = 0; i < n; i++) {
        int td = degree_q[i];
        ++degrees_cnt_q[td];
        if(td > max_degree_q) max_degree_q = td;
    }

    int *degrees_cnt_g = degree_q.data();
    memset(degrees_cnt_g, 0, sizeof(int)*g->n);
    for(ui i = 0; i < g->n; i++) {
        ui td = degree_g[i];
        ++degrees_cnt_g[td];
        if(td > max_degree_g) max_degree_g = td;
    }

    ui de = 0, ie = 0;
    while(max_degree_q > 0 && max_degree_g > 0) {
        if(degrees_cnt_q[max_degree_q] == 0) {
            --max_degree_q;
            continue;
        }
        if(degrees_cnt_g[max_degree_g] == 0) {
            --max_degree_g;
            continue;
        }

        ui td = std::min(degrees_cnt_q[max_degree_q], degrees_cnt_g[max_degree_g]);
        if(max_degree_q > max_degree_g) de += td * (max_degree_q - max_degree_g);
        else ie += td * (max_degree_g - max_degree_q);
        degrees_cnt_q[max_degree_q] -= td;
        degrees_cnt_g[max_degree_g] -= td;
    }
    while(max_degree_q > 0) {
        de += max_degree_q * degrees_cnt_q[max_degree_q];
        --max_degree_q;
    }
    while(max_degree_g > 0) {
        ie += max_degree_g * degrees_cnt_g[max_degree_g];
        --max_degree_g;
    }
    de = (de + 1) / 2; ie = (ie + 1) / 2;

    ui edge_lb = de + ie;
    if(de * 2 + g->m / 2 > m / 2 && de * 2 + g->m / 2 - m / 2 > edge_lb)
        edge_lb = de * 2 + g->m / 2 - m / 2;
    if(ie * 2 + m / 2 > g->m / 2 && ie * 2 + m / 2 - g->m / 2 > edge_lb)
        edge_lb = ie * 2 + m / 2 - g->m / 2;
    if(lb + edge_lb > verify_upper_bound) return lb + edge_lb;

    ui common_elabel_cnt = 0;
    for(ui i = 0; i < pstarts[n]; i++) ++elabel_cnt[elabels[i]];
    for(ui i = 0; i < g->pstarts[g->n]; i++) {
        ui el = g->elabels[i];
        if(elabel_cnt[el] > 0) {
            --elabel_cnt[el];
            ++common_elabel_cnt;
        }
    }
    // No need to reset elabel_cnt since it is a local variable
    common_elabel_cnt /= 2;
    if(de + g->m / 2 - common_elabel_cnt > edge_lb)
        edge_lb = de + g->m / 2 - common_elabel_cnt;
    if(ie + m / 2 - common_elabel_cnt > edge_lb)
        edge_lb = de + m / 2 - common_elabel_cnt;
    ui e_cnt = m;
    if(g->m > e_cnt) e_cnt = g->m;
    e_cnt /= 2;
    if(e_cnt - common_elabel_cnt > edge_lb)
        edge_lb = e_cnt - common_elabel_cnt;

    return lb + edge_lb;
}

void Graph::print_graph() const {
    std::cout << "Graph ID: " << id << std::endl;
    std::cout << "Number of nodes: " << n << std::endl;
    std::cout << "Number of edges: " << m << std::endl;

    // Print vertices and their labels, preferably on one line
    std::cout << "Nodes and their labels: ";
    for (ui i = 0; i < n; ++i) {
        std::cout << "(" << i << ": " << vlabels[i] << ") ";
    }
    std::cout << std::endl;

    // Print edges and their labels, preferably on one line
    std::cout << "Edges and their labels: ";
    std::unordered_set<std::string> printed_edges; // avoid duplicate printing of undirected edges
    for (ui i = 0; i < n; ++i) {
        for (ui j = pstarts[i]; j < pstarts[i + 1]; ++j) {
            ui u = i;
            ui v = edges[j];
            ui label = elabels[j];

            // To avoid duplicate edge printing in undirected graph, sort by vertex id
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

    // Save vertex labels
    os << "NodeLabels\n";
    for (ui i = 0; i < n; ++i) {
        os << vlabels[i] << " ";
    }
    os << "\n";

    // Save edge information
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

    // Read ID
    getline_trim(is, line);
    if (line.find("ID:") == 0) {
        id = line.substr(line.find(":") + 2);
    }
    // if (!std::getline(is, line)) {
    //     std::cerr << "Error: Missing ID line." << std::endl;
    //     id = static_cast<ui>(-1);
    //     return;
    // }

    // if (line.find("ID:") == 0) {
    //     // Extract the part after ':' in the ID line
    //     size_t colon_pos = line.find(":");
    //     std::string id_str;
    //     if (colon_pos != std::string::npos) {
    //         id_str = line.substr(colon_pos + 1);
    //     } else {
    //         id_str.clear();
    //     }

    //     // Strip leading whitespace from id_str
    //     id_str.erase(0, id_str.find_first_not_of(" \t\r\n"));

    //     if (id_str.empty()) {
    //         // If no number after ID line, set to -1
    //         id = static_cast<ui>(-1);
    //     } else {
    //         // Try parsing as number, set to -1 on failure
    //         try {
    //             id = std::stoul(id_str);
    //         } catch (...) {
    //             id = static_cast<ui>(-1);
    //         }
    //     }
    // } else {
    //     // If no ID line or format mismatch, set to -1
    //     id = static_cast<ui>(-1);
    // }

    // Read vertex and edge counts
    getline_trim(is, line);
    if (line.find("NumNodes:") == 0) {
        n = std::stoul(line.substr(line.find(":") + 2));
    }
    getline_trim(is, line);
    if (line.find("NumEdges:") == 0) {
        m = std::stoul(line.substr(line.find(":") + 2));
    }

    // Allocate memory
    pstarts = new ui[n + 1];
    vlabels = new ui[n];
    edges = new ui[m];
    elabels = new ui[m];

    // Read vertex labels
    getline_trim(is, line); // NodeLabels
    getline_trim(is, line);
    std::stringstream ss(line);
    for (ui i = 0; i < n; ++i) {
        ss >> vlabels[i];
    }

    // Initialize pstarts
    for (ui i = 0; i <= n; ++i) {
        pstarts[i] = 0;
    }

    // Read edge information
    getline_trim(is, line); // Edges
    ui edge_idx = 0;
    while (getline_trim(is, line) && line != "GraphDataEnd") {
        std::stringstream edge_ss(line);
        ui src, dst, label;
        edge_ss >> src >> dst >> label;

        // Save edge information
        edges[edge_idx] = dst;
        elabels[edge_idx] = label;

        // Update pstarts
        pstarts[src + 1]++;
        edge_idx++;
    }

    // Accumulate pstarts
    for (ui i = 1; i <= n; ++i) {
        pstarts[i] += pstarts[i - 1];
    }
}


std::string Graph::compute_wl_hash(ui iterations) const {
    // Initialize vertex labels
    std::vector<std::string> labels(n);
    for (ui i = 0; i < n; ++i) {
        labels[i] = std::to_string(vlabels[i]);
    }

    // Iteratively update vertex labels
    for (ui iter = 0; iter < iterations; ++iter) {
        // Collect new labels
        std::unordered_map<std::string, ui> label_counter;
        std::vector<std::string> new_labels(n);

        for (ui i = 0; i < n; ++i) {
            // Get neighbor labels
            std::vector<std::string> neighbor_labels;
            for (ui j = pstarts[i]; j < pstarts[i + 1]; ++j) {
                ui neighbor = edges[j];
                neighbor_labels.push_back(labels[neighbor]);
            }

            // Sort neighbor labels
            std::sort(neighbor_labels.begin(), neighbor_labels.end());

            // Generate new label
            std::stringstream ss;
            ss << labels[i]; // current vertex's label
            for (const auto& lbl : neighbor_labels) {
                ss << "_" << lbl;
            }

            std::string new_label = ss.str();
            new_labels[i] = new_label;

            // Count new label occurrences
            label_counter[new_label]++;
        }

        // Map new labels to compact integer labels (hashing)
        std::unordered_map<std::string, ui> label_mapping;
        ui label_id = 0;
        for (const auto& pair : label_counter) {
            label_mapping[pair.first] = label_id++;
        }

        // Update vertex labels
        for (ui i = 0; i < n; ++i) {
            labels[i] = std::to_string(label_mapping[new_labels[i]]);
        }
    }

    // Generate graph hash value
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

    // Create lookup table for vertex mapping
    std::unordered_map<ui, ui> this_to_other;
    std::unordered_map<ui, ui> other_to_this;

    // Record unmapped vertices and assign new ids
    std::unordered_map<ui, ui> new_node_ids;
    ui next_new_node_id = n; // start from anchor graph's vertex count



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
            this_to_other[v] = u;   // note: u and v are swapped here
            other_to_this[u] = v;
        }
    }

    // 1. Compute vertex edit cost
    int node_substitution_cost = 0;
    int node_deletion_cost = 0;
    int node_insertion_cost = 0;

    // 1.1 Vertex substitution and deletion cost
    for (ui u = 0; u < n; ++u) {
        if (this_to_other.count(u)) {
            ui v = this_to_other[u];
            if (vlabels[u] != other.vlabels[v]) {
                node_substitution_cost += 1; // different vertex labels, substitution cost is 1
                // Record vertex substitution operation
                edit_operations.push_back({EditOperation::NODE_SUBSTITUTION, static_cast<int>(u), 0, static_cast<int>(vlabels[u]), static_cast<int>(other.vlabels[v])});
            }
        } else {
            // Vertex deletion cost
            node_deletion_cost += 1;
            // Record vertex deletion operation
            edit_operations.push_back({EditOperation::NODE_DELETION, static_cast<int>(u), 0, static_cast<int>(vlabels[u]), 0});
        }
    }

    // 1.2 Vertex insertion cost
    for (ui v = 0; v < other.n; ++v) {
        if (!other_to_this.count(v)) {
            node_insertion_cost += 1;
            // // Directly use vertex ID from the target graph
            // ui new_node_id = v;
            ui new_node_id = next_new_node_id++;
            new_node_ids[v] = new_node_id;
            // Record vertex insertion operation
            edit_operations.push_back({EditOperation::NODE_INSERTION, static_cast<int>(new_node_id), 0, 0, static_cast<int>(other.vlabels[v])});
        }
    }


    cost += node_substitution_cost + node_deletion_cost + node_insertion_cost;

    // 2. Compute edge edit cost
    int edge_substitution_cost = 0;
    int edge_deletion_cost = 0;
    int edge_insertion_cost = 0;

    // Use set to avoid double counting
    struct pair_hash {
        size_t operator()(const std::pair<ui, ui>& p) const {
            return std::hash<ui>()(p.first) ^ std::hash<ui>()(p.second);
        }
    };

    std::unordered_set<std::pair<ui, ui>, pair_hash> processed_edges;

    // 2.1 Process edges in current graph (edge substitution and deletion cost)
    for (ui u = 0; u < n; ++u) {
        for (ui idx = pstarts[u]; idx < pstarts[u + 1]; ++idx) {
            ui u_neighbor = edges[idx];

            // Avoid double counting
            ui u_min = std::min(u, u_neighbor);
            ui u_max = std::max(u, u_neighbor);
            if (processed_edges.count({u_min, u_max}) > 0) continue;
            processed_edges.insert({u_min, u_max});

            ui edge_label = elabels[idx];

            if (this_to_other.count(u) && this_to_other.count(u_neighbor)) {
                ui v = this_to_other[u];
                ui v_neighbor = this_to_other[u_neighbor];
                // Check if the corresponding edge exists in the other graph
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
                    edge_deletion_cost += 1; // edge deletion cost
                    // Record edge deletion operation
                    edit_operations.push_back({EditOperation::EDGE_DELETION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(edge_label), 0});
                } else if (edge_label != other_edge_label) {
                    edge_substitution_cost += 1; // edge substitution cost
                    // Record edge substitution operation
                    edit_operations.push_back({EditOperation::EDGE_SUBSTITUTION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(edge_label), static_cast<int>(other_edge_label)});
                }
                // If edge exists with same label, no cost needed
            } else {
                edge_deletion_cost += 1; // edge deletion cost
                // Record edge deletion operation
                edit_operations.push_back({EditOperation::EDGE_DELETION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(edge_label), 0});
            }
        }
    }

    // 2.2 Process edges in the other graph (edge insertion cost)
    processed_edges.clear();
    for (ui v = 0; v < other.n; ++v) {
        ui u_mapped = other_to_this.count(v) ? other_to_this[v] : new_node_ids[v];
        for (ui idx = other.pstarts[v]; idx < other.pstarts[v + 1]; ++idx) {
            ui v_neighbor = other.edges[idx];
            ui u_neighbor_mapped = other_to_this.count(v_neighbor) ? other_to_this[v_neighbor] : new_node_ids[v_neighbor];

            // Avoid double counting
            ui u_min = std::min(u_mapped, u_neighbor_mapped);
            ui u_max = std::max(u_mapped, u_neighbor_mapped);
            if (processed_edges.count({u_min, u_max}) > 0) continue;
            processed_edges.insert({u_min, u_max});

            ui edge_label = other.elabels[idx];

            if (u_mapped < n && u_neighbor_mapped < n) {
                // Check if the corresponding edge exists in the current graph
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
                    edge_insertion_cost += 1; // edge insertion cost
                    // Record edge insertion operation
                    edit_operations.push_back({EditOperation::EDGE_INSERTION, static_cast<int>(u_min), static_cast<int>(u_max), 0, static_cast<int>(edge_label)});
                } else if (this_edge_label != edge_label) {
                    edge_substitution_cost += 1; // edge substitution cost
                    // Record edge substitution operation
                    edit_operations.push_back({EditOperation::EDGE_SUBSTITUTION, static_cast<int>(u_min), static_cast<int>(u_max), static_cast<int>(this_edge_label), static_cast<int>(edge_label)});
                }
                // If edge exists with same label, no cost needed
            } else {
                edge_insertion_cost += 1; // edge insertion cost
                // Record edge insertion operation
                edit_operations.push_back({EditOperation::EDGE_INSERTION, static_cast<int>(u_min), static_cast<int>(u_max), 0, static_cast<int>(edge_label)});
            }
        }
    }

    cost += edge_substitution_cost + edge_deletion_cost + edge_insertion_cost;

    return cost;
}




void Graph::initialize_vectors_from_arrays() {
    // Initialize vertex label vector
    adjacency_list.clear();  
    adjacency_list.resize(n);
    
    vlabels_vec.resize(n);
    for (ui i = 0; i < n; ++i) {
        vlabels_vec[i] = vlabels[i];
    }

    // Initialize adjacency list
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


void Graph::convert_vectors_to_arrays() {
    // Count valid vertices and build mapping
    n = 0;
    std::vector<ui> node_mapping(vlabels_vec.size(), INF); // old index to new index mapping

    for (ui i = 0; i < vlabels_vec.size(); ++i) {
        if (vlabels_vec[i] != INF) { // skip deleted vertices
            node_mapping[i] = n;
            n++;
        }
    }

    // Count valid edges
    m = 0;
    for (ui i = 0; i < adjacency_list.size(); ++i) {
        if (vlabels_vec[i] == INF) continue; // skip deleted vertices
        for (const auto& p : adjacency_list[i]) {
            ui neighbor = p.first;
            if (vlabels_vec[neighbor] == INF) continue; // skip deleted vertices
            if (i <= neighbor) {
                m++; // undirected graph, count each edge once
            }
        }
    }

    // Allocate new arrays
    delete[] vlabels;
    delete[] pstarts;
    delete[] edges;
    delete[] elabels;

    vlabels = new ui[n];
    pstarts = new ui[n + 1];
    edges = new ui[m];
    elabels = new ui[m];

    // Fill vlabels
    ui idx = 0;
    for (ui i = 0; i < vlabels_vec.size(); ++i) {
        if (vlabels_vec[i] != INF) {
            vlabels[idx] = vlabels_vec[i];
            idx++;
        }
    }

    // Fill edges and elabels
    ui edge_idx = 0;
    pstarts[0] = 0;
    idx = 0;
    for (ui i = 0; i < adjacency_list.size(); ++i) {
        if (vlabels_vec[i] == INF) continue;

        ui new_i = node_mapping[i];
        for (const auto& p : adjacency_list[i]) {
            ui neighbor = p.first;
            ui edge_label = p.second;
            if (vlabels_vec[neighbor] == INF) continue;

            ui new_neighbor = node_mapping[neighbor];

            // Avoid adding duplicate edges (undirected graph)
            if (new_i <= new_neighbor) {
                edges[edge_idx] = new_neighbor;
                elabels[edge_idx] = edge_label;
                edge_idx++;
            }
        }
        pstarts[idx + 1] = edge_idx;
        idx++;
    }

    // Update edge count
    m = edge_idx;
}

void Graph::draw_single_graph(
    const Graph &g,
    const std::string &out_dir,
    const std::string &prefix
)
{
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // Construct dot and png filenames
    std::string dot_file = out_dir + "/" + prefix + "_" + g.id + ".dot";
    std::string png_file = out_dir + "/" + prefix + "_" + g.id + ".png";

    std::ofstream ofs(dot_file);
    if (!ofs.is_open()) {
        std::cerr << "[draw_single_graph] cannot open " << dot_file << "\n";
        return;
    }

    // Undirected graph => "graph G { ... }"
    ofs << "graph G {\n";
    ofs << "  rankdir=LR;\n";        // optional, left-to-right layout
    ofs << "  node [shape=circle];\n";

    // 1) Only show vlabels_vec[u] as vertex label
    //    Do not show vertex index 
    //    Assuming g.vlabels_vec[u] is an integer label
    for (ui u = 0; u < g.n; u++) {
        // If out of bounds, use default 0 or another marker
        ui nodeLabel = 0;
        if (u < g.vlabels_vec.size()) {
            nodeLabel = g.vlabels_vec[u];
        }
        ofs << "  " << u 
            << " [label=\"" << nodeLabel << "\"];\n";
    }

    // 2) Iterate edges => do not display edge label
    //    Only draw (u < v) to avoid duplicates
    for (ui u = 0; u < g.adjacency_list.size(); u++) {
        for (auto &p : g.adjacency_list[u]) {
            ui v = p.first;
            if (v > u) {
                ofs << "  " << u << " -- " << v << ";\n";
            }
        }
    }

    ofs << "}\n";
    ofs.close();
    std::cout << "[draw_single_graph] wrote " << dot_file << "\n";

    // 3) Call dot to generate PNG
    std::string cmd = "dot -Tpng " + dot_file + " -o " + png_file;
    int ret = std::system(cmd.c_str());
    if (ret == 0) {
        std::cout << "[draw_single_graph] => " << png_file << " generated.\n";
    } else {
        std::cerr << "[WARN] dot command failed with code=" << ret << "\n";
    }
}

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
            adjacency_list[node_id] = {}; // empty adjacency list for new vertex
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
            // Delete edges connected to this vertex
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
            // Check if edge already exists u - v，to avoid duplicate insertion
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
            // For undirected graph, also update v's adjacency list
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
    // Output current graph state
    // std::cout << "After applying operation, PseudoGraph has " << vlabels.size() << " nodes and " << adjacency_list.size() << " adjacency entries." << std::endl;
}

Graph PseudoGraph::to_graph() const {
    Graph g;
    // If id is empty, set a default value
    if (id.empty()) {
        g.id = "PseudoGraph_" + std::to_string(vlabels.size());
    } else {
        g.id = id;
    }

    // If no vertices, return empty graph
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

    // Collect all vertex IDs
    std::vector<ui> node_ids;
    node_ids.reserve(vlabels.size());
    for (const auto& kv : vlabels) {
        node_ids.push_back(kv.first);
    }
    // Sort vertex IDs
    std::sort(node_ids.begin(), node_ids.end());

    // Build mapping from original node_id to new index
    ui n = (ui)vlabels.size();
    std::vector<int> id_to_index(node_ids.back() + 1, -1);
    for (ui i = 0; i < n; i++) {
        ui node_id = node_ids[i];
        id_to_index[node_id] = (int)i;
    }

    g.n = n;
    g.vlabels = new ui[n];

    // Fill vlabels into new graph
    for (ui i = 0; i < n; i++) {
        ui node_id = node_ids[i];
        g.vlabels[i] = vlabels.at(node_id);
    }

    // Count total edges (each edge appears twice in undirected graph; no dedup needed since we directly map the PseudoGraph)
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

    // Initialize pstarts
    for (ui i = 0; i <= g.n; i++) {
        g.pstarts[i] = 0;
    }

    // Compute edge count per vertex to fill pstarts (edges are stored bidirectionally)
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

    // Fill edges and elabels
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

    // Check if edge count matches
    if (current_pos.back() != g.m) {
        std::cerr << "Error: Edge count mismatch in to_graph(). Expected " << g.m << " got " << current_pos.back() << std::endl;
    }

    return g;
}


bool PseudoGraph::can_apply_operation(const EditOperation& op) const {
    switch (op.type) {
        case EditOperation::NODE_INSERTION:
            // Check if vertex already exists
            return vlabels.find(op.u) == vlabels.end();

        case EditOperation::NODE_DELETION:
            // Check if vertex exists
            if (vlabels.find(op.u) == vlabels.end()) {
                return false;
            }

            // Check if vertex has connected edges
            {
                auto it = adjacency_list.find(op.u);
                if (it != adjacency_list.end() && !it->second.empty()) {
                    // Vertex has connected edges, cannot delete
                    // std::cerr << "Cannot delete node " << op.u << " because it has connected edges." << std::endl;
                    return false;
                }
            }

            // Vertex has no connected edges, can delete
            return true;

        case EditOperation::EDGE_INSERTION:
            // Check if both vertices exist and if edge already exists
            return vlabels.find(op.u) != vlabels.end() &&
                   vlabels.find(op.v) != vlabels.end() &&
                   !edge_exists(op.u, op.v);

        case EditOperation::EDGE_DELETION:
            // Check if edge exists
            return edge_exists(op.u, op.v);

        case EditOperation::NODE_SUBSTITUTION:
            // Check if vertex exists
            return vlabels.find(op.u) != vlabels.end();

        case EditOperation::EDGE_SUBSTITUTION:
            // Check if edge exists
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
    // Collect all vertex IDs and sort them for ordered printing
    std::vector<ui> node_ids;
    node_ids.reserve(vlabels.size());
    for (const auto& kv : vlabels) {
        node_ids.push_back(kv.first);
    }
    std::sort(node_ids.begin(), node_ids.end());

    // Count vertices and edges
    ui n = static_cast<ui>(node_ids.size());

    // Edge count: sum adjacency lists for each vertex, then divide by 2 (undirected graph)
    ui total_edges = 0;
    for (auto node_id : node_ids) {
        auto it = adjacency_list.find(node_id);
        if (it != adjacency_list.end()) {
            total_edges += static_cast<ui>(it->second.size());
        }
    }
    // Each undirected edge appears once in each endpoint's adjacency list, so divide by 2
    ui m = total_edges / 2;

    std::cout << "PseudoGraph ID: " << id << std::endl;
    std::cout << "Number of nodes: " << n << std::endl;
    std::cout << "Number of edges: " << m << std::endl;

    // Print vertices and their labels
    std::cout << "Nodes and their labels: ";
    for (auto node_id : node_ids) {
        ui label = vlabels.at(node_id);
        std::cout << "(" << node_id << ": " << label << ") ";
    }
    std::cout << std::endl;

    // Print edges and their labels
    std::cout << "Edges and their labels: ";
    std::unordered_set<std::string> printed_edges; // avoid duplicate printing of undirected edges
    for (auto u_id : node_ids) {
        auto it = adjacency_list.find(u_id);
        if (it == adjacency_list.end()) continue; // no adjacent vertices
        for (const auto& nbr : it->second) {
            ui v_id = nbr.first;
            ui label = nbr.second;
            // Sort (u,v) to avoid duplicate printing of undirected edges
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



