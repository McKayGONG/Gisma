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

// Iteration count variables removed - now controlled via Application/GismaSearchEngine class members
// set from --app_max_iter command-line parameter (default: 2300)

// Implementation of the static file-open function
FILE* Utility::open_file(const char* file_name, const char* mode) {
    FILE* f = fopen(file_name, mode);
    if (f == nullptr) {
        printf("Cannot open file: %s\n", file_name);
        exit(1);
    }
    return f;
}

// Implementation of the static integer-to-string function
std::string Utility::integer_to_string(long long number) {
    std::vector<ui> sequence;
    if (number == 0) {
        sequence.pb(0);
    }
    while (number > 0) {
        sequence.pb(number % 1000);
        number /= 1000;
    }

    char buf[6]; // extra char for null terminator
    std::string res;
    for (ui i = sequence.size(); i > 0; i--) {
        if (i == sequence.size()) {
            sprintf(buf, "%u", sequence[i - 1]);
        }
        else {
            sprintf(buf, ",%03u", sequence[i - 1]);
        }
        res += std::string(buf);
    }
    return res;
}


// Implementation of the static function to load graph data from a txt file
ui Utility::load_db(const char* file_name, std::vector<Graph*>& graphs, std::map<std::string, ui>& vM, std::map<std::string, ui>& eM) {
    FILE* fin = Utility::open_file(file_name, "r");

    const ui MAX_LINE = 1024;
    char line[MAX_LINE];
    if (fgets(line, MAX_LINE, fin) == NULL) {
        fclose(fin);
        return 0;
    }

    ui max_n = 0;
    ui graph_id = 0;  // sequential numbering starting from 0
    ui max_vlabel = 0;
    ui max_elabel = 0;
    while (line[0] == 'I' && line[1] == 'D') {  // check for "ID" prefix
        line[0] = 'x';  // mark current line as processed

        std::vector<std::pair<int, ui>> vertices;
        std::vector<std::pair<std::pair<int, int>, ui>> edges;
        while (fgets(line, MAX_LINE, fin) != NULL && !(line[0] == 'I' && line[1] == 'D')) {
            // if (line[0] == 'v') {
            //     int a;
            //     char buf[128];
            //     sscanf(line + 2, "%d%s", &a, buf);
            //     vertices.pb(mp(a, label2int(buf, vM)));
            //     ui lab_id = label2int_AIDS(buf, vM);
            //     vertices.emplace_back(a, lab_id);
                

            // } else if (line[0] == 'e') {
            //     int a, b;
            //     char buf[128];
            //     sscanf(line + 2, "%d%d%s", &a, &b, buf);
            //     ui edge_label = label2int(buf, eM);
            //     edges.pb(mp(mp(a, b), edge_label));
            //     edges.pb(mp(mp(b, a), edge_label));
            if (line[0] == 'v') {
                int a;
                ui lab_id;  // ui type since labels are integers
                sscanf(line + 2, "%d%d", &a, &lab_id);  // read vertex id and integer label
                if (lab_id > max_vlabel) max_vlabel = lab_id;
                // std::cout << "max_vlabel: " << max_vlabel << std::endl;
                vertices.emplace_back(a, lab_id);  // add vertex and label to vertices
                // std::cout << "Vertex: " << a << " Label: " << lab_id << std::endl;
            
            } else if (line[0] == 'e') {
                int a, b;
                ui edge_label;  // edge label is also an integer
                sscanf(line + 2, "%d%d%d", &a, &b, &edge_label);  // read two vertices and edge label
                if (edge_label > max_elabel) max_elabel = edge_label;
                edges.pb(mp(mp(a, b), edge_label));  // add edge with label
                edges.pb(mp(mp(b, a), edge_label));  // add reverse edge
            } else {
                std::cerr << "!!! Unrecognized first letter in a line when loading DB!\n";
            }
            line[0] = 'x';  // mark current line as processed
        }

        // Sort and validate vertices
        std::sort(vertices.begin(), vertices.end());
        for (ui i = 0; i < vertices.size(); i++) {
            assert(vertices[i].first == static_cast<int>(i));
        }
        if (vertices.size() > max_n) max_n = vertices.size();

        // Sort and validate edges
        std::sort(edges.begin(), edges.end());
        for (ui i = 0; i < edges.size(); i++) {
            assert(edges[i].first.first >= 0 && edges[i].first.first < vertices.size());
            assert(edges[i].first.second >= 0 && edges[i].first.second < vertices.size());
            if (i > 0) {
                assert(edges[i].first != edges[i - 1].first);
            }
            assert(edges[i].second < eM.size());
        }

        // Use sequential graph_id starting from 0 instead of the id from the file
        std::string id = std::to_string(graph_id);
        graph_id++;  // increment to ensure the next graph's id is sequential

        // Create a new Graph object and add it to the graphs vector
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



// Implementation of the function to generate index_name with short_name and parameters, keeping one decimal place
std::string Utility::get_index_name(const std::string& dataset, double alpha, double tau, double error_tolerance_index, size_t graph_size) {
    ;
    
    // Use std::ostringstream to keep one decimal place
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << dataset << "_" << graph_size << "_"
        << alpha << "_" << tau << "_" << error_tolerance_index;
    
    return oss.str();  // return the formatted index_name
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
        
        // Read query_id and distance
        if (!(iss >> query_id >> distance >> bracket) || bracket != '[') {
            std::cerr << "Error: Invalid line format: " << line << std::endl;
            continue;
        }

        std::vector<int> graph_ids;
        int graph_id;

        // Read graph_ids
        while (iss >> graph_id) {
            graph_ids.push_back(graph_id);
            if (iss.peek() == ',') { // skip comma
                iss.ignore();
            }
        }

        ground_truth[query_id][distance] = graph_ids;
    }

    infile.close();
    return true;
}

std::vector<int> Utility::get_ids_within_range(const std::map<int, std::map<double, std::vector<int>>>& ground_truth, double range) {
    std::vector<int> ids;

    for (const auto& query : ground_truth) {
        int query_id = query.first;
        const auto& distances = query.second;

        for (const auto& distance_entry : distances) {
            double distance = distance_entry.first;
            if (distance <= range) {
                const auto& graph_ids = distance_entry.second;
                ids.insert(ids.end(), graph_ids.begin(), graph_ids.end());
            }
        }
    }

    return ids;
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

// Predict GED using embedding vectors with simple distance-based mapping
double Utility::predict_ged_with_embeddings(const std::vector<float>& emb1, const std::vector<float>& emb2) {
    return euclidean_distance(emb1, emb2);
}