#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "Utility.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>





struct EditOperation {
    enum OperationType {
        NODE_SUBSTITUTION,    // modify node label
        NODE_DELETION,        // delete node
        NODE_INSERTION,       // insert node
        EDGE_SUBSTITUTION,    // modify edge label (existing; clarified usage)
        EDGE_DELETION,        // delete edge
        EDGE_INSERTION,       // insert edge
        NONE = -1
    };
    
    OperationType type;
    int u;          // for node ops: node ID; for edge ops: first node ID (smaller)
    int v;          // for edge ops: second node ID (larger); for node ops: set to -1 or 0
    int old_label;  // original label (for substitution, deletion operations)
    int new_label;  // new label (for substitution, insertion operations)

    EditOperation() : type(NONE), u(0), v(-1), old_label(0), new_label(0) {}

    EditOperation(OperationType op_type, int u_val, int v_val, int old_lbl, int new_lbl)
        : type(op_type), u(u_val), v(v_val), old_label(old_lbl), new_label(new_lbl) {}
    
    // Convenience constructors
    // Node operations
    static EditOperation NodeSubstitution(int node_id, int old_label, int new_label) {
        return EditOperation(NODE_SUBSTITUTION, node_id, -1, old_label, new_label);
    }
    
    static EditOperation NodeDeletion(int node_id, int label) {
        return EditOperation(NODE_DELETION, node_id, -1, label, 0);
    }
    
    static EditOperation NodeInsertion(int node_id, int label) {
        return EditOperation(NODE_INSERTION, node_id, -1, 0, label);
    }
    
    // Edge operations
    static EditOperation EdgeSubstitution(int u, int v, int old_label, int new_label) {
        return EditOperation(EDGE_SUBSTITUTION, std::min(u,v), std::max(u,v), old_label, new_label);
    }
    
    static EditOperation EdgeDeletion(int u, int v, int label) {
        return EditOperation(EDGE_DELETION, std::min(u,v), std::max(u,v), label, 0);
    }
    
    static EditOperation EdgeInsertion(int u, int v, int label) {
        return EditOperation(EDGE_INSERTION, std::min(u,v), std::max(u,v), 0, label);
    }
    
    // Check if this is a node operation
    bool is_node_operation() const {
        return type == NODE_SUBSTITUTION || type == NODE_DELETION || type == NODE_INSERTION;
    }
    
    // Check if this is an edge operation
    bool is_edge_operation() const {
        return type == EDGE_SUBSTITUTION || type == EDGE_DELETION || type == EDGE_INSERTION;
    }
    
    // Check if this changes node count
    bool changes_node_count() const {
        return type == NODE_DELETION || type == NODE_INSERTION;
    }
    
    // Check if this changes edge count
    bool changes_edge_count() const {
        return type == EDGE_DELETION || type == EDGE_INSERTION;
    }
    
    // Improved toString method
    std::string to_string() const {
        std::stringstream ss;
        switch (type) {
            case NODE_SUBSTITUTION:
                ss << "NODE_SUBST(v" << u << ": " << old_label << "->" << new_label << ")";
                break;
            case NODE_DELETION:
                ss << "NODE_DEL(v" << u << ", label=" << old_label << ")";
                break;
            case NODE_INSERTION:
                ss << "NODE_INS(v" << u << ", label=" << new_label << ")";
                break;
            case EDGE_SUBSTITUTION:
                ss << "EDGE_SUBST((" << u << "," << v << "): " << old_label << "->" << new_label << ")";
                break;
            case EDGE_DELETION:
                ss << "EDGE_DEL((" << u << "," << v << "), label=" << old_label << ")";
                break;
            case EDGE_INSERTION:
                ss << "EDGE_INS((" << u << "," << v << "), label=" << new_label << ")";
                break;
            default:
                ss << "NONE";
                break;
        }
        return ss.str();
    }
    
    // Detailed debug output
    std::string to_debug_string() const {
        std::stringstream ss;
        ss << "[type=";
        switch (type) {
            case NODE_SUBSTITUTION: ss << "NODE_SUBSTITUTION"; break;
            case NODE_DELETION: ss << "NODE_DELETION"; break;
            case NODE_INSERTION: ss << "NODE_INSERTION"; break;
            case EDGE_SUBSTITUTION: ss << "EDGE_SUBSTITUTION"; break;
            case EDGE_DELETION: ss << "EDGE_DELETION"; break;
            case EDGE_INSERTION: ss << "EDGE_INSERTION"; break;
            default: ss << "NONE"; break;
        }
        ss << ", u=" << u;
        ss << ", v=" << v;
        ss << ", old_label=" << old_label;
        ss << ", new_label=" << new_label;
        ss << "]";
        return ss.str();
    }
};

// Hash function unchanged, already correctly handles EDGE_SUBSTITUTION
struct EditOperationHash {
    size_t operator()(const EditOperation& op) const {
        size_t h1 = std::hash<int>()(op.type);
        size_t h4 = std::hash<int>()(op.old_label);
        size_t h5 = std::hash<int>()(op.new_label);

        if (op.is_edge_operation()) {
            // For edge operations, node order does not matter
            size_t node_hash = std::hash<int>()(op.u) ^ std::hash<int>()(op.v);
            return h1 ^ node_hash ^ h4 ^ h5;
        } else {
            // For node operations
            size_t h2 = std::hash<int>()(op.u);
            return h1 ^ h2 ^ h4 ^ h5;
        }
    }
};

// Equal function unchanged, already correctly handles EDGE_SUBSTITUTION
struct EditOperationEqual {
    bool operator()(const EditOperation& lhs, const EditOperation& rhs) const {
        if (lhs.type != rhs.type) return false;
        if (lhs.old_label != rhs.old_label || lhs.new_label != rhs.new_label) return false;
        
        if (lhs.is_edge_operation()) {
            // For edge operations, check if nodes are the same regardless of order
            return (lhs.u == rhs.u && lhs.v == rhs.v) || (lhs.u == rhs.v && lhs.v == rhs.u);
        } else {
            // For node operations
            return lhs.u == rhs.u;
        }
    }
};


class Graph {
public:
    std::string id;
    ui n, m;
    ui* pstarts;
    ui* edges;
    ui* vlabels;
    ui* elabels;
    static size_t FEATURE_DIM;
    std::vector<ui> vlabels_vec; // vertex label vector, sized to the root graph's vertex count
    std::vector<std::vector<std::pair<ui, ui>>> adjacency_list; // adjacency list, each element is a vector storing (neighbor, edge_label)
public:
// static void* operator new(size_t size){
//     static Graph *begin=nullptr,*end=nullptr;
//     if (begin==end) begin=(Graph*)malloc(10000*size),end=begin+10000;
//     return begin++;
// }
    Graph();
    Graph(const std::string& _id, const std::vector<std::pair<int, ui> >& _vertices, const std::vector<std::pair<std::pair<int, int>, ui> >& _edges);
    // Copy constructor
    Graph(const Graph& other);

    // Assignment operator
    Graph& operator=(const Graph& other);

    ~Graph();

    void write_graph(FILE* fout, const std::vector<std::string>& _vlabels, const std::vector<std::string>& _elabels, bool bss);
    bool is_connected();
    int size_based_bound(Graph* g);
    int vertex_label_bound(Graph* g, size_t vlabel_count_size);
    int degree_difference_bound(Graph* g, size_t max_n);
    int edge_label_bound(Graph* g, size_t elabel_count_size);
   

    int ged_lower_bound_filter(Graph* g, ui verify_upper_bound, int* vlabel_cnt, int* elabel_cnt, int* degree_q, int* degree_g, int* tmp);
    int ged_lower_bound_filter(Graph* g, ui verify_upper_bound, size_t vlabel_count_size, size_t elabel_count_size, size_t max_n);
    int ged_lower_bound_filter_ori(Graph* g, ui verify_upper_bound, size_t vlabel_count_size, size_t elabel_count_size, size_t max_n);

    void print_graph() const;

    void save_to_stream(std::ostream& os) const;
    void load_from_stream(std::istream& is);


    std::string compute_wl_hash(ui iterations = 3) const;

    int compute_mapping_cost(const Graph& other, const std::vector<std::pair<ui, ui>>& mapping, std::vector<EditOperation>& edit_operations) const;

    // Initialize vector data (from raw arrays)
    void initialize_vectors_from_arrays();

    // Convert vector-based graph representation to array representation
    void convert_vectors_to_arrays();

  
    // Embedding vectors will be loaded from GREED pre-computed bin files
    

    
    void draw_single_graph(const Graph &g, const std::string &out_dir, const std::string &prefix);

};

class PseudoGraph {
public:
    std::string id; // graph ID
    std::unordered_map<ui, ui> vlabels; // mapping from node ID to node label
    std::unordered_map<ui, std::vector<std::pair<ui, ui>>> adjacency_list; // mapping from node ID to (neighbor node ID, edge label)

    // Constructors
    PseudoGraph();
    PseudoGraph(const Graph& graph); // construct PseudoGraph from Graph
    PseudoGraph(const PseudoGraph& other) = default;

    // Methods
    void apply_edit_operation(const EditOperation& op);

    // Convert PseudoGraph to standard Graph
    Graph to_graph() const;

    bool can_apply_operation(const EditOperation& op) const;
    void print_pseudo_graph() const;
private:
    bool edge_exists(ui u, ui v) const;
    
};


#endif // _GRAPH_H_
