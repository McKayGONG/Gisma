#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "Utility.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>





struct EditOperation {
    enum OperationType {
        NODE_SUBSTITUTION,    // 修改节点标签
        NODE_DELETION,        // 删除节点
        NODE_INSERTION,       // 插入节点
        EDGE_SUBSTITUTION,    // 修改边标签（已存在，明确其用途）
        EDGE_DELETION,        // 删除边
        EDGE_INSERTION,       // 插入边
        NONE = -1
    };
    
    OperationType type;
    int u;          // 对于节点操作：节点ID；对于边操作：第一个节点ID（较小的）
    int v;          // 对于边操作：第二个节点ID（较大的）；对于节点操作：设为-1或0
    int old_label;  // 原始标签（对于替换、删除操作）
    int new_label;  // 新的标签（对于替换、插入操作）

    EditOperation() : type(NONE), u(0), v(-1), old_label(0), new_label(0) {}

    EditOperation(OperationType op_type, int u_val, int v_val, int old_lbl, int new_lbl)
        : type(op_type), u(u_val), v(v_val), old_label(old_lbl), new_label(new_lbl) {}
    
    // 便捷构造函数
    // 节点操作
    static EditOperation NodeSubstitution(int node_id, int old_label, int new_label) {
        return EditOperation(NODE_SUBSTITUTION, node_id, -1, old_label, new_label);
    }
    
    static EditOperation NodeDeletion(int node_id, int label) {
        return EditOperation(NODE_DELETION, node_id, -1, label, 0);
    }
    
    static EditOperation NodeInsertion(int node_id, int label) {
        return EditOperation(NODE_INSERTION, node_id, -1, 0, label);
    }
    
    // 边操作
    static EditOperation EdgeSubstitution(int u, int v, int old_label, int new_label) {
        return EditOperation(EDGE_SUBSTITUTION, std::min(u,v), std::max(u,v), old_label, new_label);
    }
    
    static EditOperation EdgeDeletion(int u, int v, int label) {
        return EditOperation(EDGE_DELETION, std::min(u,v), std::max(u,v), label, 0);
    }
    
    static EditOperation EdgeInsertion(int u, int v, int label) {
        return EditOperation(EDGE_INSERTION, std::min(u,v), std::max(u,v), 0, label);
    }
    
    // 判断是否是节点操作
    bool is_node_operation() const {
        return type == NODE_SUBSTITUTION || type == NODE_DELETION || type == NODE_INSERTION;
    }
    
    // 判断是否是边操作
    bool is_edge_operation() const {
        return type == EDGE_SUBSTITUTION || type == EDGE_DELETION || type == EDGE_INSERTION;
    }
    
    // 判断是否改变节点数
    bool changes_node_count() const {
        return type == NODE_DELETION || type == NODE_INSERTION;
    }
    
    // 判断是否改变边数
    bool changes_edge_count() const {
        return type == EDGE_DELETION || type == EDGE_INSERTION;
    }
    
    // 改进的toString方法
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
    
    // 详细的调试输出
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

// Hash函数保持不变，已经正确处理了EDGE_SUBSTITUTION
struct EditOperationHash {
    size_t operator()(const EditOperation& op) const {
        size_t h1 = std::hash<int>()(op.type);
        size_t h4 = std::hash<int>()(op.old_label);
        size_t h5 = std::hash<int>()(op.new_label);

        if (op.is_edge_operation()) {
            // 对于边操作，节点顺序无关
            size_t node_hash = std::hash<int>()(op.u) ^ std::hash<int>()(op.v);
            return h1 ^ node_hash ^ h4 ^ h5;
        } else {
            // 对于节点操作
            size_t h2 = std::hash<int>()(op.u);
            return h1 ^ h2 ^ h4 ^ h5;
        }
    }
};

// Equal函数保持不变，已经正确处理了EDGE_SUBSTITUTION
struct EditOperationEqual {
    bool operator()(const EditOperation& lhs, const EditOperation& rhs) const {
        if (lhs.type != rhs.type) return false;
        if (lhs.old_label != rhs.old_label || lhs.new_label != rhs.new_label) return false;
        
        if (lhs.is_edge_operation()) {
            // 对于边操作，检查节点是否相同，顺序不限
            return (lhs.u == rhs.u && lhs.v == rhs.v) || (lhs.u == rhs.v && lhs.v == rhs.u);
        } else {
            // 对于节点操作
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
    std::vector<ui> vlabels_vec; // 节点标签向量，大小为根图的节点数量
    std::vector<std::vector<std::pair<ui, ui>>> adjacency_list; // 邻接列表，每个元素是一个向量，存储 (neighbor, edge_label)
public:
    Graph();
    Graph(const std::string& _id, const std::vector<std::pair<int, ui> >& _vertices, const std::vector<std::pair<std::pair<int, int>, ui> >& _edges);
    // 拷贝构造函数
    Graph(const Graph& other);

    // 赋值运算符
    Graph& operator=(const Graph& other);

    ~Graph();

    int size_based_bound(Graph* g);
    int vertex_label_bound(Graph* g, size_t vlabel_count_size);
    int degree_difference_bound(Graph* g, size_t max_n);
    int edge_label_bound(Graph* g, size_t elabel_count_size);
   

    int ged_lower_bound_filter(Graph* g, ui verify_upper_bound, int* vlabel_cnt, int* elabel_cnt, int* degree_q, int* degree_g, int* tmp);
    int ged_lower_bound_filter(Graph* g, ui verify_upper_bound, size_t vlabel_count_size, size_t elabel_count_size, size_t max_n);

    void print_graph() const;

    void save_to_stream(std::ostream& os) const;
    void load_from_stream(std::istream& is);


    std::string compute_wl_hash(ui iterations = 3) const;

    int compute_mapping_cost(const Graph& other, const std::vector<std::pair<ui, ui>>& mapping, std::vector<EditOperation>& edit_operations) const;

    // 初始化向量数据（从原始数组）
    void initialize_vectors_from_arrays();

    // 将向量表示的图转换为数组表示

  
    // Embedding vectors will be loaded from GREED pre-computed bin files
    

    

};

class PseudoGraph {
public:
    std::string id; // 图的 ID
    std::unordered_map<ui, ui> vlabels; // 节点 ID 到节点标签的映射
    std::unordered_map<ui, std::vector<std::pair<ui, ui>>> adjacency_list; // 节点 ID 到（邻居节点 ID，边标签）的映射

    // 构造函数
    PseudoGraph();
    PseudoGraph(const Graph& graph); // 从 Graph 构造 PseudoGraph
    PseudoGraph(const PseudoGraph& other) = default;

    // 方法
    void apply_edit_operation(const EditOperation& op);

    // 将 PseudoGraph 转换为标准的 Graph
    Graph to_graph() const;

    bool can_apply_operation(const EditOperation& op) const;
    void print_pseudo_graph() const;
private:
    bool edge_exists(ui u, ui v) const;
    
};


#endif // _GRAPH_H_
