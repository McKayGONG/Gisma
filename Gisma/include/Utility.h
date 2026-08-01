#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <queue>
#include <set>
#include <map>
#include <memory> // 智能指针
#include <iomanip> // std::setprecision

typedef unsigned int ui;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define pb push_back
#define mp std::make_pair // 修改为std::make_pair
#define mmax(a,b) ((a)>(b)?(a):(b))
#define mmin(a,b) ((a)<(b)?(a):(b))

// =============================================================================
// SEARCH STRATEGY SELECTION (Priority: USE_UNIFIED_DFS > USE_SIMPLE_DFS > default with_reuse)
// =============================================================================
// #define USE_UNIFIED_DFS
// #define USE_SIMPLE_DFS

// =============================================================================
// SIMPLE DFS OPTIMIZATIONS (Currently Active)
// =============================================================================
#define USE_SUBTREE_PRUNING 1  // Subtree pruning: 预判剩余步数是否足够达到tau
#define USE_DYNAMIC_DEPTH_PROBE
#define USE_ESTIMATE_LB_OPTIMIZATION 1
#define USE_BASELINE_FOR_DISTANT_CHILDREN 1

// =============================================================================
// UNIFIED DFS & SEARCH TREE REUSE RELATED (Not used with SIMPLE_DFS)
// =============================================================================
#define REUSE_IN_SEQUENCE 0  // 1:在sequence中复用，0:在sequence中不复用
#define MAX_GED_GAP 3  // 启用搜索树重用
#define MAX_MARGIN 3
// _USE_LSa_AS_LAYER_ 已改为 runtime 开关：--disable_lsa_layer（Application::disable_lsa_layer，默认层开启）
#define USE_LSa_REFRESH 1  // 1:使用LSa重计算，0:不使用LSa重计算
#define _USE_LSa_ESTIMATE_BMao_

#define USE_SIBLING_INTERSECTION 1
#define INTERSECTION_WITH_FULL_SNAPSHOT 1
#define PUSH_BACK_AFTER_RECOMPUTE 1  // 1:压回heap, 0:直接展开
#define USE_LAZY_RECOMPUTE 1  // 1:使用lazy recompute, 0:不使用
#define VERIFY_REUSE_WITH_BASELINE

// =============================================================================
// GENERAL ALGORITHM CONFIGURATIONS
// =============================================================================
#define _EXPAND_ALL_
#define _EARLY_STOP_
#define _UPPER_BOUND_


#define USE_FILTERS_FOR_APP_BMAO 1  // 1:App-BMao搜索中使用过滤器，0:直接app baseline开始算
#define APP_TEST_EXACT_RESULT 0  // 1:App_test返回精确GED结果，0:App_test只作为verification工具（类似baseline）
// COMPUTE_ASTAR_ONLY_FOR_DATA_GRAPH 已废弃，现在由 use_ept_filters 参数控制：
// use_ept_filters=true: 只对有completed_db_graph_ids的节点（叶节点）计算GED
// use_ept_filters=false: 对所有节点计算GED





// =============================================================================
// ALGORITHM LIMITS & COUNTERS
// =============================================================================
// Iteration limits are now controlled via Application/GismaSearchEngine class members
// set from --app_max_iter command-line parameter (default: 2300)
// AIDS: 2300; PubChem: 3000; SYN: 200
// exact_max_iter is controlled via --exact_max_iter parameter (default: 1000000)

// =============================================================================
// DEBUG & EXPERIMENTAL OPTIONS (Commented out for production)
// =============================================================================
// #define STEP_TRACE
// #define show_time_detail
// #define MAX_EPT_DEPTH 20  // 或其他值
// #define DEBUG_PRUNING
// #define OVERALL_UB_NO_IMPROVE_LIMIT 1000
// #define DEPTH_FIRST_PRIORITY
// #define DEBUG_VERBOSE  // 控制详细调试输出
// #define DEBUG_QUERY_82  // 启用Query 82的详细调试输出
// #define ALL_EDGE_LABELS_SAME  // 已轉為 runtime flag（Application::all_edge_labels_same）；預設 false
const double hybrid_ratio = 1.0;

const int INF = 10000000;
const double EPS = 1e-5; 
const ui block_size = 1024;

#include <cassert>

// 前向声明其他类
class Graph;
class Node;
class Anchor;
class NetDag;

class Utility {
public:
    // 打开文件的静态函数
    static FILE* open_file(const char* file_name, const char* mode);

    // 整数转字符串的静态函数

    // 从txt文件中加载图数据的静态函数
    static ui load_db(const char* file_name, std::vector<Graph*>& graphs, std::map<std::string, ui>& vM, std::map<std::string, ui>& eM);

    // 标签转整数的辅助函数
    static ui label2int(const char* label, std::map<std::string, ui>& label_map);

    // 获取路径末尾文件名并提取前四位作为短名称
    static std::string get_short_name_from_path(const std::string& path);

    // 生成包含short_name和参数的index_name，并保留一位小数
    static std::string get_index_name(const std::string& db_name, double alpha, double tau, double error_tolerance_index, size_t graph_size);

    // 获取上三角矩阵中 (i, j) 的索引
    static size_t get_upper_tri_index(int N, int i, int j);

    static bool load_exact_ground_truth(const std::string& file_path,
                                        std::map<int, std::map<double, std::vector<int>>>& ground_truth);


    // Vector distance calculation
    static double euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2);

    // Embedding vector loading
    static bool load_embeddings(const std::string& embedding_file, std::vector<std::vector<float>>& embeddings);

    // E1: construction 计时输出文件路径。空 = 不统计（默认）。由 main() 从 --timing_log 设置。
    static std::string e1_timing_path;
    // append one structured timing record (tab-separated key=value) to e1_timing_path.
    // 若 e1_timing_path 为空则直接返回（不创建任何文件）。
    static void append_e1_timing(const std::string& record);

};

#endif // _UTILITY_H_
