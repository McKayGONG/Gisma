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
#include <memory> // smart pointers
#include <iomanip> // std::setprecision

typedef unsigned int ui;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define pb push_back
#define mp std::make_pair // changed to std::make_pair
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
#define USE_SUBTREE_PRUNING 1  // Subtree pruning: predict if remaining steps suffice to reach tau
#define USE_DYNAMIC_DEPTH_PROBE
#define USE_ESTIMATE_LB_OPTIMIZATION 1
#define USE_BASELINE_FOR_DISTANT_CHILDREN 1

// =============================================================================
// UNIFIED DFS & SEARCH TREE REUSE RELATED (Not used with SIMPLE_DFS)
// =============================================================================
#define REUSE_IN_SEQUENCE 0  // 1: reuse in sequence, 0: no reuse in sequence
#define MAX_GED_GAP 3  // enable search tree reuse
#define MAX_MARGIN 3
#define _USE_LSa_AS_LAYER_
#define USE_LSa_REFRESH 1  // 1: use LSa recomputation, 0: do not use
#define _USE_LSa_ESTIMATE_BMao_

#define USE_SIBLING_INTERSECTION 1
#define INTERSECTION_WITH_FULL_SNAPSHOT 1
#define PUSH_BACK_AFTER_RECOMPUTE 1  // 1: push back to heap, 0: expand directly
#define USE_LAZY_RECOMPUTE 1  // 1: use lazy recompute, 0: do not use
#define VERIFY_REUSE_WITH_BASELINE

// =============================================================================
// GENERAL ALGORITHM CONFIGURATIONS
// =============================================================================
#define _EXPAND_ALL_
#define _EARLY_STOP_
#define _UPPER_BOUND_


#define USE_FILTERS_FOR_APP_BMAO 1  // 1: use filters in App-BMao search, 0: start directly with app baseline
#define APP_TEST_EXACT_RESULT 0  // 1: App_test returns exact GED result, 0: App_test only as verification tool (similar to baseline)
// COMPUTE_ASTAR_ONLY_FOR_DATA_GRAPH is deprecated, now controlled by use_ept_filters parameter:
// use_ept_filters=true: only compute GED for nodes with completed_db_graph_ids (leaf nodes)
// use_ept_filters=false: compute GED for all nodes





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
// #define MAX_EPT_DEPTH 20  // or other values
// #define DEBUG_PRUNING
// #define OVERALL_UB_NO_IMPROVE_LIMIT 1000
// #define DEPTH_FIRST_PRIORITY
// #define DEBUG_VERBOSE  // control verbose debug output
// #define DEBUG_QUERY_82  // enable detailed debug output for Query 82
// #define ALL_EDGE_LABELS_SAME  // enable optimization for identical edge labels
const double hybrid_ratio = 1.0;

const int INF = 10000000;
const double EPS = 1e-5; 
const ui block_size = 1024;

#include <cassert>

// Forward declarations of other classes
class Graph;
class Node;
class Anchor;
class NetDag;

class Utility {
public:
    // Static function to open a file
    static FILE* open_file(const char* file_name, const char* mode);

    // Static function to convert integer to string
    static std::string integer_to_string(long long number);

    // Static function to load graph data from txt file
    static ui load_db(const char* file_name, std::vector<Graph*>& graphs, std::map<std::string, ui>& vM, std::map<std::string, ui>& eM);

    // Helper function to convert label to integer
    static ui label2int(const char* label, std::map<std::string, ui>& label_map);

    // Extract filename from path tail and take first four characters as short name
    static std::string get_short_name_from_path(const std::string& path);

    // Generate index_name containing short_name and parameters, keeping one decimal place
    static std::string get_index_name(const std::string& db_name, double alpha, double tau, double error_tolerance_index, size_t graph_size);

    // Get the index of (i, j) in upper triangular matrix
    static size_t get_upper_tri_index(int N, int i, int j);

    static bool load_exact_ground_truth(const std::string& file_path,
                                        std::map<int, std::map<double, std::vector<int>>>& ground_truth);

    static std::vector<int> get_ids_within_range(const std::map<int, std::map<double, std::vector<int>>>& ground_truth, double range);

    // Vector distance calculation
    static double euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2);

    // Embedding vector loading
    static bool load_embeddings(const std::string& embedding_file, std::vector<std::vector<float>>& embeddings);
    static double predict_ged_with_embeddings(const std::vector<float>& emb1, const std::vector<float>& emb2);

};

#endif // _UTILITY_H_
