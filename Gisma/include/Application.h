#ifndef _APPLICATION_H_
#define _APPLICATION_H_

#include "Utility.h"
#include "Graph.h"
#include "ReuseSearchTree.h"      // * added
#include <unordered_set>
#include <climits>    // USHRT_MAX

const ushort DUMMY_VAL = UINT16_MAX;

enum LB_Method { LSa, BMa, BMao };

struct State {
    State *parent;
    State *pre_sibling;
    ushort level, image;
    ushort mapped_cost, lower_bound;

    ushort vl_lb;        // existing: vertex label lower bound
    ushort vl_common;    // existing: label gap

    ushort mc_cross;     // ** added: ancestor-current cross-edge cost
    uchar  vlabel_same;  // ** added: 0/1 whether vertex labels are equal

    ushort cs_cnt;
	std::vector<State*> _children;   /* temporary children chain used during snapshot reuse */
	bool reused_leaf = false;  // flag: whether this is a reused leaf node
	bool need_recompute = false;  // ** added: flag whether LB needs recomputation
#ifdef _EXPAND_ALL_
    ushort *siblings;
    ushort siblings_n;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    ushort siblings_total_n;  // full siblings count (including non-intersection part)
#endif
#endif
    bool   kept = false;;
	bool eq_ub_tag   = false;
#ifdef _USE_LSa_ESTIMATE_BMao_
	ui lsa_lb;
#endif
};





void print_state(const State* state);



class Application {
private:
	
	ui q_n;
	ui *q_starts, *q_edges;
	ui *q_vlabels, *q_elabels;

	ui g_n;
	ui *g_starts, *g_edges;
	ui *g_vlabels, *g_elabels;

	ui vlabels_n, elabels_n;

	ui verify_upper_bound;
	ui upper_bound;
	ui overall_lb = INF;
	ui overall_ub = INF;
	int margin = 0;
	bool disable_lsa_pruning = false;  // disable LSa pruning (still save lsa_lb for reuse)
	bool disable_reuse_lsa = false;    // disable LSa recomputation in reuse

	LB_Method lb_method;

	bool q_g_swapped;

	ui *MO;
	ui search_n;
	ui search_n_for_IS;

	char *visX, *visY;
	int *mx, *my;
	ui *BX;
	ui *candidates;
	ui *queue, *prev;

	std::vector<State *> states_memory;
	std::vector<State *> states_pool;
	ui states_pool_n;

	std::vector<ushort *> siblings_memory;
	std::vector<ushort *> siblings_pool;
	ui siblings_pool_n;

	int *elabels_map, *vlabels_map;
	short *elabels_matrix;
	uchar *q_matrix;

	ushort *visited_siblings; // these two arrays only used in DFS
	ushort *visited_siblings_n;

	/***** for bipartite matching based lower bounds ****/
	ui *cost;
	int *lx, *ly, *slack, *slackmy;

	std::pair<int, int> *children;

	long long search_space;
	/* Lightweight search tree snapshot: one "small card" per node */
	std::vector<SearchNodeLite> search_snapshot;
	std::vector<State*> open_heap;           // current A* heap (min-heap)
    std::vector<State*> full_mapping_nodes;  // leaves that completed full-mapping
	std::vector<State*> boundary_nodes;

	// A* iteration limits (replaces global APP_CNT/ASTAR_CNT/REUSE_CNT)
	int app_max_iter;       // A* algorithm max iterations
	int exact_max_iter;     // exact computation max iterations (for training data generation)

    // Utility functions
    bool check_query_edge_exists(ui u, ui v, ui& label);
    bool check_db_edge_exists(ui u, ui v, ui& label);
public:
    Application(ui _verify_upper_bound, const char *lower_bound, int _app_max_iter = 2300, int _exact_max_iter = 1000000);
    ~Application();

    void init(const std::vector<std::pair<int,int> > &g_v, const std::vector<std::pair<std::pair<int,int>,int> > &g_e, const std::vector<std::pair<int,int> > &q_v, const std::vector<std::pair<std::pair<int,int>,int> > &q_e);
    void init(const Graph *g, const Graph *q);

    ui DFS(State *node = NULL);
    ui App(std::vector<std::pair<ui, ui>> *mapping_ptr = NULL, int *lb_ptr = NULL);
	ui AStar_baseline(std::vector<std::pair<ui, ui>> *mapping_ptr = NULL, int *lb_ptr = NULL);
	ui App_test(std::vector<std::pair<ui, ui>> *mapping_ptr = NULL, int *lb_ptr = NULL);
	ui App_baseline(std::vector<std::pair<ui, ui>> *mapping_ptr = NULL, int *lb_ptr = NULL);
	ui AppForComputation(std::vector<std::pair<ui, ui>> *mapping_ptr = NULL, int *lb_ptr = NULL);


    ui compute_ged_of_BX();
    void get_mapping(std::vector<std::pair<ui,ui> > &mapping);

	// static ui label2int(const char *str, std::map<std::string, ui> &M);
	// static Graph* create_graph(const char* id, unsigned int n, unsigned int m, int* vertices_data, int* edges_data, std::map<std::string, ui>& vM, std::map<std::string, ui>& eM);
	// static Graph* create_graph(const std::string& id, unsigned int n, unsigned int m, int* vertices_data, int* edges_data, std::map<std::string, ui>& vM, std::map<std::string, ui>& eM);
    static Graph* create_graph(const char* id, unsigned int n, unsigned int m, int* vertices_data, int* edges_data);
    static void delete_graph(Graph* g);
    // static double compute_ged(Graph* graph1, Graph* graph2, ui upper_bound_value = INF);
	static double verify_ged(Graph* graph1, Graph* graph2, ui upper_bound_value = INF);
	static double compute_ged(Graph* graph1, Graph* graph2, ui upper_bound_value = INF);

	void print_graph(const Graph *g, const char* graph_name);

	long long get_search_space() { return search_space; }

	void extract_snapshot(SearchSnapshot& out);
	/*------------------------------------------------------------
     * Reuse previous round snapshot to continue search (g -> g' with one micro-edit)
     *    snap  : OPEN+CLOSED snapshot from previous round search on g
     *    tau   : new round upper-bound
     *    delta : describes the single-step edit g->g', can be used for dynamic LB patching
     *  return  : new upper_bound (i.e., GED or pruning upper bound)
     *-----------------------------------------------------------*/
    ui app_reuse(const SearchSnapshot& snap,
                 ui tau,
                 ui ged_gap = 1);

	void compute_mapping_order_reuse();
	/* --- Lower bound recomputation (LSa specific) --- */
	void recompute_lsa_lb(State* now);
	
	void refresh_lsa_snapshot_lb(State* root, ui snapshot_nodes);
	void refresh_snapshot_lb_batch(State* root, ui snapshot_nodes);
	inline bool is_real_node(const State* st) {
		return st->image < g_n && st->level < q_n;
	}
	std::pair<ui, ui> lsa_lower_bound_exact(State* node);
	std::pair<ui, ui> bruteforce_LSa_lower_bound(State* node);
	void recompute_node_lazy(State* now);
	ui get_overall_lb() const { return overall_lb; }
	ui get_overall_ub() const { return overall_ub; }
	void set_margin(int m) { margin = m; }
    int get_margin() const { return margin; }
	void set_disable_lsa_pruning(bool v) { disable_lsa_pruning = v; }
	void set_disable_reuse_lsa(bool v) { disable_reuse_lsa = v; }
private:
	void preprocess() ;

	void add_to_pool(State *st) ;
	void add_to_heap(State *st, ui &heap_n, std::vector<State*> &heap) ;
	State* get_a_new_state_node() ;
	void put_a_state_to_pool(State *st) ;
	ushort* get_a_new_siblings_node() ;
	void put_a_sibling_to_pool(ushort *sibling) ;
	void verify_induced_ged(State *now) ;
	void verify_LS_lower_bound(State *now) ;
	ui relabel(ui len1, ui *array1, ui len2, ui *array2) ;
	ui search_index(ui val, std::vector<ui> &array, ui array_len) ;

	void compute_mapping_order() ;
	void generate_best_extension(State *parent, State *now) ; // compute the best child of a state
	void generate_best_extension_baseline(State *parent, State *now) ; // compute the best child of a state
	void generate_best_extension_test(State *parent, State *now) ; // compute the best child of a state
	void compute_mapped_cost_and_upper_bound(State *now, ui n, ui *candidates, int *mapping) ;
	void compute_mapped_cost_and_upper_bound_baseline(State *now, ui n, ui *candidates, int *mapping) ;
	void compute_mapped_cost_and_upper_bound_test(State *now, ui n, ui *candidates, int *mapping) ;
	void construct_sibling(State *pre_sibling, State *now) ; // compute the best ungenerated sibling of a state
	void construct_sibling_baseline(State *pre_sibling, State *now);
	void construct_sibling_test(State *pre_sibling, State *now);
	void extend_to_full_mapping(State *parent, State *now) ;
	void extend_to_full_mapping_baseline(State *parent, State *now) ;
	void extend_to_full_mapping_test(State *parent, State *now) ;

	void compute_mapped_cost(State *now) ;

	void compute_best_extension_LSa(State *now, ui candidate_n, ui *candidates, ui pre_siblings) ;
	void compute_best_extension_BM(char anchor_aware, State *now, ui candidate_n, ui *candidates, ui pre_siblings, char no_siblings, char IS = 0) ;
	void compute_best_extension_BM_baseline(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings, char no_siblings, char IS = 0) ;
	void compute_best_extension_BM_test(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings, char no_siblings, char IS = 0) ;
	void compute_best_extension_BMa(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings) ;

	ui Hungarian(char initialization, ui n, ui *cost) ; // minimum cost bipartite matching
	void heap_top_down(ui idx, ui heap_n, std::vector<std::pair<double, int> > &heap, ui *pos) ;
	void heap_bottom_up(ui idx, std::vector<std::pair<double,int> > &heap, ui *pos) ;
	void heap_top_down(ui idx, ui heap_n, std::vector<State*> &heap) ;
	void heap_bottom_up(ui idx, std::vector<State*> &heap) ;
	inline bool state_has_higher_priority(State* a, State* b) ;

    /* Pass in the vertex and its corresponding State together */
    void patch_vertex(State* st, bool forward);

    /* --- Helper: fill mapped_cost / lower_bound for node under current book state --- */
    void calc_lb_from_tables(State* node);

	/* ====== Used internally by ReuseLSa_DFS ====== */
	size_t VCAP = 0;            // vlabels_map capacity (moved from file-scope static to instance to avoid thread contention)
	size_t ECAP = 0;            // ecnt / elabels_matrix column count
	size_t MCAP = 0;            // elabels_matrix column count (==ECAP)
	std::vector<int> ecnt;      // edge label surplus/deficit

	void ensure_vlabel_cap(ui lbl);
	void ensure_elabel_cap(ui lbl);
	ui compute_mapped_cost_from_scratch(State* node);
	ui compute_independent_lower_bound_LSa(State* node);
	ui compute_mapped_cost_baseline(State* node);
	ui compute_independent_lower_bound_LSa_baseline(State* node);
	// LSa lower bound computation based on paper definition
	

	// Helper function: compute distance between two multisets
	ui compute_multiset_distance(const std::map<ui, int>& set1, 
							const std::map<ui, int>& set2);
	ui compute_full_mapping_cost_baseline(State* leaf_node);
	ui compute_full_mapping_cost(State* node);
	void compute_best_extension_LSa_baseline(State *now, ui candidate_n, ui *candidates, ui pre_siblings) ;
	void compute_best_extension_BMa_baseline(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings) ;
	
	
	// The following declarations are added to the public section of Application.h:

	/*
	* Batch processing function declarations for BM algorithm
	*/

	// Simplified interface: build candidate list, then call core function
	void compute_specified_children_BM(
		char anchor_aware,
		State *parent,
		ui *specified_images,
		ui specified_count,
		ui *results
	);

	// Core implementation: specified-node version adapted from compute_best_extension_BM
	void compute_best_extension_BM_specified(
		char anchor_aware,
		State *parent,
		ui n, ui *candidates,
		ui *specified_images,
		ui specified_count,
		ui *results
	);

	// Primary version: batch refresh function for BM algorithm search tree
	void refresh_bm_snapshot_lb_batch(State* root, 
                                     ui snap_nodes, 
                                     ui ged_gap);
};

#endif
