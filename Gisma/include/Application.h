#ifndef _APPLICATION_H_
#define _APPLICATION_H_

#include "Utility.h"
#include "Graph.h"
#include "ReuseSearchTree.h"      // * 新增
#include <unordered_set>
#include <unordered_map>
#include <climits>    // USHRT_MAX

const ushort DUMMY_VAL = UINT16_MAX;

enum LB_Method { LSa, BMa, BMao };

struct State {
    State *parent;
    State *pre_sibling;
    ushort level, image;
    ushort mapped_cost, lower_bound;

    ushort cs_cnt;
	bool reused_leaf = false;  // 标记是否为复用的叶子节点
	bool need_recompute = false;  // ** 新增：标记是否需要重新计算LB
#ifdef _EXPAND_ALL_
    ushort *siblings;
    ushort siblings_n;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    ushort siblings_total_n;  // 完整的siblings数量（包含非交集部分）
#endif
#endif
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
	// --- LSa policy flags (interact non-trivially) ---
	// lazy_parent_lsa (default true): App() 純 BMao；extract_snapshot 時才對存留 leaves 算 LSa。
	//                                 ★ 此時 disable_lsa_pruning 無效 (App 根本沒算 LSa)。
	// disable_lsa_pruning           : 僅在 lazy=false 時生效；控 App() 是否用 LSa prune (仍保 lsa_lb)。
	// disable_reuse_lsa             : 控 child refresh — false=LSa 公式 (默認)；true=triangle (old_lb - ged_gap)。
	bool disable_lsa_pruning = false;
	bool disable_reuse_lsa = false;
	bool exact_value_mode = false;  // 預設 false：early-exit when ≤tau 找到。設 true：搜到底算精確 GED。
	bool lazy_parent_lsa = true;
	bool all_edge_labels_same = true;  // 跳過 edge label 比較；默認開（Gisma GED 模型无边标签）。set_all_edge_labels_same 覆盖
	bool refresh_bmao_recompute = false; // 若 true：refresh 階段對 leaf 用 BMao Hungarian 精確重算 (eager)，跳過 LSa/EA/triangle 估計
	bool bmao_recomp_intersect = false;  // 配合 refresh_bmao_recompute=true：true 走 recompute_node_lazy (帶交集策略)，false 走 refresh_leaf_lb_only_BM (純 g')
	bool skip_lazy_recompute = false;    // 若 true：astar pop 後跳過 lazy recompute，直接用 refresh 階段算的 lb extend
	// Parent-side flag：compute_best_extension_BM sibling loop 用更嚴的 break `lb > tau`
	// （省 parent boundary sibling 計算）。設給 parent App instance。
	bool early_stop_at_tau = false;
	// Child-side flag：recompute_node_lazy 跳過 intersection 篩選，直接用 child 算的 new_siblings。
	// 配合 parent early_stop_at_tau 使用（parent 少 sibling 時 intersection 經常 empty）。
	bool skip_intersection_in_reuse = false;
	// 若非 null，compute_best_extension_BM sibling loop 在「所有 target image 都算到」後 early break。
	// 用於 recompute_node_lazy：snapshot 帶來的 old_siblings 構成 intersection target，後續不在 target 內的 H(0) 工作可省。
	const std::unordered_set<ui>* lazy_recompute_target_siblings = nullptr;

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
	/* 轻量搜索树快照：每个结点 1 张"小卡片" */
	std::vector<SearchNodeLite> search_snapshot;
	std::vector<State*> open_heap;           // 当前 A* 的堆（小根）
    std::vector<State*> full_mapping_nodes;  // 已完成 full-mapping 的叶子
	std::vector<State*> boundary_nodes;

	// A* iteration limits (replaces global APP_CNT/ASTAR_CNT/REUSE_CNT)
	int app_max_iter;       // A*算法的最大迭代次数
	int exact_max_iter;     // 精确计算的最大迭代次数（训练数据生成）
    
    // 工具函数（如果还没有声明）
    bool check_query_edge_exists(ui u, ui v, ui& label);
    bool check_db_edge_exists(ui u, ui v, ui& label);

    // children 列表的存储：从 State 成员移到这张外部表，让搜索热路径的 State 不再携带 children vector。
    // key = State*（pool 复用地址）；生产者（extract_snapshot / app_reuse rebuild）用前先 clear 对应节点，
    // 复用地址的残留 entry 不会被消费者读到（消费者只读已重建的节点）。
    std::unordered_map<State*, std::vector<State*>> _children_map;
    std::vector<State*>& children_of(State* st) { return _children_map[st]; }
public:
    Application(ui _verify_upper_bound, const char *lower_bound, int _app_max_iter = 2300, int _exact_max_iter = 1000000);
    ~Application();

    // Profiling counters per phase (0=parent App+margin, 1=parent App_baseline,
    // 2=child app_reuse, 3=child App_baseline, 4=unknown/other)
    static int prof_phase;
    // Runtime 替代原 #define _USE_LSa_AS_LAYER_（App_test 每节点额外算一遍独立 LSa 当剪枝层）。
    // 默认 true = 层关闭（实测为死重量）；--lsa_layer 1 可开回宏时期行为。
    static bool disable_lsa_layer;
    static size_t prof_h1_count[5];
    static size_t prof_h0_count[5];
    static size_t prof_lsa_count[5];
    static double prof_h1_us[5];
    static double prof_h0_us[5];
    static double prof_lsa_us[5];
    // app_reuse() per-section timers
    static size_t prof_reuse_calls;
    static size_t prof_reuse_snapshot_nodes;
    static double prof_reuse_init_us;
    static double prof_reuse_rebuild_us;
    static double prof_reuse_refresh_us;
    static double prof_reuse_leaves_us;
    static double prof_reuse_astar_us;
    // astar inner breakdown (sums to prof_reuse_astar_us)
    static double prof_reuse_astar_pop_us;
    static double prof_reuse_astar_recompute_us;
    static double prof_reuse_astar_sibling_us;
    static double prof_reuse_astar_child_us;
    static double prof_reuse_astar_bookkeep_us;
    // state flow counters (per app_reuse() call summed)
    static size_t prof_reuse_snap_leaves;
    static size_t prof_reuse_leaves_pruned;     // refresh 後 lb >= ub+margin 直接 prune
    static size_t prof_reuse_leaves_to_heap;    // lb < ub，進 heap
    static size_t prof_reuse_leaves_boundary;   // ub <= lb < ub+margin
    static size_t prof_reuse_leaves_invalid;    // image >= g_n (dummy)
    static size_t prof_reuse_astar_iter;        // main loop pop 次數
    static size_t prof_reuse_astar_recomp_pruned;  // lazy recompute 後 prune 的
    // Diagnostic: app_reuse exit reasons
    static size_t prof_reuse_astar_exhausted;   // A* finished without finding <=tau solution
    // Intersection-aware break stats (only counted when lazy_recompute_target_siblings != nullptr)
    static size_t prof_recompute_calls;             // # times BM sibling loop ran with target
    static size_t prof_recompute_target_total_sum;  // sum of target_total across calls
    static size_t prof_recompute_iters_sum;         // sum of actual H(0) iterations performed
    static size_t prof_recompute_n_sum;             // sum of n (candidates count) — upper bound on iters
    static size_t prof_recompute_found_sum;         // sum of found_in_target at exit
    static size_t prof_recompute_typeB_sum;         // sum of (non-target images computed and stored)
    static size_t prof_recompute_break_lb;          // # calls broke via _EARLY_STOP_
    static size_t prof_recompute_break_intersect;   // # calls broke via intersection-full
    static size_t prof_recompute_break_loop_end;    // # calls ran to natural end (i == n)
    // Per-call last exit reason: 0=ok, 1=mo_size_mismatch (gate bug),
    // 3=astar_exhausted, 4=no_dummy (rebuild failed), 5=state_pool_exhausted
    static int last_reuse_exit_reason;
    static void reset_profile() {
        for (int i = 0; i < 5; ++i) {
            prof_h1_count[i] = prof_h0_count[i] = prof_lsa_count[i] = 0;
            prof_h1_us[i] = prof_h0_us[i] = prof_lsa_us[i] = 0;
        }
        prof_reuse_calls = 0;
        prof_reuse_snapshot_nodes = 0;
        prof_reuse_init_us = prof_reuse_rebuild_us = prof_reuse_refresh_us = 0;
        prof_reuse_leaves_us = prof_reuse_astar_us = 0;
        prof_reuse_astar_pop_us = prof_reuse_astar_recompute_us = 0;
        prof_reuse_astar_sibling_us = prof_reuse_astar_child_us = 0;
        prof_reuse_astar_bookkeep_us = 0;
        prof_reuse_snap_leaves = 0;
        prof_reuse_leaves_pruned = prof_reuse_leaves_to_heap = 0;
        prof_reuse_leaves_boundary = prof_reuse_leaves_invalid = 0;
        prof_reuse_astar_iter = prof_reuse_astar_recomp_pruned = 0;
        prof_reuse_astar_exhausted = 0;
        prof_recompute_calls = 0;
        prof_recompute_target_total_sum = 0;
        prof_recompute_iters_sum = 0;
        prof_recompute_n_sum = 0;
        prof_recompute_found_sum = 0;
        prof_recompute_typeB_sum = 0;
        prof_recompute_break_lb = 0;
        prof_recompute_break_intersect = 0;
        prof_recompute_break_loop_end = 0;
    }
    static void print_profile() {
        const char* names[5] = {"P.App+m", "P.AppBase", "C.reuse", "C.AppBase", "other"};
        printf("[PROFILE] %-12s %10s %12s %10s %12s %10s %12s\n",
               "phase", "H(1)_cnt", "H(1)_ms", "H(0)_cnt", "H(0)_ms", "LSa_cnt", "LSa_ms");
        for (int i = 0; i < 5; ++i) {
            if (prof_h1_count[i] + prof_h0_count[i] + prof_lsa_count[i] == 0) continue;
            printf("[PROFILE] %-12s %10zu %12.1f %10zu %12.1f %10zu %12.1f\n",
                   names[i], prof_h1_count[i], prof_h1_us[i]/1000.0,
                   prof_h0_count[i], prof_h0_us[i]/1000.0,
                   prof_lsa_count[i], prof_lsa_us[i]/1000.0);
        }
        if (prof_reuse_calls > 0) {
            double tot_us = prof_reuse_init_us + prof_reuse_rebuild_us + prof_reuse_refresh_us
                          + prof_reuse_leaves_us + prof_reuse_astar_us;
            double tot_ms = tot_us / 1000.0;
            printf("[PROFILE REUSE] %zu calls, %zu snapshot nodes total (%.1f avg), total %.1f ms\n",
                   prof_reuse_calls, prof_reuse_snapshot_nodes,
                   (double)prof_reuse_snapshot_nodes / prof_reuse_calls, tot_ms);
            auto pct = [tot_us](double v){ return tot_us > 0 ? 100.0*v/tot_us : 0.0; };
            printf("[PROFILE REUSE] %-10s %10s %10s\n", "section", "ms", "%");
            printf("[PROFILE REUSE] %-10s %10.1f %9.1f%%\n", "init",    prof_reuse_init_us/1000.0,    pct(prof_reuse_init_us));
            printf("[PROFILE REUSE] %-10s %10.1f %9.1f%%\n", "rebuild", prof_reuse_rebuild_us/1000.0, pct(prof_reuse_rebuild_us));
            printf("[PROFILE REUSE] %-10s %10.1f %9.1f%%\n", "refresh", prof_reuse_refresh_us/1000.0, pct(prof_reuse_refresh_us));
            printf("[PROFILE REUSE] %-10s %10.1f %9.1f%%\n", "leaves",  prof_reuse_leaves_us/1000.0,  pct(prof_reuse_leaves_us));
            printf("[PROFILE REUSE] %-10s %10.1f %9.1f%%\n", "astar",   prof_reuse_astar_us/1000.0,   pct(prof_reuse_astar_us));
            double astar_us = prof_reuse_astar_us;
            auto pct_a = [astar_us](double v){ return astar_us > 0 ? 100.0*v/astar_us : 0.0; };
            printf("[PROFILE REUSE]   astar inner breakdown (%% of astar):\n");
            printf("[PROFILE REUSE]   %-10s %10.1f %9.1f%%\n", "pop",       prof_reuse_astar_pop_us/1000.0,       pct_a(prof_reuse_astar_pop_us));
            printf("[PROFILE REUSE]   %-10s %10.1f %9.1f%%\n", "recompute", prof_reuse_astar_recompute_us/1000.0, pct_a(prof_reuse_astar_recompute_us));
            printf("[PROFILE REUSE]   %-10s %10.1f %9.1f%%\n", "sibling",   prof_reuse_astar_sibling_us/1000.0,   pct_a(prof_reuse_astar_sibling_us));
            printf("[PROFILE REUSE]   %-10s %10.1f %9.1f%%\n", "child",     prof_reuse_astar_child_us/1000.0,     pct_a(prof_reuse_astar_child_us));
            printf("[PROFILE REUSE]   %-10s %10.1f %9.1f%%\n", "bookkeep",  prof_reuse_astar_bookkeep_us/1000.0,  pct_a(prof_reuse_astar_bookkeep_us));
            // State flow accounting
            size_t leaves_dispatched = prof_reuse_leaves_pruned + prof_reuse_leaves_to_heap
                                     + prof_reuse_leaves_boundary + prof_reuse_leaves_invalid;
            printf("[PROFILE FLOW] snap_leaves=%zu  dispatched=%zu\n",
                   prof_reuse_snap_leaves, leaves_dispatched);
            printf("[PROFILE FLOW]   pruned (lb>=ub+margin) : %zu\n", prof_reuse_leaves_pruned);
            printf("[PROFILE FLOW]   to_heap (lb<ub)        : %zu\n", prof_reuse_leaves_to_heap);
            printf("[PROFILE FLOW]   boundary               : %zu\n", prof_reuse_leaves_boundary);
            printf("[PROFILE FLOW]   invalid (image>=g_n)   : %zu\n", prof_reuse_leaves_invalid);
            printf("[PROFILE FLOW] astar main iter   : %zu\n", prof_reuse_astar_iter);
            printf("[PROFILE FLOW] astar recomp_pruned: %zu\n", prof_reuse_astar_recomp_pruned);
            printf("[PROFILE EXIT] astar_exhausted=%zu\n", prof_reuse_astar_exhausted);
            if (prof_recompute_calls > 0) {
                double C = (double)prof_recompute_calls;
                double avg_target = (double)prof_recompute_target_total_sum / C;
                double avg_iters  = (double)prof_recompute_iters_sum / C;
                double avg_n      = (double)prof_recompute_n_sum / C;
                double avg_found  = (double)prof_recompute_found_sum / C;
                double avg_typeB  = (double)prof_recompute_typeB_sum / C;
                printf("[PROFILE BMI] recompute calls=%zu\n", prof_recompute_calls);
                printf("[PROFILE BMI]   avg n (candidates)            : %.2f\n", avg_n);
                printf("[PROFILE BMI]   avg target_total (含 self)    : %.2f\n", avg_target);
                printf("[PROFILE BMI]   avg H(0) iters actually run   : %.2f\n", avg_iters);
                printf("[PROFILE BMI]   avg target image found at exit: %.2f / %.2f (%.1f%%)\n",
                       avg_found, avg_target, avg_target > 0 ? 100.0*avg_found/avg_target : 0.0);
                printf("[PROFILE BMI]   avg non-target H(0) (Type B)  : %.2f  (%.1f%% of iters)\n",
                       avg_typeB, avg_iters > 0 ? 100.0*avg_typeB/avg_iters : 0.0);
                printf("[PROFILE BMI]   break reasons (calls / %%):\n");
                printf("[PROFILE BMI]     _EARLY_STOP_ (lb >= ub)     : %zu (%.1f%%)\n",
                       prof_recompute_break_lb, 100.0*prof_recompute_break_lb/C);
                printf("[PROFILE BMI]     intersection full           : %zu (%.1f%%)\n",
                       prof_recompute_break_intersect, 100.0*prof_recompute_break_intersect/C);
                printf("[PROFILE BMI]     ran to loop end (no break)  : %zu (%.1f%%)\n",
                       prof_recompute_break_loop_end, 100.0*prof_recompute_break_loop_end/C);
            }
        }
    }

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
	static double compute_ged(Graph* graph1, Graph* graph2, ui upper_bound_value = INF);

	void print_graph(const Graph *g, const char* graph_name);

	long long get_search_space() { return search_space; }

	void extract_snapshot(SearchSnapshot& out);
	/*------------------------------------------------------------
     * ② 复用上一轮快照继续搜索（g -> g′ 仅一次微编辑）
     *    snap  : g 上一轮搜索终止时的 OPEN+CLOSED 快照
     *    tau   : 新一轮 upper-bound
     *    delta : 描述 g->g′ 的单步编辑，后续可用于动态 LB 补丁
     *  返回值   : 新的 upper_bound (即 GED 或剪枝上界)
     *-----------------------------------------------------------*/
    ui app_reuse(const SearchSnapshot& snap,
                 ui tau,
                 ui ged_gap = 1);

	void compute_mapping_order_reuse();
	/* --- 下界重计算（LSa 专用） --- */
	void recompute_lsa_lb(State* now);
	
	inline bool is_real_node(const State* st) {
		return st->image < g_n && st->level < q_n;
	}
	std::pair<ui, ui> lsa_lower_bound_exact(State* node);
	std::pair<ui, ui> bruteforce_LSa_lower_bound(State* node);
	void recompute_node_lazy(State* now);
	// BMao-recomp 用：純算 leaf 在 g' 上的 BMao lb（無 reuse 交集策略，不寫 siblings）。
	// 直接構造 candidates → compute_best_extension_BM(no_siblings=1)。
	void refresh_leaf_lb_only_BM(State* now);
	// 记号: AppForComputation 是否因耗尽 exact_max_iter 而退出(true=未定, false=界可信)。
	// NetDagConstructor 用它把 timeout(未定) 和 over_threshold(确证超阈值) 分开。
	bool exact_hit_iter_limit = false;
	bool hit_iter_limit() const { return exact_hit_iter_limit; }
	ui get_overall_lb() const { return overall_lb; }
	// E8 Gisma-LSa: 把另一个 Application(lb=LSa) 算出的 overall_lb 注入本对象，
	// 供 dfs_traverse 的 LB Propagation / Subtree Pruning 使用（仍是合法下界）。
	void set_overall_lb_for_propagation(ui v) { overall_lb = v; }
	ui get_overall_ub() const { return overall_ub; }
	void set_margin(int m) { margin = m; }
    int get_margin() const { return margin; }
	void set_disable_lsa_pruning(bool v) { disable_lsa_pruning = v; }
	void set_disable_reuse_lsa(bool v) { disable_reuse_lsa = v; }
	void set_exact_value_mode(bool v) { exact_value_mode = v; }
	void set_lazy_parent_lsa(bool v) { lazy_parent_lsa = v; }
	void set_all_edge_labels_same(bool v) { all_edge_labels_same = v; }
	void set_refresh_bmao_recompute(bool v) { refresh_bmao_recompute = v; }
	void set_bmao_recomp_intersect(bool v) { bmao_recomp_intersect = v; }
	void set_skip_lazy_recompute(bool v) { skip_lazy_recompute = v; }
	void set_early_stop_at_tau(bool v) { early_stop_at_tau = v; }
	void set_skip_intersection_in_reuse(bool v) { skip_intersection_in_reuse = v; }
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

    /* 把 vertex 及其对应 State 一起传进来 */
    void patch_vertex(State* st, bool forward);

    /* --- 辅助：在当前账本状态下，给 node 填 mapped_cost / lower_bound --- */
    void calc_lb_from_tables(State* node);

	/* ====== ReuseLSa_DFS 内部用 ====== */
	size_t VCAP = 0;            // vlabels_map 容量（原 file-scope static，移入实例避免线程竞争）
	size_t ECAP = 0;            // ecnt / elabels_matrix 列数
	size_t MCAP = 0;            // elabels_matrix 列数（==ECAP）
	std::vector<int> ecnt;      // 边标签余缺

	void ensure_vlabel_cap(ui lbl);
	void ensure_elabel_cap(ui lbl);
	ui compute_mapped_cost_from_scratch(State* node);
	ui compute_independent_lower_bound_LSa(State* node);
	ui compute_mapped_cost_baseline(State* node);
	ui compute_independent_lower_bound_LSa_baseline(State* node);
	// 基于文章定义的LSa下界计算
	

	// 辅助函数：计算两个多重集的距离
	ui compute_multiset_distance(const std::map<ui, int>& set1, 
							const std::map<ui, int>& set2);
	void compute_best_extension_LSa_baseline(State *now, ui candidate_n, ui *candidates, ui pre_siblings) ;
	void compute_best_extension_BMa_baseline(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings) ;
	
	
	// 在Application.h的public部分添加以下声明：

	/*
	* BM算法的batch处理函数声明
	*/

	// 简化接口：处理候选列表构建，然后调用核心函数
	void compute_specified_children_BM(
		char anchor_aware,
		State *parent,
		ui *specified_images,
		ui specified_count,
		ui *results
	);

	// 核心实现：直接基于compute_best_extension_BM改造的指定节点版本
	void compute_best_extension_BM_specified(
		char anchor_aware,
		State *parent,
		ui n, ui *candidates,
		ui *specified_images,
		ui specified_count,
		ui *results
	);

	// 主推版本：BM算法的搜索树批量刷新函数
	void refresh_bm_snapshot_lb_batch(State* root,
                                     ui snap_nodes,
                                     ui ged_gap);
};

#endif
