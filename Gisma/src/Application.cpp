#include "Application.h"
#include "Utility.h"
#include "ReuseSearchTree.h"     // placed right after Application.h

#include <chrono>
#include <fstream>
#include <unordered_map>

using namespace std;

// ========== AStar Trace Support ==========
static ofstream* g_astar_trace_file = nullptr;
static int g_astar_trace_counter = 0;
static unordered_map<State*, int> g_astar_state_id_map;

void astar_trace_init(const string& filename) {
    if (g_astar_trace_file) {
        g_astar_trace_file->close();
        delete g_astar_trace_file;
    }
    g_astar_trace_file = new ofstream(filename);
    g_astar_trace_counter = 0;
    g_astar_state_id_map.clear();

    if (g_astar_trace_file->is_open()) {
        *g_astar_trace_file << "{\"nodes\":[\n";
    }
}

void astar_trace_node(State* state, State* parent, const ui* MO, bool is_expanded, bool is_pruned, bool is_complete) {
    if (!g_astar_trace_file || !g_astar_trace_file->is_open()) return;

    int node_id = g_astar_trace_counter++;
    g_astar_state_id_map[state] = node_id;

    int parent_id = -1;
    if (parent && g_astar_state_id_map.count(parent)) {
        parent_id = g_astar_state_id_map[parent];
    }

    if (node_id > 0) *g_astar_trace_file << ",\n";

    *g_astar_trace_file << "{\"id\":" << node_id
                        << ",\"parent\":" << parent_id
                        << ",\"level\":" << state->level
                        << ",\"cost\":" << state->mapped_cost
                        << ",\"lb\":" << state->lower_bound
                        << ",\"expanded\":" << (is_expanded ? "true" : "false")
                        << ",\"pruned\":" << (is_pruned ? "true" : "false")
                        << ",\"complete\":" << (is_complete ? "true" : "false")
                        << ",\"mapping\":[";

    // Reconstruct mapping by backtracking parent chain
    // Collect path from root to current node
    std::vector<State*> path;
    State* curr = state;
    while (curr != nullptr) {
        path.push_back(curr);
        curr = curr->parent;
    }

    // Reverse path, root to current
    std::reverse(path.begin(), path.end());

    // Output mapping [q_node, g_node]
    // path contains all nodes; each node's level indicates level+1 query nodes have been mapped
    bool first = true;
    for (size_t i = 0; i < path.size(); i++) {
        // path[i]'s level indicates this node maps to the level-th query vertex (0-indexed)
        // i.e., mapping MO[level] to path[i]->image
        if (!first) *g_astar_trace_file << ",";
        ui q_node = MO[path[i]->level];
        ui g_node = path[i]->image;
        *g_astar_trace_file << "[" << q_node << "," << g_node << "]";
        first = false;
    }

    *g_astar_trace_file << "]}";
    g_astar_trace_file->flush();
}

void astar_trace_close() {
    if (g_astar_trace_file && g_astar_trace_file->is_open()) {
        *g_astar_trace_file << "\n]}\n";
        g_astar_trace_file->close();
        delete g_astar_trace_file;
        g_astar_trace_file = nullptr;
    }
    g_astar_state_id_map.clear();
}
// ========== End of AStar Trace Support ==========

void print_state(const State* state) {
    if (state == nullptr) {
        std::cout << "State is nullptr" << std::endl;
        return;
    }

    printf("State details: Parent: %p, Previous Sibling: %p, Level: %u, Image: %u, Mapped Cost: %u, Lower Bound: %u, CS Count: %u",
           state->parent, state->pre_sibling, state->level, state->image, state->mapped_cost, state->lower_bound, state->cs_cnt);

#ifdef _EXPAND_ALL_
    printf(", Siblings: ");
    if (state->siblings != nullptr) {
        for (ushort i = 0; i < state->siblings_n; ++i) {
            printf("%u ", state->siblings[i]);
        }
    } else {
        printf("nullptr");
    }
    printf(", Siblings Count: %u", state->siblings_n);
#endif
    printf("\n");
}

Application::Application(ui _verify_upper_bound, const char *lower_bound, int _app_max_iter, int _exact_max_iter) {
	q_n = g_n = 0;

	q_starts = q_edges = q_vlabels = q_elabels = NULL;
	g_starts = g_edges = g_vlabels = g_elabels = NULL;

	vlabels_n = elabels_n = 0;

	verify_upper_bound = _verify_upper_bound;
	upper_bound = verify_upper_bound+1;
	app_max_iter = _app_max_iter;
	exact_max_iter = _exact_max_iter;

	MO = NULL;
	search_n_for_IS = search_n = 0;

	search_space = 0;

	q_matrix = NULL;
	elabels_map = vlabels_map = NULL;

	visX = visY = NULL;
	mx = my = NULL;
	BX = NULL;
	queue = prev = NULL;
	candidates = NULL;

	q_g_swapped = false;

	states_pool_n = siblings_pool_n = 0;

	visited_siblings = NULL;
	visited_siblings_n = NULL;

	cost = NULL;
	lx = ly = slack = slackmy = NULL;

	elabels_matrix = NULL;

	children = NULL;

	// ========== Additional initialization ==========
	
	// 1. Container initialization (vectors auto-initialize to empty, but explicit clearing is safer)
	open_heap.clear();
	full_mapping_nodes.clear();
	boundary_nodes.clear();
	
	// 2. Memory pool related
	states_pool.clear();
	siblings_pool.clear();
	states_memory.clear();
	siblings_memory.clear();
	
	// 3. Other possible members (depending on your code)
	// mapping_found (if this member exists)
	// best_mapping (if this member exists)
	
	// ========== Original code ==========
	if(strcmp(lower_bound, "LSa") == 0) lb_method = LSa;
	else if(strcmp(lower_bound, "BMao") == 0) lb_method = BMao;
	else if(strcmp(lower_bound, "BMa") == 0) lb_method = BMa;
	else {
		lb_method = LSa;
		printf("Lower bound %s is not available and LSa is used by default!\n", lower_bound);
	}
}

Application::~Application() {
	// ========== Clean up plain arrays ==========
	if(q_starts != NULL) {
		delete[] q_starts;
		q_starts = NULL;
	}
	if(q_edges != NULL) {
		delete[] q_edges;
		q_edges = NULL;
	}
	if(q_vlabels != NULL) {
		delete[] q_vlabels;
		q_vlabels = NULL;
	}
	if(q_elabels != NULL) {
		delete[] q_elabels;
		q_elabels = NULL;
	}

	if(g_starts != NULL) {
		delete[] g_starts;
		g_starts = NULL;
	}
	if(g_edges != NULL) {
		delete[] g_edges;
		g_edges = NULL;
	}
	if(g_vlabels != NULL) {
		delete[] g_vlabels;
		g_vlabels = NULL;
	}
	if(g_elabels != NULL) {
		delete[] g_elabels;
		g_elabels = NULL;
	}
	if(elabels_matrix != NULL) {
		delete[] elabels_matrix;
		elabels_matrix = NULL;
	}
	if(visited_siblings != NULL) {
		delete[] visited_siblings;
		visited_siblings = NULL;
	}
	if(visited_siblings_n != NULL) {
		delete[] visited_siblings_n;
		visited_siblings_n = NULL;
	}
	if(MO != NULL) {
		delete[] MO;
		MO = NULL;
	}
	if(visX != NULL) {
		delete[] visX;
		visX = NULL;
	}
	if(visY != NULL) {
		delete[] visY;
		visY = NULL;
	}
	if(mx != NULL) {
		delete[] mx;
		mx = NULL;
	}
	if(my != NULL) {
		delete[] my;
		my = NULL;
	}
	if(BX != NULL) {
		delete[] BX;
		BX = NULL;
	}
	if(queue != NULL) {
		delete[] queue;
		queue = NULL;
	}
	if(prev != NULL) {
		delete[] prev;
		prev = NULL;
	}
	if(candidates != NULL) {
		delete[] candidates;
		candidates = NULL;
	}
	if(cost != NULL) {
		delete[] cost;
		cost = NULL;
	}
	if(lx != NULL) {
		delete[] lx;
		lx = NULL;
	}
	if(ly != NULL) {
		delete[] ly;
		ly = NULL;
	}
	if(slack != NULL) {
		delete[] slack;
		slack = NULL;
	}
	if(slackmy != NULL) {
		delete[] slackmy;
		slackmy = NULL;
	}
	if(q_matrix != NULL) {
		delete[] q_matrix;
		q_matrix = NULL;
	}
	if(elabels_map != NULL) {
		delete[] elabels_map;
		elabels_map = NULL;
	}
	if(vlabels_map != NULL) {
		delete[] vlabels_map;
		vlabels_map = NULL;
	}
	if(children != NULL) {
		delete[] children;
		children = NULL;
	}

	// ========== Clean up memory pool ==========
	for(ui i = 0;i < states_memory.size();i ++) {
		delete[] states_memory[i];
		states_memory[i] = NULL;
	}
	states_memory.clear();
	
	for(ui i = 0;i < siblings_memory.size();i ++) {
		delete[] siblings_memory[i];
		siblings_memory[i] = NULL;
	}
	siblings_memory.clear();
	
	// ========== Additional: clean up containers ==========
	// These containers hold pointers, but pointed-to objects are managed by memory pool
	// so only the containers themselves need clearing
	open_heap.clear();
	full_mapping_nodes.clear();
    boundary_nodes.clear();
	
	// Clear pool vectors (memory already freed through states_memory and siblings_memory)
	states_pool.clear();
	siblings_pool.clear();
	
	// Reset counters
	states_pool_n = 0;
	siblings_pool_n = 0;
}

// Initialize with Graph objects
void Application::init(const Graph *g, const Graph *q) {
    search_snapshot.clear();          // Clean up previous search_snapshot state
    open_heap.clear();                // Clean up search state containers
    full_mapping_nodes.clear();
    boundary_nodes.clear();
    states_pool.clear();              // Clean up memory pool
    siblings_pool.clear();
    for(ui i = 0; i < states_memory.size(); i++) {
        delete[] states_memory[i];
    }
    states_memory.clear();            // Clean up memory management containers
    for(ui i = 0; i < siblings_memory.size(); i++) {
        delete[] siblings_memory[i];
    }
    siblings_memory.clear();
    states_pool_n = 0;
    siblings_pool_n = 0;
    q_n = q->n;
    q_starts = new ui[q_n+1]; q_vlabels = new ui[q_n];
    q_edges = new ui[q->m]; q_elabels = new ui[q->m];
    memcpy(q_starts, q->pstarts, sizeof(ui)*(q_n+1));
    memcpy(q_vlabels, q->vlabels, sizeof(ui)*q_n);
    memcpy(q_edges, q->edges, sizeof(ui)*q->m);
    memcpy(q_elabels, q->elabels, sizeof(ui)*q->m);

    g_n = g->n;
    g_starts = new ui[g_n+1]; g_vlabels = new ui[g_n];
    g_edges = new ui[g->m]; g_elabels = new ui[g->m];
    memcpy(g_starts, g->pstarts, sizeof(ui)*(g_n+1));
    memcpy(g_vlabels, g->vlabels, sizeof(ui)*g_n);
    memcpy(g_edges, g->edges, sizeof(ui)*g->m);
    memcpy(g_elabels, g->elabels, sizeof(ui)*g->m);

    preprocess();
}

void Application::init(const std::vector<std::pair<int,int> > &g_v, const std::vector<std::pair<std::pair<int,int>,int> > &g_e, const std::vector<std::pair<int,int> > &q_v, const std::vector<std::pair<std::pair<int,int>,int> > &q_e) {
	search_snapshot.clear();          // * Key: clear before each new search
	q_n = q_v.size();
	q_starts = new ui[q_n+1]; q_vlabels = new ui[q_n];
	for(ui i = 0;i < q_n;i ++) q_vlabels[i] = q_v[i].second;

	q_edges = new ui[q_e.size()]; q_elabels = new ui[q_e.size()];
	ui idx = 0;
	for(ui i = 0;i < q_e.size();i ++) {
		q_edges[i] = q_e[i].first.second;
		q_elabels[i] = q_e[i].second;
		if(i == 0||q_e[i].first.first != q_e[i-1].first.first) {
			while(idx <= q_e[i].first.first) q_starts[idx ++] = i;
		}
	}
	while(idx <= q_n) q_starts[idx ++] = q_e.size();

	g_n = g_v.size();
	g_starts = new ui[g_n+1]; g_vlabels = new ui[g_n];
	for(ui i = 0;i < g_n;i ++) g_vlabels[i] = g_v[i].second;

	g_edges = new ui[g_e.size()]; g_elabels = new ui[g_e.size()];
	idx = 0;
	for(ui i = 0;i < g_e.size();i ++) {
		g_edges[i] = g_e[i].first.second;
		g_elabels[i] = g_e[i].second;
		if(i == 0||g_e[i].first.first != g_e[i-1].first.first) {
			while(idx <= g_e[i].first.first) g_starts[idx ++] = i;
		}
	}
	while(idx <= g_n) g_starts[idx ++] = g_e.size();

	preprocess();
}

void Application::print_graph(const Graph *g, const char* graph_name) {
    std::cout << "Graph: " << graph_name << std::endl;
    // std::cout << "ID: " << g->id << std::endl;
    std::cout << "Nodes (n): " << g->n << std::endl;
    std::cout << "Edges (m): " << g->m << std::endl;

    std::cout << "Nodes: ";
    for (ui i = 0; i < g->n; ++i) {
        std::cout << "(" << i << ", " << g->vlabels[i] << ") ";
    }
    std::cout << std::endl;

    std::cout << "Edges: ";
    for (ui i = 0; i < g->n; ++i) {
        for (ui j = g->pstarts[i]; j < g->pstarts[i+1]; ++j) {
            std::cout << "(" << i << ", " << g->edges[j] << ", " << g->elabels[j] << ") ";
        }
    }
    std::cout << std::endl;
}





Graph* Application::create_graph(const char* id, unsigned int n, unsigned int m, int* vertices_data, int* edges_data) {
    std::string graph_id(id);
    std::vector<std::pair<int, ui>> vertices;
    std::vector<std::pair<std::pair<int, int>, ui>> edges;

    for (ui i = 0; i < n; ++i) {
        int node = vertices_data[2 * i];
        ui label = static_cast<ui>(vertices_data[2 * i + 1]);
        vertices.push_back(std::make_pair(node, label));
    }

    for (ui i = 0; i < m; ++i) {
        int u = edges_data[3 * i];
        int v = edges_data[3 * i + 1];
        ui label = static_cast<ui>(edges_data[3 * i + 2]);
        edges.push_back(std::make_pair(std::make_pair(u, v), label));
        edges.push_back(std::make_pair(std::make_pair(v, u), label));
    }

    std::sort(vertices.begin(), vertices.end());
    for (ui i = 0; i < vertices.size(); ++i) {
        assert(vertices[i].first == i);
    }

    std::sort(edges.begin(), edges.end());
    for (ui i = 0; i < edges.size(); ++i) {
        assert(edges[i].first.first >= 0 && edges[i].first.first < vertices.size());
        assert(edges[i].first.second >= 0 && edges[i].first.second < vertices.size());
        if (i > 0) {
            assert(edges[i].first != edges[i - 1].first);
        }
        assert(edges[i].second < vertices.size());
    }

    return new Graph(graph_id, vertices, edges);
}



void Application::delete_graph(Graph* g) {
    delete g;
}



// (reverted) remove helper namespace

double Application::verify_ged(Graph* graph1, Graph* graph2, ui upper_bound_value) {
	// return 1;
	if (upper_bound_value != INF){
		// : use fixed 1000-size buffers as in original code
		int *vlabel_cnt1 = new int[1000]();
		int *elabel_cnt1 = new int[1000]();
		int *degree_q1 = new int[1000]();
		int *degree_g1 = new int[1000]();
		int *tmp1 = new int[1000]();
		ui lower_bound = graph1->ged_lower_bound_filter(graph2, upper_bound_value, vlabel_cnt1, elabel_cnt1, degree_q1, degree_g1, tmp1);
		delete[] vlabel_cnt1;
		delete[] elabel_cnt1;
		delete[] degree_q1;
		delete[] degree_g1;
		delete[] tmp1;
		if (lower_bound > upper_bound_value) {
			return INF - 1;
		}
	}
    Application *app = new Application(upper_bound_value, "BMao");
    app->init(graph1, graph2);
	// Print graph information
	// app->print_graph(graph1, "graph1");
	// app->print_graph(graph2, "graph2");
    double result = app->App();
	
	
    delete app;

    return result;
}

double Application::compute_ged(Graph* graph1, Graph* graph2, ui upper_bound_value) {
	
	if (upper_bound_value != INF){
		int *vlabel_cnt1 = new int[1000]();
		int *elabel_cnt1 = new int[1000]();
		int *degree_q1 = new int[1000]();
		int *degree_g1 = new int[1000]();
		int *tmp1 = new int[1000]();

		ui lower_bound = graph1->ged_lower_bound_filter(graph2, upper_bound_value, vlabel_cnt1, elabel_cnt1, degree_q1, degree_g1, tmp1);

		delete[] vlabel_cnt1;
		delete[] elabel_cnt1;
		delete[] degree_q1;
		delete[] degree_g1;
		delete[] tmp1;
		
		if (lower_bound > upper_bound_value) {
			return INF - 1;
		}
	}
	// return 1;
    Application *app = new Application(upper_bound_value, "BMao");
    app->init(graph1, graph2);
	// Print graph information
	// app->print_graph(graph1, "graph1");
	// app->print_graph(graph2, "graph2");
    double result = app->AppForComputation(nullptr, nullptr);
	
	
    delete app;

    return result;
}



std::map<std::string, ui> vM;
std::map<std::string, ui> eM;

void Application::preprocess() {
	q_g_swapped = false;
	if(q_n > g_n) {
#ifndef NDEBUG
		printf("q and g are swapped!\n");
#endif
		swap(q_n, g_n);
		swap(q_starts, g_starts);
		swap(q_edges, g_edges);
		swap(q_vlabels, g_vlabels);
		swap(q_elabels, g_elabels);
		q_g_swapped = true;
	}

	vlabels_n = relabel(q_n, q_vlabels, g_n, g_vlabels);
	elabels_n = relabel(q_starts[q_n], q_elabels, g_starts[g_n], g_elabels);

	if(vlabels_map != NULL) delete[] vlabels_map;
	// vlabels_map = new int[vlabels_n];
    vlabels_map = NULL;  // Keep as NULL
	// memset(vlabels_map, 0, sizeof(int)*vlabels_n);

	if(elabels_map != NULL) delete[] elabels_map;
	elabels_map = new int[elabels_n];
	memset(elabels_map, 0, sizeof(int)*elabels_n);

	if(visX != NULL) delete[] visX;
	visX = new char[g_n];
	memset(visX, 0, sizeof(char)*g_n);

	if(visY != NULL) delete[] visY;
	visY = new char[g_n];
	memset(visY, 0, sizeof(char)*g_n);

	// to handle large graphs, this should be replaced by a hash map
	if(q_matrix != NULL) delete[] q_matrix;
	q_matrix = new uchar[q_n*q_n];
	for(ui i = 0;i < q_n;i ++) {
		uchar *t_array = q_matrix+i*q_n;
		for(ui j = 0;j < q_n;j ++) t_array[j] = elabels_n;
		for(ui j = q_starts[i];j < q_starts[i+1];j ++) t_array[q_edges[j]] = q_elabels[j];
	}

	if(candidates != NULL) delete[] candidates;
	candidates = new ui[g_n];
	if(queue != NULL) delete[] queue;
	queue = new ui[g_n];
	if(prev != NULL) delete[] prev;
	prev = new ui[g_n];
	if(mx != NULL) delete[] mx;
	mx = new int[g_n];
	if(my != NULL) delete[] my;
	my = new int[g_n];
	if(BX != NULL) delete[] BX;
	BX = new ui[q_n];

	// Lazy-allocated children may overflow if g_n grows, free old ones here
	if(children != NULL) { delete[] children; children = NULL; }

	if(lb_method == BMa||lb_method == BMao) {
		if(cost != NULL) delete[] cost;
		cost = new ui[g_n*g_n];
		if(lx != NULL) delete[] lx;
		lx = new int[g_n];
		if(ly != NULL) delete[] ly;
		ly = new int[g_n];
		if(slack != NULL) delete[] slack;
		slack = new int[g_n];
		if(slackmy != NULL) delete[] slackmy;
		slackmy = new int[g_n];
	}
}

ui Application::relabel(ui len1, ui *array1, ui len2, ui *array2) {
	vector<ui> labels;
	labels.reserve(len1+len2);
	for(ui i = 0;i < len1;i ++) labels.pb(array1[i]);
	for(ui i = 0;i < len2;i ++) labels.pb(array2[i]);

	sort(labels.begin(), labels.end());
	ui labels_n = 1;
	for(ui i = 1;i < labels.size();i ++) if(labels[i] != labels[i-1]) {
		labels[labels_n ++] = labels[i];
	}

	for(ui i = 0;i < len1;i ++) array1[i] = search_index(array1[i], labels, labels_n);
	for(ui i = 0;i < len2;i ++) array2[i] = search_index(array2[i], labels, labels_n);

	return labels_n;
}

ui Application::search_index(ui val, std::vector<ui> &array, ui array_len) {
	assert(array[0] <= val);
	ui low = 0, high = array_len-1;
	while(low <= high) {
		ui mid = (low+high)/2;
		if(array[mid] == val) return mid;
		if(array[mid] < val) low = mid + 1;
		else high = mid - 1;
	}
	printf("%u is not in the array!\n", val);
	return 0;
}

ui Application::DFS(State *node) {
	if(MO == NULL) compute_mapping_order();

	ui start_level = 0;
	if(node != NULL) start_level = node->level;
	State **stack = NULL;
	if(search_n > start_level) stack = new State*[search_n - start_level];

	ui start = 0;
	if(node != NULL) {
		start = 1;
		stack[0] = node;
	}
	State *tt = NULL;
	if(search_n > start_level + start) tt = new State[search_n-start_level-start];
	for(ui i = start;i < search_n - start_level;i ++) {
		stack[i] = &tt[i-start];
#ifdef _EXPAND_ALL_
		stack[i]->siblings = NULL;
		if(g_n > i + start_level + 1) stack[i]->siblings = new ushort[(g_n - i - start_level - 1)*2];
#endif
	}

#ifndef _EXPAND_ALL_
	if(visited_siblings == NULL) {
		visited_siblings = new ushort[q_n*g_n];
		visited_siblings_n = new ushort[q_n];
	}
#endif

	if(node == NULL) generate_best_extension(NULL, stack[0]);

	State full;

#ifndef NDEBUG
	ui *cnt = new ui[q_n+g_n+q_starts[q_n]+g_starts[g_n]];
	memset(cnt, 0, sizeof(ui)*(q_n+g_n+q_starts[q_n]+g_starts[g_n]));
	ui total = 0;
#endif

	int idx = 0;
	while(idx >= 0&&(verify_upper_bound == INF||upper_bound > verify_upper_bound)) {
		if(stack[idx]->image == g_n||stack[idx]->lower_bound >= upper_bound) {
			-- idx;
			if(idx >= 0&&stack[idx]->lower_bound < upper_bound) {
#ifndef NDEBUG
				++ cnt[stack[idx]->lower_bound];
				++ total;
#endif
				construct_sibling(NULL, stack[idx]);
			}
			continue;
		}

		if(stack[idx]->level+1 == search_n) {
#ifndef NDEBUG
			++ cnt[stack[idx]->lower_bound];
			++ total;
#endif
			extend_to_full_mapping(stack[idx], &full);
			construct_sibling(NULL, stack[idx]);
		}
		else {
			generate_best_extension(stack[idx], stack[idx+1]);
			++ idx;
		}
	}

#ifndef NDEBUG
	ui smaller = 0, exact = cnt[upper_bound];
	for(ui i = 0;i < upper_bound;i ++) smaller += cnt[i];
	delete[] cnt; cnt = NULL;
	printf("Extended #states smaller: %d, equal: %d, larger: %d\n", smaller, exact, total-smaller-exact);
#endif

#ifdef _EXPAND_ALL_
	for(ui i = 0;i < search_n - start_level - start;i ++) if(tt[i].siblings != NULL) {
		if(tt[i].siblings != NULL) delete[] tt[i].siblings;
		tt[i].siblings = NULL;
	}
#endif
	if(tt != NULL) delete[] tt;
	tt = NULL;

	for(ui i = 0;i < search_n - start_level;i ++) stack[i] = NULL;
	if(stack != NULL) delete[] stack;
	stack = NULL;

	return upper_bound;
}




ui Application::App(std::vector<std::pair<ui,ui>> *mapping_ptr, int *lb_ptr)
{
    // Initialize overall_lb and terminal_frontier_min_lb
    overall_lb = INF;
    ui terminal_frontier_min_lb = INF;

    // ========== AStar Trace Init ==========
    const char* trace_env = std::getenv("ASTAR_TRACE_FILE");
    if (trace_env) {
        astar_trace_init(trace_env);
        std::cout << "[TRACE] AStar trace enabled: " << trace_env << std::endl;
    }
    // ========================================

    // Lazy-load mapping order
    if (MO == nullptr) compute_mapping_order();

    // Clear global containers
    open_heap.clear();
    full_mapping_nodes.clear();
    boundary_nodes.clear();  // Unified boundary node container

    // Helper function: protect entire ancestor chain
    auto protect_ancestor_chain = [](State* node) {
        State* current = node;
        while (current != nullptr) {
            current->kept = true;
            current = current->parent;
        }
    };

    // Create root node
    State *root = get_a_new_state_node();
    root->cs_cnt = 0;
#ifdef _EXPAND_ALL_
    root->siblings = get_a_new_siblings_node();
    root->siblings_n = 0;
#endif

    generate_best_extension(nullptr, root);

    // ========== Trace root node ==========
    if (g_astar_trace_file) {
        astar_trace_node(root, nullptr, MO, false, false, false);
    }
    // ==================================

    // Compute LSa for root node
#ifdef _USE_LSa_ESTIMATE_BMao_
    if (root->image < g_n) {
        root->lsa_lb = compute_independent_lower_bound_LSa_baseline(root);
        if (!disable_lsa_pruning && root->lsa_lb > root->lower_bound) {
            root->lower_bound = root->lsa_lb;
        }
    }
#endif

    // Initialize heap
    ui heap_n = 1;
    open_heap.push_back(root);

    State full;
    if (mapping_ptr) *lb_ptr = -1;

    // A* main loop
    int cnt = 0;

    while (heap_n > 0 && open_heap[0]->lower_bound < upper_bound && (verify_upper_bound == INF || upper_bound > verify_upper_bound))
    {
        cnt++;
        if(app_max_iter > 0 && cnt > app_max_iter) break;
        
        // Pop best node from heap
        State *now = open_heap[0];
        open_heap[0] = open_heap[--heap_n];
        heap_top_down(0, heap_n, open_heap);

        // Generate sibling nodes
        State *sib = get_a_new_state_node();
        sib->cs_cnt = 0;
#ifdef _EXPAND_ALL_
        sib->siblings = now->siblings;
        sib->siblings_n = now->siblings_n;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        sib->siblings_total_n = now->siblings_total_n;
#endif
        now->siblings = nullptr;
        now->siblings_n = 0;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        now->siblings_total_n = 0;
#endif
        sib->parent = now->parent;
        sib->level = now->level;
        construct_sibling(nullptr, sib);
#else
        construct_sibling(now, sib);
        ++now->cs_cnt;
#endif
        
        // Compute LSa for sibling nodes
#ifdef _USE_LSa_ESTIMATE_BMao_
        if (sib->image < g_n) {
            sib->lsa_lb = compute_independent_lower_bound_LSa_baseline(sib);
            if (!disable_lsa_pruning && sib->lsa_lb > sib->lower_bound) {
                sib->lower_bound = sib->lsa_lb;
            }
        }
#endif
        
        // Process sibling nodes
        if (sib->image < g_n) {
            if (sib->lower_bound < upper_bound) {
                // Add to open_heap to continue search
                add_to_heap(sib, heap_n, open_heap);
                if (now->parent) {
                    ++now->parent->cs_cnt;
                }
                if (sib->parent) sib->parent->_children.push_back(sib);
            } else if (sib->lower_bound < upper_bound + margin) {
                // Boundary node: lb in [upper_bound, upper_bound + margin) range
                if(sib->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = sib->lower_bound;
                }
                protect_ancestor_chain(sib);
                boundary_nodes.push_back(sib);
                if (now->parent) {
                    ++now->parent->cs_cnt;
                }
                if (sib->parent) sib->parent->_children.push_back(sib);
            } else {
                // Pruning: lb >= upper_bound + margin
                if(sib->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = sib->lower_bound;
                }
#ifdef _EXPAND_ALL_
                put_a_sibling_to_pool(sib->siblings);
                sib->siblings = nullptr;
#else
                --now->cs_cnt;
#endif
                put_a_state_to_pool(sib);
            }
        } else {
            // Invalid node (image >= g_n)
            if(sib->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = sib->lower_bound;
            }
#ifdef _EXPAND_ALL_
            put_a_sibling_to_pool(sib->siblings);
            sib->siblings = nullptr;
#else
            --now->cs_cnt;
#endif
            put_a_state_to_pool(sib);
        }
        
        // Process child nodes or complete mapping
        if (now->level + 1 == search_n) {
            // Reached search depth, extend to complete mapping
            if(now->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = now->lower_bound;
            }
            protect_ancestor_chain(now);
            full_mapping_nodes.push_back(now);
            extend_to_full_mapping(now, &full);
        } else {
            // Generate child nodes
            State *child = get_a_new_state_node();
            child->cs_cnt = 0;
#ifdef _EXPAND_ALL_
            child->siblings = get_a_new_siblings_node();
            child->siblings_n = 0;
#endif
            ui old_UB = upper_bound;
            generate_best_extension(now, child);
            if (upper_bound < old_UB) {
                break;
            }
            
            // Compute LSa for child nodes
#ifdef _USE_LSa_ESTIMATE_BMao_
            if (child->image < g_n) {
                child->lsa_lb = compute_independent_lower_bound_LSa_baseline(child);
                if (!disable_lsa_pruning && child->lsa_lb > child->lower_bound) {
                    child->lower_bound = child->lsa_lb;
                }
            }
#endif
            
            ++now->cs_cnt;

            // ========== Trace child nodes ==========
            if (g_astar_trace_file && child->image < g_n) {
                bool is_pruned = (child->lower_bound >= upper_bound + margin);
                bool is_complete = (child->level == search_n);
                astar_trace_node(child, now, MO, false, is_pruned, is_complete);
            }
            // ==================================

            // Process child nodes
            if (child->image < g_n) {
                if (child->lower_bound < upper_bound) {
                    // Add to open_heap to continue search
                    add_to_heap(child, heap_n, open_heap);
                    child->parent->_children.push_back(child);
                } else if (child->lower_bound < upper_bound + margin) {
                    // Boundary node: lb in [upper_bound, upper_bound + margin) range
                    if(child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
                    protect_ancestor_chain(child);
                    boundary_nodes.push_back(child);
                    child->parent->_children.push_back(child);
                } else {
                    // Pruning: lb >= upper_bound + margin
                    if(child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
#ifdef _EXPAND_ALL_
                    put_a_sibling_to_pool(child->siblings);
                    child->siblings = nullptr;
#endif
                    put_a_state_to_pool(child);
                    --now->cs_cnt;
                }
            } else {
                // Invalid node (image >= g_n)
                if(child->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = child->lower_bound;
                }
#ifdef _EXPAND_ALL_
                put_a_sibling_to_pool(child->siblings);
                child->siblings = nullptr;
#endif
                put_a_state_to_pool(child);
                --now->cs_cnt;
            }
        }

        if (now->cs_cnt == 0 && !now->kept) {
            add_to_pool(now);
        }
    }
    
    // Set final overall_lb
    if(heap_n > 0) {
        overall_lb = open_heap[0]->lower_bound;  // active frontier
    } else {
        overall_lb = terminal_frontier_min_lb;   // terminal frontier
    }

    // Protect ancestor chains of remaining nodes in heap
    for (ui i = 0; i < heap_n; ++i) {
        protect_ancestor_chain(open_heap[i]);
    }
    
    // Adjust open_heap size, keep only valid nodes
    open_heap.resize(heap_n);

    if (mapping_ptr && *lb_ptr < 0) *lb_ptr = upper_bound;

    // ========== Trace Close ==========
    if (g_astar_trace_file) {
        astar_trace_close();
        std::cout << "[TRACE] AStar trace completed, total nodes: " << g_astar_trace_counter << std::endl;
    }
    // ==================================

    return upper_bound;
}

#include <chrono>
#include <iomanip>
using namespace std::chrono;
ui Application::AStar_baseline(vector<pair<ui,ui> > *mapping_ptr, int *lb_ptr) {
	// ========== AStar Trace Init ==========
	const char* trace_env = std::getenv("ASTAR_TRACE_FILE");
	if (trace_env) {
		astar_trace_init(trace_env);
		std::cout << "[TRACE] AStar_baseline trace enabled: " << trace_env << std::endl;
		// Disable verify_upper_bound limit to let search fully execute for complete trace
		verify_upper_bound = INF;
		std::cout << "[TRACE] Disabled verify_upper_bound for complete trace" << std::endl;
	}
	// ========================================

	if(MO == NULL) compute_mapping_order();
	State *root = get_a_new_state_node();
	root->cs_cnt = 0;
#ifdef _EXPAND_ALL_
	root->siblings = get_a_new_siblings_node();
#endif
	generate_best_extension_baseline(NULL, root);

	// ========== Trace root node ==========
	if (g_astar_trace_file) {
		astar_trace_node(root, nullptr, MO, false, false, false);
	}
	// ==================================

	vector<State*> heap;
	ui heap_n = 1;
	heap.pb(root);

	State full;

	// ========== DEBUG: Trace loop info ==========
	if (g_astar_trace_file) {
		std::cout << "[DEBUG] Entering AStar loop. heap_n=" << heap_n
		          << ", root->lb=" << root->lower_bound
		          << ", upper_bound=" << upper_bound
		          << ", verify_upper_bound=" << verify_upper_bound
		          << ", INF=" << INF << std::endl;
		std::cout << "[DEBUG] Loop condition check:" << std::endl;
		std::cout << "  heap_n > 0? " << (heap_n > 0) << std::endl;
		std::cout << "  heap[0]->lb < ub? " << (heap[0]->lower_bound < upper_bound) << std::endl;
		std::cout << "  verify check? " << ((verify_upper_bound == INF || upper_bound > verify_upper_bound)) << std::endl;
	}
	// ==========================================

	while(heap_n > 0 && heap[0]->lower_bound < upper_bound &&
	      (verify_upper_bound == INF || upper_bound > verify_upper_bound)) {

		State *now = heap[0];

		// ========== DEBUG: Trace current state ==========
		if (g_astar_trace_file) {
			std::cout << "[DEBUG] Processing node: level=" << now->level
			          << ", lb=" << now->lower_bound
			          << ", image=" << now->image << std::endl;
		}
		// ==========================================
		heap[0] = heap[--heap_n];
		heap_top_down(0, heap_n, heap);

		// ========== DEBUG ==========
		if (g_astar_trace_file) {
			std::cout << "[DEBUG] After pop: heap_n=" << heap_n << std::endl;
		}
		// ============================

		if(mapping_ptr != NULL && (*mapping_ptr).size() == now->level+1) {
			bool ok = true;
			State *tmp = now;
			
			for(ui i = 0; i <= now->level && ok; i++) {
				if((*mapping_ptr)[now->level-i].second != tmp->image) ok = false;
				tmp = tmp->parent;
			}

			if(ok) {
				*lb_ptr = now->lower_bound;
				break;
			}
		}

		State *sibling = get_a_new_state_node();
		sibling->cs_cnt = 0;
#ifdef _EXPAND_ALL_
		sibling->siblings = now->siblings;
		sibling->siblings_n = now->siblings_n;
		now->siblings = NULL;
		now->siblings_n = 0;
		sibling->parent = now->parent;
		sibling->level = now->level;
		construct_sibling_baseline(NULL, sibling);  // using baseline version
#else
		construct_sibling_baseline(now, sibling);   // using baseline version
		++now->cs_cnt;
#endif

		// ========== Trace sibling nodes ==========
		if (g_astar_trace_file && sibling->image < g_n) {
			bool is_pruned = (sibling->lower_bound >= upper_bound);
			bool is_complete = (sibling->level == search_n);
			astar_trace_node(sibling, now->parent, MO, false, is_pruned, is_complete);
		}
		// ==================================

		if(sibling->image < g_n && sibling->lower_bound < upper_bound) {
			add_to_heap(sibling, heap_n, heap);
			if(now->parent != NULL) ++now->parent->cs_cnt;
		}
		else {
			put_a_state_to_pool(sibling);
#ifdef _EXPAND_ALL_
			put_a_sibling_to_pool(sibling->siblings);
			sibling->siblings = NULL;
#else
			--now->cs_cnt;
#endif
		}

		// ========== DEBUG ==========
		if (g_astar_trace_file) {
			std::cout << "[DEBUG] now->level=" << now->level
			          << ", search_n=" << search_n
			          << ", level+1==search_n? " << (now->level+1 == search_n ? "YES" : "NO") << std::endl;
		}
		// ============================

		if(now->level+1 == search_n) {
			if (g_astar_trace_file) std::cout << "[DEBUG] Calling extend_to_full_mapping_baseline" << std::endl;
			extend_to_full_mapping_baseline(now, &full);
		}
		else {
			if (g_astar_trace_file) std::cout << "[DEBUG] Generating child node" << std::endl;
			State *child = get_a_new_state_node();
			child->cs_cnt = 0;
#ifdef _EXPAND_ALL_
			child->siblings = get_a_new_siblings_node();
#endif
			generate_best_extension_baseline(now, child);
			now->cs_cnt++;

			// ========== Trace child nodes ==========
			if (g_astar_trace_file && child->image < g_n) {
				bool is_pruned = (child->lower_bound >= upper_bound);
				bool is_complete = (child->level == search_n);
				astar_trace_node(child, now, MO, false, is_pruned, is_complete);
			}
			// ==================================

			if(child->image < g_n && child->lower_bound < upper_bound) {
				add_to_heap(child, heap_n, heap);
			}
			else {
#ifdef _EXPAND_ALL_
				put_a_sibling_to_pool(child->siblings);
				child->siblings = NULL;
#endif
				put_a_state_to_pool(child);
				--now->cs_cnt;
			}
		}

		if(now->cs_cnt == 0) add_to_pool(now);

		while(heap_n > 0 && heap[heap_n-1]->lower_bound >= upper_bound) {
			--heap_n;
			if(heap[heap_n]->cs_cnt == 0) add_to_pool(heap[heap_n]);
		}
	}

	if(mapping_ptr != NULL && (*lb_ptr) < 0) {
		*lb_ptr = upper_bound;
	}

	// ========== Trace Close ==========
	if (g_astar_trace_file) {
		astar_trace_close();
		std::cout << "[TRACE] AStar_baseline trace completed, total nodes: " << g_astar_trace_counter << std::endl;
	}
	// ==================================

	return upper_bound;
}

ui Application::App_test(vector<pair<ui,ui> > *mapping_ptr, int *lb_ptr) {
    
    if(MO == NULL) compute_mapping_order();
    
    // ========== Initialization ==========
    overall_lb = INF;
    overall_ub = INF;
    ui terminal_frontier_min_lb = INF;

    State *root = get_a_new_state_node();
    root->cs_cnt = 0;
#ifdef _EXPAND_ALL_
    root->siblings = get_a_new_siblings_node();
#endif
    
    generate_best_extension_test(NULL, root);
    
    // No longer update overall_lb here since root will be added to heap

    vector<State*> heap;
    ui heap_n = 1;
    heap.pb(root);

    State full;

    int cnt = 0;

    while(heap_n > 0 && heap[0]->lower_bound < upper_bound
#if !APP_TEST_EXACT_RESULT
          && (verify_upper_bound == INF || upper_bound > verify_upper_bound)
#endif
          ) {

        cnt++;
        if(app_max_iter > 0 && cnt > app_max_iter) break;

        State *now = heap[0];
        heap[0] = heap[--heap_n];
        heap_top_down(0, heap_n, heap);
        

        if(mapping_ptr != NULL && (*mapping_ptr).size() == now->level+1) {
            bool ok = true;
            State *tmp = now;
            
            for(ui i = 0; i <= now->level && ok; i++) {
                if((*mapping_ptr)[now->level-i].second != tmp->image) ok = false;
                tmp = tmp->parent;
            }

            if(ok) {
                *lb_ptr = now->lower_bound;
                break;
            }
        }

        State *sibling = get_a_new_state_node();
        sibling->cs_cnt = 0;
#ifdef _EXPAND_ALL_
        sibling->siblings = now->siblings;
        sibling->siblings_n = now->siblings_n;
        now->siblings = NULL;
        now->siblings_n = 0;
        sibling->parent = now->parent;
        sibling->level = now->level;
        construct_sibling_test(NULL, sibling);
#else
        construct_sibling_test(now, sibling);
        ++now->cs_cnt;
#endif
        
        // Determine if sibling can be added to heap
        if(sibling->image < g_n && sibling->lower_bound < upper_bound) {
            add_to_heap(sibling, heap_n, heap);
            if(now->parent != NULL) ++now->parent->cs_cnt;
        }
        else {
            // sibling cannot be added to heap, record as terminal frontier
            if(sibling->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = sibling->lower_bound;
            }
            put_a_state_to_pool(sibling);
#ifdef _EXPAND_ALL_
            put_a_sibling_to_pool(sibling->siblings);
            sibling->siblings = NULL;
#else
            --now->cs_cnt;
#endif
        }

        if(now->level+1 == search_n) {
            // Nodes reaching search depth are also terminal frontier
            if(now->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = now->lower_bound;
            }
            
            extend_to_full_mapping_test(now, &full);
            // extend_to_full_mapping will call compute_mapped_cost_and_upper_bound
            // which will update overall_ub
        }
        else {
#ifdef _USE_LSa_AS_LAYER_
            ui lsa_lb = compute_independent_lower_bound_LSa_baseline(now);
    
            if(lsa_lb >= upper_bound||lsa_lb > verify_upper_bound) {
                // printf("now->lower_bound: %u, lsa_lb: %u\n", now->lower_bound, lsa_lb);
                now->lower_bound = lsa_lb;
                now->image = g_n;
                
                // this node becomes a terminal node
                if(now->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = now->lower_bound;
                }
            } else 
#endif
            {
                State *child = get_a_new_state_node();
                child->cs_cnt = 0;
    #ifdef _EXPAND_ALL_
                child->siblings = get_a_new_siblings_node();
    #endif
                
                generate_best_extension_test(now, child);
                now->cs_cnt++;

                // Determine if child can be added to heap
                if(child->image < g_n && child->lower_bound < upper_bound) {
                    add_to_heap(child, heap_n, heap);
                }
                else {
                    // child cannot be added to heap, record as terminal frontier
                    if(child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
    #ifdef _EXPAND_ALL_
                    put_a_sibling_to_pool(child->siblings);
                    child->siblings = NULL;
    #endif
                    put_a_state_to_pool(child);
                    --now->cs_cnt;
                }
            }
        }

        if(now->cs_cnt == 0) add_to_pool(now);

        while(heap_n > 0 && heap[heap_n-1]->lower_bound >= upper_bound) {
            --heap_n;
            if(heap[heap_n]->cs_cnt == 0) add_to_pool(heap[heap_n]);
        }
    }

    // ========== Set final overall_lb ==========
    if(heap_n > 0) {
        overall_lb = heap[0]->lower_bound;  // active frontier
    } else {
        overall_lb = terminal_frontier_min_lb;  // terminal frontier
    }

    if(mapping_ptr != NULL && (*lb_ptr) < 0) {
        *lb_ptr = upper_bound;
    }


    return upper_bound;
}

ui Application::AppForComputation(vector<pair<ui,ui> > *mapping_ptr, int *lb_ptr) {
    if(MO == NULL) compute_mapping_order();
    
    // Initialization - simplified version for training data generation
    overall_lb = INF;
    overall_ub = INF;
    ui terminal_frontier_min_lb = INF;
    
    State *root = get_a_new_state_node();
    root->cs_cnt = 0;
#ifdef _EXPAND_ALL_
    root->siblings = get_a_new_siblings_node();
#endif
    
    generate_best_extension_test(NULL, root);
    
    vector<State*> heap;
    ui heap_n = 1;
    heap.pb(root);

    State full;

    int cnt = 0;
    bool reached_limit = false; // flag for whether loop limit is reached

    // Remove verify_upper_bound early-stop condition to ensure finding true optimum
    while(heap_n > 0 && heap[0]->lower_bound < upper_bound) {

        cnt++;
        if(cnt >= exact_max_iter) {
            // Count limit reached, mark as inexact solution
            reached_limit = true;
            break;
        }



        State *now = heap[0];
        heap[0] = heap[--heap_n];
        heap_top_down(0, heap_n, heap);

        if(mapping_ptr != NULL && (*mapping_ptr).size() == now->level+1) {
            bool ok = true;
            State *tmp = now;
            
            for(ui i = 0; i <= now->level && ok; i++) {
                if((*mapping_ptr)[now->level-i].second != tmp->image) ok = false;
                tmp = tmp->parent;
            }

            if(ok) {
                *lb_ptr = now->lower_bound;
                break;
            }
        }

        State *sibling = get_a_new_state_node();
        sibling->cs_cnt = 0;
#ifdef _EXPAND_ALL_
        sibling->siblings = now->siblings;
        sibling->siblings_n = now->siblings_n;
        now->siblings = NULL;
        now->siblings_n = 0;
        sibling->parent = now->parent;
        sibling->level = now->level;
        construct_sibling_test(NULL, sibling);
#else
        construct_sibling_test(now, sibling);
        ++now->cs_cnt;
#endif
        
        if(sibling->image < g_n && sibling->lower_bound < upper_bound) {
            add_to_heap(sibling, heap_n, heap);
            if(now->parent != NULL) ++now->parent->cs_cnt;
        }
        else {
            if(sibling->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = sibling->lower_bound;
            }
            put_a_state_to_pool(sibling);
#ifdef _EXPAND_ALL_
            put_a_sibling_to_pool(sibling->siblings);
            sibling->siblings = NULL;
#else
            --now->cs_cnt;
#endif
        }

        if(now->level+1 == search_n) {
            if(now->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = now->lower_bound;
            }
            extend_to_full_mapping_test(now, &full);
        }
        else {
            {
                State *child = get_a_new_state_node();
                child->cs_cnt = 0;
#ifdef _EXPAND_ALL_
                child->siblings = get_a_new_siblings_node();
#endif
                
                generate_best_extension_test(now, child);
                now->cs_cnt++;

                if(child->image < g_n && child->lower_bound < upper_bound) {
                    add_to_heap(child, heap_n, heap);
                }
                else {
                    if(child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
#ifdef _EXPAND_ALL_
                    put_a_sibling_to_pool(child->siblings);
                    child->siblings = NULL;
#endif
                    put_a_state_to_pool(child);
                    --now->cs_cnt;
                }
            }
        }

        if(now->cs_cnt == 0) add_to_pool(now);

        while(heap_n > 0 && heap[heap_n-1]->lower_bound >= upper_bound) {
            --heap_n;
            if(heap[heap_n]->cs_cnt == 0) add_to_pool(heap[heap_n]);
        }
    }

    // Set final overall_lb
    if(heap_n > 0) {
        overall_lb = heap[0]->lower_bound;
    } else {
        overall_lb = terminal_frontier_min_lb;
    }

    if(mapping_ptr != NULL && (*lb_ptr) < 0) {
        *lb_ptr = upper_bound;
    }

#if defined(APP_CNT_FOR_GENERATE_TRAINING_DATA) && APP_CNT_FOR_GENERATE_TRAINING_DATA > 0
    // If loop limit reached, return -1 indicating exact GED not found
    if(reached_limit) {
        return (ui)(-1); // return special value indicating inexact solution
    }
#endif

    return upper_bound;
}

ui Application::App_baseline(vector<pair<ui,ui> > *mapping_ptr, int *lb_ptr) {
    if(MO == NULL) compute_mapping_order();
    State *root = get_a_new_state_node();
    root->cs_cnt = 0;
#ifdef _EXPAND_ALL_
    root->siblings = get_a_new_siblings_node();
#endif
    
    generate_best_extension_baseline(NULL, root);

    vector<State*> heap;
    ui heap_n = 1;
    heap.pb(root);

    State full;

    int cnt = 0;

    while(heap_n > 0 && heap[0]->lower_bound < upper_bound &&
          (verify_upper_bound == INF || upper_bound > verify_upper_bound)) {

    cnt++;
    if(app_max_iter > 0 && cnt > app_max_iter) break;

        State *now = heap[0];
        heap[0] = heap[--heap_n];
        heap_top_down(0, heap_n, heap);

        if(mapping_ptr != NULL && (*mapping_ptr).size() == now->level+1) {
            bool ok = true;
            State *tmp = now;
            
            for(ui i = 0; i <= now->level && ok; i++) {
                if((*mapping_ptr)[now->level-i].second != tmp->image) ok = false;
                tmp = tmp->parent;
            }

            if(ok) {
                *lb_ptr = now->lower_bound;
                break;
            }
        }

        State *sibling = get_a_new_state_node();
        sibling->cs_cnt = 0;
#ifdef _EXPAND_ALL_
        sibling->siblings = now->siblings;
        sibling->siblings_n = now->siblings_n;
        now->siblings = NULL;
        now->siblings_n = 0;
        sibling->parent = now->parent;
        sibling->level = now->level;
        construct_sibling_baseline(NULL, sibling);  // using baseline version
#else
        construct_sibling_baseline(now, sibling);   // using baseline version
        ++now->cs_cnt;
#endif
        
        if(sibling->image < g_n && sibling->lower_bound < upper_bound) {
            add_to_heap(sibling, heap_n, heap);
            if(now->parent != NULL) ++now->parent->cs_cnt;
        }
        else {
            put_a_state_to_pool(sibling);
#ifdef _EXPAND_ALL_
            put_a_sibling_to_pool(sibling->siblings);
            sibling->siblings = NULL;
#else
            --now->cs_cnt;
#endif
        }

        if(now->level+1 == search_n) {
            extend_to_full_mapping_baseline(now, &full);
        }
        else {
            State *child = get_a_new_state_node();
            child->cs_cnt = 0;
#ifdef _EXPAND_ALL_
            child->siblings = get_a_new_siblings_node();
#endif
            generate_best_extension_baseline(now, child);
            now->cs_cnt++;

            if(child->image < g_n && child->lower_bound < upper_bound) {
                add_to_heap(child, heap_n, heap);
            }
            else {
#ifdef _EXPAND_ALL_
                put_a_sibling_to_pool(child->siblings);
                child->siblings = NULL;
#endif
                put_a_state_to_pool(child);
                --now->cs_cnt;
            }
        }

        if(now->cs_cnt == 0) add_to_pool(now);

        while(heap_n > 0 && heap[heap_n-1]->lower_bound >= upper_bound) {
            --heap_n;
            if(heap[heap_n]->cs_cnt == 0) add_to_pool(heap[heap_n]);
        }
    }

    if(mapping_ptr != NULL && (*lb_ptr) < 0) {
        *lb_ptr = upper_bound;
    }

    return upper_bound;
}


void Application::add_to_pool(State *st) {
    // If node is protected, return directly
    if (st->kept) return;
    
    put_a_state_to_pool(st);
    assert(states_pool[states_pool_n-1] == st);
    
    for(ui i = states_pool_n-1; i < states_pool_n; i++) {
        st = states_pool[i];
        
#ifdef _EXPAND_ALL_
        if(st->siblings != NULL) {
            put_a_sibling_to_pool(st->siblings);
            st->siblings = NULL;
        }
#endif
        
        if(st->pre_sibling != NULL) {
            assert(st->pre_sibling->cs_cnt > 0);
            if((-- st->pre_sibling->cs_cnt) == 0) {
                if (!st->pre_sibling->kept) {
                    put_a_state_to_pool(st->pre_sibling);
                    st->pre_sibling = NULL;
                }
            }
        }
        
        if(st->parent != NULL) {
            assert(st->parent->cs_cnt > 0);
            if((-- st->parent->cs_cnt) == 0) {
                if (!st->parent->kept) {
                    put_a_state_to_pool(st->parent);
                    st->parent = NULL;
                }
            }
        }
    }
}

void Application::add_to_heap(State *st, ui &heap_n, vector<State*> &heap) {
	if(heap.size() == heap_n) heap.pb(st);
	else heap[heap_n] = st;
	++ heap_n;
	heap_bottom_up(heap_n-1, heap);

    if (lb_method == LSa) {
        search_snapshot.push_back({static_cast<int16_t>(st->level), static_cast<int16_t>(st->image),
                                static_cast<uint16_t>(st->lower_bound),
                                static_cast<uint16_t>(st->mapped_cost), 1}); // is_open = 1
    } else if (lb_method == BMao) {
        search_snapshot.push_back({static_cast<int16_t>(st->level), static_cast<int16_t>(st->image),
                                static_cast<uint16_t>(st->lower_bound),
                                static_cast<uint16_t>(st->mapped_cost), 1}); // is_open = 1
    }

}

State* Application::get_a_new_state_node() {
	if(states_pool_n == 0) {
		State *new_block = new State[block_size];
		for(ui i = 0;i < block_size;i ++) {
			if(states_pool.size() == states_pool_n) states_pool.pb(&new_block[i]);
			else states_pool[states_pool_n] = &new_block[i];
			++ states_pool_n;
		}
		states_memory.pb(new_block);
	}
	assert(states_pool_n > 0);
	-- states_pool_n;
	return states_pool[states_pool_n];
}

void Application::put_a_state_to_pool(State *st) {
	if(states_pool_n == states_pool.size()) states_pool.pb(st);
	else states_pool[states_pool_n] = st;
	++ states_pool_n;
}


ushort* Application::get_a_new_siblings_node() {
    // return new ushort[2*g_n];
    // printf("get_a_new_siblings_node called: siblings_pool_n=%u\n", siblings_pool_n);
    
    if(siblings_pool_n == 0) {
        ui t_block_size = (block_size*8)/g_n+1;
        // printf("Allocating new block: t_block_size=%u, size=%u\n", 
        //        t_block_size, t_block_size*2*g_n);
        
        ushort *new_block = new ushort[t_block_size*2*g_n];
        if (!new_block) {
            printf("Memory allocation failed!\n");
            return nullptr;
        }
        
        for(ui i = 0;i < t_block_size;i ++) {
            if(siblings_pool.size() == siblings_pool_n) siblings_pool.pb(new_block+i*2*g_n);
            else siblings_pool[siblings_pool_n] = new_block+i*2*g_n;
            ++ siblings_pool_n;
        }
        siblings_memory.pb(new_block);
    }
    
    assert(siblings_pool_n > 0);
    -- siblings_pool_n;
    ushort* result = siblings_pool[siblings_pool_n];
    // printf("Returning: %p\n", result);
    return result;
}

void Application::put_a_sibling_to_pool(ushort *sibling) {
    if (!sibling) return;
	if(siblings_pool_n == siblings_pool.size()) siblings_pool.pb(sibling);
	else siblings_pool[siblings_pool_n] = sibling;
	++ siblings_pool_n;
}

void Application::compute_mapping_order() {
	// If MO is empty, allocate memory for it
	if(MO == NULL) MO = new ui[q_n];

	// Get total edge count of graph g
	ui g_m = g_starts[g_n];
	// Allocate memory for vertex label and edge label counts, initialize to 0
	ui *vlabels_cnt = new ui[vlabels_n];
	ui *elabels_cnt = new ui[elabels_n];
	memset(vlabels_cnt, 0, sizeof(ui)*vlabels_n);
	memset(elabels_cnt, 0, sizeof(ui)*elabels_n);

	// Count frequency of each vertex and edge label in graph g
	for(ui i = 0; i < g_n; i++) {
		++vlabels_cnt[g_vlabels[i]];
		for(ui j = g_starts[i]; j < g_starts[i+1]; j++) ++elabels_cnt[g_elabels[j]];
	}

	// Find the root vertex with maximum weight in graph q
	ui root = 0;
	double root_weight = 0;
	for(ui i = 0; i < q_n; i++) {
		// Compute vertex weight: lower frequency means higher weight
		double weight = 1 - vlabels_cnt[q_vlabels[i]] / double(g_n);
		for(ui j = q_starts[i]; j < q_starts[i+1]; j++) {
			weight += 1 - elabels_cnt[q_elabels[j]] / double(g_m);
		}
		// Update vertex with maximum weight
		if(weight > root_weight) {
			root = i;
			root_weight = weight;
		}
	}

	// Initialize heap for maintaining vertex priority
	vector<pair<double, int>> heap(q_n);
	for(ui i = 0; i < q_n; i++) {
		if(i == root) heap[i].first = root_weight;
		else heap[i].first = 0;
		heap[i].second = i;
	}
	// Place vertex with maximum weight at heap top
	swap(heap[0], heap[root]);

	// Initialize position array for tracking vertex position in heap
	ui *pos = new ui[q_n];
	for(ui i = 0; i < q_n; i++) pos[heap[i].second] = i;

	// Build max-heap sorted by weight
	ui heap_n = q_n;
	for(ui i = 0; i < q_n; i++) {
		ui u = heap[0].second;
		MO[i] = u;
		pos[u] = heap_n - 1;
		heap[0] = heap[--heap_n];
		pos[heap[0].second] = 0;
		heap_top_down(0, heap_n, heap, pos);
		// Update adjacent vertices' weights and adjust heap
		for(ui j = q_starts[u]; j < q_starts[u+1]; j++) if(pos[q_edges[j]] < heap_n) {
			ui idx = pos[q_edges[j]];
			if(heap[idx].first < EPS) heap[idx].first += 1 - vlabels_cnt[q_vlabels[q_edges[j]]] / double(g_n);
			heap[idx].first += 1 - elabels_cnt[q_elabels[j]] / double(g_m);
			heap_bottom_up(idx, heap, pos);
		}
	}

	// Debug output matching order
#ifndef NDEBUG
	//for(ui i = 0; i < q_n; i++) MO[i] = i;
#endif

	// Compute independent set size
	for(ui i = 0; i < q_n; i++) pos[MO[i]] = i;
	ui edges_cnt = 0;
	for(ui i = 0; i < q_n; i++) {
		ui u = MO[i];
		for(ui j = q_starts[u]; j < q_starts[u+1]; j++) if(pos[q_edges[j]] > i) edges_cnt += 2;
		//if(edges_cnt >= q_starts[q_n]) printf("%d %d\n", edges_cnt, q_starts[q_n]);
		if(edges_cnt == q_starts[q_n]) {
			search_n = i + 1;
			break;
		}
	}

	search_n_for_IS = search_n;

	if(lb_method != BMao) search_n = q_n;
	// search_n = q_n;

#ifndef NDEBUG
	printf("Matching order: (%d", MO[0]);
	for(ui i = 1; i < q_n; i++) printf(", %d", MO[i]);
	printf(")\n");
	printf("Independent set size: %u\n", q_n - search_n);
#endif

	// Free allocated memory
	delete[] pos;
	delete[] vlabels_cnt;
	delete[] elabels_cnt;
}



void Application::heap_top_down(ui idx, ui heap_n, vector<pair<double, int> > &heap, ui *pos) {
	pair<double, int> tmp = heap[idx];
	while(2*idx+1 < heap_n) {
		ui i = 2*idx+1;
		if(i+1 < heap_n&&heap[i+1].first > heap[i].first) ++ i;
		if(heap[i].first > tmp.first) {
			heap[idx] = heap[i];
			pos[heap[idx].second] = idx;
			idx = i;
		}
		else break;
	}
	heap[idx] = tmp;
	pos[tmp.second] = idx;
}



void Application::heap_bottom_up(ui idx, vector<pair<double,int> > &heap, ui *pos) {
	pair<double, int> tmp = heap[idx];
	while(idx > 0) {
		ui i = (idx-1)/2;
		if(heap[i].first < tmp.first) {
			heap[idx] = heap[i];
			pos[heap[idx].second] = idx;
			idx = i;
		}
		else break;
	}
	heap[idx] = tmp;
	pos[tmp.second] = idx;
}

bool Application::state_has_higher_priority(State* a, State* b) {
#ifdef DEPTH_FIRST_PRIORITY
	// Depth-first strategy：level -> lower_bound -> need_recompute -> image -> mapped_cost
	if(a->level != b->level) return a->level > b->level;
	if(a->lower_bound != b->lower_bound) return a->lower_bound < b->lower_bound;
	if(a->need_recompute != b->need_recompute) return !a->need_recompute && b->need_recompute;
	// Additional tie-breakers for deterministic ordering
	if(a->image != b->image) return a->image < b->image;
	if(a->mapped_cost != b->mapped_cost) return a->mapped_cost < b->mapped_cost;
	// Final tie-breaker: pointer address (for states that are truly identical)
	return a < b;
#else
	// Default strategy：lower_bound -> level -> need_recompute -> image -> mapped_cost
	if(a->lower_bound != b->lower_bound) return a->lower_bound < b->lower_bound;
	if(a->level != b->level) return a->level > b->level;
	if(a->need_recompute != b->need_recompute) return !a->need_recompute && b->need_recompute;
	// Additional tie-breakers for deterministic ordering
	if(a->image != b->image) return a->image < b->image;
	if(a->mapped_cost != b->mapped_cost) return a->mapped_cost < b->mapped_cost;
	// Final tie-breaker: pointer address (for states that are truly identical)
	return a < b;
#endif
}

void Application::heap_top_down(ui idx, ui heap_n, vector<State*> &heap) {
	State *tmp = heap[idx];
	while(2*idx+1 < heap_n) {
		ui i = 2*idx+1;
		// Select child with higher priority
		if(i+1 < heap_n && state_has_higher_priority(heap[i+1], heap[i])) {
			++ i;
		}
		
		// Compare selected child with tmp
		if(state_has_higher_priority(heap[i], tmp)) {
			heap[idx] = heap[i];
			idx = i;
		} else {
			break;
		}
	}
	heap[idx] = tmp;
}

void Application::heap_bottom_up(ui idx, vector<State*> &heap) {
	State *tmp = heap[idx];
	while(idx > 0) {
		ui i = (idx-1)/2;
		
		// Compare tmp with parent
		if(state_has_higher_priority(tmp, heap[i])) {
			heap[idx] = heap[i];
			idx = i;
		} else {
			break;
		}
	}
	heap[idx] = tmp;
}



/*------------------------------------------------------------
 *  Application::generate_best_extension
 *-----------------------------------------------------------*/
void Application::generate_best_extension(State *parent, State *now)
{
    // printf("generate_best_extension: now->level=%u, now->image=%u, now->siblings=%p\n",
    //        now->level, now->image, now->siblings);
    /* ---------- 1. Inherit basic properties ---------- */
    if (parent != nullptr) {
        now->level       = parent->level + 1;
        now->mapped_cost = parent->mapped_cost;
    } else {
        now->level       = 0;
        now->mapped_cost = 0;
    }
    now->parent      = parent;
    now->pre_sibling = nullptr;

    /* ---------- 2. Mark used image (skip dummy-root) ---------- */
    while (parent != nullptr) {
        if (parent->image < g_n)         /* * mark only real nodes */
            visY[parent->image] = 1;
        parent = parent->parent;
    }

    /* ---------- 3. Collect candidates ---------- */
    ui candidates_n = 0;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i])
            candidates[candidates_n++] = i;
        else
            visY[i] = 0;                 // clear to 0, for next iteration
    }

    ui pre_siblings = 0;

    if (visited_siblings_n != nullptr)
        visited_siblings_n[now->level] = 0;
    if (children == nullptr)
        children = new pair<int,int>[g_n];

    /* ---------- 4. Select best extension based on lower bound method ---------- */
    if      (lb_method == LSa)  compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMao) compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == BMa)  compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
    else
        printf("lb_method is wrong in (generate_best_extension)!\n");
}

void Application::generate_best_extension_baseline(State *parent, State *now) {
	if(parent != NULL) {
		now->level = parent->level + 1;
		now->mapped_cost = parent->mapped_cost;
	}
	else {
		now->level = 0;
		now->mapped_cost = 0;
	}
	now->parent = parent;
	now->pre_sibling = NULL;

	while(parent != NULL) {
		visY[parent->image] = 1;
		parent = parent->parent;
	}
	ui candidates_n = 0;
	for(ui i = 0;i < g_n;i ++) {
		if(!visY[i]) candidates[candidates_n ++] = i;
		else visY[i] = 0;
	}
	ui pre_siblings = 0;

	if(visited_siblings_n != NULL) visited_siblings_n[now->level] = 0;
	if(children == NULL) children = new pair<int,int>[g_n];

	if(lb_method == LSa) compute_best_extension_LSa_baseline(now, candidates_n, candidates, pre_siblings);
	else if(lb_method == BMao) compute_best_extension_BM_baseline(1, now, candidates_n, candidates, pre_siblings, 0);
	else if(lb_method == BMa) compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
	else printf("lb_method is wrong in (generate_best_extension)!\n");
}


void Application::generate_best_extension_test(State *parent, State *now) {
	if(parent != NULL) {
		now->level = parent->level + 1;
		now->mapped_cost = parent->mapped_cost;
	}
	else {
		now->level = 0;
		now->mapped_cost = 0;
	}
	now->parent = parent;
	now->pre_sibling = NULL;

	while(parent != NULL) {
		visY[parent->image] = 1;
		parent = parent->parent;
	}
	ui candidates_n = 0;
	for(ui i = 0;i < g_n;i ++) {
		if(!visY[i]) candidates[candidates_n ++] = i;
		else visY[i] = 0;
	}
	ui pre_siblings = 0;

	if(visited_siblings_n != NULL) visited_siblings_n[now->level] = 0;
	if(children == NULL) children = new pair<int,int>[g_n];

	if(lb_method == LSa) compute_best_extension_LSa_baseline(now, candidates_n, candidates, pre_siblings);
	else if(lb_method == BMao) compute_best_extension_BM_test(1, now, candidates_n, candidates, pre_siblings, 0);
	else if(lb_method == BMa) compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
	else printf("lb_method is wrong in (generate_best_extension)!\n");
}


void Application::compute_best_extension_LSa_baseline(State *now,
                                             ui n,             // candidates[] count
                                             ui *candidates,   // candidate G vertex id list
                                             ui pre_siblings)  // (if expanding siblings) number of processed siblings
{
/* ------------------------------------------------------------ *
 * 0. Debug: auxiliary arrays must be all-zero
 * ------------------------------------------------------------ */
#ifndef NDEBUG
    for (ui i = 0; i < g_n; ++i) {
        assert(visX[i] == 0);
        assert(visY[i] == 0);
    }
    if (vlabels_map)  for (ui i = 0; i < vlabels_n;  ++i) assert(vlabels_map[i]  == 0);
    for (ui i = 0; i < elabels_n; ++i)           assert(elabels_map[i] == 0);
    if (elabels_matrix)
        for (ui i = 0; i < q_n * elabels_n; ++i) assert(elabels_matrix[i] == 0);
#endif

/* ------------------------------------------------------------ *
 * 1. If auxiliary buffer not allocated, new and zero-fill
 * ------------------------------------------------------------ */
    if (!vlabels_map) {
        vlabels_map = new int[vlabels_n]{0};
    }
    if (!elabels_matrix) {
        elabels_matrix = new short[q_n * elabels_n]{0};
    }

/* ------------------------------------------------------------ *
 * 2. Temporary arrays sizeX / sizeY
 * ------------------------------------------------------------ */
    ui *sizeX = queue;   // length ≥ q_n
    ui *sizeY = prev;    // length ≥ g_n

/* ------------------------------------------------------------ *
 * 3. Pre-scan query side
 * ------------------------------------------------------------ */
    ui idx = now->level;
    ui u   = MO[idx];
    ui u_cross_cnt = 0, u_inner_cnt = 0;

    /* 3.1 visX marking */
    for (ui i = 0; i <= idx; ++i) visX[MO[i]] = 1;

    /* 3.2 sizeX & label difference */
    for (ui i = 0; i <= idx; ++i) {
        ui v = MO[i];
        sizeX[v] = 0;
        for (ui j = q_starts[v]; j < q_starts[v+1]; ++j) {
            if (!visX[q_edges[j]]) {
                ++sizeX[v];
                --elabels_matrix[v*elabels_n + q_elabels[j]];
            }
        }
    }
    
    /* 3.3 u cross-edge count */
    for (ui j = q_starts[u]; j < q_starts[u+1]; ++j)
        if (visX[q_edges[j]]) ++u_cross_cnt;

    /* 3.4 unmapped region internal edge count */
    for (ui i = idx+1; i < q_n; ++i)
        for (ui j = q_starts[MO[i]]; j < q_starts[MO[i]+1]; ++j) {
            ui uprime = q_edges[j];
            if (!visX[uprime] && MO[i] < uprime) {
                --elabels_map[q_elabels[j]];
                ++u_inner_cnt;
            }
        }

    /* 3.5 vertex label surplus/deficit */
    for (ui i = idx+1; i < q_n; ++i)
        --vlabels_map[q_vlabels[MO[i]]];

    /* 3.6 Clean up visX */
    for (ui i = 0; i <= idx; ++i) visX[MO[i]] = 0;

/* ------------------------------------------------------------ *
 * 4. Mark occupied G-side vertices in visY / my
 * ------------------------------------------------------------ */
    for (State *parent = now->parent; parent; parent = parent->parent) {
        if (parent->image >= g_n) continue;              /* * skip dummy */
        visY[parent->image] = 1;
        my[parent->image]  = MO[parent->level];
    }

/* ------------------------------------------------------------ *
 * 5. Compute sizeY & elabels_matrix correction for ancestor mapping
 * ------------------------------------------------------------ */
    ui ged_common = 0;
    for (State *st = now->parent; st; st = st->parent) {
        if (st->image >= g_n) continue;                  /* * skip dummy */
        ui v = st->image;
        ui common = 0;
        sizeY[v] = 0;

        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            if (!visY[g_edges[j]]) {
                short &cnt = elabels_matrix[MO[st->level]*elabels_n +
                                            g_elabels[j]];
                if (cnt < 0) ++common;
                ++cnt;
                ++sizeY[v];
            }
        }
        ui t_ged = sizeX[MO[st->level]];
        if (sizeY[v] > t_ged) t_ged = sizeY[v];
        ged_common += t_ged - common;
    }

/* ------------------------------------------------------------ *
 * 6. Scan candidate G vertices, update label surplus/deficit statistics
 * ------------------------------------------------------------ */
    ui vl_common = 0, el_common = 0, v_total_cnt = 0;
    for (ui i = 0; i < n; ++i) {
        ui v = candidates[i];
        if (vlabels_map[g_vlabels[v]] < 0) ++vl_common;
        ++vlabels_map[g_vlabels[v]];

        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            ui vprime = g_edges[j];
            if (!visY[vprime] && v < vprime) {
                if (elabels_map[g_elabels[j]] < 0) ++el_common;
                ++elabels_map[g_elabels[j]];
                ++v_total_cnt;
            }
        }
    }

/* ------------------------------------------------------------ *
 * 7. Compute vertex label GED upper bound
 * ------------------------------------------------------------ */
    ui mapped_cost = (now->parent ? now->parent->mapped_cost : 0);

    ui vl_ged = q_n - idx - 1;
    if (g_n > q_n) vl_ged = g_n - idx - 1;
    vl_ged -= vl_common;

/* ------------------------------------------------------------ *
 * 8. children[] stores (-score, v)
 * ------------------------------------------------------------ */
    if (!children) children = new std::pair<int,int>[g_n];

/* ------------------------------------------------------------ *
 * 9. Iterate candidate G vertices v
 * ------------------------------------------------------------ */
    for (ui i = pre_siblings; i < n; ++i) {
        ui v = candidates[i];

        /* 9.1 compute lower bound lb */
        ui lb = vl_ged + ged_common;
        if (vlabels_map[g_vlabels[v]] <= 0) ++lb;

        ui cross_ged = u_cross_cnt;
        ui inner_ged = v_total_cnt;
        if (g_vlabels[v] != q_vlabels[u]) ++cross_ged;

        ui common = 0, t_size = 0;
        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            ui vprime = g_edges[j];
            if (visY[vprime]) {
                ++cross_ged;
                if (q_matrix[u*q_n + my[vprime]] == g_elabels[j])
                    cross_ged -= 2;
                else if (q_matrix[u*q_n + my[vprime]] < elabels_n)
                    --cross_ged;

                if (elabels_matrix[my[vprime]*elabels_n + g_elabels[j]] < 0)
                    ++lb;
                if (sizeY[vprime] > sizeX[my[vprime]]) --lb;

                continue;
            }

            --inner_ged;
            if (elabels_map[g_elabels[j]] <= 0) --el_common;
            --elabels_map[g_elabels[j]];

            ++t_size;
            if (elabels_matrix[u*elabels_n + g_elabels[j]] < 0) ++common;
            ++elabels_matrix[u*elabels_n + g_elabels[j]];
        }

        if (sizeX[u] > t_size) t_size = sizeX[u];
        lb += t_size - common;

        if (u_inner_cnt > inner_ged) inner_ged = u_inner_cnt;
        inner_ged -= el_common;

        /* 9.2 save (-score, v) */
        children[i-pre_siblings].first  = -mapped_cost - cross_ged - inner_ged - lb;
        children[i-pre_siblings].second = v;

        /* 9.3 rollback difference */
        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j)
            if (!visY[g_edges[j]]) {
                if (elabels_map[g_elabels[j]] < 0) ++el_common;
                ++elabels_map[g_elabels[j]];
                --elabels_matrix[u*elabels_n + g_elabels[j]];
            }
    }

/* ------------------------------------------------------------ *
 * 10. Select highest-scored candidate as now->image
 * ------------------------------------------------------------ */
    std::sort(children, children + (n - pre_siblings));
    now->image       = children[n - pre_siblings - 1].second;
    now->lower_bound = -children[n - pre_siblings - 1].first;

/* ------------------------------------------------------------ *
 * 11. Update mapped_cost
 * ------------------------------------------------------------ */
    ui cross_ged = u_cross_cnt;
    for (ui j = g_starts[now->image]; j < g_starts[now->image+1]; ++j)
        if (visY[g_edges[j]]) {
            ++cross_ged;
            if (q_matrix[u*q_n + my[g_edges[j]]] == g_elabels[j]) cross_ged -= 2;
            else if (q_matrix[u*q_n + my[g_edges[j]]] < elabels_n) --cross_ged;
        }
    if (q_vlabels[u] != g_vlabels[now->image]) ++cross_ged;
    now->mapped_cost = mapped_cost + cross_ged;

/* ------------------------------------------------------------ *
 * 12. Debug verification & cleanup
 * ------------------------------------------------------------ */
#ifndef NDEBUG
    verify_induced_ged(now);
#endif

    for (ui i = 0; i <= idx; ++i) {
        ui v = MO[i];
        for (ui j = q_starts[v]; j < q_starts[v+1]; ++j)
            elabels_matrix[v*elabels_n + q_elabels[j]] = 0;
    }
    for (State *st = now->parent; st; st = st->parent) {
        if (st->image >= g_n) continue;                  /* * skip dummy */
        ui v = st->image;
        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j)
            elabels_matrix[my[v]*elabels_n + g_elabels[j]] = 0;
    }
    for (ui i = 0; i < n; ++i) {
        vlabels_map[g_vlabels[candidates[i]]] = 0;
        for (ui j = g_starts[candidates[i]]; j < g_starts[candidates[i]+1]; ++j)
            elabels_map[g_elabels[j]] = 0;
    }
    for (ui i = idx+1; i < q_n; ++i) {
        vlabels_map[q_vlabels[MO[i]]] = 0;
        for (ui j = q_starts[MO[i]]; j < q_starts[MO[i]+1]; ++j)
            elabels_map[q_elabels[j]] = 0;
    }
    for (State *parent = now->parent; parent; parent = parent->parent)
        if (parent->image < g_n) visY[parent->image] = 0;        /* * */

/* ------------------------------------------------------------ *
 * 13. Pruning: lb ≥ UB -> dummy
 * ------------------------------------------------------------ */
    ++search_space;
    if (now->lower_bound >= upper_bound ||
        now->lower_bound > verify_upper_bound) {
        now->image = g_n;
        return;
    }

// Add before step 14 of compute_best_extension_LSa:

#ifdef _EXPAND_ALL_
    /* 14. If expanding all siblings, fill siblings[] */
  
    ushort *t_array = now->siblings;
    now->siblings_n = 0;
    for (ui i = 0; i < n - pre_siblings - 1; ++i) {
        t_array[now->siblings_n++] = children[i].second;
        t_array[now->siblings_n++] = -children[i].first;
    }
#endif
}



void Application::compute_best_extension_LSa(State *now,
                                             ui n,             // candidates[] count
                                             ui *candidates,   // candidate G vertex id list
                                             ui pre_siblings)  // (if expanding siblings) number of processed siblings
{
    // DISABLED: Segfault fix - use baseline version like in ori_without version
    return;
/* ------------------------------------------------------------ *
 * 0. Debug: auxiliary arrays must be all-zero
 * ------------------------------------------------------------ */
#ifndef NDEBUG
    for (ui i = 0; i < g_n; ++i) {
        assert(visX[i] == 0);
        assert(visY[i] == 0);
    }
    if (vlabels_map)  for (ui i = 0; i < vlabels_n;  ++i) assert(vlabels_map[i]  == 0);
    for (ui i = 0; i < elabels_n; ++i)           assert(elabels_map[i] == 0);
    if (elabels_matrix)
        for (ui i = 0; i < q_n * elabels_n; ++i) assert(elabels_matrix[i] == 0);
#endif

/* ------------------------------------------------------------ *
 * 1. If auxiliary buffer not allocated, new and zero-fill
 * ------------------------------------------------------------ */
    if (!vlabels_map) {
        vlabels_map = new int[vlabels_n]{0};
    }
    if (!elabels_matrix) {
        elabels_matrix = new short[q_n * elabels_n]{0};
    }

/* ------------------------------------------------------------ *
 * 2. Temporary arrays sizeX / sizeY
 * ------------------------------------------------------------ */
    ui *sizeX = queue;   // length ≥ q_n
    ui *sizeY = prev;    // length ≥ g_n

/* ------------------------------------------------------------ *
 * 3. Pre-scan query side
 * ------------------------------------------------------------ */
    ui idx = now->level;
    ui u   = MO[idx];
    ui u_cross_cnt = 0, u_inner_cnt = 0;

    /* 3.1 visX marking */
    for (ui i = 0; i <= idx; ++i) visX[MO[i]] = 1;

    /* 3.2 sizeX & label difference */
    for (ui i = 0; i <= idx; ++i) {
        ui v = MO[i];
        sizeX[v] = 0;
        for (ui j = q_starts[v]; j < q_starts[v+1]; ++j) {
            if (!visX[q_edges[j]]) {
                ++sizeX[v];
                --elabels_matrix[v*elabels_n + q_elabels[j]];
            }
        }
    }

    /* 3.3 u cross-edge count */
    for (ui j = q_starts[u]; j < q_starts[u+1]; ++j)
        if (visX[q_edges[j]]) ++u_cross_cnt;

    /* 3.4 unmapped region internal edge count */
    for (ui i = idx+1; i < q_n; ++i)
        for (ui j = q_starts[MO[i]]; j < q_starts[MO[i]+1]; ++j) {
            ui uprime = q_edges[j];
            if (!visX[uprime] && MO[i] < uprime) {
                --elabels_map[q_elabels[j]];
                ++u_inner_cnt;
            }
        }

    /* 3.5 vertex label surplus/deficit */
    for (ui i = idx+1; i < q_n; ++i)
        --vlabels_map[q_vlabels[MO[i]]];

    /* 3.6 Clean up visX */
    for (ui i = 0; i <= idx; ++i) visX[MO[i]] = 0;

/* ------------------------------------------------------------ *
 * 4. Mark occupied G-side vertices in visY / my
 * ------------------------------------------------------------ */
    for (State *parent = now->parent; parent; parent = parent->parent) {
        if (parent->image >= g_n) continue;              /* * skip dummy */
        visY[parent->image] = 1;
        my[parent->image]  = MO[parent->level];
    }

/* ------------------------------------------------------------ *
 * 5. Compute sizeY & elabels_matrix correction for ancestor mapping
 * ------------------------------------------------------------ */
    ui ged_common = 0;
    for (State *st = now->parent; st; st = st->parent) {
        if (st->image >= g_n) continue;                  /* * skip dummy */
        ui v = st->image;
        ui common = 0;
        sizeY[v] = 0;

        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            if (!visY[g_edges[j]]) {
                short &cnt = elabels_matrix[MO[st->level]*elabels_n +
                                            g_elabels[j]];
                if (cnt < 0) ++common;
                ++cnt;
                ++sizeY[v];
            }
        }
        ui t_ged = sizeX[MO[st->level]];
        if (sizeY[v] > t_ged) t_ged = sizeY[v];
        ged_common += t_ged - common;
    }

/* ------------------------------------------------------------ *
 * 6. Scan candidate G vertices, update label surplus/deficit statistics
 * ------------------------------------------------------------ */
    ui vl_common = 0, el_common = 0, v_total_cnt = 0;
    for (ui i = 0; i < n; ++i) {
        ui v = candidates[i];
        if (vlabels_map[g_vlabels[v]] < 0) ++vl_common;
        ++vlabels_map[g_vlabels[v]];

        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            ui vprime = g_edges[j];
            if (!visY[vprime] && v < vprime) {
                if (elabels_map[g_elabels[j]] < 0) ++el_common;
                ++elabels_map[g_elabels[j]];
                ++v_total_cnt;
            }
        }
    }

/* ------------------------------------------------------------ *
 * 7. Compute vertex label GED upper bound
 * ------------------------------------------------------------ */
    ui mapped_cost = (now->parent ? now->parent->mapped_cost : 0);

    ui vl_ged = q_n - idx - 1;
    if (g_n > q_n) vl_ged = g_n - idx - 1;
    vl_ged -= vl_common;

/* ------------------------------------------------------------ *
 * 8. children[] stores (-score, v)
 *    [Optimization] choose strategy based on whether sibling expansion is needed
 * ------------------------------------------------------------ */
    if (!children) children = new std::pair<int,int>[g_n];

#ifdef _EXPAND_ALL_
    // When sibling expansion needed, use extended struct to save all info
    struct ExtendedChild {
        int neg_score;
        ui vertex_id;
        ui cross_ged;
        
        bool operator<(const ExtendedChild& other) const {
            return neg_score < other.neg_score;
        }
    };
    
    // Use alloca to allocate on stack (faster, auto-freed)
    ExtendedChild *ext_children = (ExtendedChild*)alloca(
        sizeof(ExtendedChild) * (n - pre_siblings));
#else
    // When no sibling expansion needed, only record best candidate
    ui best_cross_ged = 0;
    int best_score = INT_MIN;
    ui best_idx = 0;
#endif

/* ------------------------------------------------------------ *
 * 9. Iterate candidate G vertices v
 * ------------------------------------------------------------ */
    for (ui i = pre_siblings; i < n; ++i) {
        ui v = candidates[i];

        /* 9.1 compute lower bound lb */
        ui lb = vl_ged + ged_common;
        if (vlabels_map[g_vlabels[v]] <= 0) ++lb;

        ui cross_ged = u_cross_cnt;
        ui inner_ged = v_total_cnt;
        if (g_vlabels[v] != q_vlabels[u]) ++cross_ged;

        ui common = 0, t_size = 0;
        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            ui vprime = g_edges[j];
            if (visY[vprime]) {
                ++cross_ged;
                if (q_matrix[u*q_n + my[vprime]] == g_elabels[j])
                    cross_ged -= 2;
                else if (q_matrix[u*q_n + my[vprime]] < elabels_n)
                    --cross_ged;

                if (elabels_matrix[my[vprime]*elabels_n + g_elabels[j]] < 0)
                    ++lb;
                if (sizeY[vprime] > sizeX[my[vprime]]) --lb;

                continue;
            }

            --inner_ged;
            if (elabels_map[g_elabels[j]] <= 0) --el_common;
            --elabels_map[g_elabels[j]];

            ++t_size;
            if (elabels_matrix[u*elabels_n + g_elabels[j]] < 0) ++common;
            ++elabels_matrix[u*elabels_n + g_elabels[j]];
        }

        if (sizeX[u] > t_size) t_size = sizeX[u];
        lb += t_size - common;

        if (u_inner_cnt > inner_ged) inner_ged = u_inner_cnt;
        inner_ged -= el_common;

        /* 9.2 save results */
        int score = -mapped_cost - cross_ged - inner_ged - lb;
        
#ifdef _EXPAND_ALL_
        ui idx_child = i - pre_siblings;
        ext_children[idx_child].neg_score = score;
        ext_children[idx_child].vertex_id = v;
        ext_children[idx_child].cross_ged = cross_ged;
#else
        // only record best candidate
        if (score > best_score) {
            best_score = score;
            best_cross_ged = cross_ged;
            best_idx = i - pre_siblings;
        }
#endif
        
        children[i-pre_siblings].first  = score;
        children[i-pre_siblings].second = v;

        /* 9.3 rollback difference */
        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j)
            if (!visY[g_edges[j]]) {
                if (elabels_map[g_elabels[j]] < 0) ++el_common;
                ++elabels_map[g_elabels[j]];
                --elabels_matrix[u*elabels_n + g_elabels[j]];
            }
    }

/* ------------------------------------------------------------ *
 * 10. Select highest-scored candidate as now->image
 * ------------------------------------------------------------ */
#ifdef _EXPAND_ALL_
    ui num_children = n - pre_siblings;
    std::sort(ext_children, ext_children + num_children);
    
    ExtendedChild &best = ext_children[num_children - 1];
    now->image = best.vertex_id;
    now->lower_bound = -best.neg_score;
    now->mapped_cost = mapped_cost + best.cross_ged;
#else
    now->image = children[best_idx].second;
    now->lower_bound = -best_score;
    now->mapped_cost = mapped_cost + best_cross_ged;
#endif

/* ------------------------------------------------------------ *
 * 12. Debug verification & cleanup
 * ------------------------------------------------------------ */
#ifndef NDEBUG
    verify_induced_ged(now);
    
#endif

    for (ui i = 0; i <= idx; ++i) {
        ui v = MO[i];
        for (ui j = q_starts[v]; j < q_starts[v+1]; ++j)
            elabels_matrix[v*elabels_n + q_elabels[j]] = 0;
    }
    for (State *st = now->parent; st; st = st->parent) {
        if (st->image >= g_n) continue;                  /* * skip dummy */
        ui v = st->image;
        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j)
            elabels_matrix[my[v]*elabels_n + g_elabels[j]] = 0;
    }
    for (ui i = 0; i < n; ++i) {
        vlabels_map[g_vlabels[candidates[i]]] = 0;
        for (ui j = g_starts[candidates[i]]; j < g_starts[candidates[i]+1]; ++j)
            elabels_map[g_elabels[j]] = 0;
    }
    for (ui i = idx+1; i < q_n; ++i) {
        vlabels_map[q_vlabels[MO[i]]] = 0;
        for (ui j = q_starts[MO[i]]; j < q_starts[MO[i]+1]; ++j)
            elabels_map[q_elabels[j]] = 0;
    }
    for (State *parent = now->parent; parent; parent = parent->parent)
        if (parent->image < g_n) visY[parent->image] = 0;        /* * */

/* ------------------------------------------------------------ *
 * 13. Pruning: lb ≥ UB -> dummy
 * ------------------------------------------------------------ */
    ++search_space;
    if (now->lower_bound >= upper_bound ||
        now->lower_bound > verify_upper_bound) {
        now->image = g_n;
        return;
    }

#ifdef _EXPAND_ALL_
    /* 14. If expanding all siblings, fill siblings[]
     *     [Optimization] copy directly from sorted array
     */
    // Update children array order
    for (ui i = 0; i < num_children; ++i) {
        children[i].first = ext_children[i].neg_score;
        children[i].second = ext_children[i].vertex_id;
    }
    
    ushort *t_array = now->siblings;
    now->siblings_n = 0;
    for (ui i = 0; i < num_children - 1; ++i) {
        t_array[now->siblings_n++] = children[i].second;
        t_array[now->siblings_n++] = -children[i].first;
    }
#endif
}

void Application::compute_best_extension_BM_baseline(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings, char no_siblings, char IS) {
#ifndef NDEBUG
	for(ui i = 0;i < g_n;i ++) {
		assert(visX[i] == 0);
		assert(visY[i] == 0);
	}
#endif
	for(ui i = 0;i < now->level;i ++) visX[MO[i]] = 1;
	for(State *parent = now->parent;parent != NULL;parent = parent->parent) {
		visY[parent->image] = 1;
		my[parent->image] = MO[parent->level];
	}

	ui start_idx = now->level;
	for(ui i = 0;i < n;i ++) {
		ui *t_array = cost + i*n;
		if(start_idx + i >= q_n) {
			for(ui j = 0;j < n;j ++) {
				ui &cot = t_array[j];
				cot = 2;
				for(ui k = g_starts[candidates[j]];k < g_starts[candidates[j]+1];k ++) {
					if(visY[g_edges[k]]&&anchor_aware) cot += 2;
					else ++ cot;
				}
			}
			continue;
		}

		ui u = MO[start_idx+i];
		ui anchored_edges = 0;
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) {
			if(anchor_aware&&visX[q_edges[j]]) ++ anchored_edges;
			else ++ elabels_map[q_elabels[j]];
		}
		for(ui j = 0;j < n;j ++) {
			if(i == 0&&j < pre_siblings) {
				t_array[j] = INF;
				continue;
			}
			ui v = candidates[j];
			ui &cot = t_array[j];
			if(anchor_aware) cot = anchored_edges;
			else cot = 0;
			ui total = 0, common = 0;
			ui queue_n = 0;
			//if(n == 4&&i == 3&&now->parent->image == 3) printf("%d cot: %d\n", j, cot);
			for(ui k = g_starts[v];k < g_starts[v+1];k ++) {
				if(anchor_aware&&visY[g_edges[k]]) {
					++ cot;
					if(q_matrix[u*q_n+my[g_edges[k]]] == g_elabels[k]) cot -= 2;
					else if(q_matrix[u*q_n+my[g_edges[k]]] < elabels_n) -- cot;
				}
				else {
					++ total;
					if(elabels_map[g_elabels[k]]) {
						++ common;
						-- elabels_map[g_elabels[k]];
						queue[queue_n ++] = g_elabels[k];
					}
				}
			}
			cot *= 2;
			//if(n == 4&&i == 3&&now->parent->image == 3) printf("%d: %d, %d, %d\n", candidates[j], cot, total, common);
			if(q_starts[u+1] - q_starts[u] - anchored_edges*anchor_aware > total) total = q_starts[u+1] - q_starts[u] - anchored_edges*anchor_aware;
			cot += total - common;
			for(ui k = 0;k < queue_n;k ++) ++ elabels_map[queue[k]];

			if(q_vlabels[u] != g_vlabels[v]) cot += 2;
		}
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) elabels_map[q_elabels[j]] = 0;
	}
	for(State *parent = now->parent;parent != NULL;parent = parent->parent) visY[parent->image] = 0;
	for(ui i = 0;i < now->level;i ++) visX[MO[i]] = 0;

	if(!IS) ++ search_space;

#ifndef NDEBUG
	//if(n < q_n) printf("%d ", now->parent->image);
	//printf("candidates: %d:", n);
	//for(ui i = 0;i < n;i ++) printf(" %d", candidates[i]);
	//printf("\n");
#endif

	now->lower_bound = now->mapped_cost + (Hungarian(1, n, cost)+1)/2;
	if(now->lower_bound >= upper_bound||now->lower_bound > verify_upper_bound) {
		now->image = g_n;
		return ;
	}
	now->image = candidates[mx[0]];

#ifndef NDEBUG
	//printf("Lower bound (%d-%d): %d\n", MO[now->level], candidates[mx[0]], now->lower_bound);
#endif

	if(IS) {
		if(now->lower_bound < upper_bound) {
			upper_bound = now->lower_bound;
			for(State *st = now->parent;st != NULL;st = st->parent) BX[MO[st->level]] = st->image;
			for(ui i = 0;i+now->level < q_n;i ++) BX[MO[i+now->level]] = candidates[mx[i]];

#ifndef NDEBUG
			//printf("supposed GED in IS: %d\n", upper_bound);
			assert(upper_bound == compute_ged_of_BX());
#endif
		}
		return ;
	}

	compute_mapped_cost_and_upper_bound_baseline(now, n, candidates, mx);

	if(no_siblings) return ;

#ifdef _EXPAND_ALL_
	ui p_mapping_cost = 0;
	if(now->parent != NULL) p_mapping_cost = now->parent->mapped_cost;

	ushort *t_array = now->siblings;
	now->siblings_n = 0;
	for(ui i = 1;i < n;i ++) {
		//++ search_space;
		cost[mx[0]] = INF; my[mx[0]] = -1; mx[0] = -1;
		ui lb = p_mapping_cost + (Hungarian(0, n, cost)+1)/2;
#ifdef _EARLY_STOP_
		if(lb >= upper_bound||lb > verify_upper_bound) break;
#endif

		t_array[now->siblings_n++] = candidates[mx[0]];
		t_array[now->siblings_n++] = lb;
	}
	ui tn = now->siblings_n;
	for(ui i = 0;i + 2 < tn;i += 2) {
		swap(t_array[i], t_array[tn-2]);
		swap(t_array[i+1], t_array[tn-1]);
		tn -= 2;
	}
#endif
}

// Add macro definition at file top

// #define TIME_PROFILE       // Uncomment to enable time profiling

#ifdef TIME_PROFILE
#include <chrono>
#include <iomanip>
#endif

void Application::compute_best_extension_BM_test(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings, char no_siblings, char IS) {
    
#ifdef TIME_PROFILE
    using namespace std::chrono;
    auto total_start = high_resolution_clock::now();
    auto cost_matrix_start = high_resolution_clock::now();
#endif

#ifndef NDEBUG
	for(ui i = 0;i < g_n;i ++) {
		assert(visX[i] == 0);
		assert(visY[i] == 0);
	}
#endif
	for(ui i = 0;i < now->level;i ++) visX[MO[i]] = 1;
	for(State *parent = now->parent;parent != NULL;parent = parent->parent) {
		visY[parent->image] = 1;
		my[parent->image] = MO[parent->level];
	}

	ui start_idx = now->level;
	
	// ========== Optimized cost matrix computation section ==========
	for(ui i = 0;i < n;i ++) {
		ui *t_array = cost + i*n;
		if(start_idx + i >= q_n) {
			// Case where query vertices are exhausted
			for(ui j = 0;j < n;j ++) {
				ui &cot = t_array[j];
				cot = 2;
#ifdef ALL_EDGE_LABELS_SAME
				// Optimized version: compute degree directly
				ui v_degree = g_starts[candidates[j]+1] - g_starts[candidates[j]];
				if(anchor_aware) {
					// Count anchored edges
					ui anchored_count = 0;
					for(ui k = g_starts[candidates[j]]; k < g_starts[candidates[j]+1]; k++) {
						if(visY[g_edges[k]]) anchored_count++;
					}
					cot += 2 * anchored_count + (v_degree - anchored_count);
				} else {
					cot += v_degree;
				}
#else
				// Original version
				for(ui k = g_starts[candidates[j]];k < g_starts[candidates[j]+1];k ++) {
					if(visY[g_edges[k]]&&anchor_aware) cot += 2;
					else ++ cot;
				}
#endif
			}
			continue;
		}

		ui u = MO[start_idx+i];
		
#ifdef ALL_EDGE_LABELS_SAME
		// Optimized version: pre-compute info for u
		ui u_degree = q_starts[u+1] - q_starts[u];
		ui u_anchored_count = 0;
		
		if(anchor_aware) {
			for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
				if(visX[q_edges[j]]) u_anchored_count++;
			}
		}
		ui u_non_anchored = u_degree - u_anchored_count;
#else
		// Original version: need to count edge labels
		ui anchored_edges = 0;
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) {
			if(anchor_aware&&visX[q_edges[j]]) ++ anchored_edges;
			else ++ elabels_map[q_elabels[j]];
		}
#endif
		
		for(ui j = 0;j < n;j ++) {
			if(i == 0&&j < pre_siblings) {
				t_array[j] = INF;
				continue;
			}
			ui v = candidates[j];
			ui &cot = t_array[j];
			
#ifdef ALL_EDGE_LABELS_SAME
			// Optimized version
			if(anchor_aware) {
				cot = u_anchored_count;
				
				// Process all edges of v in one pass
				ui v_degree = g_starts[v+1] - g_starts[v];
				ui v_anchored_count = 0;
				ui v_anchored_matched = 0;
				
				for(ui k = g_starts[v]; k < g_starts[v+1]; k++) {
					if(visY[g_edges[k]]) {
						v_anchored_count++;
						// Check if edge exists (no need to compare labels)
						if(q_matrix[u*q_n + my[g_edges[k]]] < elabels_n) {
							v_anchored_matched++;
						}
					}
				}
				
				// Compute anchored edge cost
				cot += v_anchored_count;
				cot -= 2 * v_anchored_matched;
				cot *= 2;
				
				// Non-anchored edge cost (simple degree difference)
				ui v_non_anchored = v_degree - v_anchored_count;
				ui diff = (u_non_anchored > v_non_anchored) ? 
						  (u_non_anchored - v_non_anchored) : 
						  (v_non_anchored - u_non_anchored);
				cot += diff;
			} else {
				// Simplified version without considering anchor
				cot = 0;
				ui v_degree = g_starts[v+1] - g_starts[v];
				ui diff = (u_degree > v_degree) ? 
						  (u_degree - v_degree) : 
						  (v_degree - u_degree);
				cot = 2 * diff;
			}
#else
			// Original version
			if(anchor_aware) cot = anchored_edges;
			else cot = 0;
			ui total = 0, common = 0;
			ui queue_n = 0;
			
			for(ui k = g_starts[v];k < g_starts[v+1];k ++) {
				if(anchor_aware&&visY[g_edges[k]]) {
					++ cot;
					if(q_matrix[u*q_n+my[g_edges[k]]] == g_elabels[k]) cot -= 2;
					else if(q_matrix[u*q_n+my[g_edges[k]]] < elabels_n) -- cot;
				}
				else {
					++ total;
					if(elabels_map[g_elabels[k]]) {
						++ common;
						-- elabels_map[g_elabels[k]];
						queue[queue_n ++] = g_elabels[k];
					}
				}
			}
			cot *= 2;
			
			if(q_starts[u+1] - q_starts[u] - anchored_edges*anchor_aware > total) 
				total = q_starts[u+1] - q_starts[u] - anchored_edges*anchor_aware;
			cot += total - common;
			for(ui k = 0;k < queue_n;k ++) ++ elabels_map[queue[k]];
#endif
			
			// Vertex label difference
			if(q_vlabels[u] != g_vlabels[v]) cot += 2;
		}
		
#ifndef ALL_EDGE_LABELS_SAME
		// Cleanup only needed in original version
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) elabels_map[q_elabels[j]] = 0;
#endif
	}
	
#ifdef TIME_PROFILE
    auto cost_matrix_end = high_resolution_clock::now();
    auto hungarian_start = high_resolution_clock::now();
#endif

	for(State *parent = now->parent;parent != NULL;parent = parent->parent) visY[parent->image] = 0;
	for(ui i = 0;i < now->level;i ++) visX[MO[i]] = 0;

	if(!IS) ++ search_space;

	// ========== Hungarian algorithm section ==========
	now->lower_bound = now->mapped_cost + (Hungarian(1, n, cost)+1)/2;
	
#ifdef TIME_PROFILE
    auto hungarian_end = high_resolution_clock::now();
    
    // Compute time
    auto cost_matrix_time = duration_cast<nanoseconds>(cost_matrix_end - cost_matrix_start).count();
    auto hungarian_time = duration_cast<nanoseconds>(hungarian_end - hungarian_start).count();
    auto total_time = duration_cast<nanoseconds>(hungarian_end - total_start).count();
    
    // Print statistics
    printf("[TIME_PROFILE] compute_best_extension_BM_test (level=%u, n=%u, anchor=%d):\n", 
           now->level, n, (int)anchor_aware);
    printf("  Cost Matrix: %10lld ns (%6.2f%%)\n", 
           cost_matrix_time, 100.0 * cost_matrix_time / total_time);
    printf("  Hungarian:   %10lld ns (%6.2f%%)\n", 
           hungarian_time, 100.0 * hungarian_time / total_time);
    printf("  Total:       %10lld ns\n", total_time);
    
#ifdef ALL_EDGE_LABELS_SAME
    printf("  [Optimized: ALL_EDGE_LABELS_SAME enabled]\n");
#endif
    
    // Accumulated statistics
    static long long total_cost_matrix_time = 0;
    static long long total_hungarian_time = 0;
    static long long total_calls = 0;
    
    total_cost_matrix_time += cost_matrix_time;
    total_hungarian_time += hungarian_time;
    total_calls++;
    
    if (total_calls % 1000 == 0) {
        printf("\n[TIME_PROFILE] Cumulative stats after %lld calls:\n", total_calls);
        printf("  Avg Cost Matrix: %10lld ns (%6.2f%%)\n", 
               total_cost_matrix_time / total_calls,
               100.0 * total_cost_matrix_time / (total_cost_matrix_time + total_hungarian_time));
        printf("  Avg Hungarian:   %10lld ns (%6.2f%%)\n", 
               total_hungarian_time / total_calls,
               100.0 * total_hungarian_time / (total_cost_matrix_time + total_hungarian_time));
        printf("  Cost/Hungarian ratio: %.2f\n\n", 
               (double)total_cost_matrix_time / total_hungarian_time);
    }
#endif

	if(now->lower_bound >= upper_bound||now->lower_bound > verify_upper_bound) {
		now->image = g_n;
		return ;
	}
	now->image = candidates[mx[0]];

    // ui tmp_lb = compute_independent_lower_bound_LSa_baseline(now);
    
    // if(tmp_lb >= upper_bound||tmp_lb > verify_upper_bound) {
    //     // printf("now->lower_bound: %u, tmp_lb: %u\n", now->lower_bound, tmp_lb);
    //     now->lower_bound = tmp_lb;
	// 	now->image = g_n;
	// 	return ;
	// }

	if(IS) {
		if(now->lower_bound < upper_bound) {
			upper_bound = now->lower_bound;
			for(State *st = now->parent;st != NULL;st = st->parent) BX[MO[st->level]] = st->image;
			for(ui i = 0;i+now->level < q_n;i ++) BX[MO[i+now->level]] = candidates[mx[i]];

#ifndef NDEBUG
			assert(upper_bound == compute_ged_of_BX());
#endif
		}
		return ;
	}

	compute_mapped_cost_and_upper_bound_test(now, n, candidates, mx);
    
	if(no_siblings) return ;

#ifdef _EXPAND_ALL_
	ui p_mapping_cost = 0;
	if(now->parent != NULL) p_mapping_cost = now->parent->mapped_cost;

	ushort *t_array = now->siblings;
	now->siblings_n = 0;

#ifdef TIME_PROFILE
    auto sibling_start = high_resolution_clock::now();
    long long sibling_hungarian_time = 0;
    int sibling_hungarian_calls = 0;
#endif

	for(ui i = 1;i < n;i ++) {
		cost[mx[0]] = INF; my[mx[0]] = -1; mx[0] = -1;
		
#ifdef TIME_PROFILE
        auto sibling_hung_start = high_resolution_clock::now();
#endif
		
		ui lb = p_mapping_cost + (Hungarian(0, n, cost)+1)/2;
		
#ifdef TIME_PROFILE
        auto sibling_hung_end = high_resolution_clock::now();
        sibling_hungarian_time += duration_cast<nanoseconds>(sibling_hung_end - sibling_hung_start).count();
        sibling_hungarian_calls++;
#endif
		
#ifdef _EARLY_STOP_
		if(lb >= upper_bound||lb > verify_upper_bound) break;
#endif

		t_array[now->siblings_n++] = candidates[mx[0]];
		t_array[now->siblings_n++] = lb;
	}
	
#ifdef TIME_PROFILE
    auto sibling_end = high_resolution_clock::now();
    auto total_sibling_time = duration_cast<nanoseconds>(sibling_end - sibling_start).count();
    
    if(sibling_hungarian_calls > 0) {
        printf("  Sibling generation: %10lld ns total\n", total_sibling_time);
        printf("    - Hungarian calls: %d\n", sibling_hungarian_calls);
        printf("    - Hungarian time:  %10lld ns (%6.2f%%)\n", 
               sibling_hungarian_time, 100.0 * sibling_hungarian_time / total_sibling_time);
    }
#endif

	ui tn = now->siblings_n;
	for(ui i = 0;i + 2 < tn;i += 2) {
		swap(t_array[i], t_array[tn-2]);
		swap(t_array[i+1], t_array[tn-1]);
		tn -= 2;
	}
#endif
}
/*==============================================================*
 |      Application::compute_best_extension_BM (BM / BMao)      |
 *==============================================================*/


void Application::compute_best_extension_BM(char anchor_aware,
                                            State *now,
                                            ui n, ui *candidates,
                                            ui pre_siblings,
                                            char no_siblings,
                                            char IS)  // Function signature unchanged!
{
#ifndef NDEBUG
    for (ui i = 0; i < g_n; ++i) {        // debug: ensure bitmap is all 0
        assert(visX[i] == 0);
        assert(visY[i] == 0);
    }
#endif
    
    /* ---------- 1. Mark mapped query vertices ---------- */
    for (ui i = 0; i < now->level; ++i) visX[MO[i]] = 1;

    /* ---------- 2. Mark ancestor-mapped G vertices ---------- */
    for (State *p = now->parent; p != NULL; p = p->parent) {
        if (p->image >= g_n) continue;            // * skip dummy-root
        visY[p->image] = 1;
        my[p->image]   = MO[p->level];
    }

    ui start_idx = now->level;

    /* ---------- 3. Iterate candidate G vertices (cost matrix computation) ---------- */
    for (ui i = 0; i < n; ++i) {
        ui *t_array = cost + i * n;

        /* 3.1 If query vertices are exhausted */
        if (start_idx + i >= q_n) {
            for (ui j = 0; j < n; ++j) {
                ui &cot = t_array[j];
                cot = 2;
                
#ifdef ALL_EDGE_LABELS_SAME
                
                // Optimized version: compute degree directly
                ui v_degree = g_starts[candidates[j]+1] - g_starts[candidates[j]];
                if (anchor_aware) {
                    // Count anchored edges
                    ui anchored_count = 0;
                    for (ui k = g_starts[candidates[j]]; k < g_starts[candidates[j]+1]; ++k) {
                        if (visY[g_edges[k]]) anchored_count++;
                    }
                    cot += 2 * anchored_count + (v_degree - anchored_count);
                } else {
                    cot += v_degree;
                }
#else
                // Original version
                for (ui k = g_starts[candidates[j]]; k < g_starts[candidates[j]+1]; ++k) {
                    if (anchor_aware && visY[g_edges[k]]) cot += 2;
                    else ++cot;
                }
#endif
            }
            continue;
        }

        /* 3.2 Normal computation */
        ui u = MO[start_idx + i];
        
#ifdef ALL_EDGE_LABELS_SAME
        // Optimized version: pre-compute info for u
        ui u_degree = q_starts[u+1] - q_starts[u];
        ui u_anchored_count = 0;
        
        if (anchor_aware) {
            for (ui j = q_starts[u]; j < q_starts[u+1]; ++j) {
                if (visX[q_edges[j]]) u_anchored_count++;
            }
        }
        ui u_non_anchored = u_degree - u_anchored_count;
#else
        // Original version: need to count edge labels
        ui anchored_edges = 0;
        for (ui j = q_starts[u]; j < q_starts[u+1]; ++j) {
            if (anchor_aware && visX[q_edges[j]]) ++anchored_edges;
            else ++elabels_map[q_elabels[j]];
        }
#endif

        for (ui j = 0; j < n; ++j) {
            if (i == 0 && j < pre_siblings) {
                t_array[j] = INF;
                continue;
            }
            ui v = candidates[j];
            ui &cot = t_array[j];
            
#ifdef ALL_EDGE_LABELS_SAME
            // Optimized version
            if (anchor_aware) {
                cot = u_anchored_count;
                
                // Process all edges of v in one pass
                ui v_degree = g_starts[v+1] - g_starts[v];
                ui v_anchored_count = 0;
                ui v_anchored_matched = 0;
                
                for (ui k = g_starts[v]; k < g_starts[v+1]; ++k) {
                    if (visY[g_edges[k]]) {
                        v_anchored_count++;
                        // Check if edge exists (no need to compare labels)
                        if (q_matrix[u*q_n + my[g_edges[k]]] < elabels_n) {
                            v_anchored_matched++;
                        }
                    }
                }
                
                // Compute anchored edge cost
                cot += v_anchored_count;
                cot -= 2 * v_anchored_matched;
                cot *= 2;
                
                // Non-anchored edge cost (simple degree difference)
                ui v_non_anchored = v_degree - v_anchored_count;
                ui diff = (u_non_anchored > v_non_anchored) ? 
                          (u_non_anchored - v_non_anchored) : 
                          (v_non_anchored - u_non_anchored);
                cot += diff;
            } else {
                // Simplified version without considering anchor
                cot = 0;
                ui v_degree = g_starts[v+1] - g_starts[v];
                ui diff = (u_degree > v_degree) ? 
                          (u_degree - v_degree) : 
                          (v_degree - u_degree);
                cot = 2 * diff;
            }
#else
            // Original version
            cot = (anchor_aware ? anchored_edges : 0);

            ui total = 0, common = 0, queue_n = 0;

            for (ui k = g_starts[v]; k < g_starts[v+1]; ++k) {
                if (anchor_aware && visY[g_edges[k]]) {
                    ++cot;

                    if (q_matrix[u*q_n + my[g_edges[k]]] == g_elabels[k])      cot -= 2;
                    else if (q_matrix[u*q_n + my[g_edges[k]]] < elabels_n)     --cot;
                } else {
                    ++total;
                    if (elabels_map[g_elabels[k]]) {
                        ++common;
                        --elabels_map[g_elabels[k]];
                        queue[queue_n++] = g_elabels[k];
                    }
                }
            }
            cot *= 2;
            ui q_deg = q_starts[u+1] - q_starts[u] - anchored_edges * anchor_aware;
            if (q_deg > total) total = q_deg;
            cot += total - common;

            for (ui k = 0; k < queue_n; ++k) ++elabels_map[queue[k]];
#endif
            
            // Vertex label difference (common to both versions)
            if (q_vlabels[u] != g_vlabels[v]) cot += 2;
        }
        
#ifndef ALL_EDGE_LABELS_SAME
        // Cleanup only needed in original version
        for (ui j = q_starts[u]; j < q_starts[u+1]; ++j) elabels_map[q_elabels[j]] = 0;
#endif
    }

    /* ---------- 4. Clean up visY / visX ---------- */
    for (State *p = now->parent; p != NULL; p = p->parent)
        if (p->image < g_n) visY[p->image] = 0;          // *
    for (ui i = 0; i < now->level; ++i) visX[MO[i]] = 0;

    if (!IS) ++search_space;

    /* ---------- 5. Compute lower bound (Hungarian algorithm) ---------- */
    now->lower_bound = now->mapped_cost + (Hungarian(1, n, cost) + 1) / 2;

    /* ---------- 6. Pruning decision based on class member margin ---------- */
    // Use this->margin (or just margin, since in member function)
    // margin=0: use baseline strict condition (>=)
    // margin>0: relax condition, allow keeping more nodes
    bool should_prune = false;
    
    if (margin == 0) {
        // Baseline behavior: >= upper_bound or > verify_upper_bound
        should_prune = (now->lower_bound >= upper_bound || 
                       now->lower_bound > verify_upper_bound);
    } else {
        // Relaxed version: add margin tolerance
        should_prune = (now->lower_bound > upper_bound + margin - 1 || 
                       now->lower_bound > verify_upper_bound + margin);
    }
    
    if (should_prune) {
        now->image = g_n;    // dummy
        return;
    }
    
    now->image = candidates[mx[0]];
    
    /* ---------- 7. IS mode: directly update UB ---------- */
    if (IS) {
        if (now->lower_bound < upper_bound) {
            upper_bound = now->lower_bound;
            for (State *st = now->parent; st != NULL; st = st->parent) {
                if (st->level >= q_n) {
                    assert(false);
                    break;
                }
                BX[MO[st->level]] = st->image;
            }
            for (ui i = 0; i + now->level < q_n; ++i)
                BX[MO[i + now->level]] = candidates[mx[i]];
    #ifndef NDEBUG
            assert(upper_bound == compute_ged_of_BX());
    #endif
        }
        return;
    }
  
    /* ---------- 8. Compute mapped_cost & update UB ---------- */
    // Clear visX and visY before calling compute_mapped_cost_and_upper_bound_test (it has assertions requiring them to be 0)
    memset(visX, 0, sizeof(char)*g_n);
    memset(visY, 0, sizeof(char)*g_n);

    ui old_upper_bound = upper_bound;  // record old upper bound
    compute_mapped_cost_and_upper_bound_test(now, n, candidates, mx);
    
    // Record node that updated upper bound (if needed)
    if (upper_bound < old_upper_bound) {
        full_mapping_nodes.push_back(now);
    }
    
    if (no_siblings) return;

#ifdef _EXPAND_ALL_
    /* ---------- 9. sibling queue (expand all siblings) ---------- */
    ui p_mapping_cost = (now->parent ? now->parent->mapped_cost : 0);
    ushort *t_array   = now->siblings;
    now->siblings_n   = 0;

    for (ui i = 1; i < n; ++i) {
        cost[mx[0]] = INF; my[mx[0]] = -1; mx[0] = -1;
        ui lb = p_mapping_cost + (Hungarian(0, n, cost) + 1) / 2;
        
#ifdef _EARLY_STOP_
        // Apply same margin strategy to siblings (using class member margin)
        bool sibling_should_prune = false;
        
        if (margin == 0) {
            sibling_should_prune = (lb >= upper_bound || lb > verify_upper_bound);
        } else {
            sibling_should_prune = (lb > upper_bound + margin - 1 || 
                                   lb > verify_upper_bound + margin);
        }
        
        if (sibling_should_prune) break;
#endif

        t_array[now->siblings_n++] = candidates[mx[0]];
        t_array[now->siblings_n++] = lb;
    }
    
    /* Put the smallest lb at array end (stack effect) */
    ui tn = now->siblings_n;
    for (ui i = 0; i + 2 < tn; i += 2) {
        swap(t_array[i],     t_array[tn - 2]);
        swap(t_array[i + 1], t_array[tn - 1]);
        tn -= 2;
    }
    
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    now->siblings_total_n = now->siblings_n;  // At initial generation, both are equal
#endif
#endif
}



void Application::compute_best_extension_BMa(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings) {
	State tmp;
	tmp.parent = now;
	tmp.level = now->level+1;
	tmp.pre_sibling = NULL;

	long long old_search_space = search_space;

	if(children == NULL) children = new pair<int,int>[g_n];
	for(ui i = pre_siblings;i < n;i ++) {
		now->image = candidates[i];
		swap(candidates[i], candidates[n-1]);

		compute_mapped_cost(now);
		tmp.mapped_cost = now->mapped_cost;
		compute_best_extension_BM(anchor_aware, &tmp, n-1, candidates, 0, 1);
		children[i-pre_siblings].first = -tmp.lower_bound;
		children[i-pre_siblings].second = now->image;

		swap(candidates[i], candidates[n-1]);
	}

	search_space = old_search_space + 1;

	sort(children, children + n-pre_siblings);
	now->image = children[n-pre_siblings-1].second;
	now->lower_bound = -children[n-pre_siblings-1].first;
	compute_mapped_cost(now);

	if(now->lower_bound >= upper_bound||now->lower_bound > verify_upper_bound) {
		now->image = g_n;
		return ;
	}

#ifdef _EXPAND_ALL_
	ushort *t_array = now->siblings;
	now->siblings_n = 0;
	for(ui i = 0;i < n-pre_siblings-1;i ++) {
		//++ search_space;
		t_array[now->siblings_n++] = children[i].second;
		t_array[now->siblings_n++] = -children[i].first;
	}
#endif
}

void Application::compute_mapped_cost_and_upper_bound_baseline(State *now, ui n, ui *candidates, int *mapping) {
	ui cot = now->mapped_cost;
	ui start_level = now->level;

#ifndef NDEBUG
	for(ui i = 0;i < g_n;i ++) {
		assert(visX[i] == 0);
		assert(visY[i] == 0);
	}
#endif

	assert((now->parent == NULL&&cot == 0)||(now->parent != NULL&&cot == now->parent->mapped_cost));
	for(State *parent = now->parent;parent != NULL;parent = parent->parent) {
		if (parent->image >= g_n || parent->level >= q_n) continue;  // Skip dummy root
		visY[parent->image] = 1;
		queue[parent->image] = MO[parent->level];
	}
	for(ui i = 0;i < start_level;i ++) visX[MO[i]] = 1;

	//printf("mapped_cost: %d, start_level: %d, n: %d\n", cot, start_level, n);
	//if(start_level == 1) printf("%d %d\n", q_vlabels[MO[0]], g_vlabels[now->parent->image]);

	for(ui i = 0;i < n;i ++) {
		if(i+start_level >= q_n) {
			ui v = candidates[mapping[i]];
			++ cot;
			for(ui j = g_starts[v];j < g_starts[v+1];j ++) if(visY[g_edges[j]]) ++ cot;
			visY[v] = 1;
			//printf("here\n");
			continue;
		}
		ui u = MO[i+start_level], v = candidates[mapping[i]];
		if(q_vlabels[u] != g_vlabels[v]) ++ cot;
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) if(visX[q_edges[j]]) ++ cot;
		for(ui j = g_starts[v];j < g_starts[v+1];j ++) if(visY[g_edges[j]]) {
			++ cot;
			if(q_matrix[u*q_n+queue[g_edges[j]]] == g_elabels[j]) cot -= 2;
			else if(q_matrix[u*q_n+queue[g_edges[j]]] < elabels_n) -- cot;
		}
		if(i == 0) now->mapped_cost = cot;
		visX[u] = visY[v] = 1;
		queue[v] = u;
	}

#ifndef NDEBUG
	verify_induced_ged(now);
#endif

	memset(visX, 0, sizeof(char)*g_n);
	memset(visY, 0, sizeof(char)*g_n);

#ifndef NDEBUG
	//for(State *parent = now->parent;parent != NULL;parent = parent->parent) {
	//	queue[parent->level] = parent->image;
	//}
	//printf("Map (");
	//for(ui i = 0;i < start_level;i ++) printf("%d, ", queue[i]);
	//printf("%d;", now->image);
	//for(ui i = 1;i < n;i ++) printf(" %d", candidates[mapping[i]]);
	//printf("): cost %d, lower_bound %d\n", cot, now->lower_bound);
#endif

    
#ifdef _UPPER_BOUND_
	if(cot < upper_bound) {
		upper_bound = cot;
		for(State *st = now->parent;st != NULL;st = st->parent) {
			if (st->level >= q_n || st->image >= g_n) continue;  // Skip dummy root
			BX[MO[st->level]] = st->image;
		}
		for(ui i = 0;i+now->level < q_n;i ++) BX[MO[i+now->level]] = candidates[mapping[i]];

		//printf("supposed GED in upper_bound: %d (level: %d)\n", upper_bound, now->level);
		assert(upper_bound == compute_ged_of_BX());
	}
#endif
}

void Application::compute_mapped_cost_and_upper_bound_test(State *now, ui n, ui *candidates, int *mapping) {
	ui cot = now->mapped_cost;
	ui start_level = now->level;

#ifndef NDEBUG
	for(ui i = 0;i < g_n;i ++) {
		assert(visX[i] == 0);
		assert(visY[i] == 0);
	}
#endif

	assert((now->parent == NULL&&cot == 0)||(now->parent != NULL&&cot == now->parent->mapped_cost));
	for(State *parent = now->parent;parent != NULL;parent = parent->parent) {
		if (parent->image >= g_n || parent->level >= q_n) continue;  // Skip dummy root
		visY[parent->image] = 1;
		queue[parent->image] = MO[parent->level];
	}
	for(ui i = 0;i < start_level;i ++) visX[MO[i]] = 1;

	//printf("mapped_cost: %d, start_level: %d, n: %d\n", cot, start_level, n);
	//if(start_level == 1) printf("%d %d\n", q_vlabels[MO[0]], g_vlabels[now->parent->image]);

	for(ui i = 0;i < n;i ++) {
		if(i+start_level >= q_n) {
			ui v = candidates[mapping[i]];
			++ cot;
			for(ui j = g_starts[v];j < g_starts[v+1];j ++) if(visY[g_edges[j]]) ++ cot;
			visY[v] = 1;
			//printf("here\n");
			continue;
		}
		ui u = MO[i+start_level], v = candidates[mapping[i]];
		if(q_vlabels[u] != g_vlabels[v]) ++ cot;
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) if(visX[q_edges[j]]) ++ cot;
		for(ui j = g_starts[v];j < g_starts[v+1];j ++) if(visY[g_edges[j]]) {
			++ cot;
			if(q_matrix[u*q_n+queue[g_edges[j]]] == g_elabels[j]) cot -= 2;
			else if(q_matrix[u*q_n+queue[g_edges[j]]] < elabels_n) -- cot;
		}
		if(i == 0) now->mapped_cost = cot;
		visX[u] = visY[v] = 1;
		queue[v] = u;
	}

#ifndef NDEBUG
	verify_induced_ged(now);
#endif

	memset(visX, 0, sizeof(char)*g_n);
	memset(visY, 0, sizeof(char)*g_n);

#ifndef NDEBUG
	//for(State *parent = now->parent;parent != NULL;parent = parent->parent) {
	//	queue[parent->level] = parent->image;
	//}
	//printf("Map (");
	//for(ui i = 0;i < start_level;i ++) printf("%d, ", queue[i]);
	//printf("%d;", now->image);
	//for(ui i = 1;i < n;i ++) printf(" %d", candidates[mapping[i]]);
	//printf("): cost %d, lower_bound %d\n", cot, now->lower_bound);
#endif

    if (cot < overall_ub) {
        overall_ub = cot;
#ifndef NDEBUG
        printf("[UPDATE] New overall_ub = %u (from mapped_cost)\n", overall_ub);
#endif
    }
    
#ifdef _UPPER_BOUND_
	if(cot < upper_bound) {
		upper_bound = cot;
		for(State *st = now->parent;st != NULL;st = st->parent) {
			if (st->level >= q_n || st->image >= g_n) continue;  // Skip dummy root
			BX[MO[st->level]] = st->image;
		}
		for(ui i = 0;i+now->level < q_n;i ++) BX[MO[i+now->level]] = candidates[mapping[i]];

		//printf("supposed GED in upper_bound: %d (level: %d)\n", upper_bound, now->level);
		assert(upper_bound == compute_ged_of_BX());
	}
#endif
}
/* ------------------------------------------------------------------
 *  compute_mapped_cost_and_upper_bound  ——  compute now's mapped_cost,
 *  and update global upper_bound / BX based on mapping[].
 * ------------------------------------------------------------------*/
void Application::compute_mapped_cost_and_upper_bound(
        State *now, ui n, ui *candidates, int *mapping) {
    const ui DUMMY = UINT32_MAX;                // queue unwritten marker

    /* 0. Zero-fill temporary arrays (queue/visX/visY) */
    std::fill(queue, queue + g_n, DUMMY);
    std::memset(visX , 0, sizeof(char)*g_n);
    std::memset(visY , 0, sizeof(char)*g_n);

    ui cot         = now->mapped_cost;          // accumulated cost
    ui start_level = now->level;

    /* 1. Write ancestor mapping to queue / visY (skip dummy-root) */
    for (State *p = now->parent; p; p = p->parent) {
        if (p->image >= g_n) continue;          // dummy-root
        visY[p->image]  = 1;
        queue[p->image] = MO[p->level];
    }
    for (ui i = 0; i < start_level; ++i) visX[MO[i]] = 1;

    /* 2. Process current layer and subsequent n-1 rows of mapping */
    for (ui i = 0; i < n; ++i) {
        ui  v = candidates[mapping[i]];
        bool padding_row = (i + start_level >= q_n);

        if (padding_row) {                      // only compute row cost, no further expansion
            ++cot;
            for (ui j = g_starts[v]; j < g_starts[v+1]; ++j)
                if (visY[g_edges[j]]) ++cot;
            visY[v] = 1;
            continue;
        }

        ui u = MO[i + start_level];             // query vertex

        if (q_vlabels[u] != g_vlabels[v]) ++cot;

        for (ui j = q_starts[u]; j < q_starts[u+1]; ++j)
            if (visX[q_edges[j]]) ++cot;

        for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
            ui nb = g_edges[j];
            if (!visY[nb]) continue;            // only care about cross-edges to mapped G vertices

            ++cot;
            ui q_nb = queue[nb];
            if (q_nb != DUMMY && q_nb < q_n) {  // corresponding Q vertex exists in queue
                ushort qlbl = q_matrix[u*q_n + q_nb];
                ushort glbl = g_elabels[j];
                if (qlbl == glbl)           cot -= 2;
                else if (qlbl < elabels_n) --cot;
            }
        }

        if (i == 0) now->mapped_cost = cot;

        visX[u]  = 1;
        visY[v]  = 1;
        queue[v] = u;
    }

#ifndef NDEBUG
    verify_induced_ged(now);
#endif

    /* 3. Clean up queue / vis markers again after use (maintain all-zero to not break subsequent calls) */
    std::fill(queue, queue + g_n, DUMMY);
    std::memset(visX , 0, sizeof(char)*g_n);
    std::memset(visY , 0, sizeof(char)*g_n);

    /* 4. Update overall_ub (unconditional, record true optimum) */
    
    if (cot < overall_ub) {
        overall_ub = cot;
#ifndef NDEBUG
        printf("[UPDATE] New overall_ub = %u (from mapped_cost)\n", overall_ub);
#endif
    }

    /* 5. If better GED obtained (relative to threshold), write BX and tighten upper_bound */
#ifdef _UPPER_BOUND_
    if (cot < upper_bound) {
        upper_bound = cot;

        /* 5-A. Write ancestor part to BX, skip dummy-root */
        for (State *st = now->parent; st; st = st->parent) {
            if (st->image >= g_n || st->level >= q_n) continue;
            BX[ MO[ st->level ] ] = st->image;
        }

        /* 5-B. Write current branch mapping to BX */
        for (ui i = 0; i + now->level < q_n; ++i)
            BX[ MO[i + now->level] ] = candidates[mapping[i]];

#ifndef NDEBUG
        assert(upper_bound == compute_ged_of_BX());
#endif
    }
#endif
}


/*==============================================================*
 |            Application::compute_mapped_cost                  |
 *==============================================================*/
void Application::compute_mapped_cost(State *now)
{
#ifndef NDEBUG
    for (ui i = 0; i < g_n; ++i) {                   // all auxiliary bitmaps should be 0
        assert(visX[i] == 0);
        assert(visY[i] == 0);
    }
#endif

    /* ---------- 1. parent chain base cost ---------- */
    ui cot = (now->parent ? now->parent->mapped_cost : 0);

    /* ---------- 2. Mark ancestor mapping to visY / my ---------- */
    std::unordered_set<State*> visited;
    int depth = 0;
    for (State *parent = now->parent; parent != NULL; parent = parent->parent) {
        if (visited.find(parent) != visited.end()) {
            fprintf(stderr, "WARNING: cycle detected at depth %d!\n", depth);
            // Print cycle info
            fprintf(stderr, "Current node: level=%u, image=%u\n", now->level, now->image);
            State* p = parent;
            int i = 0;
            do {
                fprintf(stderr, "  [%d] level=%u, image=%u, parent=%p\n", 
                        i++, p->level, p->image, p->parent);
                p = p->parent;
            } while (p != parent && i < 10);
            break;
        }
        visited.insert(parent);
        depth++;
        
        if (parent->image >= g_n) continue;
        visY[parent->image] = 1;
        my[parent->image] = MO[parent->level];
    }

    /* ---------- 3. Mark mapped query vertices to visX ---------- */
    for (ui i = 0; i < now->level; ++i) visX[MO[i]] = 1;

    /* ---------- 4. Compute cross GED ---------- */
    ui u = MO[now->level];
    ui v = now->image;

    if (q_vlabels[u] != g_vlabels[v]) ++cot;

    /* 4.1 u ↔ mapped Q vertices */
    for (ui j = q_starts[u]; j < q_starts[u+1]; ++j)
        if (visX[q_edges[j]]) ++cot;

    /* 4.2 v ↔ mapped G vertices */
    for (ui j = g_starts[v]; j < g_starts[v+1]; ++j)
        if (visY[g_edges[j]]) {
            ++cot;
            if (q_matrix[u*q_n + my[g_edges[j]]] == g_elabels[j])
                cot -= 2;
            else if (q_matrix[u*q_n + my[g_edges[j]]] < elabels_n)
                --cot;
        }

    now->mapped_cost = cot;

    /* ---------- 5. Clean up markers ---------- */
    for (State *parent = now->parent; parent != NULL; parent = parent->parent)
        if (parent->image < g_n) visY[parent->image] = 0;   // *
    for (ui i = 0; i < now->level; ++i) visX[MO[i]] = 0;

#ifndef NDEBUG
    verify_induced_ged(now);
#endif
}

void Application::construct_sibling_baseline(State *pre_sibling, State *now)
{
    now->pre_sibling = pre_sibling;

#ifdef _EXPAND_ALL_
    /*========================  Expand all siblings  ========================*/
    ushort &tn = now->siblings_n;
    if (tn == 0) {
        now->image = g_n;                 // empty，mark dummy
        // now->lower_bound = INF;
    } else {
        now->image       = now->siblings[tn - 2];
        now->lower_bound = now->siblings[tn - 1];
        compute_mapped_cost(now);
        tn -= 2;
#ifndef NDEBUG
        if (lb_method == LS) verify_LS_lower_bound(now);
#endif
    }

#else   /*=====================  Normal sibling branch  ===================*/

    /* ---------- 0. Inherit basic info ---------- */
    if (pre_sibling != NULL) {
        now->parent = pre_sibling->parent;
        now->level  = pre_sibling->level;
    }
    now->mapped_cost = (now->parent ? now->parent->mapped_cost : 0);

    /* ---------- 1. Collect visited siblings ---------- */
    ui pre_siblings = 0;
    if (pre_sibling == NULL) {
        assert(visited_siblings_n != NULL);
        visited_siblings[now->level * g_n + (visited_siblings_n[now->level]++)]
            = now->image;
        for (ui i = 0; i < visited_siblings_n[now->level]; ++i)
            candidates[pre_siblings++] =
                visited_siblings[now->level * g_n + i];
    } else {
        for (State *st = pre_sibling; st != NULL; st = st->pre_sibling)
            candidates[pre_siblings++] = st->image;
    }

    /* ---------- 2. Mark occupied G-side vertices ---------- */
    for (ui i = 0; i < pre_siblings; ++i) visY[candidates[i]] = 1;
    for (State *st = now->parent; st != NULL; st = st->parent)
        if (st->image < g_n) visY[st->image] = 1;          // * filter dummy

    /* ---------- 3. Generate candidate list ---------- */
    ui candidates_n = pre_siblings;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i]) candidates[candidates_n++] = i;
        else visY[i] = 0;                                 // clear to 0
    }
    if (candidates_n <= pre_siblings) {                    // no new candidates
        now->image = g_n;                                  // dummy
        return;
    }

    /* ---------- 4. Select best sibling ---------- */
    if      (lb_method == LS)    compute_best_extension_LS (now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BM)    compute_best_extension_BM (0,  now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == SM)    compute_best_extension_SM (0,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == LSa)   compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMao)  compute_best_extension_BM_baseline (1,  now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == SMa)   compute_best_extension_SM (1,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMa)   compute_best_extension_BMa(1,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == Hybrid) {
        if (search_space < hyrbid_ratio * g_n * g_n)
            compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
        else {
            search_n = search_n_for_IS;
            compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0);
        }
    } else {
        printf("lb_method is wrong in construct_sibling!\n");
    }
#endif  /* _EXPAND_ALL_ */
}

void Application::construct_sibling_test(State *pre_sibling, State *now)
{
    now->pre_sibling = pre_sibling;

#ifdef _EXPAND_ALL_
    /*========================  Expand all siblings  ========================*/
    ushort &tn = now->siblings_n;
    if (tn == 0) {
        now->image = g_n;                 // empty，mark dummy
        // now->lower_bound = INF;
    } else {
        now->image       = now->siblings[tn - 2];
        now->lower_bound = now->siblings[tn - 1];
        compute_mapped_cost(now);
        tn -= 2;
#ifndef NDEBUG
        if (lb_method == LS) verify_LS_lower_bound(now);
#endif
    }

#else   /*=====================  Normal sibling branch  ===================*/

    /* ---------- 0. Inherit basic info ---------- */
    if (pre_sibling != NULL) {
        now->parent = pre_sibling->parent;
        now->level  = pre_sibling->level;
    }
    now->mapped_cost = (now->parent ? now->parent->mapped_cost : 0);

    /* ---------- 1. Collect visited siblings ---------- */
    ui pre_siblings = 0;
    if (pre_sibling == NULL) {
        assert(visited_siblings_n != NULL);
        visited_siblings[now->level * g_n + (visited_siblings_n[now->level]++)]
            = now->image;
        for (ui i = 0; i < visited_siblings_n[now->level]; ++i)
            candidates[pre_siblings++] =
                visited_siblings[now->level * g_n + i];
    } else {
        for (State *st = pre_sibling; st != NULL; st = st->pre_sibling)
            candidates[pre_siblings++] = st->image;
    }

    /* ---------- 2. Mark occupied G-side vertices ---------- */
    for (ui i = 0; i < pre_siblings; ++i) visY[candidates[i]] = 1;
    for (State *st = now->parent; st != NULL; st = st->parent)
        if (st->image < g_n) visY[st->image] = 1;          // * filter dummy

    /* ---------- 3. Generate candidate list ---------- */
    ui candidates_n = pre_siblings;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i]) candidates[candidates_n++] = i;
        else visY[i] = 0;                                 // clear to 0
    }
    if (candidates_n <= pre_siblings) {                    // no new candidates
        now->image = g_n;                                  // dummy
        return;
    }

    /* ---------- 4. Select best sibling ---------- */
    if      (lb_method == LS)    compute_best_extension_LS (now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BM)    compute_best_extension_BM (0,  now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == SM)    compute_best_extension_SM (0,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == LSa)   compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMao)  compute_best_extension_BM_test (1,  now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == SMa)   compute_best_extension_SM (1,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMa)   compute_best_extension_BMa(1,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == Hybrid) {
        if (search_space < hyrbid_ratio * g_n * g_n)
            compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
        else {
            search_n = search_n_for_IS;
            compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0);
        }
    } else {
        printf("lb_method is wrong in construct_sibling!\n");
    }
#endif  /* _EXPAND_ALL_ */
}
/*==============================================================*
 |            Application::construct_sibling                    |
 *==============================================================*/
void Application::construct_sibling(State *pre_sibling, State *now)
{
    now->pre_sibling = pre_sibling;

#ifdef _EXPAND_ALL_
    /*========================  Expand all siblings  ========================*/
    ushort &tn = now->siblings_n;
    
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    // If siblings_total_n exists, use it to determine read start position
    ushort total_n = now->siblings_total_n > 0 ? now->siblings_total_n : now->siblings_n;
    // Compute intersectionsection start position
    ushort intersection_start = total_n - tn;
    
    if (tn == 0) {
        now->image = g_n;                 // empty，mark dummy
        // now->lower_bound = INF;
    } else {
        // Take last element from end of intersection section (the optimal one)
        now->image       = now->siblings[intersection_start + tn - 2];
        now->lower_bound = now->siblings[intersection_start + tn - 1];
        compute_mapped_cost(now);
        tn -= 2;
        
        // Update siblings_total_n（total reduced by 2）
        now->siblings_total_n -= 2;
        
#ifndef NDEBUG
        if (lb_method == LS) verify_LS_lower_bound(now);
#endif
    }
#else
    // Original logic: take from array end
    if (tn == 0) {
        now->image = g_n;                 // empty，mark dummy
        // now->lower_bound = INF;
    } else {
        now->image       = now->siblings[tn - 2];
        now->lower_bound = now->siblings[tn - 1];
        compute_mapped_cost(now);
        tn -= 2;
#ifndef NDEBUG
        if (lb_method == LS) verify_LS_lower_bound(now);
#endif
    }
#endif

#else   /*=====================  Normal sibling branch  ===================*/

    /* ---------- 0. Inherit basic info ---------- */
    if (pre_sibling != NULL) {
        now->parent = pre_sibling->parent;
        now->level  = pre_sibling->level;
    }
    now->mapped_cost = (now->parent ? now->parent->mapped_cost : 0);

    /* ---------- 1. Collect visited siblings ---------- */
    ui pre_siblings = 0;
    if (pre_sibling == NULL) {
        assert(visited_siblings_n != NULL);
        visited_siblings[now->level * g_n + (visited_siblings_n[now->level]++)]
            = now->image;
        for (ui i = 0; i < visited_siblings_n[now->level]; ++i)
            candidates[pre_siblings++] =
                visited_siblings[now->level * g_n + i];
    } else {
        for (State *st = pre_sibling; st != NULL; st = st->pre_sibling)
            candidates[pre_siblings++] = st->image;
    }

    /* ---------- 2. Mark occupied G-side vertices ---------- */
    for (ui i = 0; i < pre_siblings; ++i) visY[candidates[i]] = 1;
    for (State *st = now->parent; st != NULL; st = st->parent)
        if (st->image < g_n) visY[st->image] = 1;          // * filter dummy

    /* ---------- 3. Generate candidate list ---------- */
    ui candidates_n = pre_siblings;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i]) candidates[candidates_n++] = i;
        else visY[i] = 0;                                 // clear to 0
    }
    if (candidates_n <= pre_siblings) {                    // no new candidates
        now->image = g_n;                                  // dummy
        return;
    }

    /* ---------- 4. Select best sibling ---------- */
    if      (lb_method == LS)    compute_best_extension_LS (now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BM)    compute_best_extension_BM (0,  now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == SM)    compute_best_extension_SM (0,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == LSa)   compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMao)  compute_best_extension_BM (1,  now, candidates_n, candidates, pre_siblings, 0);
    else if (lb_method == SMa)   compute_best_extension_SM (1,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == BMa)   compute_best_extension_BMa(1,  now, candidates_n, candidates, pre_siblings);
    else if (lb_method == Hybrid) {
        if (search_space < hyrbid_ratio * g_n * g_n)
            compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
        else {
            search_n = search_n_for_IS;
            compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0);
        }
    } else {
        printf("lb_method is wrong in construct_sibling!\n");
    }
#endif  /* _EXPAND_ALL_ */
}


void Application::extend_to_full_mapping(State *parent, State *now) {
    assert(parent != NULL);
    // printf("extend_to_full_mapping: now->level=%u, now->siblings=%p\n",
    //        now->level, now->siblings);
    
    if(lb_method == BMao) {
        // Set now's properties
        now->level = parent->level + 1;
        now->mapped_cost = parent->mapped_cost;
        now->parent = parent;
        now->pre_sibling = NULL;
        
        assert(now->level == search_n);
        
        // Mark vertices used in ancestor chain
        State* temp_parent = parent;
        while(temp_parent != NULL) {
            if(temp_parent->image < g_n) {
                visY[temp_parent->image] = 1;
            }
            temp_parent = temp_parent->parent;
        }
        
        // Collect unused candidate vertices
        ui candidates_n = 0;
        for(ui i = 0; i < g_n; i++) {
            if(!visY[i]) {
                candidates[candidates_n++] = i;
            } else {
                visY[i] = 0;  // Clean up markers
            }
        }
        
        ui pre_siblings = 0;
        
        // Compute best extension
        compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0, 1);
    }
    else {
        assert(parent != NULL&&parent->level+1 == q_n);
        ui cot = parent->mapped_cost;
        now = parent;
        
        if(q_n < g_n) {
            // Mark mapped nodes
            while(parent != NULL) {
                if (parent->image < g_n)          // * skip dummy-root
                    visY[parent->image] = 1;
                parent = parent->parent;
            }
            
            for(ui i = 0;i < g_n;i ++) if(!visY[i]) {
                // Compute edge cost of unmapped nodes
                for(ui j = g_starts[i];j < g_starts[i+1];j ++) {
                    if(visY[g_edges[j]]) {
                        cot++;
                    }
                }
                visY[i] = 1;
            }
            
            cot += (g_n-q_n);
            
            for(ui i = 0;i < g_n;i ++) visY[i] = 0;
        }
        
        if(cot < upper_bound) {
            while(now != NULL) {
                if (now->image < g_n)             // * only record real nodes
                    BX[MO[now->level]] = now->image;
                now = now->parent;
            }
            upper_bound = cot;
            
#ifndef NDEBUG
            ui verify_cost = compute_ged_of_BX();
            assert(upper_bound == verify_cost);
#endif
        }
    }
}

void Application::extend_to_full_mapping_baseline(State *parent, State *now) {
    assert(parent != NULL);
    // printf("extend_to_full_mapping: now->level=%u, now->siblings=%p\n",
    //        now->level, now->siblings);
    
    if(lb_method == BMao) {
        // Set now's properties
        now->level = parent->level + 1;
        now->mapped_cost = parent->mapped_cost;
        now->parent = parent;
        now->pre_sibling = NULL;
        
        assert(now->level == search_n);
        
        // Mark vertices used in ancestor chain
        State* temp_parent = parent;
        while(temp_parent != NULL) {
            if(temp_parent->image < g_n) {
                visY[temp_parent->image] = 1;
            }
            temp_parent = temp_parent->parent;
        }
        
        // Collect unused candidate vertices
        ui candidates_n = 0;
        for(ui i = 0; i < g_n; i++) {
            if(!visY[i]) {
                candidates[candidates_n++] = i;
            } else {
                visY[i] = 0;  // Clean up markers
            }
        }
        
        ui pre_siblings = 0;
        
        // Compute best extension
        compute_best_extension_BM_baseline(1, now, candidates_n, candidates, pre_siblings, 0, 1);
    }
    else {
        assert(parent != NULL&&parent->level+1 == q_n);
        ui cot = parent->mapped_cost;
        now = parent;
        
        if(q_n < g_n) {
            // Mark mapped nodes
            while(parent != NULL) {
                if (parent->image < g_n)          // * skip dummy-root
                    visY[parent->image] = 1;
                parent = parent->parent;
            }
            
            for(ui i = 0;i < g_n;i ++) if(!visY[i]) {
                // Compute edge cost of unmapped nodes
                for(ui j = g_starts[i];j < g_starts[i+1];j ++) {
                    if(visY[g_edges[j]]) {
                        cot++;
                    }
                }
                visY[i] = 1;
            }
            
            cot += (g_n-q_n);
            
            for(ui i = 0;i < g_n;i ++) visY[i] = 0;
        }
        
        if(cot < upper_bound) {
            while(now != NULL) {
                if (now->image < g_n)             // * only record real nodes
                    BX[MO[now->level]] = now->image;
                now = now->parent;
            }
            upper_bound = cot;
            
#ifndef NDEBUG
            ui verify_cost = compute_ged_of_BX();
            assert(upper_bound == verify_cost);
#endif
        }
    }
}

void Application::extend_to_full_mapping_test(State *parent, State *now) {
    assert(parent != NULL);
    // printf("extend_to_full_mapping: now->level=%u, now->siblings=%p\n",
    //        now->level, now->siblings);
    
    if(lb_method == BMao) {
        // Set now's properties
        now->level = parent->level + 1;
        now->mapped_cost = parent->mapped_cost;
        now->parent = parent;
        now->pre_sibling = NULL;
        
        assert(now->level == search_n);
        
        // Mark vertices used in ancestor chain
        State* temp_parent = parent;
        while(temp_parent != NULL) {
            if(temp_parent->image < g_n) {
                visY[temp_parent->image] = 1;
            }
            temp_parent = temp_parent->parent;
        }
        
        // Collect unused candidate vertices
        ui candidates_n = 0;
        for(ui i = 0; i < g_n; i++) {
            if(!visY[i]) {
                candidates[candidates_n++] = i;
            } else {
                visY[i] = 0;  // Clean up markers
            }
        }
        
        ui pre_siblings = 0;
        
        // Compute best extension
        compute_best_extension_BM_test(1, now, candidates_n, candidates, pre_siblings, 0, 1);
    }
    else {
        assert(parent != NULL&&parent->level+1 == q_n);
        ui cot = parent->mapped_cost;
        now = parent;
        
        if(q_n < g_n) {
            // Mark mapped nodes
            while(parent != NULL) {
                if (parent->image < g_n)          // * skip dummy-root
                    visY[parent->image] = 1;
                parent = parent->parent;
            }
            
            for(ui i = 0;i < g_n;i ++) if(!visY[i]) {
                // Compute edge cost of unmapped nodes
                for(ui j = g_starts[i];j < g_starts[i+1];j ++) {
                    if(visY[g_edges[j]]) {
                        cot++;
                    }
                }
                visY[i] = 1;
            }
            
            cot += (g_n-q_n);
            
            for(ui i = 0;i < g_n;i ++) visY[i] = 0;
        }
        
        if(cot < upper_bound) {
            while(now != NULL) {
                if (now->image < g_n)             // * only record real nodes
                    BX[MO[now->level]] = now->image;
                now = now->parent;
            }
            upper_bound = cot;
            
#ifndef NDEBUG
            ui verify_cost = compute_ged_of_BX();
            assert(upper_bound == verify_cost);
#endif
        }
    }
}

ui Application::Hungarian(char initialization, ui n, ui *cost) {
#ifndef NDEBUG
	//printf("Cost Matrix:\n");
	//for(ui i = 0;i < n;i ++) {
	//	for(ui j = 0;j < n;j ++) printf(" %d", cost[i*n+j]);
	//	printf("\n");
	//}
#endif
	if(initialization) {//Initialization
		memset(mx, -1, sizeof(int)*n);
		memset(my, -1, sizeof(int)*n);
		memset(ly, 0, sizeof(int)*n);
		for(ui i = 0;i < n;i ++) {
			lx[i] = INF;
			ui *t_array = cost+i*n;
			for(ui j = 0;j < n;j ++) if(t_array[j] < lx[i]) lx[i] = t_array[j];
			for(ui j = 0;j < n;j ++) if(my[j] == -1&&t_array[j] == lx[i]) {
				mx[i] = j;
				my[j] = i;
				break;
			}
		}
	}

	for(ui u = 0;u < n;u ++) if(mx[u] == -1) { //Augmentation
		memset(visX, 0, sizeof(char)*n);
		memset(visY, 0, sizeof(char)*n);

		int q_n = 1;
		queue[0] = u;
		visX[u] = 1;

		ui *t_array = cost + u*n;
		for(ui i = 0;i < n;i ++) {
			slack[i] = t_array[i] - lx[u] - ly[i];
			slackmy[i] = u;
		}

		int target = n, X;
		while(true) {
			for(ui i = 0;i < q_n&&target == n;i ++) {
				ui v = queue[i];
				t_array = cost + v*n;
				for(ui j = 0;j < n;j ++) if(!visY[j]&&t_array[j] == lx[v] + ly[j]) {
					if(my[j] == -1) {
						X = v;
						target = j;
						break;
					}
					visY[j] = 1;
					X = my[j];
					visX[X] = 1;
					prev[X] = v;
					queue[q_n ++] = X;

					ui *tt_array = cost + X*n;
					for(ui k = 0;k < n;k ++) if(!visY[k]&&tt_array[k] - lx[X] - ly[k] < slack[k]) {
						slack[k] = tt_array[k] - lx[X] - ly[k];
						slackmy[k] = X;
					}
				}
			}
			if(target != n) break;

			q_n = 0;
			int delta = INF;
			for(ui i = 0;i < n;i ++) if(!visY[i]&&slack[i] < delta) delta = slack[i];
			for(ui i = 0;i < n;i ++) {
				if(visX[i]) lx[i] += delta;
				if(visY[i]) ly[i] -= delta;
				else slack[i] -= delta;
			}

			for(ui i = 0;i < n;i ++) if(!visY[i]&&slack[i] == 0) {
				if(my[i] == -1) {
					X = slackmy[i];
					target = i;
					break;
				}

				visY[i] = 1;
				if(!visX[my[i]]) {
					X = my[i];
					visX[X] = 1;
					prev[X] = slackmy[i];
					queue[q_n ++] = X;

					ui *tt_array = cost + X*n;
					for(ui k = 0;k < n;k ++) if(!visY[k]&&tt_array[k] - lx[X] - ly[k] < slack[k]) {
						slack[k] = tt_array[k] - lx[X] - ly[k];
						slackmy[k] = X;
					}
				}
			}
		}

		while(true) {
			int ty = mx[X];
			mx[X] = target;
			my[target] = X;
			if(X == u) break;

			X = prev[X];
			target = ty;
		}
	}

	memset(visX, 0, sizeof(char)*n);
	memset(visY, 0, sizeof(char)*n);

	int res = 0;
	for(ui i = 0;i < n;i ++) res += cost[i*n + mx[i]];
	return res;
}



ui Application::compute_ged_of_BX() {
	if(upper_bound > verify_upper_bound) return upper_bound;

	ui *py = new ui[g_n];
	for(ui i = 0;i < g_n;i ++) py[i] = q_n;
	for(ui i = 0;i < q_n;i ++) py[BX[i]] = i;

	ui ged = g_n - q_n;
	for(ui i = 0;i < q_n;i ++) if(q_vlabels[i] != g_vlabels[BX[i]]) ++ ged;

	ged += q_starts[q_n]/2;
	for(ui i = 0;i < g_n;i ++) for(ui j = g_starts[i];j < g_starts[i+1];j ++) {
		ui v = g_edges[j];
		if(v < i) continue;

		++ ged;
		if(py[i] >= q_n||py[v] >= q_n) continue;
		if(q_matrix[py[i]*q_n+py[v]] == g_elabels[j]) ged -= 2;
		else if(q_matrix[py[i]*q_n+py[v]] < elabels_n) -- ged;
	}
	delete[] py; py = NULL;
#ifndef NDEBUG
	printf("Verified mapping (");
	for(ui i = 0;i < q_n;i ++) {
		if(i != 0) printf(",");
		printf("%d", BX[i]);
	}
	printf(") has GED %d\n", ged);
#endif
	return ged;
}

void Application::get_mapping(vector<pair<ui,ui> > &mapping) {
	mapping.clear();
	
	// BX[query_node_id] = database_node_id 
	// Build mapping directly by query node ID order
	for(ui query_node_id = 0; query_node_id < q_n; query_node_id++) {
		mapping.pb(mp(query_node_id, BX[query_node_id]));
	}
}

void Application::verify_induced_ged(State *now) {
    // if (now->image >= g_n || now->level >= q_n) return;   // skip dummy

	ui *px = new ui[q_n];
	ui *py = new ui[g_n];

	for(ui i = 0;i < q_n;i ++) px[i] = g_n;
	for(ui i = 0;i < g_n;i ++) py[i] = q_n;
	for(State *st = now;st != NULL;st = st->parent) {
        if (st->image >= g_n || st->level >= q_n)  continue;
		px[MO[st->level]] = st->image;
		py[st->image] = MO[st->level];
	}

	ui res = 0;
	for(ui i = 0;i <= now->level;i ++) {
		ui u = MO[i], v = px[u];
		if(q_vlabels[u] != g_vlabels[v]) ++ res;
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) if(px[q_edges[j]] < g_n&&q_edges[j] > u) ++ res;
		for(ui j = g_starts[v];j < g_starts[v+1];j ++) if(py[g_edges[j]] < q_n&&g_edges[j] > v) {
			++ res;
			if(q_matrix[u*q_n+py[g_edges[j]]] == g_elabels[j]) res -= 2;
			else if(q_matrix[u*q_n+py[g_edges[j]]] < elabels_n) -- res;
		}
	}

	if(res != now->mapped_cost) printf("(%d->%d) res: %d, now->mapped_cost: %d\n", MO[now->level], now->image, res, now->mapped_cost);

	delete[] px;
	delete[] py;
}

void Application::verify_LS_lower_bound(State *now) {
    // if (now->image >= g_n || now->level >= q_n) return;   // skip dummy

	int *v_labels_cnt = new int[vlabels_n];
	int *e_labels_cnt = new int[elabels_n];
	memset(v_labels_cnt, 0, sizeof(int)*vlabels_n);
	memset(e_labels_cnt, 0, sizeof(int)*elabels_n);

	char *vis = new char[g_n];
	memset(vis, 0, sizeof(char)*g_n);
	for(State *st = now;st != NULL;st = st->parent) {
        if (st->image >= g_n || st->level >= q_n)  continue;
		vis[st->image] = 1;
	}

	int uvl_cnt = 0, vvl_cnt = 0, uel_cnt = 0, vel_cnt = 0, vl_common = 0, el_common = 0;
	for(ui i = now->level+1;i < q_n;i ++) {
		-- v_labels_cnt[q_vlabels[MO[i]]];
		++ uvl_cnt;
	}
	for(ui i = 0;i < g_n;i ++) if(!vis[i]) {
		if(v_labels_cnt[g_vlabels[i]] < 0) ++ vl_common;
		++ v_labels_cnt[g_vlabels[i]];
		++ vvl_cnt;
	}

	for(ui i = 0;i < g_n;i ++) if(!vis[i]) {
		for(ui j = g_starts[i];j < g_starts[i+1];j ++) if(vis[g_edges[j]]||(!vis[g_edges[j]]&&g_edges[j] > i)) {
			++ vel_cnt;
			-- e_labels_cnt[g_elabels[j]];
		}
	}

	memset(vis, 0, sizeof(char)*g_n);
	for(ui i = 0;i <= now->level;i ++) vis[MO[i]] = 1;

	for(ui i = now->level+1;i < q_n;i ++) {
		ui u = MO[i];
		for(ui j = q_starts[u];j < q_starts[u+1];j ++) if(vis[q_edges[j]]||(!vis[q_edges[j]]&&q_edges[j] > u)) {
			++ uel_cnt;
			if(e_labels_cnt[q_elabels[j]] < 0) ++ el_common;
			++ e_labels_cnt[q_elabels[j]];
		}
	}

	int res = now->mapped_cost;
	if(vvl_cnt > uvl_cnt) uvl_cnt = vvl_cnt;
	if(vel_cnt > uel_cnt) uel_cnt = vel_cnt;
	res += uvl_cnt - vl_common + uel_cnt - el_common;

	if(res != now->lower_bound) {
		printf("WA (%d->%d) res: %d, now->lower_bound: %d, now->mapped_cost: %d\n", MO[now->level], now->image, res, now->lower_bound, now->mapped_cost);
	}
    // printf("res = %d, now->lower_bound = %d\n", res, now->lower_bound);
	assert(res == now->lower_bound);

	delete[] vis;
	delete[] v_labels_cnt;
	delete[] e_labels_cnt;
}



