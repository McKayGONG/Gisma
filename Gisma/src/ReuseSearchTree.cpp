/*********************************************************************
 *  ReuseLSa_DFS.cpp  ――  Incremental patch refresh of snapshot tree node LBs
 *  Two key modifications:
 *    - ensure_vlabel_cap(ui)   -> Application member
 *    - ensure_elabel_cap(ui)   -> Application member
 ********************************************************************/

#include "Application.h"
#include <vector>
#include <cstring>

/* ============================================================= *
 *  VCAP / ECAP / MCAP / ecnt moved to Application instance members          *
 *  (Application.h)，resolves data race in multi-threaded parallel reuse search      *
 * ============================================================= */



void Application::ensure_vlabel_cap(ui lbl)
{
    /* --- if not yet allocated, allocate directly min(8 aligned, lbl+1) --- */
    if (vlabels_map == nullptr) {
        VCAP = ((std::max<ui>(lbl,0) + 8) & ~7u);   // at least 8, aligned
        vlabels_map = new int[VCAP]();              // all zero
        return;
    }

    /* --- otherwise check capacity and expand if needed --- */
    if (lbl < VCAP) return;

    size_t newCap = (lbl + 8) & ~7u;
    int*   newArr = new int[newCap]();
    std::memcpy(newArr, vlabels_map, sizeof(int) * VCAP);
    delete[] vlabels_map;
    vlabels_map = newArr;
    VCAP = newCap;
}

void Application::ensure_elabel_cap(ui lbl)
{
    if (elabels_matrix == nullptr) {                // <- same logic
        ECAP = MCAP = ((std::max<ui>(lbl,0)+8) & ~7u);
        ecnt.assign(ECAP, 0);
        elabels_matrix = new short[q_n * ECAP]();
        return;
    }

    if (lbl < ECAP) return;

    size_t newCap = (lbl + 8) & ~7u;
    ecnt.resize(newCap, 0);

    size_t need = (size_t)q_n * newCap;
    short* newMat = new short[need]();
    for (ui r = 0; r < q_n; ++r)
        std::memcpy(newMat + r*newCap,
                    elabels_matrix + r*ECAP,
                    sizeof(short) * ECAP);
    delete[] elabels_matrix;
    elabels_matrix = newMat;

    ECAP = MCAP = newCap;
}


// Add detailed debugging to compute_independent_lower_bound_LSa_baseline
ui Application::compute_independent_lower_bound_LSa_baseline(State *now) {
    // Special case: complete mapping
    if (now->level >= q_n) {
        return now->mapped_cost;
    }
    
    // Special case: current node is dummy mapping
    bool now_is_dummy = (now->image >= g_n);
    
    // 1. MC (Mapped Cost)
    ui mc = compute_mapped_cost_baseline(now);
    
    // 2. Prepare data structures
    int *v_labels_cnt = new int[vlabels_n];
    int *e_labels_cnt = new int[elabels_n];
    int *cross_e_labels_cnt = new int[elabels_n];
    
    memset(v_labels_cnt, 0, sizeof(int)*vlabels_n);
    memset(e_labels_cnt, 0, sizeof(int)*elabels_n);
    memset(cross_e_labels_cnt, 0, sizeof(int)*elabels_n);
    
    // Mark mapped vertices
    char *visG = new char[g_n];
    char *visQ = new char[q_n];
    memset(visG, 0, sizeof(char)*g_n);
    memset(visQ, 0, sizeof(char)*q_n);
    
    // Build mapping relationships
    ui *px = new ui[q_n];
    ui *py = new ui[g_n];
    for(ui i = 0; i < q_n; i++) px[i] = g_n;
    for(ui i = 0; i < g_n; i++) py[i] = q_n;
    
    // Rebuild mapping (excluding dummy mappings)
    for(State *st = now; st != NULL; st = st->parent) {
        if (st->image >= g_n || st->level >= q_n) continue;  // skip dummy
        visG[st->image] = 1;
        visQ[MO[st->level]] = 1;
        px[MO[st->level]] = st->image;
        py[st->image] = MO[st->level];
    }
    
    /* ------------------------------------------------------------ *
     * 2. Compute Inner LB (internal lower bound of unmapped subgraph)
     * ------------------------------------------------------------ */
    
    // 2.1 vertex multiset difference
    int uvl_cnt = 0, vvl_cnt = 0, vl_common = 0;
    
    // unmapped query vertices (considering dummy case)
    ui unmapped_start = now_is_dummy ? now->level : (now->level + 1);
    
    for(ui i = unmapped_start; i < q_n; i++) {
        --v_labels_cnt[q_vlabels[MO[i]]];
        ++uvl_cnt;
    }
    
    // unmapped data vertices
    for(ui i = 0; i < g_n; i++) 
        if(!visG[i]) {
            if(v_labels_cnt[g_vlabels[i]] < 0) ++vl_common;
            ++v_labels_cnt[g_vlabels[i]];
            ++vvl_cnt;
        }
    
    // 2.2 internal edge multiset difference
    int inner_uel_cnt = 0, inner_vel_cnt = 0, inner_el_common = 0;
    
    // internal edges of data graph
    for(ui i = 0; i < g_n; i++) {
        if(!visG[i]) {
            for(ui j = g_starts[i]; j < g_starts[i+1]; j++) {
                ui neighbor = g_edges[j];
                if(!visG[neighbor] && i < neighbor) {
                    ++inner_vel_cnt;
                    --e_labels_cnt[g_elabels[j]];
                }
            }
        }
    }
    
    // internal edges of query graph
    for(ui i = unmapped_start; i < q_n; i++) {
        ui u = MO[i];
        for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
            ui neighbor = q_edges[j];
            if(!visQ[neighbor] && u < neighbor) {
                ++inner_uel_cnt;
                if(e_labels_cnt[q_elabels[j]] < 0) ++inner_el_common;
                ++e_labels_cnt[q_elabels[j]];
            }
        }
    }
    
    ui vertex_lb = (vvl_cnt > uvl_cnt) ? vvl_cnt : uvl_cnt;
    vertex_lb -= vl_common;
    
    ui inner_edge_lb = (inner_vel_cnt > inner_uel_cnt) ? inner_vel_cnt : inner_uel_cnt;
    inner_edge_lb -= inner_el_common;
    
    ui inner_lb = vertex_lb + inner_edge_lb;
    
    /* ------------------------------------------------------------ *
     * 3. Compute Cross LB (lower bound of cross-boundary edges)
     * ------------------------------------------------------------ */
    ui cross_lb = 0;
    
    // For each mapped vertex pair (excluding dummy)
    ui cross_end = now_is_dummy ? now->level : (now->level + 1);
    
    for(ui idx = 0; idx < cross_end; idx++) {
        ui u = MO[idx];
        ui v = px[u];
        if (v >= g_n) continue;  // skip dummy mapping
        
        // Reset counters
        memset(cross_e_labels_cnt, 0, sizeof(int)*elabels_n);
        
        // Count cross-boundary edges of u
        ui u_cross_cnt = 0;
        for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
            ui neighbor = q_edges[j];
            if(!visQ[neighbor]) {
                --cross_e_labels_cnt[q_elabels[j]];
                ++u_cross_cnt;
            }
        }
        
        // Count cross-boundary edges of v
        ui v_cross_cnt = 0;
        ui cross_common = 0;
        for(ui j = g_starts[v]; j < g_starts[v+1]; j++) {
            ui neighbor = g_edges[j];
            if(!visG[neighbor]) {
                if(cross_e_labels_cnt[g_elabels[j]] < 0) ++cross_common;
                ++cross_e_labels_cnt[g_elabels[j]];
                ++v_cross_cnt;
            }
        }
        
        ui max_cross = (u_cross_cnt > v_cross_cnt) ? u_cross_cnt : v_cross_cnt;
        cross_lb += max_cross - cross_common;
    }
    
    // 4. Final LSa LB
    ui lsa_lb = mc + inner_lb + cross_lb;
    
    // Clean up memory
    delete[] v_labels_cnt;
    delete[] e_labels_cnt;
    delete[] cross_e_labels_cnt;
    delete[] visG;
    delete[] visQ;
    delete[] px;
    delete[] py;
    
    return lsa_lb;
}

// Also modify compute_mapped_cost_baseline to add dummy check
ui Application::compute_mapped_cost_baseline(State *now) {
    ui *px = new ui[q_n];
    ui *py = new ui[g_n];

    for(ui i = 0; i < q_n; i++) px[i] = g_n;
    for(ui i = 0; i < g_n; i++) py[i] = q_n;
    
    for(State *st = now; st != NULL; st = st->parent) {
        if (st->image >= g_n || st->level >= q_n) continue;
        px[MO[st->level]] = st->image;
        py[st->image] = MO[st->level];
    }

    ui res = 0;
    for(ui i = 0; i <= now->level; i++) {
        ui u = MO[i], v = px[u];
        
        // Key fix: check if v is valid
        if (v >= g_n) continue;

        if(q_vlabels[u] != g_vlabels[v]) ++res;
        
        for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
            if(px[q_edges[j]] < g_n && q_edges[j] > u) {
                ++res;
            }
        }
        
        for(ui j = g_starts[v]; j < g_starts[v+1]; j++) {
            if(py[g_edges[j]] < q_n && g_edges[j] > v) {
                ++res;
                ui matched_q = py[g_edges[j]];
                if(q_matrix[u*q_n+matched_q] == g_elabels[j]) {
                    res -= 2;
                }
                else if(q_matrix[u*q_n+matched_q] < elabels_n) {
                    --res;
                }
            }
        }
    }

    delete[] px;
    delete[] py;

    return res;
}

ui Application::compute_full_mapping_cost_baseline(State* leaf_node)
{
    // Validate input: must be complete mapping (leaf node)
    if (leaf_node == nullptr || leaf_node->level + 1 != search_n) {
        std::cerr << "[ERROR] compute_full_mapping_cost_baseline: not a complete mapping!" << std::endl;
        return DUMMY_VAL;
    }
    
    // Create independent work arrays
    char* temp_visY = new char[g_n]();
    ui* temp_mapping = new ui[q_n];  // Store mapping: temp_mapping[query_vertex] = db_vertex
    
    // Initialize mapping array with invalid values
    std::fill(temp_mapping, temp_mapping + q_n, DUMMY_VAL);
    
    // 1. Backtrack from leaf node, build complete mapping
    State* current = leaf_node;
    ui nodes_traced = 0;
    
    while (current != nullptr) {
        if (current->level != DUMMY_VAL && current->image != DUMMY_VAL) {
            // Record mapping: MO[level] -> image
            temp_mapping[MO[current->level]] = current->image;
            if (current->image < g_n) {
                temp_visY[current->image] = 1;
            }
            nodes_traced++;
        }
        current = current->parent;
    }
    
    // Verify complete mapping was obtained
    if (nodes_traced != search_n) {
        std::cerr << "[ERROR] Incomplete mapping: traced " << nodes_traced 
                  << " nodes, expected " << search_n << std::endl;
        delete[] temp_visY;
        delete[] temp_mapping;
        return DUMMY_VAL;
    }
    
    // 2. Compute mapping cost
    ui total_cost = 0;
    
    // 2.1 Vertex substitution cost
    for (ui u = 0; u < q_n; ++u) {
        if (temp_mapping[u] < g_n) {
            ui v = temp_mapping[u];
            if (q_vlabels[u] != g_vlabels[v]) {
                total_cost++;  // vertex labels do not match
            }
        }
    }
    
    // 2.2 Edge cost computation
    // Check each edge in query graph
    for (ui u1 = 0; u1 < q_n; ++u1) {
        ui v1 = temp_mapping[u1];
        if (v1 >= g_n) continue;  // u1 unmapped (should not happen)
        
        for (ui j = q_starts[u1]; j < q_starts[u1+1]; ++j) {
            ui u2 = q_edges[j];
            if (u1 >= u2) continue;  // avoid duplicate computation of undirected edges
            
            ui v2 = temp_mapping[u2];
            if (v2 >= g_n) continue;  // u2 unmapped (should not happen)
            
            ui q_label = q_elabels[j];
            
            // Find corresponding edge in data graph
            bool found = false;
            ui g_label = DUMMY_VAL;
            
            // look up v1 -> v2
            for (ui k = g_starts[v1]; k < g_starts[v1+1]; ++k) {
                if (g_edges[k] == v2) {
                    found = true;
                    g_label = g_elabels[k];
                    break;
                }
            }
            
            if (!found) {
                // edge does not exist, needs deletion
                total_cost++;
            } else if (q_label != g_label) {
                // edge exists but label differs, needs substitution
                total_cost++;
            }
            // otherwise edge fully matches, no cost
        }
    }
    
    // 2.3 Process uncovered edges in data graph
    // Check edges between all mapped vertices in data graph
    for (ui v1 = 0; v1 < g_n; ++v1) {
        if (!temp_visY[v1]) continue;  // v1 is not mapped
        
        for (ui j = g_starts[v1]; j < g_starts[v1+1]; ++j) {
            ui v2 = g_edges[j];
            if (v1 >= v2) continue;  // avoid duplicate computation
            
            if (temp_visY[v2]) {
                // v1 and v2 are both mapped, check for corresponding query edge
                // Find query vertices mapped to v1 and v2
                ui u1 = DUMMY_VAL, u2 = DUMMY_VAL;
                for (ui u = 0; u < q_n; ++u) {
                    if (temp_mapping[u] == v1) u1 = u;
                    if (temp_mapping[u] == v2) u2 = u;
                }
                
                if (u1 != DUMMY_VAL && u2 != DUMMY_VAL) {
                    // Check if query graph has edge u1-u2
                    bool has_edge = false;
                    for (ui k = q_starts[u1]; k < q_starts[u1+1]; ++k) {
                        if (q_edges[k] == u2) {
                            has_edge = true;
                            break;
                        }
                    }
                    
                    if (!has_edge) {
                        // data graph has edge but query graph does not, needs insertion
                        total_cost++;
                    }
                    // if edge exists, already handled in 2.2
                }
            } else {
                // v2 is not mapped, this edge incurs cost
                total_cost++;
            }
        }
    }
    
    // 2.4 Process unmapped vertices
    if (q_n < g_n) {
        // Need to delete excess data graph vertices
        ui unmapped_vertices = 0;
        for (ui v = 0; v < g_n; ++v) {
            if (!temp_visY[v]) {
                unmapped_vertices++;
                
                // All edges of unmapped vertex need deletion (edges to mapped vertices already computed in 2.3)
                // Here compute edges between unmapped vertices
                for (ui j = g_starts[v]; j < g_starts[v+1]; ++j) {
                    ui v2 = g_edges[j];
                    if (!temp_visY[v2] && v < v2) {
                        total_cost++;  // edge deletion cost
                    }
                }
            }
        }
        
        // vertex deletion cost
        total_cost += unmapped_vertices;
    }
    
    // Cleanup
    delete[] temp_visY;
    delete[] temp_mapping;
    
    return total_cost;
}
ui Application::compute_full_mapping_cost(State *now) {
    // 1. First compute cost of mapped portion
    ui mapped_cost = compute_mapped_cost_baseline(now);
    
    // 2. If q_n >= g_n, no unmapped g vertices, return directly
    if(q_n >= g_n) {
        return mapped_cost;
    }
    
    // 3. Build mapping array, mark mapped g vertices
    ui *visY = new ui[g_n];
    for(ui i = 0; i < g_n; i++) visY[i] = 0;
    
    // Backtrack from current state, mark all mapped g vertices
    for(State *st = now; st != NULL; st = st->parent) {
        if (st->image < g_n) {  // skip dummy node
            visY[st->image] = 1;
        }
    }
    
    // 4. Compute cost of unmapped portion
    ui unmapped_cost = 0;
    
    // 4.1 For each unmapped g vertex
    for(ui i = 0; i < g_n; i++) {
        if(!visY[i]) {  // i is unmapped vertex
            // Compute edge deletion cost for this vertex
            for(ui j = g_starts[i]; j < g_starts[i+1]; j++) {
                if(visY[g_edges[j]]) {
                    // other end of edge is mapped vertex (or previously processed unmapped vertex)
                    unmapped_cost++;
                }
            }
            // Mark immediately after processing, so subsequent unmapped vertices will compute edges with it
            visY[i] = 1;
        }
    }
    
    // 4.2 Add deletion cost of unmapped vertices
    unmapped_cost += (g_n - q_n);
    
    delete[] visY;
    
    return mapped_cost + unmapped_cost;
}
void Application::refresh_bm_snapshot_lb_batch(State* root, ui snap_nodes, ui ged_gap) {
    /* ---------- 0. Synchronize static capacity counters ---------- */
    VCAP = vlabels_n;
    ECAP = elabels_n;
    MCAP = ECAP;
    if (ecnt.size() < ECAP) ecnt.resize(ECAP, 0);

    /* ---------- 1. Work array initialization ---------- */

    if (!visX)            visX            = new char[g_n];
    if (!visY)            visY            = new char[g_n];
    if (!queue)           queue           = new ui  [q_n];
    if (!prev)            prev            = new ui  [g_n];
    if (!candidates)      candidates      = new ui  [g_n];
    if (!cost)            cost            = new ui  [g_n * g_n];
    if (!mx)              mx              = new int [g_n];
    if (!my)              my              = new int [g_n];
    if (!lx)              lx              = new int [g_n];
    if (!ly)              ly              = new int [g_n];
    if (!vlabels_map)     vlabels_map     = new int [vlabels_n];
    if (!elabels_map)     elabels_map     = new int [elabels_n];
    if (!elabels_matrix)  elabels_matrix  = new short[q_n * elabels_n];
    if (!slack)           slack           = new int [g_n];
    if (!slackmy)         slackmy         = new int [g_n];
    if (!q_matrix)        q_matrix        = new uchar[q_n * q_n];

    std::memset(visX,            0, sizeof(char) * g_n);
    std::memset(visY,            0, sizeof(char) * g_n);
    std::memset(queue,           0, sizeof(ui)   * q_n);
    std::memset(prev,            0, sizeof(ui)   * g_n);
    std::memset(cost,            0, sizeof(ui)   * g_n * g_n);
    std::memset(vlabels_map,     0, sizeof(int)  * vlabels_n);
    std::memset(elabels_map,     0, sizeof(int)  * elabels_n);
    std::memset(elabels_matrix,  0, sizeof(short)* q_n * elabels_n);
    std::fill(ecnt.begin(), ecnt.end(), 0);
    std::fill(my, my + g_n, -1);
    std::fill(mx, mx + g_n, -1);
    std::fill(lx, lx + g_n, 0);
    std::fill(ly, ly + g_n, 0);
    std::fill(slack, slack + g_n, 0);
    std::fill(slackmy, slackmy + g_n, -1);

    // Initialize q_matrix
    for(ui i = 0; i < q_n; i++) {
        uchar *t_array = q_matrix + i * q_n;
        for(ui j = 0; j < q_n; j++) t_array[j] = elabels_n;
        for(ui j = q_starts[i]; j < q_starts[i+1]; j++) {
            if (q_edges[j] < q_n) {
                t_array[q_edges[j]] = q_elabels[j];
            }
        }
    }

    /* ---------- 2. Save original upper bound ---------- */
    ui saved_upper_bound = upper_bound;
    ui saved_verify_upper_bound = verify_upper_bound;
    upper_bound = UINT_MAX;
    verify_upper_bound = UINT_MAX;

    /* ---------- 3. BFS traverse tree processing nodes ---------- */
    std::queue<State*> bfs_queue;
    bfs_queue.push(root);

    while (!bfs_queue.empty()) {
        State* current = bfs_queue.front();
        bfs_queue.pop();

        // Process dummy nodes
        if (current->level == UINT16_MAX && current->image == UINT16_MAX) {
            current->lower_bound = 0;
            current->mapped_cost = 0;
            
            
            for (State* child : current->_children) {
                if (child) bfs_queue.push(child);
            }
            continue;
        }

        /* ---------- 3.1 Compute current node's MC ---------- */

        // Verify with gold standard
        int mapped_cost_baseline = compute_mapped_cost_baseline(current);
        current->mapped_cost = mapped_cost_baseline;
        


        // **** Optimization: decide whether to use LSa based on lb value
        ui lb_threshold = saved_upper_bound + ged_gap - 1;

        // Runtime control of LSa recomputation (via --disable_reuse_lsa parameter)
        int tmp_lb = 0;
        int old_lsa_lb = 0;

        if (!disable_reuse_lsa) {
            // Use LSa recomputation
            tmp_lb = compute_independent_lower_bound_LSa_baseline(current);
    #ifdef _USE_LSa_ESTIMATE_BMao_
            old_lsa_lb = current->lsa_lb;
            current->lower_bound = current->lower_bound + tmp_lb <= old_lsa_lb ? 0 : mmax(tmp_lb,current->lower_bound + tmp_lb - old_lsa_lb);
    #else
            current->lower_bound = current->lower_bound <= ged_gap ? 0 : mmax(tmp_lb,current->lower_bound - ged_gap);
    #endif
        } else {
            // Disable LSa, use simple ged_gap subtraction
            current->lower_bound = current->lower_bound <= ged_gap ? 0 : current->lower_bound - ged_gap;
        }

        /* ---------- 3.2 Update LB values in siblings ---------- */
#ifdef _EXPAND_ALL_
        if (current->siblings && current->siblings_n > 0) {
            // siblings array structure: even indices are vertex IDs, odd indices are LB values
            for (ui i = 1; i < current->siblings_n; i += 2) {
                ui old_sibling_lb = current->siblings[i];
                if (!disable_reuse_lsa) {
    #ifdef _USE_LSa_ESTIMATE_BMao_
                    current->siblings[i] = (old_sibling_lb + tmp_lb <= old_lsa_lb) ? 0 : (old_sibling_lb + tmp_lb - old_lsa_lb);
    #else
                    current->siblings[i] = (old_sibling_lb <= ged_gap) ? 0 : (old_sibling_lb - ged_gap);
    #endif
                } else {
                    current->siblings[i] = (old_sibling_lb <= ged_gap) ? 0 : (old_sibling_lb - ged_gap);
                }
            }
        }
#endif
        

        for (State* child : current->_children) {
            if (child) bfs_queue.push(child);
        }
    }

    /* ---------- 4. Restore original upper bound ---------- */
    upper_bound = saved_upper_bound;
    verify_upper_bound = saved_verify_upper_bound;

}
// ============================================================
// 3. Fixed refresh_lsa_snapshot_lb
// ============================================================
void Application::refresh_lsa_snapshot_lb(State* root, ui /*snap_nodes*/)
{
    /* ---------- 0. Synchronize static capacity counters ---------- */
    VCAP = vlabels_n;
    ECAP = elabels_n;
    MCAP = ECAP;
    if (ecnt.size() < ECAP) ecnt.resize(ECAP, 0);

    /* ---------- 1. Work array initialization ---------- */
    if (!visX)            visX            = new char[g_n];
    if (!visY)            visY            = new char[g_n];
    if (!queue)           queue           = new ui  [q_n];
    if (!prev)            prev            = new ui  [g_n];
    if (!vlabels_map)     vlabels_map     = new int [vlabels_n];
    if (!elabels_map)     elabels_map     = new int [elabels_n];
    if (!elabels_matrix)  elabels_matrix  = new short[q_n * elabels_n];
    if (!my)              my              = new int [g_n];

    std::memset(visX,            0, sizeof(char) * g_n);
    std::memset(visY,            0, sizeof(char) * g_n);
    std::memset(queue,           0, sizeof(ui)   * q_n);
    std::memset(prev,            0, sizeof(ui)   * g_n);
    std::memset(vlabels_map,     0, sizeof(int)  * vlabels_n);
    std::memset(elabels_map,     0, sizeof(int)  * elabels_n);
    std::memset(elabels_matrix,  0, sizeof(short)* q_n * elabels_n);
    std::fill(ecnt.begin(), ecnt.end(), 0);
    std::fill(my, my + g_n, g_n);

    /* ---------- 2. Save original upper bound ---------- */
    ui saved_upper_bound = upper_bound;
    ui saved_verify_upper_bound = verify_upper_bound;
    upper_bound = UINT_MAX;
    verify_upper_bound = UINT_MAX;

    /* ---------- 3. Traverse tree, compute lower bound and mapped_cost for each node ---------- */
    std::queue<State*> bfs_queue;
    bfs_queue.push(root);
    
    ui lb_baseline_calls = 0;
    ui mapped_cost_calculations = 0;
    
    while (!bfs_queue.empty()) {
        State* node = bfs_queue.front();
        bfs_queue.pop();
        
        // Process dummy nodes
        if (node->level == DUMMY_VAL && node->image == DUMMY_VAL) {
            node->lower_bound = 0;
            node->mapped_cost = 0;
            for (State* child : node->_children) {
                bfs_queue.push(child);
            }
            continue;
        }
        
        // ===== Key: use true gold standard to compute mapped_cost =====
        node->mapped_cost = compute_mapped_cost_baseline(node);
        mapped_cost_calculations++;
        
        // Use baseline function to compute lower bound (it will call compute_mapped_cost_basic again)
        // To avoid redundant computation, modify compute_independent_lower_bound_LSa_baseline
        // to accept a mapped_cost parameter
        node->lower_bound = compute_independent_lower_bound_LSa_baseline(node);
        lb_baseline_calls++;
        
        // Add all children to queue
        for (State* child : node->_children) {
            bfs_queue.push(child);
        }
    }
    
    /* ---------- 4. Restore original upper bound ---------- */
    upper_bound = saved_upper_bound;
    verify_upper_bound = saved_verify_upper_bound;
    
    // printf("[refresh] Called compute_independent_lower_bound_LSa_baseline %u times\n", lb_baseline_calls);
    // printf("[refresh] Computed mapped_cost from scratch %u times\n", mapped_cost_calculations);
}

// Renamed to a more general function name
void Application::refresh_snapshot_lb_batch(State* root, ui /*snap_nodes*/) {
    /* ---------- 0. Synchronize static capacity counters ---------- */
    VCAP = vlabels_n;
    ECAP = elabels_n;
    MCAP = ECAP;
    if (ecnt.size() < ECAP) ecnt.resize(ECAP, 0);

    /* ---------- 1. Work array initialization ---------- */
    // general array initialization
    if (!visX)            visX            = new char[g_n];
    if (!visY)            visY            = new char[g_n];
    if (!queue)           queue           = new ui  [q_n];
    if (!prev)            prev            = new ui  [g_n];
    if (!vlabels_map)     vlabels_map     = new int [vlabels_n];
    if (!elabels_map)     elabels_map     = new int [elabels_n];
    if (!my)              my              = new int [g_n];
    if (!candidates)      candidates      = new ui  [g_n];
    
    // LSa-specific arrays
    if (lb_method == LSa) {
        if (!elabels_matrix)  elabels_matrix  = new short[q_n * elabels_n];
        if (!children)        children        = new std::pair<int,int>[g_n];
    }
    
    // BMao-specific arrays
    if (lb_method == BMao) {
        if (!cost)     cost     = new ui[g_n * g_n];
        if (!mx)       mx       = new int[g_n];
        if (!lx)       lx       = new int[g_n];
        if (!ly)       ly       = new int[g_n];
        if (!slack)    slack    = new int[g_n];
        if (!slackmy)  slackmy  = new int[g_n];
        if (!q_matrix) q_matrix = new uchar[q_n * q_n];
        
        // Initialize q_matrix
        for(ui i = 0; i < q_n; i++) {
            uchar *t_array = q_matrix + i * q_n;
            for(ui j = 0; j < q_n; j++) t_array[j] = elabels_n;
            for(ui j = q_starts[i]; j < q_starts[i+1]; j++) {
                if (q_edges[j] < q_n) {
                    t_array[q_edges[j]] = q_elabels[j];
                }
            }
        }
    }

    // Zero-fill all arrays
    std::memset(visX,         0, sizeof(char) * g_n);
    std::memset(visY,         0, sizeof(char) * g_n);
    std::memset(queue,        0, sizeof(ui)   * q_n);
    std::memset(prev,         0, sizeof(ui)   * g_n);
    std::memset(vlabels_map,  0, sizeof(int)  * vlabels_n);
    std::memset(elabels_map,  0, sizeof(int)  * elabels_n);
    std::fill(my, my + g_n, g_n);
    
    if (lb_method == LSa && elabels_matrix) {
        std::memset(elabels_matrix, 0, sizeof(short) * q_n * elabels_n);
    }
    
    if (lb_method == BMao) {
        std::memset(cost, 0, sizeof(ui) * g_n * g_n);
        std::fill(mx, mx + g_n, -1);
        std::fill(lx, lx + g_n, 0);
        std::fill(ly, ly + g_n, 0);
        std::fill(slack, slack + g_n, 0);
        std::fill(slackmy, slackmy + g_n, -1);
    }
    
    std::fill(ecnt.begin(), ecnt.end(), 0);

    /* ---------- 2. Save original upper bound ---------- */
    ui saved_upper_bound = upper_bound;
    ui saved_verify_upper_bound = verify_upper_bound;
    
    // Set to max value to avoid pruning
    upper_bound = UINT_MAX;
    verify_upper_bound = UINT_MAX;

    /* ---------- 3. Traverse tree, recompute ---------- */
    std::queue<State*> bfs_queue;
    std::set<State*> processed;
    
    // Process dummy root
    if (root->level >= q_n) {
        for (State* child : root->_children) {
            if (child != nullptr) {
                bfs_queue.push(child);
            }
        }
    } else {
        bfs_queue.push(root);
    }
    
    while (!bfs_queue.empty()) {
        State* node = bfs_queue.front();
        bfs_queue.pop();
        
        if (node == nullptr) continue;
        if (node->level >= q_n) continue;
        
        if (processed.find(node) != processed.end()) {
            continue;
        }
        processed.insert(node);
        
        // Process dummy node
        if (node->image >= g_n) {
            if (node->parent != nullptr) {
                node->mapped_cost = node->parent->mapped_cost;
            }
            for (State* child : node->_children) {
                bfs_queue.push(child);
            }
            continue;
        }
        
        // Normal node processing
        ui target_image = node->image;
        
        // Mark graph vertices used by ancestors
        std::memset(visY, 0, sizeof(char) * g_n);
        for (State* p = node->parent; p != nullptr; p = p->parent) {
            if (p->image < g_n) {
                visY[p->image] = 1;
            }
        }
        
        // Collect all unused candidates
        ui candidates_n = 0;
        ui target_pos = UINT_MAX;
        
        for (ui i = 0; i < g_n; ++i) {
            if (!visY[i]) {
                if (i == target_image) {
                    target_pos = candidates_n;
                }
                candidates[candidates_n++] = i;
            }
        }
        
        // Clean up visY
        std::memset(visY, 0, sizeof(char) * g_n);
        
        // Ensure target_image is in candidate list
        if (target_pos == UINT_MAX) {
            continue;
        }
        
        // Call computation function
        switch (lb_method) {
            case LSa:
                // LSa processing...
                break;
                
            case BMao: {
                // Ensure target is at position 0
                if (target_pos != 0) {
                    std::swap(candidates[0], candidates[target_pos]);
                }
                compute_best_extension_BM(1, node, candidates_n, candidates, 0, 1, 0);
                
                // Force using correct mapping
                node->image = target_image;
                break;
            }
                
            default:
                break;
        }
        
        // Restore upper_bound
        upper_bound = UINT_MAX;
        verify_upper_bound = UINT_MAX;
        
        // Process all child nodes
        for (State* child : node->_children) {
            if (child != nullptr) {
                bfs_queue.push(child);
            }
        }
    }
    
    /* ---------- 4. Restore original upper bound ---------- */
    upper_bound = saved_upper_bound;
    verify_upper_bound = saved_verify_upper_bound;
}

void Application::recompute_node_lazy(State* now) {
    // If complete mapping or dummy node, no recomputation needed
    if (now->level >= search_n || now->image >= g_n) {
        now->need_recompute = false;
        return;
    }
    
    // 1. Collect pre_siblings (using _children info)
    ui pre_siblings = 0;
    
    // Find siblings before current node in parent's _children
    if (now->parent && !now->parent->_children.empty()) {
        for (auto child : now->parent->_children) {
            if (child == now) break;  // reached current node, stop
            if (child->level == now->level) {  // same-level nodes
                if (pre_siblings >= g_n) {
                    break;
                }
                if (child->image < g_n) {  // exclude dummy nodes
                    candidates[pre_siblings++] = child->image;
                }
            }
        }
    }
    
    // 2. Mark occupied G-side vertices
    for (ui i = 0; i < pre_siblings; ++i) {
        if (candidates[i] < g_n) {
            visY[candidates[i]] = 1;
        }
    }
    for (State *st = now->parent; st != NULL; st = st->parent) {
        if (st->image < g_n) visY[st->image] = 1;
    }
    
    // 3. Generate complete candidate list
    ui candidates_n = pre_siblings;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i]) {
            if (candidates_n < g_n) {
                candidates[candidates_n++] = i;
            }
        }
    }
    
    // Clean up visY
    for (ui i = 0; i < g_n; ++i) {
        visY[i] = 0;
    }
    
    if (candidates_n <= pre_siblings) {
        // no new candidates, mark as dummy
        now->image = g_n;
        // Use margin to set lower bound, ensure pruning
        now->lower_bound = upper_bound + margin + 1;
        now->need_recompute = false;
        return;
    }
    
    // 4. Save original mapped_cost (from parent node)
    now->mapped_cost = (now->parent ? now->parent->mapped_cost : 0);
    
#if USE_SIBLING_INTERSECTION
    // ===== Intersection strategy =====
    
    // 4.1 Save original siblings info
    std::vector<std::pair<ui, ui>> old_siblings_vec;  // pair<image, lb>
    ui old_image = now->image;  // Save original image
    
#ifdef _EXPAND_ALL_
    // Use map for dedup, keep minimum lb for each image
    std::unordered_map<ui, ui> old_siblings_map;  // image -> min_lb
    
    // First add current node's image (if valid)
    if (old_image < g_n) {
        old_siblings_map[old_image] = now->lower_bound;
    }
    
    if (now->siblings && now->siblings_n > 0) {
        // Save original siblings, dedup (keep minimum lb)
        ui duplicate_count = 0;
        for (ui i = 0; i < now->siblings_n; i += 2) {
            if (i + 1 >= now->siblings_n) break;
            
            ui img = now->siblings[i];
            ui lb = now->siblings[i+1];
            
            // Dedup: if exists, keep smaller lb
            auto it = old_siblings_map.find(img);
            if (it != old_siblings_map.end()) {
                duplicate_count++;
                if (lb < it->second) {
                    old_siblings_map[img] = lb;
                }
            } else {
                old_siblings_map[img] = lb;
            }
        }
    }
    
    // Convert map to vector
    old_siblings_vec.reserve(old_siblings_map.size());
    for (const auto& pair : old_siblings_map) {
        old_siblings_vec.push_back({pair.first, pair.second});
    }
#else
    // Non-_EXPAND_ALL_ mode, only save current image
    if (old_image < g_n) {
        old_siblings_vec.push_back({old_image, now->lower_bound});
    }
#endif
    
    // 4.2 Temporarily clear siblings for new computation
#ifdef _EXPAND_ALL_
    // If no siblings space, allocate new
    if (now->siblings == nullptr) {
        now->siblings = get_a_new_siblings_node();
    }
    now->siblings_n = 0;
#endif
    
    // 4.3 Call compute_best_extension to compute new siblings and image
    if (lb_method == LSa) {
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    } else if (lb_method == BMao) {
        compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0, 0);
    } else if (lb_method == BMa) {
        compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
    } else {
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    }
    
#ifdef _EXPAND_ALL_
    // 4.4 Save count of fully computed siblings
    ui full_siblings_n = now->siblings_n;
    
    // 4.5 Build new siblings mapping (for fast lookup)
    std::unordered_map<ui, ui> new_siblings_map;  // image -> lb
    
    // Add currently selected image
    new_siblings_map[now->image] = now->lower_bound;
    
    // Add all images from siblings array
    for (ui i = 0; i < now->siblings_n; i += 2) {
        if (i + 1 >= now->siblings_n) break;
        new_siblings_map[now->siblings[i]] = now->siblings[i+1];
    }
    
    // 4.6 Compute intersection
    std::vector<std::pair<ui, ui>> intersection;
    
    // Check which original siblings also exist in new computation
    for (const auto& old_pair : old_siblings_vec) {
        auto it = new_siblings_map.find(old_pair.first);
        if (it != new_siblings_map.end()) {
            // In intersection, use new LB values
            intersection.push_back({old_pair.first, it->second});
        }
    }
    
    // 4.7 Process based on intersection result
    if (intersection.empty()) {
        // Intersection empty, eliminate this node
        now->image = g_n;
        // Use margin to set lower bound, ensure pruning
        now->lower_bound = upper_bound + margin + 1;
        now->siblings_n = 0;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        now->siblings_total_n = 0;
#endif
        // Mark as no longer needing recomputation
        now->need_recompute = false;
        return;
    }
    
    // 4.8 Find smallest LB in intersection as image
    ui best_image = intersection[0].first;
    ui best_lb = intersection[0].second;
    
    for (size_t i = 1; i < intersection.size(); ++i) {
        if (intersection[i].second < best_lb) {
            best_lb = intersection[i].second;
            best_image = intersection[i].first;
        }
    }
    
    // 4.9 Reorganize siblings array
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    // New strategy: non-intersection part first, intersection part second
    
    // First, collect all non-intersection siblings
    std::vector<std::pair<ui, ui>> non_intersection;
    
    // Find siblings in new computation that are not in intersection
    if (now->image != best_image && new_siblings_map.find(now->image) != new_siblings_map.end()) {
        // If current image is not best and in new computation, check if in intersection
        bool in_intersection = false;
        for (const auto& pair : intersection) {
            if (pair.first == now->image) {
                in_intersection = true;
                break;
            }
        }
        if (!in_intersection) {
            non_intersection.push_back({now->image, now->lower_bound});
        }
    }
    
    // Find siblings not in intersection from siblings array
    for (ui i = 0; i < full_siblings_n; i += 2) {
        if (i + 1 >= full_siblings_n) break;
        
        ui img = now->siblings[i];
        ui lb = now->siblings[i+1];
        bool in_intersection = false;
        for (const auto& pair : intersection) {
            if (pair.first == img) {
                in_intersection = true;
                break;
            }
        }
        if (!in_intersection) {
            non_intersection.push_back({img, lb});
        }
    }
    
    // Compute final space needed
    ui expected_total_elements = (non_intersection.size() + intersection.size() - 1) * 2;  // -1 for best_image
    if (expected_total_elements > g_n * 2) {
        // Truncate to avoid overflow
        while (expected_total_elements > g_n * 2 && !non_intersection.empty()) {
            non_intersection.pop_back();
            expected_total_elements = (non_intersection.size() + intersection.size() - 1) * 2;
        }
    }
    
    // Refill siblings array
    now->siblings_n = 0;
    
    // Put non-intersection part first
    for (const auto& pair : non_intersection) {
        if (now->siblings_n + 2 > g_n * 2) break;
        now->siblings[now->siblings_n++] = pair.first;
        now->siblings[now->siblings_n++] = pair.second;
    }
    
    // Record intersection start position
    ui intersection_start = now->siblings_n;
    
    // Then put intersection part (except best)
    for (const auto& pair : intersection) {
        if (pair.first != best_image) {
            if (now->siblings_n + 2 > g_n * 2) break;
            now->siblings[now->siblings_n++] = pair.first;
            now->siblings[now->siblings_n++] = pair.second;
        }
    }
    
    // Set siblings_total_n to full size
    now->siblings_total_n = now->siblings_n;
    
    // Set siblings_n to intersection-only size
    // Safety check to prevent underflow
    if (now->siblings_total_n >= intersection_start) {
        now->siblings_n = now->siblings_total_n - intersection_start;
    } else {
        now->siblings_n = 0;
    }
    
#else  // not using INTERSECTION_WITH_FULL_SNAPSHOT
    // Original strategy: keep only intersection
    now->siblings_n = 0;
    
    // First add non-best intersection elements
    for (const auto& pair : intersection) {
        if (pair.first != best_image) {
            if (now->siblings_n + 2 > g_n * 2) break;
            now->siblings[now->siblings_n++] = pair.first;
            now->siblings[now->siblings_n++] = pair.second;
        }
    }
#endif
    
    // Update now's image and lower_bound
    now->image = best_image;
    now->lower_bound = best_lb;
    
    // Need to recompute mapped_cost
    compute_mapped_cost(now);
    
#else  // non-_EXPAND_ALL_ mode
    // In non-_EXPAND_ALL_ mode, check if original image is still valid in new computation results
    if (old_image < g_n && now->image == old_image) {
        // Original image is still optimal, keep (using new LB)
    } else {
        // Original image no longer optimal, check if in candidates
        bool found = false;
        for (ui i = pre_siblings; i < candidates_n; ++i) {
            if (candidates[i] == old_image) {
                found = true;
                break;
            }
        }
        
        if (!found || old_image >= g_n) {
            // Original image not in valid candidates, use new optimal image
            // (this is equivalent to empty intersection case)
            // now->image has already been set to new optimum by compute_best_extension
        }
    }
#endif

#else  // USE_SIBLING_INTERSECTION == 0
    // ===== Original strategy: completely recompute =====
    
    // 5. Process siblings (safer approach)
#ifdef _EXPAND_ALL_
    // If no siblings, allocate new; if exists, reuse directly
    if (now->siblings == nullptr) {
        now->siblings = get_a_new_siblings_node();
    }
    now->siblings_n = 0;  // Clear count, let function refill
#endif
    
    // 6. Call compute_best_extension to recompute (no_siblings=0, generate siblings)
    if (lb_method == LSa) {
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    } else if (lb_method == BMao) {
        compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0, 0);  // anchor_aware=1, no_siblings=0, IS=0
    } else if (lb_method == BMa) {
        compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
    } else {
        // For other lb_methods, use LSa as default
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    }
    
#endif  // USE_SIBLING_INTERSECTION
    
    // 7. Mark as no longer needing recomputation
    now->need_recompute = false;
    
    // Now this node is like a freshly computed node with optimal image and siblings
}


void Application::extract_snapshot(SearchSnapshot& out) {
    /* ---------- 0. Basic info ---------- */
    out.v.clear();
    out.ub = upper_bound;
    out.overall_lb = overall_lb;
    upper_bound = verify_upper_bound + 1;
    out.mo.assign(MO, MO + q_n);
    
#ifdef _EXPAND_ALL_
    out.siblings_data.clear();
    uint32_t siblings_offset_counter = 0;  // track current offset
#endif

    /* ---------- 1. Collect nodes needing preservation ---------- */
    std::unordered_set<State*> need_set;    // for deduplication
    std::vector<State*> need_nodes;         // save all needed nodes
    std::queue<State*> pending;             // BFS queue, for tracing ancestors
    std::unordered_set<State*> active_set;  // mark active nodes (selected nodes)

    /* ---------- 1-A. Collect all active nodes and their parents ---------- */
    auto mark_node_and_parent = [&](State* st) {
        if (!st) return;
        
        // filter out nodes with too-high lower bounds
        if (st->lower_bound >= upper_bound + margin) {
            return;
        }
        
        // mark as active node
        active_set.insert(st);
        
        // add node itself to preservation set
        if (need_set.insert(st).second) {
            need_nodes.push_back(st);
            pending.push(st);  // add to BFS queue for ancestor tracing
        }
        
        // also process parent nodes
        if (st->parent && st->parent->lower_bound < upper_bound + margin) {
            if (need_set.insert(st->parent).second) {
                need_nodes.push_back(st->parent);
                pending.push(st->parent);
            }
        }
    };

    // Collect nodes and their parents from containers
    for (State* st : open_heap) {
        mark_node_and_parent(st);
    }
    
    for (State* st : full_mapping_nodes) {
        mark_node_and_parent(st);
    }
    
    for (State* st : boundary_nodes) {
        mark_node_and_parent(st);
    }

    /* ---------- 1-B. BFS trace all ancestors ---------- */
    while (!pending.empty()) {
        State* cur = pending.front();
        pending.pop();
        
        if (cur->parent && need_set.insert(cur->parent).second) {
            need_nodes.push_back(cur->parent);
            pending.push(cur->parent);
        }
    }

    /* ---------- 1-C. Create and add dummy root node ---------- */
    State dummy_state;
    State* dummy = &dummy_state;
    dummy->parent = nullptr;
    dummy->level = DUMMY_VAL;
    dummy->image = DUMMY_VAL;
    dummy->lower_bound = 0;
    dummy->mapped_cost = 0;
    dummy->_children.clear();
#ifdef _EXPAND_ALL_
    dummy->siblings = nullptr;
    dummy->siblings_n = 0;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    dummy->siblings_total_n = 0;
#endif
#ifdef _USE_LSa_ESTIMATE_BMao_
    dummy->lsa_lb = 0;
#endif
#endif
    
    need_set.insert(dummy);
    need_nodes.push_back(dummy);

    /* ---------- 2. Rebuild parent-child relationships ---------- */
    // Clear all nodes' children lists
    for (State* st : need_nodes) {
        st->_children.clear();
    }
    
    // Rebuild children lists
    for (State* st : need_nodes) {
        if (st == dummy) continue;
        
        if (st->parent && need_set.count(st->parent)) {
            st->parent->_children.push_back(st);
        } else if (!st->parent) {
            // orphan nodes connected to dummy
            st->parent = dummy;
            dummy->_children.push_back(st);
        }
    }

    /* ---------- 3. Pre-order traversal output to snapshot ---------- */
    std::vector<State*> stack;
    stack.push_back(dummy);
    
    while (!stack.empty()) {
        State* cur = stack.back();
        stack.pop_back();
        
        // Only output nodes in need_set
        if (need_set.count(cur)) {
            // Simplified version: only mark whether it is an active node
            uint8_t is_open = active_set.count(cur) ? 1 : 0;
            
            SearchNodeLite node_lite{};  // first initialize all to 0

            // then assign one by one
            node_lite.level = cur->level;
            node_lite.image = cur->image;
            node_lite.lb = static_cast<uint16_t>(cur->lower_bound);
            node_lite.mc = static_cast<uint16_t>(cur->mapped_cost);
            node_lite.is_open = is_open;
            node_lite.vl_lb = static_cast<uint16_t>(cur->vl_lb);
            node_lite.vl_common = static_cast<uint16_t>(cur->vl_common);
            node_lite.mc_cross = cur->mc_cross;
            node_lite.vlabel_same = cur->vlabel_same;

#ifdef _USE_LSa_ESTIMATE_BMao_
            node_lite.lsa_lb = cur->lsa_lb;
#endif

#ifdef _EXPAND_ALL_
            // Determine if it is a dummy node via level or image
            if (cur->level == DUMMY_VAL || cur->image == DUMMY_VAL) {
                // This is a dummy node, do not process siblings
                node_lite.siblings_n = 0;
                node_lite.siblings_offset = 0;
            } 
            // Only leaf nodes may have siblings
            else if (cur->_children.empty() && cur->siblings != nullptr) {
                // Use siblings_total_n to save complete siblings info
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
                ui actual_siblings_n = cur->siblings_total_n > 0 ? cur->siblings_total_n : cur->siblings_n;
#else
                ui actual_siblings_n = cur->siblings_n;
#endif
                
                if (actual_siblings_n > 0) {
                    node_lite.siblings_n = actual_siblings_n;
                    node_lite.siblings_offset = siblings_offset_counter;
                    
                    for (ui i = 0; i < actual_siblings_n; i++) {
                        out.siblings_data.push_back(cur->siblings[i]);
                    }
                    siblings_offset_counter += actual_siblings_n;
                } else {
                    node_lite.siblings_n = 0;
                    node_lite.siblings_offset = 0;
                }
            } else {
                node_lite.siblings_n = 0;
                node_lite.siblings_offset = 0;
            }
#endif
            
            out.v.push_back(node_lite);
        }
       
        // Push in reverse order to ensure pre-order traversal
        for (auto it = cur->_children.rbegin(); it != cur->_children.rend(); ++it) {
            stack.push_back(*it);
        }
    }
}

/*------------------------------------------------------------
 * compute_mapping_order_reuse()
 *  - If MO == nullptr -> run original full flow (old function body)
 *  - If MO already has content (from snapshot) -> only recompute search_n / search_n_for_IS
 *-----------------------------------------------------------*/
void Application::compute_mapping_order_reuse()
{
    /* ---------- Case 1: first entry, MO not yet generated ---------- */
    if (MO == nullptr) {
        // printf("extract[compute_mapping_order_reuse] MO is nullptr\n");
        compute_mapping_order();        // directly call old function (original logic)
        return;
    }
    // printf("extract[compute_mapping_order_reuse] MO is not nullptr\n");
    /* ---------- Case 2: MO already provided by snapshot ---------- */
    /* only use it to compute independent-set size (search_n) --------- */

    /* 1. Build pos[]: position of query vertex in MO */
    ui* pos = new ui[q_n];
    for (ui i = 0; i < q_n; ++i) pos[MO[i]] = i;

    /* 2. Count non-overlapping edges per vertex, find minimum i that “covers all edges” */
    ui edges_cnt = 0;
    for (ui i = 0; i < q_n; ++i) {
        ui u = MO[i];
        for (ui j = q_starts[u]; j < q_starts[u+1]; ++j)
            if (pos[q_edges[j]] > i)       // count once only
                edges_cnt += 2;
        if (edges_cnt == q_starts[q_n]) {  // all edges accounted for
            search_n = i + 1;              // prefix size
            break;
        }
    }
    search_n_for_IS = search_n;            // BMa/BMao also used by
    if (lb_method != BMao) search_n = q_n; // overwrite per original rule

#ifndef NDEBUG
    printf("[reuse-MO] search_n = %u, IS = %u\n",
           search_n, q_n - search_n);
#endif

    delete[] pos;
}

// #define REUSE_DEBUG_OVERALL_LB
ui Application::app_reuse(const SearchSnapshot& snap,
                          ui tau,
                          ui ged_gap) {

    // Check MO array
    if (!snap.mo.empty() && snap.mo.size() < q_n) {
        return tau + 1;
    }

    // Initialize overall_lb and terminal_frontier_min_lb
    // Even if snapshot is empty, can still use overall_lb info
    overall_lb = (snap.overall_lb > ged_gap) ? (snap.overall_lb - ged_gap) : 0;
    ui terminal_frontier_min_lb = INF;

    // If snapshot is empty, downgrade to normal AStar but inherit overall_lb and MO
    if (snap.v.empty()) {
        // Do not return failure directly, continue with normal A* search
        // This allows using overall_lb as initial lower bound
        // Note: no search tree to reuse at this point, but at least LB and MO info available
    }

    /*------------------------------------------------------------------*/
    /* 0. Initialization: inherit upper bound & mapping order                                    */
    /*------------------------------------------------------------------*/
    upper_bound = tau + 1;
    State* best_solution = nullptr;
    
    // Set search_n
    search_n = q_n;
    if (g_n < q_n) search_n = g_n;
    
    if (!snap.mo.empty()) {
        if (!MO) {
            MO = new ui[q_n];
        }
        std::memcpy(MO, snap.mo.data(), sizeof(ui) * q_n);
    }
    
    compute_mapping_order_reuse();
    
    // Clear global containers
    open_heap.clear();
    full_mapping_nodes.clear();
    boundary_nodes.clear();
    
    State full;
    
    // Helper function: protect entire ancestor chain
    auto protect_ancestor_chain = [](State* node) {
        State* current = node;
        while (current != nullptr) {
            current->kept = true;
            current = current->parent;
        }
    };
    
    ui heap_n = 0;

    /*------------------------------------------------------------------*/
    /* 1. Rebuild tree structure from snapshot                                            */
    /*------------------------------------------------------------------*/
    State* dummy = nullptr;
    std::vector<State*> all_nodes;
    std::vector<State*> path_stack;
    
    // Rebuild nodes in pre-order traversal order
    for (size_t i = 0; i < snap.v.size(); ++i) {
        const auto& rec = snap.v[i];
        
        State* st = get_a_new_state_node();
        if (!st) {
            return tau + 1;
        }
        
        *st = {};
        st->_children.clear();
        
        st->level = rec.level;
        st->image = rec.image;
        st->lower_bound = rec.lb;
        st->mapped_cost = rec.mc;
        st->vl_lb = rec.vl_lb;
        st->vl_common = rec.vl_common;
        st->mc_cross = rec.mc_cross;
        st->vlabel_same = rec.vlabel_same;
        st->cs_cnt = 0;
        st->reused_leaf = false;
        st->kept = false;
        st->need_recompute = false;
#ifdef _USE_LSa_ESTIMATE_BMao_
        st->lsa_lb = rec.lsa_lb;
#endif
#ifdef _EXPAND_ALL_
        st->siblings_n = rec.siblings_n;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        st->siblings_total_n = rec.siblings_n;
#endif
        if (rec.siblings_n > 0) {
            st->siblings = get_a_new_siblings_node();
            
            if (st->siblings && rec.siblings_offset < snap.siblings_data.size()) {
                size_t copy_size = std::min(
                    static_cast<size_t>(rec.siblings_n),
                    snap.siblings_data.size() - rec.siblings_offset
                );
                std::memcpy(st->siblings,
                            &snap.siblings_data[rec.siblings_offset],
                            copy_size * sizeof(ushort));
            }
        } else {
            st->siblings = nullptr;
        }
#endif
        
        // Process dummy node
        if (st->level == DUMMY_VAL && st->image == DUMMY_VAL) {
            dummy = st;
            st->parent = nullptr;
            st->cs_cnt = 1;
            path_stack.clear();
            path_stack.push_back(st);
        } else {
            while (path_stack.size() > st->level + 1) {
                path_stack.pop_back();
            }
            
            if (!path_stack.empty()) {
                st->parent = path_stack.back();
                st->parent->_children.push_back(st);
                ++st->parent->cs_cnt;
            }
            
            path_stack.push_back(st);
        }
        
        all_nodes.push_back(st);
    }
    
    if (!dummy) {
        return upper_bound;
    }
    
    /*------------------------------------------------------------------*/
    /* 2. Refresh lower bounds of all nodes                                            */
    /*------------------------------------------------------------------*/
    refresh_bm_snapshot_lb_batch(dummy, all_nodes.size(), ged_gap);

    /*------------------------------------------------------------------*/
    /* 3. Process leaf nodes                                                  */
    /*------------------------------------------------------------------*/
    dummy->cs_cnt = dummy->_children.size();
    
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        State* node = all_nodes[i];
        if (node == dummy) continue;
        
        if (node->_children.empty()) {
            node->need_recompute = true;
            
            if (node->lower_bound >= upper_bound + margin) {
                // Pruning: lb >= upper_bound + margin
                if(node->level != DUMMY_VAL && node->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = node->lower_bound;
                }
                
                if (node->parent) {
                    --node->parent->cs_cnt;
                    if (node->parent->cs_cnt == 0 && node->parent != dummy) {
                        add_to_pool(node->parent);
                    }
                }
            } else {
                if (node->image >= g_n) {
                    // invalid node
                    if (node->parent) {
                        --node->parent->cs_cnt;
                        if (node->parent->cs_cnt == 0 && node->parent != dummy) {
                            add_to_pool(node->parent);
                        }
                    }
                } else if (node->lower_bound < upper_bound) {
                    // continue search
                    node->reused_leaf = true;
                    add_to_heap(node, heap_n, open_heap);
                } else {
                    // Boundary node: lb in [upper_bound, upper_bound + margin) range
                    if(node->level != DUMMY_VAL && node->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = node->lower_bound;
                    }
                    protect_ancestor_chain(node);
                    boundary_nodes.push_back(node);
                }
            }
        }
    }
    
    /*------------------------------------------------------------------*/
    /* 4. A* search main loop                                                  */
    /*------------------------------------------------------------------*/
    int cnt = 0;

    while (heap_n > 0 && open_heap[0]->lower_bound < upper_bound &&
           (verify_upper_bound == INF || upper_bound > verify_upper_bound))
    {
        cnt++;
        if(app_max_iter > 0 && cnt > app_max_iter) break;

        State *now = open_heap[0];
        
        open_heap[0] = open_heap[--heap_n];
        heap_top_down(0, heap_n, open_heap);
        
        // Lazy compute
        if (now->need_recompute) {
            if (now->image >= g_n) {
                now->need_recompute = false;
                if (now->cs_cnt == 0 && !now->kept) {
                    add_to_pool(now);
                }
                continue;
            }
            
            ui old_lb = now->lower_bound;
            recompute_node_lazy(now);

            if (now->lower_bound >= upper_bound) {
                if(now->level != DUMMY_VAL && now->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = now->lower_bound;
                }
                
                if (now->lower_bound < upper_bound + margin) {
                    // boundary node
                    protect_ancestor_chain(now);
                    boundary_nodes.push_back(now);
                } else {
                    // pruning
                    if (now->cs_cnt == 0 && !now->kept) {
                        add_to_pool(now);
                    }
                }
                continue;
            }
            
#if PUSH_BACK_AFTER_RECOMPUTE
            if (now->image < g_n) {
                add_to_heap(now, heap_n, open_heap);
            }
            continue;
#endif
        }

        // Generate sibling nodes
        State *sib = get_a_new_state_node();
        sib->cs_cnt = 0;
        sib->need_recompute = false;

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
                   
        if (sib->image < g_n) {
            if (sib->lower_bound < upper_bound) {
                // continue search
                add_to_heap(sib, heap_n, open_heap);
                if (now->parent) ++now->parent->cs_cnt;
                if (sib->parent) sib->parent->_children.push_back(sib);
            } else if (sib->lower_bound < upper_bound + margin) {
                // Boundary node: lb in [upper_bound, upper_bound + margin) range
                if(sib->level != DUMMY_VAL && sib->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = sib->lower_bound;
                }
                protect_ancestor_chain(sib);
                boundary_nodes.push_back(sib);
                if (now->parent) ++now->parent->cs_cnt;
                if (sib->parent) sib->parent->_children.push_back(sib);
            } else {
                // Pruning: lb >= upper_bound + margin
                if(sib->level != DUMMY_VAL && sib->lower_bound < terminal_frontier_min_lb) {
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
            // invalid node
#ifdef _EXPAND_ALL_
            put_a_sibling_to_pool(sib->siblings);
            sib->siblings = nullptr;
#else
            --now->cs_cnt;
#endif
            put_a_state_to_pool(sib);
        }

        // child/full-mapping branch
        if (now->level + 1 == search_n) {
            // complete mapping
            if(now->level != DUMMY_VAL && now->lower_bound < terminal_frontier_min_lb) {
                terminal_frontier_min_lb = now->lower_bound;
            }
            protect_ancestor_chain(now);
            full_mapping_nodes.push_back(now);
            extend_to_full_mapping(now, &full);
            
            if (upper_bound < tau + 1) {
                best_solution = now;
            }
        } else {
            if (now->level + 1 >= q_n) {
                if (now->cs_cnt == 0 && !now->kept) {
                    add_to_pool(now);
                }
                continue;
            }
            
            State *child = get_a_new_state_node();
            if (!child) {
                if (now->cs_cnt == 0 && !now->kept) {
                    add_to_pool(now);
                }
                continue;
            }
            
            child->cs_cnt = 0;
            child->need_recompute = false;
            
#ifdef _EXPAND_ALL_
            child->siblings = get_a_new_siblings_node();
#endif
                        
            generate_best_extension(now, child);
            ++now->cs_cnt;
            
            if (child->image < g_n) {
                if (child->lower_bound < upper_bound) {
                    // continue search
                    add_to_heap(child, heap_n, open_heap);
                    child->parent->_children.push_back(child);
                } else if (child->lower_bound < upper_bound + margin) {
                    // Boundary node: lb in [upper_bound, upper_bound + margin) range
                    if(child->level != DUMMY_VAL && child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
                    protect_ancestor_chain(child);
                    boundary_nodes.push_back(child);
                    child->parent->_children.push_back(child);
                } else {
                    // Pruning: lb >= upper_bound + margin
                    if(child->level != DUMMY_VAL && child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
#ifdef _EXPAND_ALL_
                    if (child->siblings) {
                        put_a_sibling_to_pool(child->siblings);
                        child->siblings = nullptr;
                    }
#endif
                    put_a_state_to_pool(child);
                    --now->cs_cnt;
                }
            } else {
                // invalid node
#ifdef _EXPAND_ALL_
                if (child->siblings) {
                    put_a_sibling_to_pool(child->siblings);
                    child->siblings = nullptr;
                }
#endif
                put_a_state_to_pool(child);
                --now->cs_cnt;
            }
        }

        if (now->cs_cnt == 0 && !now->kept) {
            add_to_pool(now);
        }

        // lazy deletion
        while (heap_n > 0 && open_heap[heap_n-1]->lower_bound >= upper_bound) {
            --heap_n;
            if (open_heap[heap_n]->cs_cnt == 0 && !open_heap[heap_n]->kept) {
                add_to_pool(open_heap[heap_n]);
            }
        }
    }
    
    // Set final overall_lb
    if(heap_n > 0) {
        overall_lb = open_heap[0]->lower_bound;  // active frontier
    } else if(terminal_frontier_min_lb < INF){
        overall_lb = terminal_frontier_min_lb;   // terminal frontier
    }
    
    // Protect ancestor chains of remaining nodes in heap
    for (ui i = 0; i < heap_n; ++i) {
        protect_ancestor_chain(open_heap[i]);
    }
    open_heap.resize(heap_n);
    
    return upper_bound;
}