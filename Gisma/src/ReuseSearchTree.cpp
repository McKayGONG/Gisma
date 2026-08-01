/*********************************************************************
 *  ReuseLSa_DFS.cpp  ――  增量补丁刷新快照树各节点 LB
 *  两处关键修改：
 *    - ensure_vlabel_cap(ui)   -> Application 成员
 *    - ensure_elabel_cap(ui)   -> Application 成员
 ********************************************************************/

#include "Application.h"
#include <vector>
#include <cstring>
#include <chrono>

/* ============================================================= *
 *  VCAP / ECAP / MCAP / ecnt 已移至 Application 实例成员          *
 *  (Application.h)，解决多线程并行 reuse 搜索时的数据竞争问题      *
 * ============================================================= */



void Application::ensure_vlabel_cap(ui lbl)
{
    /* --- 若还没分配，直接分配 min(8 对齐, lbl+1) --- */
    if (vlabels_map == nullptr) {
        VCAP = ((std::max<ui>(lbl,0) + 8) & ~7u);   // 至少 8，对齐
        vlabels_map = new int[VCAP]();              // 全 0
        return;
    }

    /* --- 否则按容量判断是否扩容 --- */
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
    if (elabels_matrix == nullptr) {                // <- 同理
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


// 在 compute_independent_lower_bound_LSa_baseline 中添加详细调试
ui Application::compute_independent_lower_bound_LSa_baseline(State *now) {
    prof_lsa_count[prof_phase]++;
    struct _LSaScope {
        std::chrono::steady_clock::time_point t0;
        int phase;
        _LSaScope(int p) : t0(std::chrono::steady_clock::now()), phase(p) {}
        ~_LSaScope() {
            auto t1 = std::chrono::steady_clock::now();
            double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            Application::prof_lsa_us[phase] += ns / 1000.0;
        }
    } _scope(prof_phase);
    // 特殊情况：完全映射
    if (now->level >= q_n) {
        return now->mapped_cost;
    }

    // 特殊情况：当前节点是dummy映射
    bool now_is_dummy = (now->image >= g_n);
    
    // 1. MC (Mapped Cost)
    ui mc = compute_mapped_cost_baseline(now);
    
    // 2. 准备数据结构
    int *v_labels_cnt = new int[vlabels_n];
    int *e_labels_cnt = new int[elabels_n];
    int *cross_e_labels_cnt = new int[elabels_n];
    
    memset(v_labels_cnt, 0, sizeof(int)*vlabels_n);
    memset(e_labels_cnt, 0, sizeof(int)*elabels_n);
    memset(cross_e_labels_cnt, 0, sizeof(int)*elabels_n);
    
    // 标记已映射顶点
    char *visG = new char[g_n];
    char *visQ = new char[q_n];
    memset(visG, 0, sizeof(char)*g_n);
    memset(visQ, 0, sizeof(char)*q_n);
    
    // 构建映射关系
    ui *px = new ui[q_n];
    ui *py = new ui[g_n];
    for(ui i = 0; i < q_n; i++) px[i] = g_n;
    for(ui i = 0; i < g_n; i++) py[i] = q_n;
    
    // 重建映射（排除dummy映射）
    for(State *st = now; st != NULL; st = st->parent) {
        if (st->image >= g_n || st->level >= q_n) continue;  // 跳过dummy
        visG[st->image] = 1;
        visQ[MO[st->level]] = 1;
        px[MO[st->level]] = st->image;
        py[st->image] = MO[st->level];
    }
    
    /* ------------------------------------------------------------ *
     * 2. 计算 Inner LB（未映射子图的内部下界）
     * ------------------------------------------------------------ */
    
    // 2.1 顶点多重集差
    int uvl_cnt = 0, vvl_cnt = 0, vl_common = 0;
    
    // 未映射查询顶点（考虑dummy情况）
    ui unmapped_start = now_is_dummy ? now->level : (now->level + 1);
    
    for(ui i = unmapped_start; i < q_n; i++) {
        --v_labels_cnt[q_vlabels[MO[i]]];
        ++uvl_cnt;
    }
    
    // 未映射数据顶点
    for(ui i = 0; i < g_n; i++) 
        if(!visG[i]) {
            if(v_labels_cnt[g_vlabels[i]] < 0) ++vl_common;
            ++v_labels_cnt[g_vlabels[i]];
            ++vvl_cnt;
        }
    
    // 2.2 内部边多重集差
    int inner_uel_cnt = 0, inner_vel_cnt = 0, inner_el_common = 0;

    if (all_edge_labels_same) {
        // 所有边同标签：多重集差退化为 |count差|，common = min(两边count)，不需 label 计数
        for(ui i = 0; i < g_n; i++) {
            if(!visG[i]) {
                for(ui j = g_starts[i]; j < g_starts[i+1]; j++) {
                    ui neighbor = g_edges[j];
                    if(!visG[neighbor] && i < neighbor) ++inner_vel_cnt;
                }
            }
        }
        for(ui i = unmapped_start; i < q_n; i++) {
            ui u = MO[i];
            for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
                ui neighbor = q_edges[j];
                if(!visQ[neighbor] && u < neighbor) ++inner_uel_cnt;
            }
        }
        inner_el_common = (inner_vel_cnt < inner_uel_cnt) ? inner_vel_cnt : inner_uel_cnt;
    } else {
        // 数据图的内部边
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

        // 查询图的内部边
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
    }
    
    ui vertex_lb = (vvl_cnt > uvl_cnt) ? vvl_cnt : uvl_cnt;
    vertex_lb -= vl_common;
    
    ui inner_edge_lb = (inner_vel_cnt > inner_uel_cnt) ? inner_vel_cnt : inner_uel_cnt;
    inner_edge_lb -= inner_el_common;
    
    ui inner_lb = vertex_lb + inner_edge_lb;
    
    /* ------------------------------------------------------------ *
     * 3. 计算 Cross LB（跨界边的下界）
     * ------------------------------------------------------------ */
    ui cross_lb = 0;
    
    // 对每个已映射顶点对（排除dummy）
    ui cross_end = now_is_dummy ? now->level : (now->level + 1);
    
    for(ui idx = 0; idx < cross_end; idx++) {
        ui u = MO[idx];
        ui v = px[u];
        if (v >= g_n) continue;  // 跳过dummy映射

        ui u_cross_cnt = 0, v_cross_cnt = 0, cross_common = 0;

        if (all_edge_labels_same) {
            // 全同标签：cross_common = min(u_cross_cnt, v_cross_cnt)，不需 label 计数
            for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
                if(!visQ[q_edges[j]]) ++u_cross_cnt;
            }
            for(ui j = g_starts[v]; j < g_starts[v+1]; j++) {
                if(!visG[g_edges[j]]) ++v_cross_cnt;
            }
            cross_common = (u_cross_cnt < v_cross_cnt) ? u_cross_cnt : v_cross_cnt;
        } else {
            // 重置计数器
            memset(cross_e_labels_cnt, 0, sizeof(int)*elabels_n);

            // 统计u的跨界边
            for(ui j = q_starts[u]; j < q_starts[u+1]; j++) {
                ui neighbor = q_edges[j];
                if(!visQ[neighbor]) {
                    --cross_e_labels_cnt[q_elabels[j]];
                    ++u_cross_cnt;
                }
            }

            // 统计v的跨界边
            for(ui j = g_starts[v]; j < g_starts[v+1]; j++) {
                ui neighbor = g_edges[j];
                if(!visG[neighbor]) {
                    if(cross_e_labels_cnt[g_elabels[j]] < 0) ++cross_common;
                    ++cross_e_labels_cnt[g_elabels[j]];
                    ++v_cross_cnt;
                }
            }
        }

        ui max_cross = (u_cross_cnt > v_cross_cnt) ? u_cross_cnt : v_cross_cnt;
        cross_lb += max_cross - cross_common;
    }
    
    // 4. 最终的 LSa LB
    ui lsa_lb = mc + inner_lb + cross_lb;
    
    // 清理内存
    delete[] v_labels_cnt;
    delete[] e_labels_cnt;
    delete[] cross_e_labels_cnt;
    delete[] visG;
    delete[] visQ;
    delete[] px;
    delete[] py;
    
    return lsa_lb;
}

// 同时修改 compute_mapped_cost_baseline 添加dummy检查
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
        
        // 关键修复：检查v是否有效
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

#define DEBUG_REFRESH 0  // 设为1启用调试，设为0关闭


void Application::refresh_bm_snapshot_lb_batch(State* root, ui snap_nodes, ui ged_gap) {
    /* ---------- 0. 同步静态容量计数 ---------- */
    VCAP = vlabels_n;
    ECAP = elabels_n;
    MCAP = ECAP;
    if (ecnt.size() < ECAP) ecnt.resize(ECAP, 0);

    /* ---------- 1. 工作数组初始化 ---------- */

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

    // 初始化q_matrix
    for(ui i = 0; i < q_n; i++) {
        uchar *t_array = q_matrix + i * q_n;
        for(ui j = 0; j < q_n; j++) t_array[j] = elabels_n;
        for(ui j = q_starts[i]; j < q_starts[i+1]; j++) {
            if (q_edges[j] < q_n) {
                t_array[q_edges[j]] = q_elabels[j];
            }
        }
    }

    /* ---------- 2. 保存原始上界 ---------- */
    ui saved_upper_bound = upper_bound;
    ui saved_verify_upper_bound = verify_upper_bound;
    upper_bound = UINT_MAX;
    verify_upper_bound = UINT_MAX;

    /* ---------- 3. BFS遍历树处理节点 ---------- */
    std::queue<State*> bfs_queue;
    bfs_queue.push(root);

    while (!bfs_queue.empty()) {
        State* current = bfs_queue.front();
        bfs_queue.pop();

        // 处理 dummy 节点
        if (current->level == UINT16_MAX && current->image == UINT16_MAX) {
            current->lower_bound = 0;
            current->mapped_cost = 0;
            
            
            for (State* child : children_of(current)) {
                if (child) bfs_queue.push(child);
            }
            continue;
        }

        /* ---------- 3.1 计算当前节点的MC ---------- */

        // 用金标准验证
        int mapped_cost_baseline = compute_mapped_cost_baseline(current);
        current->mapped_cost = mapped_cost_baseline;

        // Internal nodes: lower_bound is never read downstream (only leaves get
        // pushed to open_heap), and siblings arrays are only stored on leaves.
        // mc is needed (used as baseline by construct_sibling / generate_best_extension
        // when astar later expands a leaf descendant), so we keep mc above.
        if (!children_of(current).empty()) {
            for (State* child : children_of(current)) {
                if (child) bfs_queue.push(child);
            }
            continue;
        }

        // BMao-recomp: refresh 階段精確算 BMao lb。兩個變體：
        //   bmao_recomp_intersect=false (默認): refresh_leaf_lb_only_BM (純 g')
        //   bmao_recomp_intersect=true:        recompute_node_lazy (帶 g ∩ g' 交集策略)
        if (refresh_bmao_recompute) {
            ui save_ub = upper_bound;
            upper_bound = saved_upper_bound;
            if (bmao_recomp_intersect) {
                recompute_node_lazy(current);
            } else {
                refresh_leaf_lb_only_BM(current);
            }
            upper_bound = save_ub;
            continue;
        }

        int tmp_lb = 0;
        int old_lsa_lb = 0;

        if (!disable_reuse_lsa) {
            // LSa 重计算 estimate
            tmp_lb = compute_independent_lower_bound_LSa_baseline(current);
    #ifdef _USE_LSa_ESTIMATE_BMao_
            old_lsa_lb = current->lsa_lb;
            current->lower_bound = current->lower_bound + tmp_lb <= old_lsa_lb ? 0 : mmax(tmp_lb,current->lower_bound + tmp_lb - old_lsa_lb);
    #else
            current->lower_bound = current->lower_bound <= ged_gap ? 0 : mmax(tmp_lb,current->lower_bound - ged_gap);
    #endif
        } else {
            // disable_reuse_lsa=true: triangle (簡單 ged_gap 减法)
            current->lower_bound = current->lower_bound <= ged_gap ? 0 : current->lower_bound - ged_gap;
        }

        /* ---------- 3.2 更新siblings中的LB值 ---------- */
#ifdef _EXPAND_ALL_
        if (current->siblings && current->siblings_n > 0) {
            // siblings数组的结构：偶数索引是顶点ID，奇数索引是LB值
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
        

        for (State* child : children_of(current)) {
            if (child) bfs_queue.push(child);
        }
    }

    /* ---------- 4. 恢复原始上界 ---------- */
    upper_bound = saved_upper_bound;
    verify_upper_bound = saved_verify_upper_bound;

}
// ============================================================
// 3. 修复后的 refresh_lsa_snapshot_lb
// ============================================================

// 重命名为更通用的函数名

void Application::recompute_node_lazy(State* now) {
    // 如果是完整映射或dummy节点，不需要重计算
    if (now->level >= search_n || now->image >= g_n) {
        now->need_recompute = false;
        return;
    }
    
    // 1. 收集pre_siblings（利用_children信息）
    ui pre_siblings = 0;
    
    // 从父节点的_children中找到当前节点之前的兄弟
    if (now->parent && !children_of(now->parent).empty()) {
        for (auto child : children_of(now->parent)) {
            if (child == now) break;  // 到达当前节点，停止
            if (child->level == now->level) {  // 同级节点
                if (pre_siblings >= g_n) {
                    break;
                }
                if (child->image < g_n) {  // 排除dummy节点
                    candidates[pre_siblings++] = child->image;
                }
            }
        }
    }
    
    // 2. 标记G端占用顶点
    for (ui i = 0; i < pre_siblings; ++i) {
        if (candidates[i] < g_n) {
            visY[candidates[i]] = 1;
        }
    }
    for (State *st = now->parent; st != NULL; st = st->parent) {
        if (st->image < g_n) visY[st->image] = 1;
    }
    
    // 3. 生成完整候选列表
    ui candidates_n = pre_siblings;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i]) {
            if (candidates_n < g_n) {
                candidates[candidates_n++] = i;
            }
        }
    }
    
    // 清理visY
    for (ui i = 0; i < g_n; ++i) {
        visY[i] = 0;
    }
    
    if (candidates_n <= pre_siblings) {
        // 没有新候选，标记为dummy
        now->image = g_n;
        // 使用margin设置下界，确保被剪枝
        now->lower_bound = upper_bound + margin + 1;
        now->need_recompute = false;
        return;
    }
    
    // 4. 保存原始mapped_cost（父节点的）
    now->mapped_cost = (now->parent ? now->parent->mapped_cost : 0);
    
#if USE_SIBLING_INTERSECTION
    // ===== 交集策略 =====
    
    // 4.1 保存原siblings信息
    std::vector<std::pair<ui, ui>> old_siblings_vec;  // pair<image, lb>
    ui old_image = now->image;  // 保存原image
    
#ifdef _EXPAND_ALL_
    // 使用map去重，保留每个image的最小lb值
    std::unordered_map<ui, ui> old_siblings_map;  // image -> min_lb
    
    // 首先添加当前节点的image（如果有效）
    if (old_image < g_n) {
        old_siblings_map[old_image] = now->lower_bound;
    }
    
    if (now->siblings && now->siblings_n > 0) {
        // 保存原siblings，去重（保留最小lb）
        ui duplicate_count = 0;
        for (ui i = 0; i < now->siblings_n; i += 2) {
            if (i + 1 >= now->siblings_n) break;

            ui img = now->siblings[i];
            ui lb = now->siblings[i+1];

            // FIX 2026-05-13: snap.siblings 可能含 parent compacted index >= child g_n
            // (e.g. NODE_DEL 後 parent 最高 index 在 child 中越界)。Skip 越界 entry，
            // 不讓它流入 best_image 選擇導致下游 g_vlabels[g_n] 越界 segfault。
            if (img >= g_n) continue;

            // 去重：如果已存在，保留lb较小的
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
    
    // 将map转换为vector
    old_siblings_vec.reserve(old_siblings_map.size());
    for (const auto& pair : old_siblings_map) {
        old_siblings_vec.push_back({pair.first, pair.second});
    }
#else
    // 非_EXPAND_ALL_模式，只保存当前image
    if (old_image < g_n) {
        old_siblings_vec.push_back({old_image, now->lower_bound});
    }
#endif
    
    // 4.2 临时清空siblings，准备计算新的
#ifdef _EXPAND_ALL_
    // 如果没有siblings空间，分配新的
    if (now->siblings == nullptr) {
        now->siblings = get_a_new_siblings_node();
    }
    now->siblings_n = 0;
#endif
    
    // 4.3 调用compute_best_extension计算新的siblings和image
    // Build intersection target set (old siblings' image set). compute_best_extension_BM
    // sibling loop 在算到所有 target image 後可 early break，省掉沒用的 H(0)。
#ifdef _EXPAND_ALL_
    std::unordered_set<ui> intersect_target;
    // skip_intersection_in_reuse 模式下不用 intersection target（child 直接用 new_siblings）
    if (lb_method == BMao && !old_siblings_map.empty() && !skip_intersection_in_reuse) {
        intersect_target.reserve(old_siblings_map.size());
        for (const auto& pr : old_siblings_map) intersect_target.insert(pr.first);
        lazy_recompute_target_siblings = &intersect_target;
    }
#endif
    if (lb_method == LSa) {
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    } else if (lb_method == BMao) {
        compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0, 0);
    } else if (lb_method == BMa) {
        compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
    } else {
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    }
    lazy_recompute_target_siblings = nullptr;  // 防止其他 caller 誤用

    // skip_intersection_in_reuse：跳過 intersection 篩選，直接用 child 算的 new_siblings
    if (skip_intersection_in_reuse) {
        now->need_recompute = false;
        return;
    }

#ifdef _EXPAND_ALL_
    // 4.4 保存完整计算的siblings数量
    ui full_siblings_n = now->siblings_n;
    
    // 4.5 构建新siblings的映射（用于快速查找）
    std::unordered_map<ui, ui> new_siblings_map;  // image -> lb
    
    // 添加当前选中的image
    new_siblings_map[now->image] = now->lower_bound;
    
    // 添加siblings数组中的所有image
    for (ui i = 0; i < now->siblings_n; i += 2) {
        if (i + 1 >= now->siblings_n) break;
        new_siblings_map[now->siblings[i]] = now->siblings[i+1];
    }
    
    // 4.6 计算交集
    std::vector<std::pair<ui, ui>> intersection;
    
    // 检查原siblings中哪些在新计算中也存在
    for (const auto& old_pair : old_siblings_vec) {
        auto it = new_siblings_map.find(old_pair.first);
        if (it != new_siblings_map.end()) {
            // 在交集中，使用新的LB值
            intersection.push_back({old_pair.first, it->second});
        }
    }
    
    // 4.7 根据交集结果处理
    if (intersection.empty()) {
        // 交集为空，淘汰该节点
        now->image = g_n;
        // 使用margin设置下界，确保被剪枝
        now->lower_bound = upper_bound + margin + 1;
        now->siblings_n = 0;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        now->siblings_total_n = 0;
#endif
        // 标记不再需要重计算
        now->need_recompute = false;
        return;
    }
    
    // 4.8 找到交集中LB最小的作为image
    ui best_image = intersection[0].first;
    ui best_lb = intersection[0].second;

    for (size_t i = 1; i < intersection.size(); ++i) {
        if (intersection[i].second < best_lb) {
            best_lb = intersection[i].second;
            best_image = intersection[i].first;
        }
    }

    // FIX 2026-05-13: defensive guard — 若 intersection 受污染 (best_image >= g_n)
    // 直接 invalidate，避免下游 compute_mapped_cost(now) 用 g_vlabels[best_image] 越界
    if (best_image >= g_n) {
        now->image = g_n;
        now->lower_bound = upper_bound + margin + 1;
        now->siblings_n = 0;
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        now->siblings_total_n = 0;
#endif
        now->need_recompute = false;
        return;
    }
    
    // 4.9 重新组织siblings数组
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
    // 新策略：非交集部分在前，交集部分在后
    
    // 首先，收集所有非交集的siblings
    std::vector<std::pair<ui, ui>> non_intersection;
    
    // 从新计算的完整siblings中找出不在交集中的
    if (now->image != best_image && new_siblings_map.find(now->image) != new_siblings_map.end()) {
        // 当前image如果不是best，且在新计算中，需要检查是否在交集中
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
    
    // 从siblings数组中找出不在交集中的
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
    
    // 计算最终需要的空间
    ui expected_total_elements = (non_intersection.size() + intersection.size() - 1) * 2;  // -1 for best_image
    if (expected_total_elements > g_n * 2) {
        // 截断以避免越界
        while (expected_total_elements > g_n * 2 && !non_intersection.empty()) {
            non_intersection.pop_back();
            expected_total_elements = (non_intersection.size() + intersection.size() - 1) * 2;
        }
    }
    
    // 重新填充siblings数组
    now->siblings_n = 0;
    
    // 先放入非交集部分
    for (const auto& pair : non_intersection) {
        if (now->siblings_n + 2 > g_n * 2) break;
        now->siblings[now->siblings_n++] = pair.first;
        now->siblings[now->siblings_n++] = pair.second;
    }
    
    // 记录交集开始的位置
    ui intersection_start = now->siblings_n;
    
    // 再放入交集部分（除了best）
    for (const auto& pair : intersection) {
        if (pair.first != best_image) {
            if (now->siblings_n + 2 > g_n * 2) break;
            now->siblings[now->siblings_n++] = pair.first;
            now->siblings[now->siblings_n++] = pair.second;
        }
    }
    
    // 设置siblings_total_n为完整大小
    now->siblings_total_n = now->siblings_n;
    
    // 设置siblings_n为只包含交集部分的大小
    // 安全检查防止下溢
    if (now->siblings_total_n >= intersection_start) {
        now->siblings_n = now->siblings_total_n - intersection_start;
    } else {
        now->siblings_n = 0;
    }
    
#else  // 不使用INTERSECTION_WITH_FULL_SNAPSHOT
    // 原策略：只保留交集
    now->siblings_n = 0;
    
    // 先添加非best的交集元素
    for (const auto& pair : intersection) {
        if (pair.first != best_image) {
            if (now->siblings_n + 2 > g_n * 2) break;
            now->siblings[now->siblings_n++] = pair.first;
            now->siblings[now->siblings_n++] = pair.second;
        }
    }
#endif
    
    // 更新now的image和lower_bound
    now->image = best_image;
    now->lower_bound = best_lb;
    
    // 需要重新计算mapped_cost
    compute_mapped_cost(now);
    
#else  // 非_EXPAND_ALL_模式
    // 在非_EXPAND_ALL_模式下，检查原image是否在新计算结果中仍然有效
    if (old_image < g_n && now->image == old_image) {
        // 原image仍然是最优选择，保留（使用新的LB）
    } else {
        // 原image不再是最优选择，检查是否在候选中
        bool found = false;
        for (ui i = pre_siblings; i < candidates_n; ++i) {
            if (candidates[i] == old_image) {
                found = true;
                break;
            }
        }
        
        if (!found || old_image >= g_n) {
            // 原image不在有效候选中，使用新的最优image
            // （这相当于交集为空的情况）
            // now->image已经被compute_best_extension设置为新的最优
        }
    }
#endif

#else  // USE_SIBLING_INTERSECTION == 0
    // ===== 原策略：完全重新计算 =====
    
    // 5. 处理siblings（更安全的方式）
#ifdef _EXPAND_ALL_
    // 如果没有siblings，分配新的；如果有，直接重用
    if (now->siblings == nullptr) {
        now->siblings = get_a_new_siblings_node();
    }
    now->siblings_n = 0;  // 清空计数，让函数重新填充
#endif
    
    // 6. 调用compute_best_extension重新计算（no_siblings=0，生成siblings）
    if (lb_method == LSa) {
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    } else if (lb_method == BMao) {
        compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings, 0, 0);  // anchor_aware=1, no_siblings=0, IS=0
    } else if (lb_method == BMa) {
        compute_best_extension_BMa(1, now, candidates_n, candidates, pre_siblings);
    } else {
        // 对于其他lb_method，使用LSa作为默认
        compute_best_extension_LSa(now, candidates_n, candidates, pre_siblings);
    }
    
#endif  // USE_SIBLING_INTERSECTION
    
    // 7. 标记不再需要重计算
    now->need_recompute = false;

    // 现在这个节点已经像一个全新计算的节点一样，有最优的image和siblings
}

// BMao-recomp 用：純算 leaf 在 g' 上的精確 BMao lb，無交集策略，不重生成 siblings。
// 構造 candidates 後直接 compute_best_extension_BM(no_siblings=1)。
void Application::refresh_leaf_lb_only_BM(State* now) {
    if (now->level >= search_n || now->image >= g_n) return;

    // 1. 收集 pre_siblings (祖先層中當前節點之前的同級兄弟)
    ui pre_siblings = 0;
    if (now->parent && !children_of(now->parent).empty()) {
        for (auto child : children_of(now->parent)) {
            if (child == now) break;
            if (child->level == now->level && child->image < g_n) {
                if (pre_siblings >= g_n) break;
                candidates[pre_siblings++] = child->image;
            }
        }
    }

    // 2. 標記 G 端已佔用 (祖先 + pre_siblings)
    for (ui i = 0; i < pre_siblings; ++i) {
        if (candidates[i] < g_n) visY[candidates[i]] = 1;
    }
    for (State *st = now->parent; st != NULL; st = st->parent) {
        if (st->image < g_n) visY[st->image] = 1;
    }

    // 3. 完整候選列表 = pre_siblings + 未佔用的 G 頂點
    ui candidates_n = pre_siblings;
    for (ui i = 0; i < g_n; ++i) {
        if (!visY[i] && candidates_n < g_n) candidates[candidates_n++] = i;
    }
    for (ui i = 0; i < g_n; ++i) visY[i] = 0;  // 清理

    if (candidates_n <= pre_siblings) {
        // 沒新候選 → leaf 失效
        now->image = g_n;
        now->lower_bound = upper_bound + margin + 1;
        return;
    }

    // 4. mapped_cost 從父節點繼承（compute_best_extension_BM 會用）
    now->mapped_cost = (now->parent ? now->parent->mapped_cost : 0);

    // 5. 直接調用 BMao 算 lb：no_siblings=1 → 不寫 siblings 數組
    //    會更新 now->image, now->lower_bound, now->matching
    compute_best_extension_BM(1, now, candidates_n, candidates, pre_siblings,
                              /*no_siblings=*/1, /*IS=*/0);
}


void Application::extract_snapshot(SearchSnapshot& out) {
    /* ---------- 0. 基本信息 ---------- */
    out.v.clear();
    out.ub = upper_bound;
    out.overall_lb = overall_lb;
    upper_bound = verify_upper_bound + 1;
    out.mo.assign(MO, MO + q_n);
    
#ifdef _EXPAND_ALL_
    out.siblings_data.clear();
    uint32_t siblings_offset_counter = 0;  // 跟踪当前偏移量
#endif

    /* ---------- 1. 收集需要保存的节点 ---------- */
    std::unordered_set<State*> need_set;    // 用于去重
    std::vector<State*> need_nodes;         // 保存所有需要的节点
    std::queue<State*> pending;             // BFS队列，用于追溯祖先
    std::unordered_set<State*> active_set;  // 标记活跃节点（精选节点）

    /* ---------- 1-A. 收集所有活跃节点及其父节点 ---------- */
    auto mark_node_and_parent = [&](State* st) {
        if (!st) return;
        
        // 过滤掉下界过高的节点
        if (st->lower_bound >= upper_bound + margin) {
            return;
        }
        
        // 标记为活跃节点
        active_set.insert(st);
        
        // 将节点本身加入需要保存的集合
        if (need_set.insert(st).second) {
            need_nodes.push_back(st);
            pending.push(st);  // 加入BFS队列以追溯祖先
        }
        
        // 同时处理父节点
        if (st->parent && st->parent->lower_bound < upper_bound + margin) {
            if (need_set.insert(st->parent).second) {
                need_nodes.push_back(st->parent);
                pending.push(st->parent);
            }
        }
    };

    // 从容器中收集节点本身及其父节点
    for (State* st : open_heap) {
        mark_node_and_parent(st);
    }
    
    for (State* st : full_mapping_nodes) {
        mark_node_and_parent(st);
    }
    
    for (State* st : boundary_nodes) {
        mark_node_and_parent(st);
    }

    /* ---------- 1-B. BFS追溯所有祖先 ---------- */
    while (!pending.empty()) {
        State* cur = pending.front();
        pending.pop();
        
        if (cur->parent && need_set.insert(cur->parent).second) {
            need_nodes.push_back(cur->parent);
            pending.push(cur->parent);
        }
    }

    /* ---------- 1-C. 创建并添加dummy根节点 ---------- */
    State dummy_state;
    State* dummy = &dummy_state;
    dummy->parent = nullptr;
    dummy->level = DUMMY_VAL;
    dummy->image = DUMMY_VAL;
    dummy->lower_bound = 0;
    dummy->mapped_cost = 0;
    children_of(dummy).clear();
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

    /* ---------- 2. 重建父子关系 ---------- */
    // 清空所有节点的children列表
    for (State* st : need_nodes) {
        children_of(st).clear();
    }
    
    // 重建children列表
    for (State* st : need_nodes) {
        if (st == dummy) continue;
        
        if (st->parent && need_set.count(st->parent)) {
            children_of(st->parent).push_back(st);
        } else if (!st->parent) {
            // 孤儿节点连接到dummy
            st->parent = dummy;
            children_of(dummy).push_back(st);
        }
    }

    /* ---------- 3. 前序遍历输出到快照 ---------- */
    std::vector<State*> stack;
    stack.push_back(dummy);
    
    while (!stack.empty()) {
        State* cur = stack.back();
        stack.pop_back();
        
        // 只输出在need_set中的节点
        if (need_set.count(cur)) {
            // 简化版本：只标记是否是活跃节点
            uint8_t is_open = active_set.count(cur) ? 1 : 0;
            
            SearchNodeLite node_lite{};  // 先全部初始化为0

            // 然后逐个赋值
            node_lite.level = cur->level;
            node_lite.image = cur->image;
            node_lite.lb = static_cast<uint16_t>(cur->lower_bound);
            node_lite.mc = static_cast<uint16_t>(cur->mapped_cost);
            node_lite.is_open = is_open;

#ifdef _USE_LSa_ESTIMATE_BMao_
            // lazy_parent_lsa: 只對 leaves 算 LSa（refresh 裡僅 leaves 用）。
            // 但若 disable_reuse_lsa=true（child 將用 triangle refresh），LSa 算了也白算 → 跳過。
            if (lazy_parent_lsa && !disable_reuse_lsa
                && cur->image < g_n && cur->level < q_n
                && children_of(cur).empty()) {
                cur->lsa_lb = compute_independent_lower_bound_LSa_baseline(cur);
            } else if (lazy_parent_lsa) {
                // Internal node OR disable_reuse_lsa: lsa_lb not used; set 0 to avoid garbage.
                cur->lsa_lb = 0;
            }
            node_lite.lsa_lb = cur->lsa_lb;
#endif

#ifdef _EXPAND_ALL_
            // 通过 level 或 image 判断是否是 dummy 节点
            if (cur->level == DUMMY_VAL || cur->image == DUMMY_VAL) {
                // 这是 dummy 节点，不处理 siblings
                node_lite.siblings_n = 0;
                node_lite.siblings_offset = 0;
            } 
            // 只有叶子节点才可能有 siblings
            else if (children_of(cur).empty() && cur->siblings != nullptr) {
                // 使用siblings_total_n来保存完整的siblings信息
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
       
        // 反向压栈，保证前序遍历
        for (auto it = children_of(cur).rbegin(); it != children_of(cur).rend(); ++it) {
            stack.push_back(*it);
        }
    }
}

/*------------------------------------------------------------
 * compute_mapping_order_reuse()
 *  - 若  MO == nullptr        -> 运行原完整流程（旧函数体）
 *  - 若  MO 已有内容 (来自快照)-> 仅重算 search_n / search_n_for_IS
 *-----------------------------------------------------------*/
void Application::compute_mapping_order_reuse()
{
    /* ---------- 情况 1：第一次进入，MO 尚未生成 ---------- */
    if (MO == nullptr) {
        // printf("extract[compute_mapping_order_reuse] MO is nullptr\n");
        compute_mapping_order();        // 直接调用旧函数（原逻辑）
        return;
    }
    // printf("extract[compute_mapping_order_reuse] MO is not nullptr\n");
    /* ---------- 情况 2：MO 已由快照提供 ---------- */
    /* 只用它来计算 independent-set 大小 (search_n) --------- */

    /* 1. 构造 pos[] ：查询顶点在 MO 中的位置 */
    ui* pos = new ui[q_n];
    for (ui i = 0; i < q_n; ++i) pos[MO[i]] = i;

    /* 2. 逐顶点数未重叠边数，找到最小 i 使“覆盖所有边” */
    ui edges_cnt = 0;
    for (ui i = 0; i < q_n; ++i) {
        ui u = MO[i];
        for (ui j = q_starts[u]; j < q_starts[u+1]; ++j)
            if (pos[q_edges[j]] > i)       // 只计一次
                edges_cnt += 2;
        if (edges_cnt == q_starts[q_n]) {  // 全部边已计入
            search_n = i + 1;              // 前缀大小
            break;
        }
    }
    search_n_for_IS = search_n;            // BMa/BMao 亦要用
    if (lb_method != BMao) search_n = q_n; // 按原规则覆写

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

    // 前置 gate (GismaSearchEngine.cpp dfs_traverse) 保證父子 vertex 數相同：
    //   snap.mo.size() == q_n。若進到這裡 MO 大小不對，是 caller bug，保守 fail。
    if (!snap.mo.empty() && snap.mo.size() != q_n) {
        last_reuse_exit_reason = 1;
        return tau + 1;
    }
    last_reuse_exit_reason = 0;

    // 初始化overall_lb和terminal_frontier_min_lb
    // 即使snapshot为空，也可以利用overall_lb信息
    overall_lb = (snap.overall_lb > ged_gap) ? (snap.overall_lb - ged_gap) : 0;
    ui terminal_frontier_min_lb = INF;

    // 如果快照为空，降级为普通AStar，但继承overall_lb和MO
    if (snap.v.empty()) {
        // 不直接返回失败，而是继续执行普通A*搜索
        // 这样可以利用overall_lb作为初始下界
        // 注意：此时不会有搜索树可以复用，但至少有LB和MO信息
    }

    auto _prof_t0 = std::chrono::steady_clock::now();

    /*------------------------------------------------------------------*/
    /* 0. 初始化：继承上界 & 映射顺序                                    */
    /*------------------------------------------------------------------*/
    upper_bound = tau + 1;  // 維持 lb-based prune；exact_value_mode 只停用 early-exit while 條件
    State* best_solution = nullptr;

    // 设置 search_n
    search_n = q_n;
    if (g_n < q_n) search_n = g_n;

    if (!snap.mo.empty()) {
        if (!MO) {
            MO = new ui[q_n];
        }
        std::memcpy(MO, snap.mo.data(), sizeof(ui) * q_n);
    }
    
    compute_mapping_order_reuse();
    
    // 释放上次留下的外部引用，再清空容器
    auto release_external_refs = [&]() {
        for (auto* st : open_heap)          if (st && --st->cs_cnt == 0) add_to_pool(st);
        for (auto* st : boundary_nodes)     if (st && --st->cs_cnt == 0) add_to_pool(st);
        for (auto* st : full_mapping_nodes) if (st && --st->cs_cnt == 0) add_to_pool(st);
        open_heap.clear();
        boundary_nodes.clear();
        full_mapping_nodes.clear();
    };
    release_external_refs();

    State full;

    // cs_cnt-based 外部引用记账（取代 kept + 走父链）
    auto protect_ancestor_chain = [](State* node) {
        if (node) node->cs_cnt++;
    };
    
    ui heap_n = 0;

    auto _prof_t1 = std::chrono::steady_clock::now();
    prof_reuse_init_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_prof_t1 - _prof_t0).count() / 1000.0;

    /*------------------------------------------------------------------*/
    /* 1. 重建快照中的树结构                                            */
    /*------------------------------------------------------------------*/
    State* dummy = nullptr;
    std::vector<State*> all_nodes;
    std::vector<State*> path_stack;

    // 按照前序遍历的顺序重建节点
    for (size_t i = 0; i < snap.v.size(); ++i) {
        const auto& rec = snap.v[i];
        ushort new_level = rec.level;

        State* st = get_a_new_state_node();
        if (!st) {
            last_reuse_exit_reason = 5;  // state pool exhausted
            return tau + 1;
        }

        *st = {};
        children_of(st).clear();

        st->level = new_level;
        st->image = rec.image;
        st->lower_bound = rec.lb;
        st->mapped_cost = rec.mc;
        st->cs_cnt = 0;
        st->reused_leaf = false;
        st->need_recompute = false;
#ifdef _USE_LSa_ESTIMATE_BMao_
        st->lsa_lb = rec.lsa_lb;
#endif
#ifdef _EXPAND_ALL_
        const size_t dest_cap = static_cast<size_t>(2) * g_n;
        st->siblings_n = static_cast<ushort>(std::min(static_cast<size_t>(rec.siblings_n), dest_cap));
#if USE_SIBLING_INTERSECTION && INTERSECTION_WITH_FULL_SNAPSHOT
        st->siblings_total_n = st->siblings_n;
#endif
        if (st->siblings_n > 0) {
            st->siblings = get_a_new_siblings_node();

            if (st->siblings && rec.siblings_offset < snap.siblings_data.size()) {
                size_t copy_size = std::min({
                    static_cast<size_t>(st->siblings_n),
                    snap.siblings_data.size() - rec.siblings_offset,
                    dest_cap
                });
                std::memcpy(st->siblings,
                            &snap.siblings_data[rec.siblings_offset],
                            copy_size * sizeof(ushort));
            }
        } else {
            st->siblings = nullptr;
        }
#endif

        // 处理dummy节点
        if (st->level == DUMMY_VAL && st->image == DUMMY_VAL) {
            dummy = st;
            st->parent = nullptr;
            st->cs_cnt = 1;
            path_stack.clear();
            path_stack.push_back(st);
        } else {
            while (path_stack.size() > (size_t)(st->level + 1)) {
                path_stack.pop_back();
            }

            if (!path_stack.empty()) {
                st->parent = path_stack.back();
                children_of(st->parent).push_back(st);
                ++st->parent->cs_cnt;
            }

            path_stack.push_back(st);
        }

        all_nodes.push_back(st);
    }
    
    if (!dummy) {
        if (upper_bound > tau) last_reuse_exit_reason = 4;  // no dummy (rebuild empty)
        return upper_bound;
    }
    
    auto _prof_t2 = std::chrono::steady_clock::now();
    prof_reuse_rebuild_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_prof_t2 - _prof_t1).count() / 1000.0;

    /*------------------------------------------------------------------*/
    /* 2. 刷新所有节点的下界                                            */
    /*------------------------------------------------------------------*/
    refresh_bm_snapshot_lb_batch(dummy, all_nodes.size(), ged_gap);

    auto _prof_t3 = std::chrono::steady_clock::now();
    prof_reuse_refresh_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_prof_t3 - _prof_t2).count() / 1000.0;

    /*------------------------------------------------------------------*/
    /* 3. 处理叶子节点                                                  */
    /*------------------------------------------------------------------*/
    dummy->cs_cnt = children_of(dummy).size();
    
    for (size_t i = 0; i < all_nodes.size(); ++i) {
        State* node = all_nodes[i];
        if (node == dummy) continue;
        
        if (children_of(node).empty()) {
            prof_reuse_snap_leaves++;
            // BMao-recomp 模式下 refresh 階段已精確重算 leaf lb；不需要 astar 再 lazy recompute
            // skip_lazy_recompute 強制跳過 lazy recompute（即便 refresh 是 LSa estimate）
            node->need_recompute = !refresh_bmao_recompute && !skip_lazy_recompute;

            // full mapping leaf (level == search_n) 必須直接更新 ub。
            // astar 主循環的 child 分支只對 level+1 == search_n 觸發 extend_to_full_mapping；
            // 對 level == search_n 的 leaf 走 else 分支會立即 continue 跳過 ub 更新。

            if (node->lower_bound >= upper_bound + margin) {
                prof_reuse_leaves_pruned++;
                // 剪枝：lb >= upper_bound + margin
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
                    prof_reuse_leaves_invalid++;
                    // 无效节点
                    if (node->parent) {
                        --node->parent->cs_cnt;
                        if (node->parent->cs_cnt == 0 && node->parent != dummy) {
                            add_to_pool(node->parent);
                        }
                    }
                } else if (node->lower_bound < upper_bound) {
                    prof_reuse_leaves_to_heap++;
                    // 继续搜索
                    node->reused_leaf = true;
                    add_to_heap(node, heap_n, open_heap);
                } else {
                    prof_reuse_leaves_boundary++;
                    // 边界节点：lb在[upper_bound, upper_bound + margin)范围内
                    if(node->level != DUMMY_VAL && node->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = node->lower_bound;
                    }
                    protect_ancestor_chain(node);
                    boundary_nodes.push_back(node);
                }
            }
        }
    }
    
    auto _prof_t4 = std::chrono::steady_clock::now();
    prof_reuse_leaves_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_prof_t4 - _prof_t3).count() / 1000.0;

    /*------------------------------------------------------------------*/
    /* 4. A*搜索主循环                                                  */
    /*------------------------------------------------------------------*/
    int cnt = 0;

    while (heap_n > 0 && open_heap[0]->lower_bound < upper_bound &&
           (exact_value_mode || verify_upper_bound == INF || upper_bound > verify_upper_bound))
    {
        cnt++;
        prof_reuse_astar_iter++;
        if(app_max_iter > 0 && cnt > app_max_iter) break;

        auto _it_t0 = std::chrono::steady_clock::now();
        State *now = open_heap[0];

        open_heap[0] = open_heap[--heap_n];
        heap_top_down(0, heap_n, open_heap);

        auto _it_t1 = std::chrono::steady_clock::now();
        prof_reuse_astar_pop_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_it_t1 - _it_t0).count() / 1000.0;

        // Lazy compute
        if (now->need_recompute) {
            if (now->image >= g_n) {
                now->need_recompute = false;
                if (now->cs_cnt == 0) {
                    add_to_pool(now);
                }
                {
                    auto _t = std::chrono::steady_clock::now();
                    prof_reuse_astar_recompute_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_t - _it_t1).count() / 1000.0;
                }
                continue;
            }

            ui old_lb = now->lower_bound;
            recompute_node_lazy(now);

            if (now->lower_bound >= upper_bound) {
                prof_reuse_astar_recomp_pruned++;
                if(now->level != DUMMY_VAL && now->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = now->lower_bound;
                }

                if (now->lower_bound < upper_bound + margin) {
                    // 边界节点
                    protect_ancestor_chain(now);
                    boundary_nodes.push_back(now);
                } else {
                    // 剪枝
                    if (now->cs_cnt == 0) {
                        add_to_pool(now);
                    }
                }
                {
                    auto _t = std::chrono::steady_clock::now();
                    prof_reuse_astar_recompute_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_t - _it_t1).count() / 1000.0;
                }
                continue;
            }

#if PUSH_BACK_AFTER_RECOMPUTE
            if (now->image < g_n) {
                add_to_heap(now, heap_n, open_heap);
            }
            {
                auto _t = std::chrono::steady_clock::now();
                prof_reuse_astar_recompute_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_t - _it_t1).count() / 1000.0;
            }
            continue;
#endif
        }

        auto _it_t2 = std::chrono::steady_clock::now();
        prof_reuse_astar_recompute_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_it_t2 - _it_t1).count() / 1000.0;

        // 生成兄弟节点
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
                // 继续搜索
                add_to_heap(sib, heap_n, open_heap);
                if (now->parent) ++now->parent->cs_cnt;
                if (sib->parent) children_of(sib->parent).push_back(sib);
            } else if (sib->lower_bound < upper_bound + margin) {
                // 边界节点：lb在[upper_bound, upper_bound + margin)范围内
                if(sib->level != DUMMY_VAL && sib->lower_bound < terminal_frontier_min_lb) {
                    terminal_frontier_min_lb = sib->lower_bound;
                }
                protect_ancestor_chain(sib);
                boundary_nodes.push_back(sib);
                if (now->parent) ++now->parent->cs_cnt;
                if (sib->parent) children_of(sib->parent).push_back(sib);
            } else {
                // 剪枝：lb >= upper_bound + margin
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
            // 无效节点
#ifdef _EXPAND_ALL_
            put_a_sibling_to_pool(sib->siblings);
            sib->siblings = nullptr;
#else
            --now->cs_cnt;
#endif
            put_a_state_to_pool(sib);
        }

        auto _it_t3 = std::chrono::steady_clock::now();
        prof_reuse_astar_sibling_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_it_t3 - _it_t2).count() / 1000.0;

        // child/full-mapping分支
        if (now->level + 1 == search_n) {
            // 完整映射
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
                if (now->cs_cnt == 0) {
                    add_to_pool(now);
                }
                {
                    auto _t = std::chrono::steady_clock::now();
                    prof_reuse_astar_child_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_t - _it_t3).count() / 1000.0;
                }
                continue;
            }

            State *child = get_a_new_state_node();
            if (!child) {
                if (now->cs_cnt == 0) {
                    add_to_pool(now);
                }
                {
                    auto _t = std::chrono::steady_clock::now();
                    prof_reuse_astar_child_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_t - _it_t3).count() / 1000.0;
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
                    // 继续搜索
                    add_to_heap(child, heap_n, open_heap);
                    children_of(child->parent).push_back(child);
                } else if (child->lower_bound < upper_bound + margin) {
                    // 边界节点：lb在[upper_bound, upper_bound + margin)范围内
                    if(child->level != DUMMY_VAL && child->lower_bound < terminal_frontier_min_lb) {
                        terminal_frontier_min_lb = child->lower_bound;
                    }
                    protect_ancestor_chain(child);
                    boundary_nodes.push_back(child);
                    children_of(child->parent).push_back(child);
                } else {
                    // 剪枝：lb >= upper_bound + margin
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
                // 无效节点
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

        auto _it_t4 = std::chrono::steady_clock::now();
        prof_reuse_astar_child_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_it_t4 - _it_t3).count() / 1000.0;

        if (now->cs_cnt == 0) {
            add_to_pool(now);
        }

        // 懒惰删除
        while (heap_n > 0 && open_heap[heap_n-1]->lower_bound >= upper_bound) {
            --heap_n;
            if (open_heap[heap_n]->cs_cnt == 0) {
                add_to_pool(open_heap[heap_n]);
            }
        }

        auto _it_t5 = std::chrono::steady_clock::now();
        prof_reuse_astar_bookkeep_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_it_t5 - _it_t4).count() / 1000.0;
    }
    
    auto _prof_t5 = std::chrono::steady_clock::now();
    prof_reuse_astar_us += std::chrono::duration_cast<std::chrono::nanoseconds>(_prof_t5 - _prof_t4).count() / 1000.0;
    prof_reuse_calls++;
    prof_reuse_snapshot_nodes += snap.v.size();

    // 设置最终的overall_lb
    if(heap_n > 0) {
        overall_lb = open_heap[0]->lower_bound;  // 活跃前沿
    } else if(terminal_frontier_min_lb < INF){
        overall_lb = terminal_frontier_min_lb;   // 终止前沿
    }

    // 保护heap中剩余节点的祖先链
    for (ui i = 0; i < heap_n; ++i) {
        protect_ancestor_chain(open_heap[i]);
    }
    open_heap.resize(heap_n);

    if (upper_bound > tau) {
        ++prof_reuse_astar_exhausted;
        last_reuse_exit_reason = 3;
    }
    return upper_bound;
}
