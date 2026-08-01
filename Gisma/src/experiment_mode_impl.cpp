// Experiment mode implementation
// This will be included in main.cpp

// 全局覆盖集合，供 GismaSearchEngine 使用
std::unordered_set<int> g_ept_coverage;
std::unordered_set<int> g_extra_coverage;

// Reuse logging (used by GismaSearchEngine.cpp)
FILE* g_reuse_log = nullptr;
std::mutex g_reuse_log_mutex;

// E7: search-time memory measurement (Linux). Only called when config.e7_stats.
#ifndef _WIN32
#include <sys/resource.h>
#include <unistd.h>
static long getCurrentRSS_kb() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long total_pages = 0, rss_pages = 0;
    if (fscanf(f, "%ld %ld", &total_pages, &rss_pages) != 2) rss_pages = 0;
    fclose(f);
    return rss_pages * (sysconf(_SC_PAGESIZE) / 1024);
}
static long getPeakRSS_kb() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  // Linux: kilobytes
}
#else
static long getCurrentRSS_kb() { return 0; }   // Windows stub (E7 mem stats not needed for select-alpha)
static long getPeakRSS_kb() { return 0; }
#endif

// Task structure for mixed scheduling
struct ExperimentTask {
    std::string method;
    double tau;
    int query_idx;
    int query_local_idx;
};

void experiment_mode(const Config& config) {
    printf("[experiment_mode] Starting experiment mode...\n");
    long e7_rss_baseline_kb = 0;  // E7: RSS baseline (KB) after index load; set when config.e7_stats

    // 1. Parse tau_values
    if (config.tau_values.empty()) {
        std::cerr << "[ERROR] tau_values is empty. Please specify tau values like --tau_values \"2,4,6,8\"" << std::endl;
        return;
    }

    std::vector<double> tau_list;
    std::stringstream ss(config.tau_values);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            double tau_val = std::stod(item);
            tau_list.push_back(tau_val);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Invalid tau value: " << item << std::endl;
            return;
        }
    }

    printf("[experiment_mode] Testing %zu tau values: ", tau_list.size());
    for (size_t i = 0; i < tau_list.size(); ++i) {
        printf("%.1f%s", tau_list[i], (i < tau_list.size() - 1) ? ", " : "\n");
    }

    // 2. Parse methods (if specified)
    std::vector<std::string> methods_list;
    if (!config.methods.empty()) {
        std::stringstream ss_methods(config.methods);
        std::string method_item;
        while (std::getline(ss_methods, method_item, ',')) {
            // Trim whitespace
            method_item.erase(0, method_item.find_first_not_of(" \t"));
            method_item.erase(method_item.find_last_not_of(" \t") + 1);
            methods_list.push_back(method_item);
        }
        printf("[experiment_mode] Testing %zu methods: ", methods_list.size());
        for (size_t i = 0; i < methods_list.size(); ++i) {
            printf("%s%s", methods_list[i].c_str(), (i < methods_list.size() - 1) ? ", " : "\n");
        }
    } else {
        methods_list.push_back(config.search_method);
        printf("[experiment_mode] Using single method: %s\n", config.search_method.c_str());
    }

    // 3. Load database (shared by all methods)
    printf("[experiment_mode] Loading database graphs...\n");

    std::vector<std::shared_ptr<Node>> db_node_list;
    std::vector<Graph*> db;
    std::map<std::string, ui> vM, eM;

    auto load_start = std::chrono::high_resolution_clock::now();
    ui max_db_n = Utility::load_db(config.db_name.c_str(), db, vM, eM);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);

    printf("[experiment_mode] Database graphs loaded: %zu graphs in %ld ms\n", db.size(), load_duration.count());
    Graph::FEATURE_DIM = vM.size();

    // Convert database graphs to Node format
    db_node_list.resize(db.size());
    for (size_t i = 0; i < db.size(); ++i) {
        auto graph_ptr = db[i];
        auto node = std::make_shared<Node>();
        node->node_id = i;
        node->graph = std::make_shared<Graph>(*graph_ptr);
        node->file_name = graph_ptr->id;
        node->is_anchor = false;
        node->nearest_anchor = -1;
        node->nearest_anchor_dist = std::numeric_limits<double>::infinity();
        db_node_list[i] = node;
    }

    printf("[experiment_mode] Database nodes conversion completed\n");

    // E11 masked deletion (FreshDiskANN lazy deletion [Singh et al. 2021]): 标记 deleted_frac 比例的 db 图为已删除。
    // 删除成本 = O(1)/图(仅加入集合，不动索引结构)；搜索时导航仍用，最后从结果集过滤掉。
    std::set<int> deleted_ids;
    if (config.deleted_frac > 0.0) {
        auto del_t0 = std::chrono::high_resolution_clock::now();
        std::vector<int> all_ids; all_ids.reserve(db.size());
        for (auto* g : db) { try { all_ids.push_back(std::stoi(g->id)); } catch (...) {} }
        std::mt19937 rng(12345u);
        std::shuffle(all_ids.begin(), all_ids.end(), rng);
        size_t k = (size_t)(config.deleted_frac * all_ids.size());
        for (size_t i = 0; i < k && i < all_ids.size(); ++i) deleted_ids.insert(all_ids[i]);
        auto del_t1 = std::chrono::high_resolution_clock::now();
        double del_s = std::chrono::duration<double>(del_t1 - del_t0).count();
        printf("[E11-delete] masked %zu/%zu graphs deleted (frac=%.3f) in %.6fs = %.4f us/op (O(1) lazy mask; FreshDiskANN-style)\n",
               deleted_ids.size(), db.size(), config.deleted_frac, del_s, k ? del_s * 1e6 / (double)k : 0.0);
    }

    // 4. Load ground truth (shared by all methods)
    printf("[experiment_mode] Loading ground truth...\n");
    std::map<int, std::map<double, std::vector<int>>> ground_truth;
    if (!Utility::load_exact_ground_truth(config.ground_truth_path, ground_truth)) {
        std::cerr << "[ERROR] Failed to load ground truth from " << config.ground_truth_path << std::endl;
        return;
    }
    printf("[experiment_mode] Ground truth loaded successfully: %zu queries\n", ground_truth.size());

    // E11: 删除后 recall 应在【存活】数据库上度量(FreshDiskANN-style recall-after-deletion)。
    // 故把被删图也从 ground truth 剔除——否则分母仍含不可返回的真answer，recall 机械性掉约 deleted_frac。
    // 与下方 results 的 deleted_ids 过滤对称(同一 db-id 空间)。
    if (!deleted_ids.empty()) {
        size_t removed = 0, total = 0;
        for (auto& q : ground_truth) {
            for (auto& tg : q.second) {
                auto& v = tg.second;
                size_t before = v.size();
                v.erase(std::remove_if(v.begin(), v.end(),
                        [&deleted_ids](int id){ return deleted_ids.count(id) > 0; }), v.end());
                removed += before - v.size(); total += before;
            }
        }
        printf("[E11-delete] ground truth pruned: removed %zu/%zu gt answers in deleted set "
               "(recall now measured over surviving DB)\n", removed, total);
    }

    // 5. Load query graphs (shared by all methods)
    printf("[experiment_mode] Loading query graphs...\n");
    std::vector<std::shared_ptr<Node>> query_node_list;
    std::vector<Graph*> query_db;
    ui max_query_n = Utility::load_db(config.query_name.c_str(), query_db, vM, eM);
    printf("[experiment_mode] Query graphs loaded: %zu graphs\n", query_db.size());

    int q_start = config.q_start;
    int q_end = config.q_end;
    if (q_start == -1) q_start = 0;
    if (q_end == -1) q_end = query_db.size() - 1;

    int num_queries = q_end - q_start + 1;
    query_node_list.resize(num_queries);

    for (int idx = 0; idx < num_queries; ++idx) {
        int i = q_start + idx;
        auto graph_ptr = query_db[i];
        auto node = std::make_shared<Node>();
        node->node_id = i;
        node->graph = std::make_shared<Graph>(*graph_ptr);
        node->file_name = graph_ptr->id;
        node->is_anchor = false;
        node->nearest_anchor = -1;
        node->nearest_anchor_dist = std::numeric_limits<double>::infinity();
        query_node_list[idx] = node;
    }

    printf("[experiment_mode] Query range: [%d, %d] (%zu queries)\n", q_start, q_end, query_node_list.size());

    // 6. Conditionally load NetDag and EPT (needed for Gisma, Base+GS, Base+SS, Base_All_EPT)
    bool need_gisma_resources = false;
    for (const auto& method : methods_list) {
        if (method == "Gisma" || method == "Gisma-default" || method == "Base+GS" || method == "Base+SS" || method == "Base_All_EPT"
            || method == "astar-lsa" || method == "app-lsa" || method == "gisma-lsa") {  // E8 verifier methods 也是 Gisma 搜索，需要 NetDag/EPT
            need_gisma_resources = true;
            break;
        }
    }

    std::shared_ptr<NetDag> netdag_ptr;
    std::shared_ptr<EditPathTreeManager> ept_manager;

    if (need_gisma_resources) {
        printf("[experiment_mode] Loading NetDag (needed for Gisma/Base+GS/Base+SS)...\n");
        double alpha = config.alpha;
        double tau_index = config.tau_index;
        double error_tolerance_index = config.error_tolerance_index;

        std::string index_name = config.index_name.empty()
            ? Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db.size())
            : config.index_name;
        netdag_ptr = std::make_shared<NetDag>();

        std::string dataset_dir = "./NetDags/" + config.dataset + "/";
        std::string reassigned_dir = dataset_dir + "reassigned/";
        std::string reassigned_file_path = reassigned_dir + index_name + ".dat";

        netdag_ptr->load_from_file(*netdag_ptr, reassigned_file_path);
        printf("[experiment_mode] NetDag loaded successfully\n");

        // Fill anchor vectors
        printf("[experiment_mode] Filling anchor vectors from queues...\n");
        auto fill_start = std::chrono::high_resolution_clock::now();
        for (auto & anchorPtr : netdag_ptr->anchors) {
            anchorPtr->fill_vectors_from_queues();
        }
        auto fill_end = std::chrono::high_resolution_clock::now();
        auto fill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fill_end - fill_start);
        printf("[experiment_mode] Anchor vectors filled in %ld ms\n", fill_duration.count());

        // Load embeddings into nodes
        printf("[experiment_mode] Loading embedding vectors into nodes...\n");
        std::string embedding_file = "./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin";
        std::vector<std::vector<float>> embeddings;
        if (Utility::load_embeddings(embedding_file, embeddings)) {
            printf("[experiment_mode] Embedding vectors loaded successfully from %s\n", embedding_file.c_str());
            printf("[experiment_mode] Loaded %zu embeddings, assigning to %zu nodes...\n", embeddings.size(), netdag_ptr->nodes.size());

            // Assign embeddings to nodes
            for (size_t i = 0; i < embeddings.size() && i < netdag_ptr->nodes.size(); ++i) {
                if (netdag_ptr->nodes[i]) {
                    netdag_ptr->nodes[i]->embedding = embeddings[i];
                }
            }
            printf("[experiment_mode] Embeddings assigned to nodes successfully\n");
        } else {
            printf("[experiment_mode] Warning: Failed to load embeddings from %s\n", embedding_file.c_str());
            printf("[experiment_mode] Embedding distance checks will be disabled\n");
        }

        // Load EPT
        printf("[experiment_mode] Loading EPT (shared across all tasks)...\n");
        ept_manager = std::make_shared<EditPathTreeManager>();
        std::string ept_directory_path = "./EPFs/" + index_name + "/";
        ept_manager->load_all_epts_from_directory_parallel(ept_directory_path);
        printf("[experiment_mode] EPT loaded successfully\n");

        // 统计整个数据库的 EPT 和 Extra 覆盖范围，存储到全局变量
        g_ept_coverage.clear();
        g_extra_coverage.clear();
        for (const auto& anchor : netdag_ptr->anchors) {
            // EPT 覆盖：anchor 本身 + 所有 tree_node 的 completed_db_graph_ids
            g_ept_coverage.insert(anchor->node_id);
            EditPathTree* ept = ept_manager->get_ept(anchor->node_id);
            if (ept) {
                for (const auto& tree_node : ept->tree_nodes) {
                    for (int id : tree_node.completed_db_graph_ids) {
                        g_ept_coverage.insert(id);
                    }
                }
            }
            // Extra 覆盖：nodes_in_cluster
            auto cluster_copy = anchor->nodes_in_cluster;
            while (!cluster_copy.empty()) {
                g_extra_coverage.insert(cluster_copy.top().second);
                cluster_copy.pop();
            }
        }
        std::unordered_set<int> total_union = g_ept_coverage;
        total_union.insert(g_extra_coverage.begin(), g_extra_coverage.end());
        printf("[experiment_mode] Database Coverage: EPT=%zu graphs, Extra=%zu graphs, Union=%zu graphs (total DB=%zu)\n",
               g_ept_coverage.size(), g_extra_coverage.size(), total_union.size(), db.size());
    } else {
        printf("[experiment_mode] Skipping NetDag and EPT loading (not needed for App-BMao/AStar-BMao)\n");
    }

    // E11 #2(插入前 recall): 移除 10% 叶子图 + 从 GT 剔除它们(不在库里就不该算应找到的答案)。
    //   必须在 searcher 创建前做(searcher 会拷贝 GT)。之后 recall = over 存活 DB。
    if (config.insert_remove_only && ept_manager) {
        std::vector<int> rm_leaves = ept_manager->collect_leaf_db_ids();
        std::sort(rm_leaves.begin(), rm_leaves.end());
        rm_leaves.erase(std::unique(rm_leaves.begin(), rm_leaves.end()), rm_leaves.end());
        std::mt19937 rm_rng(777u);
        std::shuffle(rm_leaves.begin(), rm_leaves.end(), rm_rng);
        int rm_want = std::min((int)rm_leaves.size(), config.insert_count);
        std::unordered_set<int> rm_S(rm_leaves.begin(), rm_leaves.begin() + rm_want);
        std::unordered_set<int> rm_removed;
        for (ui aid : ept_manager->all_anchor_ids()) {
            EditPathTree* e = ept_manager->get_ept(aid);
            if (!e) continue;
            for (int gid : e->remove_db_graph_ids(rm_S)) rm_removed.insert(gid);
        }
        size_t gt_pruned = 0, gt_total = 0;
        for (auto& q : ground_truth) for (auto& tg : q.second) {
            auto& v = tg.second; size_t before = v.size();
            v.erase(std::remove_if(v.begin(), v.end(), [&](int id){ return rm_removed.count(id) > 0; }), v.end());
            gt_pruned += before - v.size(); gt_total += before;
        }
        printf("[E11-remove-only] removed %zu unique leaf graphs + pruned %zu/%zu GT answers. recall below = over SURVIVING DB(GT 已剔除).\n",
               rm_removed.size(), gt_pruned, gt_total);
        fflush(stdout);
    }

    // 7. Create GismaSearchEngine for each method
    std::vector<double> ged_matrix;
    bool has_ged_matrix = false;

    std::map<std::string, std::shared_ptr<GismaSearchEngine>> searchers;
    for (const auto& method : methods_list) {
        printf("[experiment_mode] Creating GismaSearchEngine for %s...\n", method.c_str());
        // Gisma-default: always use hard-coded default values, ignore command line overrides
        bool use_ept_filters_for_method = !config.disable_ept_filters;
        std::string nd_mode_for_method = config.nd_mode;
        std::string dfs_mode_for_method = config.dfs_mode;
        int max_ged_gap_for_method = config.max_ged_gap;
        int max_margin_for_method = config.max_margin;
        if (method == "Gisma-default") {
            use_ept_filters_for_method = true;
            nd_mode_for_method = "filters";
            dfs_mode_for_method = "";  // empty = default (all optimizations)
            max_ged_gap_for_method = 3;
            max_margin_for_method = 3;
        }

        auto searcher = std::make_shared<GismaSearchEngine>(
            netdag_ptr,
            config.tau_index,
            config.error_tolerance_search,
            q_start,
            q_end,
            has_ged_matrix,
            ged_matrix,
            method,  // keep original method name
            config.dataset,
            db_node_list,
            query_node_list,
            db_node_list.size(),
            db,
            ground_truth,
            vM,
            eM,
            max_db_n,
            ept_manager.get(),
            nd_mode_for_method,
            dfs_mode_for_method,
            use_ept_filters_for_method,
            config.only_compute_db_graph,
            config.app_max_iter,
            !config.disable_fast_down,
            config.exact_max_iter,
            config.nd_filter_ratio,
            config.disable_lsa_pruning,
            config.disable_reuse_lsa,
            (method == "Gisma" || method == "Gisma-default") ? config.verify_reuse : false,
            config.chain_reuse
        );
        searcher->max_ged_gap = max_ged_gap_for_method;
        searcher->max_margin = max_margin_for_method;
        searcher->all_edge_labels_same = config.all_edge_labels_same;
        searcher->exact_value_mode = config.exact_value_mode;
        searcher->early_stop_at_tau = config.early_stop_at_tau;
        // E8 verifier 作为一等 method：astar-lsa/app-lsa/gisma-lsa 各自映射到对应 ged_algorithm，
        // 使一次 --methods 就能在同一 invocation 里对比多个 verifier（每 method 独立 searcher，无并发问题）。
        // E8 verifier 完全由 method 名决定（--ged_algorithm 已删除）。
        if      (method == "astar-lsa") searcher->ged_algorithm = "LSa";        // 精确 LSa（exact）
        else if (method == "app-lsa")   searcher->ged_algorithm = "app-lsa";    // 近似 LSa，无 reuse
        else if (method == "gisma-lsa") searcher->ged_algorithm = "gisma-lsa";  // 近似 LSa + search-tree reuse
        else                            searcher->ged_algorithm = "App";        // 默认 App-BMao（reuse）
        searcher->e7_stats = config.e7_stats;  // E7: gated search-time stats
        searchers[method] = searcher;
    }
    printf("[experiment_mode] All GismaSearchEngines created successfully\n");
    if (config.e7_stats) e7_rss_baseline_kb = getCurrentRSS_kb();  // E7: RSS baseline after index/searchers loaded, before queries

    // E11 insert 性能：挑 insert_count 个 EPT 叶子图(代表真实可插入的图)，用 GS_search 定位(=我们的 insert)逐个插入计时。
    if (config.measure_insert && ept_manager) {
        std::vector<int> leaves = ept_manager->collect_leaf_db_ids();
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        std::mt19937 rng(777u);
        std::shuffle(leaves.begin(), leaves.end(), rng);
        int want = std::min((int)leaves.size(), config.insert_count);
        // id -> db 索引
        std::unordered_map<int,int> id2idx;
        for (size_t i = 0; i < db.size(); ++i) { try { id2idx[std::stoi(db[i]->id)] = (int)i; } catch (...) {} }
        auto& ins = searchers.count("Gisma") ? searchers.at("Gisma") : searchers.begin()->second;
        double ins_tau = config.tau_index;
        double tot_t = 0.0; long long tot_ndc = 0; int done = 0;
        for (int j = 0; j < want; ++j) {
            auto it = id2idx.find(leaves[j]);
            if (it == id2idx.end()) continue;
            auto qnode = db_node_list[it->second];
            SearchStats st;
            auto t0 = std::chrono::high_resolution_clock::now();
            ins->GS_search(qnode, ins_tau, st);          // 我们的 insert = giant-step 定位 (HNSW: insert=search)
            auto t1 = std::chrono::high_resolution_clock::now();
            tot_t += std::chrono::duration<double>(t1 - t0).count();
            tot_ndc += st.ND_ndc_count;
            done++;
        }
        printf("[E11-insert] placed %d EPT-leaf graphs via GS_search (tau=%.0f): avg %.6fs/graph, %.1f NDC(balls)/graph. insert=search [HNSW Malkov2018].\n",
               done, ins_tau, done ? tot_t / done : 0.0, done ? (double)tot_ndc / done : 0.0);
        printf("[E11-insert] (total leaves available=%zu)\n", leaves.size());
    }

    // E11 insert 稳定性 round-trip：把 insert_count 个 EPT 叶子图当作"还没插入"，加入 deleted_ids
    // (从搜索结果排除)但【不】剔除 ground truth → 本次 recall = "插入前"(这些图缺失，queries 找不到)。
    // baseline(本 flag 关) = "插入后" = GS_search 把图放回正确 ball 后可检索 → recall 回到 baseline。
    // 与 measure_insert 同种子(777) → 同一批 100 图，保证"插入成本"和"插入稳定性"指的是同一组。
    if (config.insert_stability && ept_manager) {
        std::vector<int> leaves = ept_manager->collect_leaf_db_ids();
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        std::mt19937 rng(777u);
        std::shuffle(leaves.begin(), leaves.end(), rng);
        int want = std::min((int)leaves.size(), config.insert_count);
        for (int j = 0; j < want; ++j) deleted_ids.insert(leaves[j]);
        printf("[E11-insert-stab] excluding %d EPT-leaf graphs from results (GT NOT pruned): "
               "recall now = 'before insert'; baseline run = 'after insert'. leaves available=%zu\n",
               want, leaves.size());
    }

    // E11 真·增量插入: 把 insert_count 个 EPT 叶子图从其所在 EPT 节点移除，再用插入算法真正改写索引结构
    //   (算 anchor->g 精确 GED 作距离, flat 挂回 root 分支)，然后在【修改后的增量索引】上跑搜索测 recall。
    //   回答 R5 D2 "增量索引 != 从头构建的 paper 索引" —— recall 在真实插入后的索引上度量(不是 baseline 顶替)。
    if ((config.insert_rebuild || config.insert_probe) && ept_manager) {
        // 1) 选 10% 叶子图(seed 777)
        std::vector<int> ir_leaves = ept_manager->collect_leaf_db_ids();
        std::sort(ir_leaves.begin(), ir_leaves.end());
        ir_leaves.erase(std::unique(ir_leaves.begin(), ir_leaves.end()), ir_leaves.end());
        std::mt19937 ir_rng(777u);
        std::shuffle(ir_leaves.begin(), ir_leaves.end(), ir_rng);
        int ir_want = std::min((int)ir_leaves.size(), config.insert_count);
        std::unordered_set<int> ir_S(ir_leaves.begin(), ir_leaves.begin() + ir_want);

        // 2) 移除阶段: 从所有 EPT 删掉 S(g 可能在多个 anchor 的 EPT 里), 收集唯一被删图
        //    诊断: 移除前记录每个 g 的家 anchor + 它在原 EPT 里的深度(level)
        std::unordered_set<int> ir_uniq;
        std::unordered_map<int, std::vector<std::pair<ui,int>>> ir_home;  // gid -> [(anchor, level)]
        for (ui aid : ept_manager->all_anchor_ids()) {
            EditPathTree* ept = ept_manager->get_ept(aid);
            if (!ept) continue;
            for (size_t ni = 0; ni < ept->tree_nodes.size(); ++ni)
                for (int cid : ept->tree_nodes[ni].completed_db_graph_ids)
                    if (ir_S.count(cid)) ir_home[cid].push_back({aid, ept->tree_nodes[ni].level});
            for (int gid : ept->remove_db_graph_ids(ir_S)) ir_uniq.insert(gid);
        }

        // 3) id -> db 索引 + 自己 load 一份 embeddings(按 db 下标, 行 i = db[i])
        std::unordered_map<int,int> ir_id2idx;
        for (size_t i = 0; i < db.size(); ++i) { try { ir_id2idx[std::stoi(db[i]->id)] = (int)i; } catch (...) {} }
        std::vector<std::vector<float>> ir_emb;
        Utility::load_embeddings("./embeddings/" + config.dataset + "/" + config.dataset + "_embeddings.bin", ir_emb);
        const std::vector<float> empty_emb;
        auto& ins = searchers.count("Gisma") ? searchers.at("Gisma") : searchers.begin()->second;
        auto sq_l2 = [](const std::vector<float>& a, const std::vector<float>& b)->double {
            if (a.empty() || a.size() != b.size()) return 1e18;
            double s = 0; for (size_t i = 0; i < a.size(); ++i) { double d = (double)a[i]-b[i]; s += d*d; } return s; };

        // 4) 真插入(批量): Phase A 并行算每个图的 (anchor, edit ops)(GS_search+embedding选anchor+AppForComputation精确edit path,
        //    彼此独立、最贵); Phase B 串行 merge 进树(必须串行, 改共享 EPT 结构)。
        std::vector<int> uniq_vec(ir_uniq.begin(), ir_uniq.end());
        struct InsertJob { ui anchor; int gid; std::vector<EditOperation> ops; double secs; bool ok; bool is_extra; int ged; double emb_chosen; double emb_home; int home_in_cand; };
        std::vector<InsertJob> ir_jobs(uniq_vec.size());
        std::atomic<size_t> ir_next(0);
        unsigned int ir_nw = config.use_parallel
            ? std::max(1u, (config.num_workers > 0 ? (unsigned int)config.num_workers : std::thread::hardware_concurrency()))
            : 1u;
        auto ir_t0 = std::chrono::high_resolution_clock::now();
        {
            std::vector<std::future<void>> ir_fts;
            for (unsigned int w = 0; w < ir_nw; ++w) {
                ir_fts.emplace_back(std::async(std::launch::async, [&]() {
                    size_t i;
                    while ((i = ir_next.fetch_add(1)) < uniq_vec.size()) {
                        InsertJob& J = ir_jobs[i]; J.ok = false; J.gid = uniq_vec[i]; J.secs = 0;
                        J.emb_chosen = -1; J.emb_home = -1; J.home_in_cand = 0;
                        auto it = ir_id2idx.find(J.gid); if (it == ir_id2idx.end()) continue;
                        int gidx = it->second; Graph* g_graph = db[gidx];
                        auto jt0 = std::chrono::high_resolution_clock::now();
                        SearchStats st;
                        auto cands = ins->GS_search(db_node_list[gidx], 20.0, st);  // 候选 tau >= EPT 赋值半径
                        const std::vector<float>& emb_g = (gidx < (int)ir_emb.size()) ? ir_emb[gidx] : empty_emb;
                        // 家在不在候选 + 家/选中的 embedding 距离(不依赖 GED, 探测用)
                        { std::unordered_set<int> cs; for (auto& c : cands) cs.insert(std::get<0>(c));
                          auto hit2 = ir_home.find(J.gid);
                          if (hit2 != ir_home.end()) for (auto& hp : hit2->second) {
                              ui ha = hp.first; if (cs.count((int)ha)) J.home_in_cand = 1;
                              auto hx = ir_id2idx.find((int)ha);
                              if (hx != ir_id2idx.end() && hx->second < (int)ir_emb.size()) {
                                  double hd = sq_l2(ir_emb[hx->second], emb_g);
                                  if (J.emb_home < 0 || hd < J.emb_home) J.emb_home = hd;
                              }
                          } }
                        J.secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - jt0).count();
                        if (config.insert_probe) { J.ok = true; continue; }  // 探测模式: 到此为止, 不算 GED/不插入
                        if (cands.empty()) continue;
                        long long best_a = -1; double best_d = 1e18;
                        for (auto& c : cands) {
                            int aid = std::get<0>(c); auto ait = ir_id2idx.find(aid);
                            if (ait == ir_id2idx.end() || ait->second >= (int)ir_emb.size()) continue;
                            const std::vector<float>& ea = ir_emb[ait->second];
                            if (ea.empty() || ea.size() != emb_g.size()) continue;
                            double d = sq_l2(ea, emb_g); if (d < best_d) { best_d = d; best_a = aid; }
                        }
                        if (best_a < 0) continue;
                        J.emb_chosen = best_d;
                        auto a_it = ir_id2idx.find((int)best_a);
                        if (a_it == ir_id2idx.end()) continue;
                        Graph* anchor_graph = db[a_it->second];
                        if (!anchor_graph) continue;
                        // max_exact_ged_for_EPT 默认 -1 表示 alpha+4(与 main.cpp 构建逻辑一致), experiment 模式未自动调整
                        double mxg = (config.max_exact_ged_for_EPT < 0) ? (config.alpha + 4.0) : config.max_exact_ged_for_EPT;
                        ui thr = (ui)mxg;
                        J.anchor = (ui)best_a;
                        // 照构建两步法: 先 LB 过滤(便宜)。LB>thr → 直接 Extra(杀掉 GED>17 的 73s 爆炸, 不算精确)。
                        ui lb = anchor_graph->ged_lower_bound_filter(g_graph, thr, vM.size(), eM.size(), max_db_n);
                        if (lb > thr) {
                            J.is_extra = true; J.ged = (int)lb;
                        } else {
                            // LB<=thr → 算精确 GED + edit path(已验证正确的方法: tau 大, get_mapping + 过滤)
                            const Graph* sm = (anchor_graph->n < g_graph->n) ? anchor_graph : g_graph;
                            const Graph* lg = (anchor_graph->n < g_graph->n) ? g_graph : anchor_graph;
                            // tau=thr → upper_bound=thr+1, AppForComputation 在 GED>thr 时封顶早停(避免算 GED-32 那种 65s)
                            Application app(thr, "BMao", 1000000);
                            app.set_all_edge_labels_same(config.all_edge_labels_same);
                            app.init(lg, sm);
                            int ged_ir = (int)app.AppForComputation(nullptr, nullptr);
                            J.ged = ged_ir;
                            if (ged_ir >= 0 && ged_ir <= (int)thr) {
                                std::vector<std::pair<ui,ui>> raw; app.get_mapping(raw);
                                std::vector<std::pair<ui,ui>> ir_map;
                                for (auto& pr : raw) if (pr.first < (ui)sm->n && pr.second < (ui)lg->n) ir_map.push_back(pr);
                                anchor_graph->compute_mapping_cost(*g_graph, ir_map, J.ops);
                                J.is_extra = false;
                            } else {
                                J.is_extra = true;   // 精确 GED>thr → Extra
                            }
                        }
                        J.ok = true;
                        auto jt1 = std::chrono::high_resolution_clock::now();
                        J.secs = std::chrono::duration<double>(jt1 - jt0).count();
                    }
                }));
            }
            for (auto& f : ir_fts) f.get();
        }
        auto ir_tA = std::chrono::high_resolution_clock::now();

        // 探测模式: 统计 home_in_cand=0 (GS_search 漏召回家 anchor) 的比例, 然后结束。
        if (config.insert_probe) {
            int probe_total = (int)uniq_vec.size(), probe_miss = 0;
            double sum_emb_home_miss = 0; int miss_with_emb = 0;
            for (auto& J : ir_jobs) {
                if (J.home_in_cand == 0) { probe_miss++; if (J.emb_home >= 0) { sum_emb_home_miss += J.emb_home; miss_with_emb++; } }
            }
            double probe_s = std::chrono::duration<double>(ir_tA - ir_t0).count();
            printf("[E11-insert-probe] %d removed graphs; home_in_cand=0 (GS_search 漏家): %d (%.2f%%); "
                   "%d 个家在候选里. probe wall=%.2fs (%u threads). 漏召回的家平均 embedding 距离=%.2f\n",
                   probe_total, probe_miss, probe_total ? 100.0 * probe_miss / probe_total : 0.0,
                   probe_total - probe_miss, probe_s, ir_nw, miss_with_emb ? sum_emb_home_miss / miss_with_emb : -1.0);
            return;
        }

        // Phase B (串行): GED<=thr → merge 进 EPT; GED>thr → push 进 anchor 的 Extra(nodes_in_cluster_vec)。
        std::vector<int> ir_inserted; std::unordered_set<ui> ir_touched;
        int ir_done = 0, ir_fail = 0, ir_extra = 0, ir_dbg = 0; long long ir_tot_ops = 0; double ir_work = 0;
        for (auto& J : ir_jobs) {
            if (!J.ok) { ir_fail++; continue; }
            if (ir_dbg < 8) { int olv = ir_home[J.gid].empty() ? -1 : ir_home[J.gid][0].second;
                fprintf(stderr, "[ir-detail] gid=%d a=%u GED=%d extra=%d pathlen=%zu origLevel=%d compute=%.3fs | embChosen=%.2f embHome=%.2f home_in_cand=%d\n",
                        J.gid, J.anchor, J.ged, (int)J.is_extra, J.ops.size(), olv, J.secs, J.emb_chosen, J.emb_home, J.home_in_cand); fflush(stderr); ir_dbg++; }
            ir_work += J.secs; ir_inserted.push_back(J.gid);
            if (J.is_extra) {
                // 进 Extra: 照构建, push 到 anchor 的 nodes_in_cluster_vec(第二元素 = db 下标)
                auto anchor_node = std::dynamic_pointer_cast<Anchor>(netdag_ptr->nodes[J.anchor]);
                if (anchor_node) { anchor_node->nodes_in_cluster_vec.push_back({(double)J.ged, ir_id2idx[J.gid]}); ir_extra++; ir_done++; }
                else ir_fail++;
            } else {
                EditPathTree* ept = ept_manager->get_ept(J.anchor);
                if (!ept) { ir_fail++; continue; }
                ept->insert_graph_merge(J.ops, db[ir_id2idx[J.gid]], J.gid);
                ir_touched.insert(J.anchor);
                ir_tot_ops += (long long)J.ops.size(); ir_done++;
            }
        }
        for (ui aid : ir_touched) { EditPathTree* e = ept_manager->get_ept(aid); if (e) e->precompute_max_subtree_depth(); }
        auto ir_t1 = std::chrono::high_resolution_clock::now();
        double ir_phaseA = std::chrono::duration<double>(ir_tA - ir_t0).count();
        double ir_phaseB = std::chrono::duration<double>(ir_t1 - ir_tA).count();
        printf("[E11-insert-rebuild] removed %zu, re-inserted %d (EPT=%d into %zu EPTs, Extra=%d) failed=%d. "
               "PhaseA(%u thr)=%.2fs PhaseB(serial)=%.4fs; compute=%.3fs/graph(work), batch wall=%.4fs/graph; EPT avg path len=%.2f.\n",
               ir_uniq.size(), ir_done, ir_done - ir_extra, ir_touched.size(), ir_extra, ir_fail, ir_nw, ir_phaseA, ir_phaseB,
               ir_done ? ir_work / ir_done : 0.0, ir_done ? (ir_phaseA + ir_phaseB) / ir_done : 0.0,
               (ir_done - ir_extra) ? (double)ir_tot_ops / (ir_done - ir_extra) : 0.0);
        fflush(stdout);
        // self-check 已删除: 用 db 成员自己当 query 是退化测法, recall(对 queries.txt) 才是真验证。
        (void)ir_inserted;
    }

    std::string base_exp_dir = "./experiment_results";
    std::string exp_dir;
    std::string latest_dir;

    if (config.save) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        #ifdef _WIN32
        localtime_s(&tm_now, &time_t_now);
        #else
        localtime_r(&time_t_now, &tm_now);
        #endif

        char timestamp[64];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm_now);

        // 时间戳目录（存档）：experiment_results/archive/{dataset}/{timestamp}/
        exp_dir = base_exp_dir + "/archive/" + config.dataset + "/" + timestamp;
        std::filesystem::create_directories(exp_dir);

        // latest 目录：experiment_results/latest/{dataset}/
        latest_dir = base_exp_dir + "/latest/" + config.dataset;
        std::filesystem::create_directories(latest_dir);

        printf("[experiment_mode] Results directory: %s\n", exp_dir.c_str());
        printf("[experiment_mode] Latest directory:  %s\n", latest_dir.c_str());
    } else {
        printf("[experiment_mode] --save not specified, skipping directory creation\n");
    }

    // 8b. Print experiment parameters summary to console
    printf("\n");
    printf("================================================================================\n");
    printf("                           EXPERIMENT PARAMETERS\n");
    printf("================================================================================\n");
    printf("Dataset:          %s\n", config.dataset.c_str());
    printf("DB size:          %zu\n", db.size());
    printf("Query range:      [%d, %d] (%d queries)\n", q_start, q_end, num_queries);
    printf("Methods:          %s\n", config.methods.empty() ? config.search_method.c_str() : config.methods.c_str());
    printf("Tau values:       %s\n", config.tau_values.c_str());
    printf("Alpha:            %.1f\n", config.alpha);
    printf("Tau index:        %.1f\n", config.tau_index);
    printf("App max iter:     %d\n", config.app_max_iter);
    printf("ND mode:          %s\n", config.nd_mode.c_str());
    printf("DFS mode:         %s\n", config.dfs_mode.empty() ? "unified" : config.dfs_mode.c_str());
    printf("Fast down:        %s\n", config.disable_fast_down ? "disabled" : "enabled");
    printf("Disable all LSa:  %s\n", config.disable_all_lsa ? "true" : "false");
    printf("  - LSa pruning:  %s\n", config.disable_lsa_pruning ? "disabled" : "enabled");
    printf("  - Reuse LSa:    %s\n", config.disable_reuse_lsa ? "disabled" : "enabled");
    printf("Parallel:         %s\n", config.use_parallel ? "true" : "false");
    printf("Reuse vs Baseline:%s\n", config.verify_reuse ? "true (compare reuse with A* baseline)" : "false");
    printf("Chain reuse:      %s\n", config.chain_reuse ? "true" : "false");
    printf("EPT filters:      %s\n", config.disable_ept_filters ? "disabled" : "enabled");
    printf("Only DB graph:    %s\n", (config.only_compute_db_graph || !config.disable_ept_filters) ? "true" : "false");
    printf("Save results:     %s\n", config.save ? "true" : "false");
    printf("================================================================================\n");
    printf("\n");

    // 9. Create all tasks (method × tau × query)
    std::vector<ExperimentTask> all_tasks;
    for (const auto& method : methods_list) {
        for (const auto& tau : tau_list) {
            for (int q_local_idx = 0; q_local_idx < num_queries; ++q_local_idx) {
                ExperimentTask task;
                task.method = method;
                task.tau = tau;
                task.query_idx = q_start + q_local_idx;
                task.query_local_idx = q_local_idx;
                all_tasks.push_back(task);
            }
        }
    }

    int total_tasks = all_tasks.size();
    printf("[experiment_mode] Created %d tasks (%zu methods x %zu tau x %d queries)\n",
           total_tasks, methods_list.size(), tau_list.size(), num_queries);

    // 10. Create subdirectories for each (method, tau) combination (skip if --save not specified)
    std::map<std::string, std::map<double, std::string>> subdirs;             // 时间戳目录
    std::map<std::string, std::map<double, std::string>> latest_subdirs;      // latest 目录 (results/ 子目录)
    std::map<std::string, std::map<double, std::string>> latest_base_subdirs; // latest 目录 (基础目录，用于写 summary)
    std::string timestamp_str;  // 保存时间戳字符串，用于写入 latest summary
    if (config.save) {
        // 从 exp_dir 提取时间戳（最后一个目录名）
        size_t last_slash = exp_dir.find_last_of("/\\");
        timestamp_str = (last_slash != std::string::npos) ? exp_dir.substr(last_slash + 1) : "";

        for (const auto& method : methods_list) {
            for (const auto& tau : tau_list) {
                std::ostringstream subdir_name;
                subdir_name << method << "_tau_" << std::fixed << std::setprecision(1) << tau;

                // 时间戳目录
                std::string subdir_path = exp_dir + "/" + subdir_name.str();
                std::filesystem::create_directories(subdir_path);
                subdirs[method][tau] = subdir_path;

                // latest 目录：基础目录 + results/ 子目录
                std::string latest_base_path = latest_dir + "/" + subdir_name.str();
                std::string latest_results_path = latest_base_path + "/results";
                std::filesystem::create_directories(latest_results_path);
                latest_base_subdirs[method][tau] = latest_base_path;
                latest_subdirs[method][tau] = latest_results_path;  // json 保存到 results/ 子目录
            }
        }
    }

    // 11. Execute all tasks in parallel (mixed scheduling)
    auto total_exp_start = std::chrono::high_resolution_clock::now();

    // Collect results for summary and optional saving
    struct TaskResult {
        std::string method;
        double tau;
        QueryDetails details;
    };
    std::vector<TaskResult> all_results;

    if (config.use_parallel) {
        unsigned int max_cores = std::thread::hardware_concurrency();
        if (max_cores == 0) max_cores = 1;
        unsigned int max_allowed = static_cast<unsigned int>(max_cores * 0.7);
        if (max_allowed == 0) max_allowed = 1;

        unsigned int num_threads;
        if (config.num_workers > 0) {
            // 使用用户指定的线程数，但不超过可用核心数的70%
            num_threads = std::min(static_cast<unsigned int>(config.num_workers), max_allowed);
        } else {
            // 自动使用可用核心数的70%
            num_threads = max_allowed;
        }

        printf("\n[experiment_mode] Using PARALLEL execution with DYNAMIC TASK QUEUE for %d threads and %d mixed tasks...\n",
               num_threads, total_tasks);

        // Open reuse log if verify_reuse is enabled
        if (config.verify_reuse) {
            g_reuse_log = fopen("reuse_log.csv", "w");
            if (g_reuse_log) {
                fprintf(g_reuse_log,
                        "query_id,anchor_id,parent_idx,child_idx,snapshot_size,ged_gap,reuse_us,baseline_us,ged,ged_baseline,tau\n");
                printf("[experiment_mode] Reuse log opened: reuse_log.csv\n");
            }
        }

        std::mutex print_mutex;
        std::atomic<int> completed_tasks(0);
        std::atomic<size_t> next_task_index(0);
        std::mutex results_mutex;

        // Create threads
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (unsigned int i = 0; i < num_threads; ++i) {
            futures.emplace_back(std::async(std::launch::async,
                [&all_tasks, &query_node_list, &searchers,
                 &print_mutex, &completed_tasks, total_tasks,
                 &all_results, &results_mutex, &next_task_index, &deleted_ids]() {

                    while (true) {
                        // Atomically get next task index
                        size_t task_idx = next_task_index.fetch_add(1, std::memory_order_relaxed);
                        if (task_idx >= (size_t)total_tasks) break;

                        const auto& task = all_tasks[task_idx];
                        auto query_node = query_node_list[task.query_local_idx];
                        auto& searcher = searchers.at(task.method);

                        // Execute search
                        SearchStats local_stats;
                        std::vector<int> results;

                        if (task.method == "Gisma" || task.method == "Gisma-default") {
                            results = searcher->Gisma_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Gisma-no-reuse") {
                            results = searcher->Gisma_no_reuse_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Gisma-only-dfs") {
                            results = searcher->Gisma_only_dfs_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Gisma-no-SP") {
                            results = searcher->Gisma_no_SP_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Gisma-no-LP") {
                            results = searcher->Gisma_no_LP_search(query_node, task.tau, local_stats);
                        } else if (task.method == "BMao_scan") {
                            results = searcher->BMao_scan_search(query_node, task.tau, local_stats);
                        } else if (task.method == "App-BMao") {
                            // App-BMao: original standalone implementation (full DB scan, no index/reuse)
                            results = searcher->App_BMao_orig_search(query_node, task.tau, local_stats);
                        } else if (task.method == "App-BMao-gisma") {
                            // App-BMao variant sharing Gisma's State (full DB scan)
                            results = searcher->App_BMao_search(query_node, task.tau, local_stats);
                        } else if (task.method == "App-BMao-orig") {
                            results = searcher->App_BMao_orig_search(query_node, task.tau, local_stats);
                        } else if (task.method == "AStar-BMao") {
                            results = searcher->AStar_BMao_search(query_node, task.tau, local_stats);
                        } else if (task.method == "AStar-scan") {
                            results = searcher->AStar_scan_search(query_node, task.tau, local_stats);
                        } else if (task.method == "AStar-scan-no-lsa") {
                            results = searcher->AStar_scan_no_lsa_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Base+GS") {
                            results = searcher->Base_GS_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Base+SS") {
                            results = searcher->Base_SS_search(query_node, task.tau, local_stats);
                        } else if (task.method == "Base_All_EPT") {
                            results = searcher->Base_All_EPT_search(query_node, task.tau, local_stats);
                        } else if (task.method == "astar-lsa") {
                            results = searcher->AStar_LSa_scan_search(query_node, task.tau, local_stats);  // full scan + exact LSa (no index)
                        } else if (task.method == "app-lsa") {
                            results = searcher->App_LSa_scan_search(query_node, task.tau, local_stats);    // full scan + approx LSa (no index)
                        } else if (task.method == "gisma-lsa") {
                            results = searcher->Gisma_search(query_node, task.tau, local_stats);           // full Gisma framework (index+reuse) + LSa
                        } else {
                            results = searcher->Gisma_search(query_node, task.tau, local_stats);
                        }

                        // E11 masked deletion: 从结果集过滤掉已删除的图(导航已用过它们)
                        if (!deleted_ids.empty() && !results.empty()) {
                            results.erase(std::remove_if(results.begin(), results.end(),
                                [&deleted_ids](int id){ return deleted_ids.count(id) > 0; }), results.end());
                        }

                        // Compute metrics
                        auto [has_gt, recall, precision, iou, intersection_count, ground_truth_size]
                            = searcher->compute_recall_precision_IoU(task.query_idx, results, task.tau);

                        // Create QueryDetails
                        double pure_compute_time = local_stats.total_lb_time() + local_stats.total_astar_time();
                        QueryDetails query_detail(task.query_idx, pure_compute_time, recall, precision, iou, has_gt, !results.empty());
                        query_detail.query_nodes = query_node->graph->n;
                        query_detail.query_edges = query_node->graph->m;
                        query_detail.result_count = (int)results.size();
                        query_detail.lb_time = local_stats.total_lb_time();
                        query_detail.astar_time = local_stats.total_astar_time();
                        // ND和EPT分别的时间
                        query_detail.nd_lb_time = local_stats.ND_lb_time;
                        query_detail.nd_astar_time = local_stats.ND_astar_time;
                        query_detail.ept_lb_time = local_stats.EPT_lb_time;
                        query_detail.ept_astar_time = local_stats.EPT_astar_time;
                        query_detail.nd_lb_count = local_stats.ND_lb_count;
                        query_detail.nd_astar_count = local_stats.ND_astar_count;
                        query_detail.nd_ndc_count = local_stats.ND_ndc_count;
                        query_detail.ept_lb_count = local_stats.EPT_lb_count;
                        query_detail.ept_astar_count = local_stats.EPT_astar_count;
                        query_detail.ept_ndc_count = local_stats.EPT_ndc_count;
                        query_detail.e7_ept_trees = local_stats.e7_ept_trees;
                        query_detail.e7_answer_depth_sum = local_stats.e7_answer_depth_sum;
                        query_detail.e7_answer_count = local_stats.e7_answer_count;
                        query_detail.total_ept_nodes = local_stats.EPT_total_nodes_visited;
                        query_detail.nodes_computed = local_stats.EPT_nodes_computed;
                        query_detail.nodes_pruned = local_stats.EPT_filter_pruned_nodes;
                        query_detail.lb_propagation_count = local_stats.lb_propagation_count;
                        query_detail.lb_pruning_count = local_stats.lb_pruning_count;
                        query_detail.subtree_pruned = local_stats.subtree_pruning_avoided_nodes;
                        query_detail.subtree_pruning_decisions = local_stats.subtree_pruning_decisions;
                        query_detail.reuse_count = local_stats.EPT_reuse_count;
                        query_detail.reuse_attempt = local_stats.EPT_reuse_attempt;
                        query_detail.reuse_success_time = local_stats.EPT_reuse_success_time;
                        query_detail.reuse_fail_no_parent_snapshot = local_stats.EPT_reuse_fail_no_parent_snapshot;
                        query_detail.reuse_fail_no_parent_lp_skipped = local_stats.EPT_reuse_fail_no_parent_lp_skipped;
                        query_detail.reuse_fail_no_parent_filter_skipped = local_stats.EPT_reuse_fail_no_parent_filter_skipped;
                        query_detail.reuse_fail_no_parent_reuse_no_chain = local_stats.EPT_reuse_fail_no_parent_reuse_no_chain;
                        query_detail.reuse_fail_no_parent_other = local_stats.EPT_reuse_fail_no_parent_other;
                        query_detail.lp_skip_reuseable = local_stats.lp_skip_reuseable;
                        query_detail.lp_skip_not_reuseable = local_stats.lp_skip_not_reuseable;
                        query_detail.reuse_fail_root_node = local_stats.EPT_reuse_fail_root_node;
                        query_detail.reuse_fail_multi_ops = local_stats.EPT_reuse_fail_multi_ops;
                        query_detail.reuse_fail_vertex_count_changed = local_stats.EPT_reuse_fail_vertex_count_changed;
                        // EXP-5: Baseline 统计
                        query_detail.baseline_app_count = local_stats.EPT_baseline_app_count;
                        query_detail.baseline_app_time = local_stats.EPT_baseline_app_time;
                        query_detail.baseline_reuse_time = local_stats.EPT_baseline_reuse_time;
                        query_detail.baseline_correct = local_stats.EPT_reuse_correct;
                        query_detail.baseline_incorrect = local_stats.EPT_reuse_incorrect;
                        query_detail.baseline_samples = local_stats.baseline_samples;
                        // db图 vs 中间图统计
                        query_detail.db_graph_lb_count = local_stats.EPT_db_graph_lb_count;
                        query_detail.db_graph_lb_time = local_stats.EPT_db_graph_lb_time;
                        query_detail.db_graph_astar_count = local_stats.EPT_db_graph_astar_count;
                        query_detail.db_graph_astar_time = local_stats.EPT_db_graph_astar_time;
                        query_detail.intermediate_graph_lb_count = local_stats.EPT_intermediate_graph_lb_count;
                        query_detail.intermediate_graph_lb_time = local_stats.EPT_intermediate_graph_lb_time;
                        query_detail.intermediate_graph_astar_count = local_stats.EPT_intermediate_graph_astar_count;
                        query_detail.intermediate_graph_astar_time = local_stats.EPT_intermediate_graph_astar_time;
                        query_detail.extra_lb_count_qd = local_stats.extra_lb_count;
                        query_detail.extra_lb_time_qd = local_stats.extra_lb_time;
                        query_detail.extra_astar_count_qd = local_stats.extra_astar_count;
                        query_detail.extra_astar_time_qd = local_stats.extra_astar_time;
                        query_detail.extra_ndc_count_qd = local_stats.extra_ndc_count;
            query_detail.margin_overhead_count_qd = local_stats.margin_overhead_count;
            query_detail.margin_overhead_with_margin_time_qd = local_stats.margin_overhead_with_margin_time;
            query_detail.margin_overhead_without_margin_time_qd = local_stats.margin_overhead_without_margin_time;
            query_detail.margin_overhead_correct_qd = local_stats.margin_overhead_correct;
            query_detail.margin_overhead_incorrect_qd = local_stats.margin_overhead_incorrect;
            query_detail.reuse_speedup_gt3x_qd = local_stats.reuse_speedup_gt3x;
            query_detail.reuse_speedup_2x_3x_qd = local_stats.reuse_speedup_2x_3x;
            query_detail.reuse_speedup_1x_2x_qd = local_stats.reuse_speedup_1x_2x;
            query_detail.reuse_speedup_05x_1x_qd = local_stats.reuse_speedup_05x_1x;
            query_detail.reuse_speedup_lt05x_qd = local_stats.reuse_speedup_lt05x;
            query_detail.reuse_snapshot_size_1_qd = local_stats.reuse_snapshot_size_1;
            query_detail.reuse_snapshot_size_2_10_qd = local_stats.reuse_snapshot_size_2_10;
            query_detail.reuse_snapshot_size_gt10_qd = local_stats.reuse_snapshot_size_gt10;
            for (int _i = 0; _i < 3; ++_i) for (int _j = 0; _j < 5; ++_j) query_detail.reuse_xtab_qd[_i][_j] = local_stats.reuse_xtab[_i][_j];
            for (int _i = 0; _i < 3; ++_i) { query_detail.reuse_xtime_qd[_i] = local_stats.reuse_xtime[_i]; query_detail.baseline_xtime_qd[_i] = local_stats.baseline_xtime[_i]; }
            for (int _i = 0; _i < QueryDetails::QD_MAX_CHAIN_DEPTH && _i < SearchStats::MAX_CHAIN_DEPTH; ++_i) {
                query_detail.chain_depth_count_qd[_i]         = local_stats.chain_depth_count[_i];
                query_detail.chain_depth_correct_qd[_i]       = local_stats.chain_depth_correct[_i];
                query_detail.chain_depth_pos_qd[_i]           = local_stats.chain_depth_pos[_i];
                query_detail.chain_depth_tp_qd[_i]            = local_stats.chain_depth_tp[_i];
                query_detail.chain_depth_reuse_time_qd[_i]    = local_stats.chain_depth_reuse_time[_i];
                query_detail.chain_depth_baseline_time_qd[_i] = local_stats.chain_depth_baseline_time[_i];
            }
                        query_detail.margin_overhead_count_qd = local_stats.margin_overhead_count;
                        query_detail.margin_overhead_with_margin_time_qd = local_stats.margin_overhead_with_margin_time;
                        query_detail.margin_overhead_without_margin_time_qd = local_stats.margin_overhead_without_margin_time;
            query_detail.margin_overhead_correct_qd = local_stats.margin_overhead_correct;
            query_detail.margin_overhead_incorrect_qd = local_stats.margin_overhead_incorrect;
            query_detail.reuse_speedup_gt3x_qd = local_stats.reuse_speedup_gt3x;
            query_detail.reuse_speedup_2x_3x_qd = local_stats.reuse_speedup_2x_3x;
            query_detail.reuse_speedup_1x_2x_qd = local_stats.reuse_speedup_1x_2x;
            query_detail.reuse_speedup_05x_1x_qd = local_stats.reuse_speedup_05x_1x;
            query_detail.reuse_speedup_lt05x_qd = local_stats.reuse_speedup_lt05x;
            query_detail.reuse_snapshot_size_1_qd = local_stats.reuse_snapshot_size_1;
            query_detail.reuse_snapshot_size_2_10_qd = local_stats.reuse_snapshot_size_2_10;
            query_detail.reuse_snapshot_size_gt10_qd = local_stats.reuse_snapshot_size_gt10;
            for (int _i = 0; _i < 3; ++_i) for (int _j = 0; _j < 5; ++_j) query_detail.reuse_xtab_qd[_i][_j] = local_stats.reuse_xtab[_i][_j];
            for (int _i = 0; _i < 3; ++_i) { query_detail.reuse_xtime_qd[_i] = local_stats.reuse_xtime[_i]; query_detail.baseline_xtime_qd[_i] = local_stats.baseline_xtime[_i]; }
            for (int _i = 0; _i < QueryDetails::QD_MAX_CHAIN_DEPTH && _i < SearchStats::MAX_CHAIN_DEPTH; ++_i) {
                query_detail.chain_depth_count_qd[_i]         = local_stats.chain_depth_count[_i];
                query_detail.chain_depth_correct_qd[_i]       = local_stats.chain_depth_correct[_i];
                query_detail.chain_depth_pos_qd[_i]           = local_stats.chain_depth_pos[_i];
                query_detail.chain_depth_tp_qd[_i]            = local_stats.chain_depth_tp[_i];
                query_detail.chain_depth_reuse_time_qd[_i]    = local_stats.chain_depth_reuse_time[_i];
                query_detail.chain_depth_baseline_time_qd[_i] = local_stats.chain_depth_baseline_time[_i];
            }
                        query_detail.margin_overhead_correct_qd = local_stats.margin_overhead_correct;
                        query_detail.margin_overhead_incorrect_qd = local_stats.margin_overhead_incorrect;
                        query_detail.reuse_speedup_gt3x_qd = local_stats.reuse_speedup_gt3x;
                        query_detail.reuse_speedup_2x_3x_qd = local_stats.reuse_speedup_2x_3x;
                        query_detail.reuse_speedup_1x_2x_qd = local_stats.reuse_speedup_1x_2x;
                        query_detail.reuse_speedup_05x_1x_qd = local_stats.reuse_speedup_05x_1x;
                        query_detail.reuse_speedup_lt05x_qd = local_stats.reuse_speedup_lt05x;
                        query_detail.reuse_snapshot_size_1_qd = local_stats.reuse_snapshot_size_1;
                        query_detail.reuse_snapshot_size_2_10_qd = local_stats.reuse_snapshot_size_2_10;
                        query_detail.reuse_snapshot_size_gt10_qd = local_stats.reuse_snapshot_size_gt10;
                        for (int _i = 0; _i < 3; ++_i) for (int _j = 0; _j < 5; ++_j) query_detail.reuse_xtab_qd[_i][_j] = local_stats.reuse_xtab[_i][_j];
            for (int _i = 0; _i < 3; ++_i) { query_detail.reuse_xtime_qd[_i] = local_stats.reuse_xtime[_i]; query_detail.baseline_xtime_qd[_i] = local_stats.baseline_xtime[_i]; }
            for (int _i = 0; _i < QueryDetails::QD_MAX_CHAIN_DEPTH && _i < SearchStats::MAX_CHAIN_DEPTH; ++_i) {
                query_detail.chain_depth_count_qd[_i]         = local_stats.chain_depth_count[_i];
                query_detail.chain_depth_correct_qd[_i]       = local_stats.chain_depth_correct[_i];
                query_detail.chain_depth_pos_qd[_i]           = local_stats.chain_depth_pos[_i];
                query_detail.chain_depth_tp_qd[_i]            = local_stats.chain_depth_tp[_i];
                query_detail.chain_depth_reuse_time_qd[_i]    = local_stats.chain_depth_reuse_time[_i];
                query_detail.chain_depth_baseline_time_qd[_i] = local_stats.chain_depth_baseline_time[_i];
            }

                        // Collect result
                        {
                            std::lock_guard<std::mutex> lock(results_mutex);
                            TaskResult result{task.method, task.tau, query_detail};
                            all_results.push_back(result);
                        }

                        // Update progress
                        int current = ++completed_tasks;
                        if (current % 50 == 0 || current == total_tasks) {
                            std::lock_guard<std::mutex> lock(print_mutex);
                            printf("[experiment_mode] Progress: %d/%d tasks completed (%.1f%%)\n",
                                   current, total_tasks, 100.0 * current / total_tasks);
                        }
                    }
                }));
        }

        // Wait for all threads
        printf("[experiment_mode] Waiting for all %d threads to complete...\n", num_threads);
        for (auto& future : futures) {
            future.get();
        }

        // Close reuse log
        if (g_reuse_log) {
            fclose(g_reuse_log);
            g_reuse_log = nullptr;
            printf("[experiment_mode] Reuse log closed: reuse_log.csv\n");
        }

        // Batch save all results (skip if --save not specified)
        if (config.save) {
            printf("[experiment_mode] All computation completed. Now saving %zu results to files...\n", all_results.size());
            auto save_start = std::chrono::high_resolution_clock::now();
            for (const auto& result : all_results) {
                // 保存到时间戳目录
                const std::string& subdir = subdirs.at(result.method).at(result.tau);
                searchers.at(result.method)->save_single_query_json(result.details, result.tau, subdir);
                // 保存到 latest 目录
                const std::string& latest_subdir = latest_subdirs.at(result.method).at(result.tau);
                searchers.at(result.method)->save_single_query_json(result.details, result.tau, latest_subdir);
            }
            auto save_end = std::chrono::high_resolution_clock::now();
            double save_time = std::chrono::duration<double>(save_end - save_start).count();
            printf("[experiment_mode] Saved all results in %.3f seconds\n", save_time);
        } else {
            printf("[experiment_mode] All computation completed. Skipping file save (--save not specified)\n");
        }

    } else {
        // Serial execution
        printf("\n[experiment_mode] Using SERIAL execution for %d tasks...\n", total_tasks);

        int completed_tasks = 0;
        for (const auto& task : all_tasks) {
            auto query_node = query_node_list[task.query_local_idx];
            auto& searcher = searchers.at(task.method);

            // Execute search
            SearchStats local_stats;
            std::vector<int> results;

            if (task.method == "Gisma" || task.method == "Gisma-default") {
                results = searcher->Gisma_search(query_node, task.tau, local_stats);
            } else if (task.method == "Gisma-no-reuse") {
                results = searcher->Gisma_no_reuse_search(query_node, task.tau, local_stats);
            } else if (task.method == "Gisma-only-dfs") {
                results = searcher->Gisma_only_dfs_search(query_node, task.tau, local_stats);
            } else if (task.method == "Gisma-no-SP") {
                results = searcher->Gisma_no_SP_search(query_node, task.tau, local_stats);
            } else if (task.method == "Gisma-no-LP") {
                results = searcher->Gisma_no_LP_search(query_node, task.tau, local_stats);
            } else if (task.method == "BMao_scan") {
                results = searcher->BMao_scan_search(query_node, task.tau, local_stats);
            } else if (task.method == "App-BMao") {
                // App-BMao: original standalone implementation (full DB scan, no index/reuse)
                results = searcher->App_BMao_orig_search(query_node, task.tau, local_stats);
            } else if (task.method == "App-BMao-gisma") {
                // App-BMao variant sharing Gisma's State (full DB scan)
                results = searcher->App_BMao_search(query_node, task.tau, local_stats);
            } else if (task.method == "App-BMao-orig") {
                results = searcher->App_BMao_orig_search(query_node, task.tau, local_stats);
            } else if (task.method == "AStar-BMao") {
                results = searcher->AStar_BMao_search(query_node, task.tau, local_stats);
            } else if (task.method == "AStar-scan") {
                results = searcher->AStar_scan_search(query_node, task.tau, local_stats);
            } else if (task.method == "AStar-scan-no-lsa") {
                results = searcher->AStar_scan_no_lsa_search(query_node, task.tau, local_stats);
            } else if (task.method == "Base+GS") {
                results = searcher->Base_GS_search(query_node, task.tau, local_stats);
            } else if (task.method == "Base+SS") {
                results = searcher->Base_SS_search(query_node, task.tau, local_stats);
            } else if (task.method == "Base_All_EPT") {
                results = searcher->Base_All_EPT_search(query_node, task.tau, local_stats);
            } else if (task.method == "astar-lsa") {
                results = searcher->AStar_LSa_scan_search(query_node, task.tau, local_stats);  // full scan + exact LSa
            } else if (task.method == "app-lsa") {
                results = searcher->App_LSa_scan_search(query_node, task.tau, local_stats);    // full scan + approx LSa
            } else if (task.method == "gisma-lsa") {
                results = searcher->Gisma_search(query_node, task.tau, local_stats);           // Gisma index + LSa
            } else {
                results = searcher->Gisma_search(query_node, task.tau, local_stats);
            }

            // E11 masked deletion: 从结果集过滤掉已删除的图
            if (!deleted_ids.empty() && !results.empty()) {
                results.erase(std::remove_if(results.begin(), results.end(),
                    [&deleted_ids](int id){ return deleted_ids.count(id) > 0; }), results.end());
            }

            // Compute metrics
            auto [has_gt, recall, precision, iou, intersection_count, ground_truth_size]
                = searcher->compute_recall_precision_IoU(task.query_idx, results, task.tau);

            // Create QueryDetails
            double pure_compute_time = local_stats.total_lb_time() + local_stats.total_astar_time();
            QueryDetails query_detail(task.query_idx, pure_compute_time, recall, precision, iou, has_gt, !results.empty());
            query_detail.query_nodes = query_node->graph->n;
            query_detail.query_edges = query_node->graph->m;
            query_detail.result_count = (int)results.size();
            query_detail.lb_time = local_stats.total_lb_time();
            query_detail.astar_time = local_stats.total_astar_time();
            // ND和EPT分别的时间
            query_detail.nd_lb_time = local_stats.ND_lb_time;
            query_detail.nd_astar_time = local_stats.ND_astar_time;
            query_detail.ept_lb_time = local_stats.EPT_lb_time;
            query_detail.ept_astar_time = local_stats.EPT_astar_time;
            query_detail.nd_lb_count = local_stats.ND_lb_count;
            query_detail.nd_astar_count = local_stats.ND_astar_count;
            query_detail.nd_ndc_count = local_stats.ND_ndc_count;
            query_detail.ept_lb_count = local_stats.EPT_lb_count;
            query_detail.ept_astar_count = local_stats.EPT_astar_count;
            query_detail.ept_ndc_count = local_stats.EPT_ndc_count;
            query_detail.e7_ept_trees = local_stats.e7_ept_trees;
            query_detail.e7_answer_depth_sum = local_stats.e7_answer_depth_sum;
            query_detail.e7_answer_count = local_stats.e7_answer_count;
            query_detail.total_ept_nodes = local_stats.EPT_total_nodes_visited;
            query_detail.nodes_computed = local_stats.EPT_nodes_computed;
            query_detail.nodes_pruned = local_stats.EPT_filter_pruned_nodes;
            query_detail.lb_propagation_count = local_stats.lb_propagation_count;
            query_detail.lb_pruning_count = local_stats.lb_pruning_count;
            query_detail.subtree_pruned = local_stats.subtree_pruning_avoided_nodes;
            query_detail.subtree_pruning_decisions = local_stats.subtree_pruning_decisions;
            query_detail.reuse_count = local_stats.EPT_reuse_count;
            query_detail.reuse_attempt = local_stats.EPT_reuse_attempt;
            query_detail.reuse_success_time = local_stats.EPT_reuse_success_time;
            query_detail.reuse_fail_no_parent_snapshot = local_stats.EPT_reuse_fail_no_parent_snapshot;
            query_detail.reuse_fail_root_node = local_stats.EPT_reuse_fail_root_node;
            query_detail.reuse_fail_multi_ops = local_stats.EPT_reuse_fail_multi_ops;
            query_detail.reuse_fail_vertex_count_changed = local_stats.EPT_reuse_fail_vertex_count_changed;
            // EXP-5: Baseline 统计
            query_detail.baseline_app_count = local_stats.EPT_baseline_app_count;
            query_detail.baseline_app_time = local_stats.EPT_baseline_app_time;
            query_detail.baseline_reuse_time = local_stats.EPT_baseline_reuse_time;
            query_detail.baseline_correct = local_stats.EPT_reuse_correct;
            query_detail.baseline_incorrect = local_stats.EPT_reuse_incorrect;
            query_detail.baseline_samples = local_stats.baseline_samples;
            // db图 vs 中间图统计
            query_detail.db_graph_lb_count = local_stats.EPT_db_graph_lb_count;
            query_detail.db_graph_lb_time = local_stats.EPT_db_graph_lb_time;
            query_detail.db_graph_astar_count = local_stats.EPT_db_graph_astar_count;
            query_detail.db_graph_astar_time = local_stats.EPT_db_graph_astar_time;
            query_detail.intermediate_graph_lb_count = local_stats.EPT_intermediate_graph_lb_count;
            query_detail.intermediate_graph_lb_time = local_stats.EPT_intermediate_graph_lb_time;
            query_detail.intermediate_graph_astar_count = local_stats.EPT_intermediate_graph_astar_count;
            query_detail.intermediate_graph_astar_time = local_stats.EPT_intermediate_graph_astar_time;
            query_detail.extra_lb_count_qd = local_stats.extra_lb_count;
            query_detail.extra_lb_time_qd = local_stats.extra_lb_time;
            query_detail.extra_astar_count_qd = local_stats.extra_astar_count;
            query_detail.extra_astar_time_qd = local_stats.extra_astar_time;
            query_detail.extra_ndc_count_qd = local_stats.extra_ndc_count;
            query_detail.margin_overhead_count_qd = local_stats.margin_overhead_count;
            query_detail.margin_overhead_with_margin_time_qd = local_stats.margin_overhead_with_margin_time;
            query_detail.margin_overhead_without_margin_time_qd = local_stats.margin_overhead_without_margin_time;
            query_detail.margin_overhead_correct_qd = local_stats.margin_overhead_correct;
            query_detail.margin_overhead_incorrect_qd = local_stats.margin_overhead_incorrect;
            query_detail.reuse_speedup_gt3x_qd = local_stats.reuse_speedup_gt3x;
            query_detail.reuse_speedup_2x_3x_qd = local_stats.reuse_speedup_2x_3x;
            query_detail.reuse_speedup_1x_2x_qd = local_stats.reuse_speedup_1x_2x;
            query_detail.reuse_speedup_05x_1x_qd = local_stats.reuse_speedup_05x_1x;
            query_detail.reuse_speedup_lt05x_qd = local_stats.reuse_speedup_lt05x;
            query_detail.reuse_snapshot_size_1_qd = local_stats.reuse_snapshot_size_1;
            query_detail.reuse_snapshot_size_2_10_qd = local_stats.reuse_snapshot_size_2_10;
            query_detail.reuse_snapshot_size_gt10_qd = local_stats.reuse_snapshot_size_gt10;
            for (int _i = 0; _i < 3; ++_i) for (int _j = 0; _j < 5; ++_j) query_detail.reuse_xtab_qd[_i][_j] = local_stats.reuse_xtab[_i][_j];
            for (int _i = 0; _i < 3; ++_i) { query_detail.reuse_xtime_qd[_i] = local_stats.reuse_xtime[_i]; query_detail.baseline_xtime_qd[_i] = local_stats.baseline_xtime[_i]; }
            for (int _i = 0; _i < QueryDetails::QD_MAX_CHAIN_DEPTH && _i < SearchStats::MAX_CHAIN_DEPTH; ++_i) {
                query_detail.chain_depth_count_qd[_i]         = local_stats.chain_depth_count[_i];
                query_detail.chain_depth_correct_qd[_i]       = local_stats.chain_depth_correct[_i];
                query_detail.chain_depth_pos_qd[_i]           = local_stats.chain_depth_pos[_i];
                query_detail.chain_depth_tp_qd[_i]            = local_stats.chain_depth_tp[_i];
                query_detail.chain_depth_reuse_time_qd[_i]    = local_stats.chain_depth_reuse_time[_i];
                query_detail.chain_depth_baseline_time_qd[_i] = local_stats.chain_depth_baseline_time[_i];
            }

            // Collect result for summary
            TaskResult result{task.method, task.tau, query_detail};
            all_results.push_back(result);

            // Save immediately (skip if --save not specified)
            if (config.save) {
                // 保存到时间戳目录
                const std::string& subdir = subdirs.at(task.method).at(task.tau);
                searcher->save_single_query_json(query_detail, task.tau, subdir);
                // 保存到 latest 目录
                const std::string& latest_subdir = latest_subdirs.at(task.method).at(task.tau);
                searcher->save_single_query_json(query_detail, task.tau, latest_subdir);
            }

            // Update progress
            completed_tasks++;
            if (completed_tasks % 50 == 0 || completed_tasks == total_tasks) {
                printf("[experiment_mode] Progress: %d/%d tasks completed (%.1f%%)\n",
                       completed_tasks, total_tasks, 100.0 * completed_tasks / total_tasks);
            }
        }
    }

    auto total_exp_end = std::chrono::high_resolution_clock::now();
    auto total_exp_duration = std::chrono::duration<double>(total_exp_end - total_exp_start);

    // Print comparison table
    printf("\n");
    printf("========================================================================================================================================================\n");
    printf("                                               COMPARISON TABLE [Dataset: %s]\n", config.dataset.c_str());
    printf("========================================================================================================================================================\n");
    printf("| %-16s | %5s | %7s | %7s | %7s | %10s | %8s | %10s | %8s | %8s | %10s | %10s | %10s |\n",
           "Method", "Tau", "Recall%", "Prec%", "IoU%", "Filter(s)", "Filter#", "Verify(s)", "Verify#", "NDC#", "ND(s)", "EPT(s)", "Total(s)");
    printf("|------------------|-------|---------|---------|---------|------------|----------|------------|----------|----------|------------|------------|------------|\n");

    for (const auto& tau : tau_list) {
        for (const auto& method : methods_list) {
            // Recompute stats for this cell
            int queries_with_gt = 0;
            int queries_with_results = 0;
            double sum_recall = 0.0, sum_precision = 0.0, sum_iou = 0.0;
            double total_time = 0.0;
            double total_nd_lb_time = 0.0, total_nd_astar_time = 0.0;
            double total_ept_lb_time = 0.0, total_ept_astar_time = 0.0;
            long long total_nd_lb_count = 0, total_nd_astar_count = 0;
            long long total_ept_lb_count = 0, total_ept_astar_count = 0;
            long long total_nd_ndc_count = 0, total_ept_ndc_count = 0;
            int total_queries = 0;

            for (const auto& result : all_results) {
                if (result.method == method && result.tau == tau) {
                    total_queries++;
                    total_time += result.details.time;
                    total_nd_lb_time += result.details.nd_lb_time;
                    total_nd_astar_time += result.details.nd_astar_time;
                    total_ept_lb_time += result.details.ept_lb_time;
                    total_ept_astar_time += result.details.ept_astar_time;
                    total_nd_lb_count += result.details.nd_lb_count;
                    total_nd_astar_count += result.details.nd_astar_count;
                    total_nd_ndc_count += result.details.nd_ndc_count;
                    total_ept_lb_count += result.details.ept_lb_count;
                    total_ept_astar_count += result.details.ept_astar_count;
                    total_ept_ndc_count += result.details.ept_ndc_count;
                    if (result.details.has_ground_truth) {
                        queries_with_gt++;
                        sum_recall += result.details.recall;
                        sum_iou += result.details.iou;
                    }
                    if (result.details.has_results) {
                        queries_with_results++;
                        sum_precision += result.details.precision;
                    }
                }
            }

            double avg_recall = queries_with_gt > 0 ? sum_recall / queries_with_gt * 100 : 0.0;
            double avg_precision = queries_with_results > 0 ? sum_precision / queries_with_results * 100 : 0.0;
            double avg_iou = queries_with_gt > 0 ? sum_iou / queries_with_gt * 100 : 0.0;
            double avg_time = total_queries > 0 ? total_time / total_queries : 0.0;
            double avg_nd_lb = total_queries > 0 ? total_nd_lb_time / total_queries : 0.0;
            double avg_nd_astar = total_queries > 0 ? total_nd_astar_time / total_queries : 0.0;
            double avg_ept_lb = total_queries > 0 ? total_ept_lb_time / total_queries : 0.0;
            double avg_ept_astar = total_queries > 0 ? total_ept_astar_time / total_queries : 0.0;
            double avg_filter_count = total_queries > 0 ? (double)(total_nd_lb_count + total_ept_lb_count) / total_queries : 0.0;
            double avg_verify_count = total_queries > 0 ? (double)(total_nd_astar_count + total_ept_astar_count) / total_queries : 0.0;
            double avg_ndc_count = total_queries > 0 ? (double)(total_nd_ndc_count + total_ept_ndc_count) / total_queries : 0.0;  // NDC = 唯一节点计数（去重）

            // Combine filter time (ND_lb + EPT_lb) and verify time (ND_astar + EPT_astar)
            double filter_time = avg_nd_lb + avg_ept_lb;
            double verify_time = avg_nd_astar + avg_ept_astar;
            // Total ND time and EPT time
            double nd_time = avg_nd_lb + avg_nd_astar;
            double ept_time = avg_ept_lb + avg_ept_astar;

            // Format time: show N/A only if time is essentially zero
            auto fmt_time = [](double t) -> std::string {
                if (t < 1e-9) return "N/A";
                char buf[32];
                snprintf(buf, sizeof(buf), "%.6f", t);
                return std::string(buf);
            };

            // Format count: show N/A only if count is zero
            auto fmt_count = [](double c) -> std::string {
                if (c < 0.001) return "N/A";
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1f", c);
                return std::string(buf);
            };

            // Check if method has ND/EPT breakdown (Gisma, Base+SS, Base+GS)
            bool has_nd_ept = (method.find("Gisma") != std::string::npos ||
                               method.find("Base+SS") != std::string::npos ||
                               method.find("Base+GS") != std::string::npos);

            std::string nd_str = has_nd_ept ? fmt_time(nd_time) : "N/A";
            std::string ept_str = has_nd_ept ? fmt_time(ept_time) : "N/A";

            printf("| %-16s | %5.1f | %7.2f | %7.2f | %7.2f | %10s | %8s | %10s | %8s | %8s | %10s | %10s | %10.6f |\n",
                   method.c_str(), tau, avg_recall, avg_precision, avg_iou,
                   fmt_time(filter_time).c_str(),
                   fmt_count(avg_filter_count).c_str(),
                   fmt_time(verify_time).c_str(),
                   fmt_count(avg_verify_count).c_str(),
                   fmt_count(avg_ndc_count).c_str(),
                   nd_str.c_str(), ept_str.c_str(),
                   avg_time);
        }
        // Separator between tau groups
        if (&tau != &tau_list.back()) {
            printf("|------------------|-------|---------|---------|---------|------------|----------|------------|----------|----------|------------|------------|------------|\n");
        }
    }
    printf("========================================================================================================================================================\n");
    printf("Note: Filter = ND_lb + EPT_lb, Verify = ND_astar + EPT_astar, NDC = unique nodes computed (deduplicated)\n");
    printf("      ND(s) = NetDag phase total time, EPT(s) = EditPathTree phase total time\n");

    // Print Gisma detailed statistics table (for methods with ND/EPT breakdown)
    bool has_gisma_methods = false;
    for (const auto& method : methods_list) {
        if (method == "Gisma" || method == "Gisma-default" || method == "Gisma-no-reuse" || method == "Gisma-only-dfs" || method == "Gisma-no-SP" || method == "Gisma-no-LP" || method == "Base+SS" || method == "Base+GS" || method == "Base_All_EPT"
            || method == "astar-lsa" || method == "app-lsa" || method == "gisma-lsa") {  // E8 verifier methods 也有 ND/EPT breakdown
            has_gisma_methods = true;
            break;
        }
    }

    if (has_gisma_methods) {
        printf("\n");
        printf("==========================================================================================================================================\n");
        printf("                                  GISMA DETAILED STATISTICS (ND vs EPT) [Dataset: %s]\n", config.dataset.c_str());
        printf("==========================================================================================================================================\n");
        printf("| %-16s | %5s | %10s | %12s | %10s | %12s | %10s | %12s | %10s | %12s |\n",
               "Method", "Tau", "ND_Flt#", "ND_Flt(s)", "ND_Vfy#", "ND_Vfy(s)", "EPT_Flt#", "EPT_Flt(s)", "EPT_Vfy#", "EPT_Vfy(s)");
        printf("|------------------|-------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|");

        for (const auto& tau : tau_list) {
            printf("\n");
            for (const auto& method : methods_list) {
                // Only show Gisma-like methods
                if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                    continue;
                }

                // Collect stats for this method/tau
                double total_nd_lb_time = 0.0, total_nd_astar_time = 0.0;
                double total_ept_lb_time = 0.0, total_ept_astar_time = 0.0;
                long long total_nd_lb_count = 0, total_nd_astar_count = 0;
                long long total_ept_lb_count = 0, total_ept_astar_count = 0;
                int total_queries = 0;

                for (const auto& result : all_results) {
                    if (result.method == method && result.tau == tau) {
                        total_queries++;
                        total_nd_lb_time += result.details.nd_lb_time;
                        total_nd_astar_time += result.details.nd_astar_time;
                        total_ept_lb_time += result.details.ept_lb_time;
                        total_ept_astar_time += result.details.ept_astar_time;
                        total_nd_lb_count += result.details.nd_lb_count;
                        total_nd_astar_count += result.details.nd_astar_count;
                        total_ept_lb_count += result.details.ept_lb_count;
                        total_ept_astar_count += result.details.ept_astar_count;
                    }
                }

                double avg_nd_lb_count = total_queries > 0 ? (double)total_nd_lb_count / total_queries : 0.0;
                double avg_nd_lb_time = total_queries > 0 ? total_nd_lb_time / total_queries : 0.0;
                double avg_nd_astar_count = total_queries > 0 ? (double)total_nd_astar_count / total_queries : 0.0;
                double avg_nd_astar_time = total_queries > 0 ? total_nd_astar_time / total_queries : 0.0;
                double avg_ept_lb_count = total_queries > 0 ? (double)total_ept_lb_count / total_queries : 0.0;
                double avg_ept_lb_time = total_queries > 0 ? total_ept_lb_time / total_queries : 0.0;
                double avg_ept_astar_count = total_queries > 0 ? (double)total_ept_astar_count / total_queries : 0.0;
                double avg_ept_astar_time = total_queries > 0 ? total_ept_astar_time / total_queries : 0.0;

                printf("| %-16s | %5.1f | %10.1f | %12.6f | %10.1f | %12.6f | %10.1f | %12.6f | %10.1f | %12.6f |\n",
                       method.c_str(), tau,
                       avg_nd_lb_count, avg_nd_lb_time,
                       avg_nd_astar_count, avg_nd_astar_time,
                       avg_ept_lb_count, avg_ept_lb_time,
                       avg_ept_astar_count, avg_ept_astar_time);
            }
            // Separator between tau groups
            if (&tau != &tau_list.back()) {
                printf("|------------------|-------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|");
            }
        }
        printf("\n==========================================================================================================================================\n");
        printf("Note: Flt = Filter (lower bound), Vfy = Verify (A*/App GED computation)\n");
        printf("      ND = NetDag phase, EPT = EditPathTree phase\n");

        // Print Three Optimizations table
        printf("\n");
        printf("===================================================================================================\n");
        printf("                         THREE OPTIMIZATIONS STATISTICS [Dataset: %s]\n", config.dataset.c_str());
        printf("===================================================================================================\n");
        printf("| %-16s | %5s | %12s | %12s | %12s | %12s | %12s |\n",
               "Method", "Tau", "SubtreePrn#", "SubtreePrnAv", "LB-Prop#", "LB-Prune#", "Reuse#");
        printf("|------------------|-------|--------------|--------------|--------------|--------------|--------------|");

        for (const auto& tau : tau_list) {
            printf("\n");
            for (const auto& method : methods_list) {
                // Only show Gisma-like methods
                if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                    continue;
                }

                // Collect stats for this method/tau
                long long total_subtree_pruning_decisions = 0;
                long long total_subtree_pruned = 0;
                long long total_lb_propagation_count = 0;
                long long total_lb_pruning_count = 0;
                long long total_reuse_count = 0;
                int total_queries = 0;

                for (const auto& result : all_results) {
                    if (result.method == method && result.tau == tau) {
                        total_queries++;
                        total_subtree_pruning_decisions += result.details.subtree_pruning_decisions;
                        total_subtree_pruned += result.details.subtree_pruned;
                        total_lb_propagation_count += result.details.lb_propagation_count;
                        total_lb_pruning_count += result.details.lb_pruning_count;
                        total_reuse_count += result.details.reuse_count;
                    }
                }

                double avg_subtree_pruning_decisions = total_queries > 0 ? (double)total_subtree_pruning_decisions / total_queries : 0.0;
                double avg_subtree_pruned = total_queries > 0 ? (double)total_subtree_pruned / total_queries : 0.0;
                double avg_lb_propagation_count = total_queries > 0 ? (double)total_lb_propagation_count / total_queries : 0.0;
                double avg_lb_pruning_count = total_queries > 0 ? (double)total_lb_pruning_count / total_queries : 0.0;
                double avg_reuse_count = total_queries > 0 ? (double)total_reuse_count / total_queries : 0.0;

                printf("| %-16s | %5.1f | %12.1f | %12.1f | %12.1f | %12.1f | %12.1f |\n",
                       method.c_str(), tau,
                       avg_subtree_pruning_decisions,
                       avg_subtree_pruned,
                       avg_lb_propagation_count,
                       avg_lb_pruning_count,
                       avg_reuse_count);
            }
            // Separator between tau groups
            if (&tau != &tau_list.back()) {
                printf("|------------------|-------|--------------|--------------|--------------|--------------|--------------|");
            }
        }
        printf("\n===================================================================================================\n");
        printf("Note: SubtreePrn# = Subtree Pruning decisions, SubtreePrnAv = nodes avoided by pruning\n");
        printf("      LB-Prop# = LB Propagation count, LB-Prune# = LB Pruning (GED computations skipped)\n");
        printf("      Reuse# = Search Tree Reuse success count\n");

        // Print DB Graph vs Intermediate Graph Statistics table
        printf("\n");
        printf("===========================================================================================================================\n");
        printf("                           DB GRAPH vs INTERMEDIATE GRAPH STATISTICS [Dataset: %s]\n", config.dataset.c_str());
        printf("===============================================================================================================================\n");
        printf("| %-16s | %5s | %9s | %10s | %9s | %10s |\n",
               "Method", "Tau", "DB_Flt#", "DB_Flt(s)", "Int_Flt#", "Int_Flt(s)");
        printf("|------------------|-------|-----------|------------|-----------|------------|");

        for (const auto& tau : tau_list) {
            printf("\n");
            for (const auto& method : methods_list) {
                // Only show Gisma-like methods
                if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                    continue;
                }

                // Collect stats for this method/tau
                long long total_db_lb_count = 0;
                double total_db_lb_time = 0.0;
                long long total_inter_lb_count = 0;
                double total_inter_lb_time = 0.0;
                int total_queries = 0;

                for (const auto& result : all_results) {
                    if (result.method == method && result.tau == tau) {
                        total_queries++;
                        total_db_lb_count += result.details.db_graph_lb_count;
                        total_db_lb_time += result.details.db_graph_lb_time;
                        total_inter_lb_count += result.details.intermediate_graph_lb_count;
                        total_inter_lb_time += result.details.intermediate_graph_lb_time;
                    }
                }

                double avg_db_lb_count = total_queries > 0 ? (double)total_db_lb_count / total_queries : 0.0;
                double avg_db_lb_time = total_queries > 0 ? total_db_lb_time / total_queries : 0.0;
                double avg_inter_lb_count = total_queries > 0 ? (double)total_inter_lb_count / total_queries : 0.0;
                double avg_inter_lb_time = total_queries > 0 ? total_inter_lb_time / total_queries : 0.0;

                printf("| %-16s | %5.1f | %9.1f | %10.6f | %9.1f | %10.6f |\n",
                       method.c_str(), tau,
                       avg_db_lb_count, avg_db_lb_time,
                       avg_inter_lb_count, avg_inter_lb_time);
            }
            // Separator between tau groups
            if (&tau != &tau_list.back()) {
                printf("|------------------|-------|-----------|------------|-----------|------------|");
            }
        }
        printf("\n===============================================================================================================================\n");
        printf("Note: DB = anchor/completed graphs, Int = intermediate graphs (edited)\n");
        printf("      Flt# = avg filter count per query, Flt(s) = avg filter time per query (seconds)\n");

        // EPT vs Extra table
        printf("\n");
        printf("===============================================================================================================================\n");
        printf("                           EPT vs EXTRA CLUSTER STATISTICS [Dataset: %s]\n", config.dataset.c_str());
        printf("===============================================================================================================================\n");
        printf("| %-16s | %5s | %9s | %10s | %9s | %10s | %9s | %10s | %9s | %10s | %9s |\n",
               "Method", "Tau", "EPT_Flt#", "EPT_Flt(s)", "EPT_Vfy#", "EPT_Vfy(s)", "Ext_Flt#", "Ext_Flt(s)", "Ext_Vfy#", "Ext_Vfy(s)", "Ext_NDC#");
        printf("|------------------|-------|-----------|------------|-----------|------------|-----------|------------|-----------|------------|-----------|");

        for (const auto& tau : tau_list) {
            printf("\n");
            for (const auto& method : methods_list) {
                if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                    continue;
                }

                long long total_extra_lb = 0, total_extra_astar = 0, total_extra_ndc = 0;
                double total_extra_lb_time = 0.0, total_extra_astar_time = 0.0;
                long long total_ept_lb = 0, total_ept_astar = 0;
                double total_ept_lb_time = 0.0, total_ept_astar_time = 0.0;
                int total_queries = 0;

                for (const auto& result : all_results) {
                    if (result.method == method && result.tau == tau) {
                        total_queries++;
                        total_extra_lb += result.details.extra_lb_count_qd;
                        total_extra_astar += result.details.extra_astar_count_qd;
                        total_extra_ndc += result.details.extra_ndc_count_qd;
                        total_extra_lb_time += result.details.extra_lb_time_qd;
                        total_extra_astar_time += result.details.extra_astar_time_qd;
                        total_ept_lb += result.details.ept_lb_count - result.details.extra_lb_count_qd;
                        total_ept_astar += result.details.ept_astar_count - result.details.extra_astar_count_qd;
                        total_ept_lb_time += result.details.ept_lb_time - result.details.extra_lb_time_qd;
                        total_ept_astar_time += result.details.ept_astar_time - result.details.extra_astar_time_qd;
                    }
                }

                if (total_queries > 0) {
                    double nq = total_queries;
                    printf("| %-16s | %5.1f | %9.1f | %10.6f | %9.1f | %10.6f | %9.1f | %10.6f | %9.1f | %10.6f | %9.1f |\n",
                           method.c_str(), tau,
                           total_ept_lb / nq, total_ept_lb_time / nq,
                           total_ept_astar / nq, total_ept_astar_time / nq,
                           total_extra_lb / nq, total_extra_lb_time / nq,
                           total_extra_astar / nq, total_extra_astar_time / nq,
                           total_extra_ndc / nq);
                }
            }
        }
        printf("\n===============================================================================================================================\n");

        // EXP-5: Search Tree Reuse Effectiveness Table (按 tau 分组)
        // 检查是否有 baseline 数据
        bool has_baseline_data = false;
        for (const auto& result : all_results) {
            if (result.details.baseline_app_count > 0) {
                has_baseline_data = true;
                break;
            }
        }

        if (has_baseline_data) {
            // Collect methods that have baseline data
            std::vector<std::string> methods_with_baseline;
            for (const auto& method : methods_list) {
                for (const auto& result : all_results) {
                    if (result.method == method && result.details.baseline_app_count > 0) {
                        methods_with_baseline.push_back(method);
                        break;
                    }
                }
            }

            for (const auto& bmethod : methods_with_baseline) {
                printf("\n");
                printf("=======================================================================================================================\n");
                printf("                     EXP-5: Search Tree Reuse Effectiveness [%s]\n", bmethod.c_str());
                printf("=======================================================================================================================\n");
                printf("| %-5s | %-20s | %-10s | %-12s | %-12s | %-10s | %-10s |\n",
                       "Tau", "Method", "Count", "PerCall(ms)", "PerQuery(ms)", "Speedup", "Accuracy%");
                printf("|-------|----------------------|------------|--------------|--------------|------------|------------|\n");

                for (const auto& tau : tau_list) {
                    size_t tau_baseline_count = 0;
                    double tau_baseline_time = 0.0;
                    size_t tau_all_reuse_count = 0;
                    double tau_all_reuse_time = 0.0;
                    double tau_astar_reuse_time = 0.0;
                    size_t tau_correct = 0;
                    size_t tau_incorrect = 0;

                    for (const auto& result : all_results) {
                        if (result.method == bmethod && result.tau == tau) {
                            tau_baseline_count += result.details.baseline_app_count;
                            tau_baseline_time += result.details.baseline_app_time;
                            tau_all_reuse_count += result.details.reuse_count;
                            tau_all_reuse_time += result.details.reuse_success_time;
                            tau_astar_reuse_time += result.details.baseline_reuse_time;
                            tau_correct += result.details.baseline_correct;
                            tau_incorrect += result.details.baseline_incorrect;
                        }
                    }

                    // Count number of queries for this method+tau
                    int num_queries = 0;
                    for (const auto& result : all_results) {
                        if (result.method == bmethod && result.tau == tau) {
                            num_queries++;
                        }
                    }

                    // Also collect margin overhead for this method+tau
                    size_t tau_margin_count = 0;
                    double tau_margin_with = 0.0, tau_margin_without = 0.0;
                    for (const auto& result : all_results) {
                        if (result.method == bmethod && result.tau == tau) {
                            tau_margin_count += result.details.margin_overhead_count_qd;
                            tau_margin_with += result.details.margin_overhead_with_margin_time_qd;
                            tau_margin_without += result.details.margin_overhead_without_margin_time_qd;
                        }
                    }

                    if (tau_baseline_count > 0 && tau_all_reuse_count > 0 && num_queries > 0) {
                        double avg_reuse_ms = (tau_all_reuse_time / tau_all_reuse_count) * 1000.0;
                        double avg_baseline_ms = (tau_baseline_time / tau_baseline_count) * 1000.0;
                        double speedup = avg_baseline_ms / avg_reuse_ms;

                        double accuracy_pct = (tau_correct + tau_incorrect > 0)
                            ? 100.0 * tau_correct / (tau_correct + tau_incorrect) : 0.0;

                        double avg_reuse_count = (double)tau_all_reuse_count / num_queries;
                        double avg_baseline_count = (double)tau_baseline_count / num_queries;

                        double avg_reuse_per_query_ms = (tau_all_reuse_time / num_queries) * 1000.0;
                        double avg_baseline_per_query_ms = (tau_baseline_time / num_queries) * 1000.0;

                        printf("| %5.1f | %-20s | %10.1f | %12.4f | %12.4f | %9.2fx | %9.2f%% |\n",
                               tau, "Reuse A*", avg_reuse_count, avg_reuse_ms, avg_reuse_per_query_ms, speedup, accuracy_pct);

                        printf("| %5s | %-20s | %10.1f | %12.4f | %12.4f | %10s | %10s |\n",
                               "", "From-scratch A*", avg_baseline_count, avg_baseline_ms, avg_baseline_per_query_ms, "(base)", "(ref)");

                        // Margin A* row: compare App(margin) vs App(no margin) on same nodes
                        if (tau_margin_count > 0) {
                            double avg_margin_ms = (tau_margin_with / tau_margin_count) * 1000.0;
                            double avg_margin_baseline_ms = (tau_margin_without / tau_margin_count) * 1000.0;
                            double avg_margin_count = (double)tau_margin_count / num_queries;
                            double margin_per_query_ms = (tau_margin_with / num_queries) * 1000.0;
                            double margin_baseline_per_query_ms = (tau_margin_without / num_queries) * 1000.0;
                            double margin_overhead = avg_margin_baseline_ms / avg_margin_ms;

                            size_t tau_margin_correct = 0, tau_margin_incorrect = 0;
                            for (const auto& result : all_results) {
                                if (result.method == bmethod && result.tau == tau) {
                                    tau_margin_correct += result.details.margin_overhead_correct_qd;
                                    tau_margin_incorrect += result.details.margin_overhead_incorrect_qd;
                                }
                            }
                            double margin_accuracy = (tau_margin_correct + tau_margin_incorrect > 0)
                                ? 100.0 * tau_margin_correct / (tau_margin_correct + tau_margin_incorrect) : 0.0;

                            printf("| %5s | %-20s | %10.1f | %12.4f | %12.4f | %9.2fx | %9.2f%% |\n",
                                   "", "A* + Snapshot", avg_margin_count, avg_margin_ms, margin_per_query_ms, margin_overhead, margin_accuracy);
                            printf("| %5s | %-20s | %10.1f | %12.4f | %12.4f | %10s | %10s |\n",
                                   "", "A* (no snapshot)", avg_margin_count, avg_margin_baseline_ms, margin_baseline_per_query_ms, "(base)", "(ref)");
                        }
                        // Summary: total time with vs without reuse
                        if (tau_margin_count > 0) {
                            double margin_per_query_ms = (tau_margin_with / num_queries) * 1000.0;
                            double margin_baseline_per_query_ms = (tau_margin_without / num_queries) * 1000.0;
                            double with_reuse_total = avg_reuse_per_query_ms + margin_per_query_ms;
                            double without_reuse_total = avg_baseline_per_query_ms + margin_baseline_per_query_ms;
                            double total_speedup = without_reuse_total / with_reuse_total;
                            printf("| %5s | %-20s | %10s | %12s | %12.4f | %10s | %10s |\n",
                                   "", "With reuse (total)", "", "", with_reuse_total, "", "");
                            printf("| %5s | %-20s | %10s | %12s | %12.4f | %10s | %10s |\n",
                                   "", "Without reuse (total)", "", "", without_reuse_total, "", "");
                            printf("| %5s | %-20s | %10s | %12s | %12s | %9.2fx | %10s |\n",
                                   "", "Overall", "", "", "", total_speedup, "");
                        }
                        printf("|-------|----------------------|------------|--------------|--------------|------------|------------|\n");
                    }
                }
                printf("=======================================================================================================================\n");
                printf("      Accuracy%% = percentage where Reuse A* and From-scratch A* agree on whether GED <= tau\n");
                printf("=======================================================================================================================\n");
            }

            // Reuse speedup distribution per method
            for (const auto& bmethod : methods_with_baseline) {
                size_t gt3=0, x2_3=0, x1_2=0, x05_1=0, lt05=0;
                size_t sz1=0, sz2_10=0, szgt10=0;
                int nq = 0;
                for (const auto& result : all_results) {
                    if (result.method == bmethod) {
                        nq++;
                        gt3 += result.details.reuse_speedup_gt3x_qd;
                        x2_3 += result.details.reuse_speedup_2x_3x_qd;
                        x1_2 += result.details.reuse_speedup_1x_2x_qd;
                        x05_1 += result.details.reuse_speedup_05x_1x_qd;
                        lt05 += result.details.reuse_speedup_lt05x_qd;
                        sz1 += result.details.reuse_snapshot_size_1_qd;
                        sz2_10 += result.details.reuse_snapshot_size_2_10_qd;
                        szgt10 += result.details.reuse_snapshot_size_gt10_qd;
                    }
                }
                size_t total = gt3 + x2_3 + x1_2 + x05_1 + lt05;
                if (total > 0 && nq > 0) {
                    double n = nq;
                    printf("\n");
                    printf("=======================================================================================================================\n");
                    printf("                     Reuse Speedup Distribution [%s] (avg per query)\n", bmethod.c_str());
                    printf("=======================================================================================================================\n");
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "> 3x (reuse much faster)", gt3/n, 100.0*gt3/total);
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "2x - 3x", x2_3/n, 100.0*x2_3/total);
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "1x - 2x (reuse faster)", x1_2/n, 100.0*x1_2/total);
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "0.5x - 1x (reuse slower)", x05_1/n, 100.0*x05_1/total);
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "< 0.5x (reuse much slower)", lt05/n, 100.0*lt05/total);
                    printf("|--------------------------------|------------|----------|\n");
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "Snapshot size = 1 (dummy)", sz1/n, 100.0*sz1/total);
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "Snapshot size 2-10", sz2_10/n, 100.0*sz2_10/total);
                    printf("| %-30s | %10.1f | %8.1f%% |\n", "Snapshot size > 10", szgt10/n, 100.0*szgt10/total);
                    printf("=======================================================================================================================\n");

                    // Cross-tab: snapshot_size × speedup
                    size_t xt[3][5] = {};
                    double r_time[3] = {}, b_time[3] = {};
                    for (const auto& result : all_results) {
                        if (result.method == bmethod) {
                            for (int i = 0; i < 3; ++i) {
                                for (int j = 0; j < 5; ++j)
                                    xt[i][j] += result.details.reuse_xtab_qd[i][j];
                                r_time[i] += result.details.reuse_xtime_qd[i];
                                b_time[i] += result.details.baseline_xtime_qd[i];
                            }
                        }
                    }
                    const char* sz_labels[3] = {"dummy (size=1)", "small (2-10)", "large (>10)"};
                    const char* sp_labels[5] = {">3x", "2-3x", "1-2x", ".5-1x", "<.5x"};
                    printf("\n");
                    printf("=======================================================================================================================\n");
                    printf("      Cross-tab: Snapshot Size x Speedup [%s] (count per query, row%% in parens)\n", bmethod.c_str());
                    printf("=======================================================================================================================\n");
                    printf("| %-16s |", "size \\ speedup");
                    for (int j = 0; j < 5; ++j) printf(" %12s |", sp_labels[j]);
                    printf(" %8s | %10s | %10s | %8s |\n", "row_sum", "reuse(ms)", "baseln(ms)", "tot_sp");
                    printf("|------------------|--------------|--------------|--------------|--------------|--------------|----------|------------|------------|----------|\n");
                    for (int i = 0; i < 3; ++i) {
                        size_t row_sum = 0;
                        for (int j = 0; j < 5; ++j) row_sum += xt[i][j];
                        printf("| %-16s |", sz_labels[i]);
                        for (int j = 0; j < 5; ++j) {
                            double pct = row_sum > 0 ? 100.0 * xt[i][j] / row_sum : 0.0;
                            printf(" %6.1f(%4.1f%%) |", (double)xt[i][j]/n, pct);
                        }
                        double r_ms = r_time[i] * 1000.0 / n;   // per query ms
                        double b_ms = b_time[i] * 1000.0 / n;
                        double tot_sp = r_time[i] > 0 ? b_time[i] / r_time[i] : 0.0;
                        printf(" %8.1f | %10.3f | %10.3f | %7.2fx |\n",
                               (double)row_sum/n, r_ms, b_ms, tot_sp);
                    }
                    printf("=======================================================================================================================\n");
                    printf("  tot_sp = sum(baseline_time) / sum(reuse_time)  — time-weighted speedup per snapshot-size bucket\n");

                    // Chain depth distribution table
                    size_t cd_count[QueryDetails::QD_MAX_CHAIN_DEPTH] = {};
                    size_t cd_correct[QueryDetails::QD_MAX_CHAIN_DEPTH] = {};
                    size_t cd_pos[QueryDetails::QD_MAX_CHAIN_DEPTH] = {};
                    size_t cd_tp[QueryDetails::QD_MAX_CHAIN_DEPTH] = {};
                    double cd_rt[QueryDetails::QD_MAX_CHAIN_DEPTH] = {};
                    double cd_bt[QueryDetails::QD_MAX_CHAIN_DEPTH] = {};
                    for (const auto& result : all_results) {
                        if (result.method == bmethod) {
                            for (int i = 0; i < QueryDetails::QD_MAX_CHAIN_DEPTH; ++i) {
                                cd_count[i]   += result.details.chain_depth_count_qd[i];
                                cd_correct[i] += result.details.chain_depth_correct_qd[i];
                                cd_pos[i]     += result.details.chain_depth_pos_qd[i];
                                cd_tp[i]      += result.details.chain_depth_tp_qd[i];
                                cd_rt[i]      += result.details.chain_depth_reuse_time_qd[i];
                                cd_bt[i]      += result.details.chain_depth_baseline_time_qd[i];
                            }
                        }
                    }
                    size_t cd_total = 0;
                    int cd_max_depth = -1;
                    for (int i = 0; i < QueryDetails::QD_MAX_CHAIN_DEPTH; ++i) {
                        cd_total += cd_count[i];
                        if (cd_count[i] > 0) cd_max_depth = i;
                    }
                    if (cd_total > 0) {
                        double n = nq;
                        printf("\n");
                        printf("=======================================================================================================================\n");
                        printf("                  Chain Reuse Depth Distribution [%s] (per query)\n", bmethod.c_str());
                        printf("=======================================================================================================================\n");
                        printf("| %-14s | %9s | %6s | %8s | %8s | %8s |\n",
                               "Depth", "Count/Q", "Pct%", "Speedup", "Recall%", "Acc%");
                        printf("|----------------|-----------|--------|----------|----------|----------|\n");
                        for (int d = 0; d <= cd_max_depth; ++d) {
                            if (cd_count[d] == 0) continue;
                            double pct = 100.0 * cd_count[d] / cd_total;
                            double sp = cd_rt[d] > 0 ? cd_bt[d] / cd_rt[d] : 0.0;
                            double recall = cd_pos[d] > 0 ? 100.0 * cd_tp[d] / cd_pos[d] : 0.0;
                            double acc = cd_count[d] > 0 ? 100.0 * cd_correct[d] / cd_count[d] : 0.0;
                            char label[32];
                            if (d == 0) snprintf(label, sizeof(label), "0 (non-chain)");
                            else snprintf(label, sizeof(label), "%d", d);
                            printf("| %-14s | %9.1f | %5.1f%% | %7.2fx | %7.2f%% | %7.2f%% |\n",
                                   label, cd_count[d]/n, pct, sp, recall, acc);
                        }
                        printf("=======================================================================================================================\n");
                        printf("  Depth 0 = non-chain (parent used fresh A*). Recall = TP / (baseline ged<=tau).\n");
                        printf("  Acc = reuse-vs-baseline same-side-of-tau agreement.\n");
                    }
                }
            }

            // Reuse fail reason breakdown per method
            for (const auto& bmethod : methods_with_baseline) {
                // Aggregate fail stats for this method
                size_t total_nodes = 0;
                size_t fail_no_parent = 0, fail_empty = 0, fail_size_one = 0;
                size_t fail_root = 0, fail_no_op = 0, fail_multi_ops = 0;
                size_t fail_vertex = 0, fail_mo = 0;
                size_t reuse_count = 0;
                int num_queries = 0;

                for (const auto& result : all_results) {
                    if (result.method == bmethod) {
                        num_queries++;
                        total_nodes += result.details.total_ept_nodes;
                        fail_no_parent += result.details.reuse_fail_no_parent_snapshot;
                        fail_root += result.details.reuse_fail_root_node;
                        fail_multi_ops += result.details.reuse_fail_multi_ops;
                        fail_vertex += result.details.reuse_fail_vertex_count_changed;
                        reuse_count += result.details.reuse_count;
                    }
                }

                if (num_queries > 0) {
                    double nq = num_queries;

                    // Aggregate no_parent sub-reasons
                    size_t np_lp = 0, np_filter = 0, np_reuse = 0, np_other = 0;
                    for (const auto& result : all_results) {
                        if (result.method == bmethod) {
                            np_lp += result.details.reuse_fail_no_parent_lp_skipped;
                            np_filter += result.details.reuse_fail_no_parent_filter_skipped;
                            np_reuse += result.details.reuse_fail_no_parent_reuse_no_chain;
                            np_other += result.details.reuse_fail_no_parent_other;
                        }
                    }

                    printf("\n");
                    printf("=======================================================================================================================\n");
                    printf("                     Reuse Fail Reason Breakdown [%s] (avg per query)\n", bmethod.c_str());
                    printf("=======================================================================================================================\n");
                    printf("| %-40s | %12.1f |\n", "Reuse Success", reuse_count / nq);
                    printf("| %-40s | %12.1f |\n", "Fail: Root Node", fail_root / nq);
                    printf("| %-40s | %12.1f |\n", "Fail: Vertex Count Changed", fail_vertex / nq);
                    printf("| %-40s | %12.1f |\n", "Fail: Multi Ops (>max_ged_gap)", fail_multi_ops / nq);
                    printf("| %-40s | %12.1f |\n", "Fail: No Parent Snapshot (total)", fail_no_parent / nq);
                    printf("|   %-38s | %12.1f |\n", "- Parent LP skipped", np_lp / nq);
                    printf("|   %-38s | %12.1f |\n", "- Parent filter skipped", np_filter / nq);
                    printf("|   %-38s | %12.1f |\n", "- Parent reuse, no chain_reuse", np_reuse / nq);
                    printf("|   %-38s | %12.1f |\n", "- Other (ged<0/INF, empty snapshot)", np_other / nq);
                    printf("=======================================================================================================================\n");
                    printf("  (sub-reasons sum to total: %.1f = %.1f + %.1f + %.1f + %.1f)\n",
                           fail_no_parent / nq, np_lp / nq, np_filter / nq, np_reuse / nq, np_other / nq);

                    // LP-skip × reuse-able cross-tab: of nodes LP skipped, how many were structurally reuse-able?
                    size_t lp_re = 0, lp_nre = 0;
                    for (const auto& result : all_results) {
                        if (result.method == bmethod) {
                            lp_re  += result.details.lp_skip_reuseable;
                            lp_nre += result.details.lp_skip_not_reuseable;
                        }
                    }
                    size_t lp_total = lp_re + lp_nre;
                    if (lp_total > 0) {
                        printf("\n");
                        printf("  LP-skipped nodes: %.1f/q total, of which structurally reuse-able = %.1f (%.1f%%), not = %.1f (%.1f%%)\n",
                               lp_total / nq, lp_re / nq, 100.0 * lp_re / lp_total,
                               lp_nre / nq, 100.0 * lp_nre / lp_total);
                    }
                }
            }

            // 显示每个 method+tau 的样本对
            for (const auto& bmethod : methods_with_baseline) {
                printf("\n");
                printf("=======================================================================================================================\n");
                printf("              Sample Pairs (Reuse A* vs From-scratch A*, up to 10 per tau) [%s]\n", bmethod.c_str());
                printf("=======================================================================================================================\n");

                for (const auto& tau : tau_list) {
                    std::vector<std::pair<double, double>> tau_samples;
                    for (const auto& result : all_results) {
                        if (result.method == bmethod && result.tau == tau) {
                            for (const auto& sample : result.details.baseline_samples) {
                                tau_samples.push_back(sample);
                                if (tau_samples.size() >= 10) break;
                            }
                            if (tau_samples.size() >= 10) break;
                        }
                    }

                    if (!tau_samples.empty()) {
                        printf("\nTau = %.1f (showing %zu samples):\n", tau, tau_samples.size());
                        printf("| %-6s | %-14s | %-14s | %-10s |\n", "Sample", "Reuse A*(ms)", "From-scratch(ms)", "Speedup");
                        printf("|--------|----------------|----------------|------------|\n");

                        for (size_t i = 0; i < tau_samples.size(); ++i) {
                            double reuse_ms = tau_samples[i].first;
                            double baseline_ms = tau_samples[i].second;
                            double speedup = (reuse_ms > 0) ? baseline_ms / reuse_ms : 0.0;
                            printf("| %6zu | %14.6f | %14.6f | %9.2fx |\n",
                                   i + 1, reuse_ms, baseline_ms, speedup);
                        }
                        printf("|--------|----------------|----------------|------------|\n");
                    }
                }
                printf("=======================================================================================================================\n");
            }
        }
    }

    // Final message
    printf("\n[experiment_mode] All experiments completed\n");
    printf("[experiment_mode] Total time: %.3f seconds\n", total_exp_duration.count());
    if (config.save) {
        printf("[experiment_mode] Results saved to: %s\n", exp_dir.c_str());

        // Write summary.txt file
        std::filesystem::path summary_path = std::filesystem::path(exp_dir) / "summary.txt";
        std::ofstream summary_file(summary_path);
        if (summary_file.is_open()) {
            printf("[DEBUG] Writing summary to: %s\n", summary_path.string().c_str());
            // Write parameters section
            summary_file << "================================================================================\n";
            summary_file << "                           EXPERIMENT PARAMETERS\n";
            summary_file << "================================================================================\n";
            summary_file << "Dataset:          " << config.dataset << "\n";
            summary_file << "DB size:          " << db.size() << "\n";
            summary_file << "Query range:      [" << q_start << ", " << q_end << "] (" << num_queries << " queries)\n";
            summary_file << "Methods:          " << config.methods << "\n";
            summary_file << "Tau values:       " << config.tau_values << "\n";
            summary_file << "Alpha:            " << config.alpha << "\n";
            summary_file << "Tau index:        " << config.tau_index << "\n";
            summary_file << "App max iter:     " << config.app_max_iter << "\n";
            summary_file << "ND mode:          " << config.nd_mode << "\n";
            summary_file << "DFS mode:         " << (config.dfs_mode.empty() ? "unified" : config.dfs_mode) << "\n";
            summary_file << "Fast down:        " << (config.disable_fast_down ? "disabled" : "enabled") << "\n";
            summary_file << "Disable all LSa:  " << (config.disable_all_lsa ? "true" : "false") << "\n";
            summary_file << "  - LSa pruning:  " << (config.disable_lsa_pruning ? "disabled" : "enabled") << "\n";
            summary_file << "  - Reuse LSa:    " << (config.disable_reuse_lsa ? "disabled" : "enabled") << "\n";
            summary_file << "Verify baseline:  " << (config.verify_reuse ? "true" : "false") << "\n";
            summary_file << "Chain reuse:      " << (config.chain_reuse ? "true" : "false") << "\n";
            summary_file << "Only DB graph:    " << ((config.only_compute_db_graph || !config.disable_ept_filters) ? "true" : "false") << "\n";
            summary_file << "Total time:       " << total_exp_duration.count() << " seconds\n";
            summary_file << "================================================================================\n\n";

            // Write main comparison table
            summary_file << "========================================================================================================================================================\n";
            summary_file << "                                               COMPARISON TABLE [Dataset: " << config.dataset << "]\n";
            summary_file << "========================================================================================================================================================\n";

            char buf[512];
            snprintf(buf, sizeof(buf), "| %-16s | %5s | %7s | %7s | %7s | %10s | %8s | %10s | %8s | %8s | %10s | %10s | %10s |\n",
                   "Method", "Tau", "Recall%", "Prec%", "IoU%", "Filter(s)", "Filter#", "Verify(s)", "Verify#", "NDC#", "ND(s)", "EPT(s)", "Total(s)");
            summary_file << buf;
            summary_file << "|------------------|-------|---------|---------|---------|------------|----------|------------|----------|----------|------------|------------|------------|\n";

            for (const auto& tau : tau_list) {
                for (const auto& method : methods_list) {
                    int queries_with_gt = 0;
                    int queries_with_results = 0;
                    double sum_recall = 0.0, sum_precision = 0.0, sum_iou = 0.0;
                    double total_time = 0.0;
                    double total_nd_lb_time = 0.0, total_nd_astar_time = 0.0;
                    double total_ept_lb_time = 0.0, total_ept_astar_time = 0.0;
                    long long total_nd_lb_count = 0, total_nd_astar_count = 0;
                    long long total_ept_lb_count = 0, total_ept_astar_count = 0;
                    long long total_nd_ndc_count = 0, total_ept_ndc_count = 0;
                    int total_queries = 0;

                    for (const auto& result : all_results) {
                        if (result.method == method && result.tau == tau) {
                            total_queries++;
                            total_time += result.details.time;
                            total_nd_lb_time += result.details.nd_lb_time;
                            total_nd_astar_time += result.details.nd_astar_time;
                            total_ept_lb_time += result.details.ept_lb_time;
                            total_ept_astar_time += result.details.ept_astar_time;
                            total_nd_lb_count += result.details.nd_lb_count;
                            total_nd_astar_count += result.details.nd_astar_count;
                            total_nd_ndc_count += result.details.nd_ndc_count;
                            total_ept_lb_count += result.details.ept_lb_count;
                            total_ept_astar_count += result.details.ept_astar_count;
                            total_ept_ndc_count += result.details.ept_ndc_count;
                            if (result.details.has_ground_truth) {
                                queries_with_gt++;
                                sum_recall += result.details.recall;
                                sum_iou += result.details.iou;
                            }
                            if (result.details.has_results) {
                                queries_with_results++;
                                sum_precision += result.details.precision;
                            }
                        }
                    }

                    double avg_recall = queries_with_gt > 0 ? sum_recall / queries_with_gt * 100 : 0.0;
                    double avg_precision = queries_with_results > 0 ? sum_precision / queries_with_results * 100 : 0.0;
                    double avg_iou = queries_with_gt > 0 ? sum_iou / queries_with_gt * 100 : 0.0;
                    double avg_time = total_queries > 0 ? total_time / total_queries : 0.0;
                    double avg_nd_lb = total_queries > 0 ? total_nd_lb_time / total_queries : 0.0;
                    double avg_nd_astar = total_queries > 0 ? total_nd_astar_time / total_queries : 0.0;
                    double avg_ept_lb = total_queries > 0 ? total_ept_lb_time / total_queries : 0.0;
                    double avg_ept_astar = total_queries > 0 ? total_ept_astar_time / total_queries : 0.0;
                    double avg_filter_count = total_queries > 0 ? (double)(total_nd_lb_count + total_ept_lb_count) / total_queries : 0.0;
                    double avg_verify_count = total_queries > 0 ? (double)(total_nd_astar_count + total_ept_astar_count) / total_queries : 0.0;
                    double avg_ndc_count = total_queries > 0 ? (double)(total_nd_ndc_count + total_ept_ndc_count) / total_queries : 0.0;

                    double filter_time = avg_nd_lb + avg_ept_lb;
                    double verify_time = avg_nd_astar + avg_ept_astar;
                    double nd_time = avg_nd_lb + avg_nd_astar;
                    double ept_time = avg_ept_lb + avg_ept_astar;

                    auto fmt_time_f = [](double t) -> std::string {
                        if (t < 1e-9) return "N/A";
                        char b[32]; snprintf(b, sizeof(b), "%.6f", t); return std::string(b);
                    };
                    auto fmt_count_f = [](double c) -> std::string {
                        if (c < 0.001) return "N/A";
                        char b[32]; snprintf(b, sizeof(b), "%.1f", c); return std::string(b);
                    };

                    bool has_nd_ept = (method.find("Gisma") != std::string::npos ||
                                       method.find("Base+SS") != std::string::npos ||
                                       method.find("Base+GS") != std::string::npos);
                    std::string nd_str = has_nd_ept ? fmt_time_f(nd_time) : "N/A";
                    std::string ept_str = has_nd_ept ? fmt_time_f(ept_time) : "N/A";

                    snprintf(buf, sizeof(buf), "| %-16s | %5.1f | %7.2f | %7.2f | %7.2f | %10s | %8s | %10s | %8s | %8s | %10s | %10s | %10.6f |\n",
                           method.c_str(), tau, avg_recall, avg_precision, avg_iou,
                           fmt_time_f(filter_time).c_str(), fmt_count_f(avg_filter_count).c_str(),
                           fmt_time_f(verify_time).c_str(), fmt_count_f(avg_verify_count).c_str(),
                           fmt_count_f(avg_ndc_count).c_str(), nd_str.c_str(), ept_str.c_str(), avg_time);
                    summary_file << buf;
                }
                if (&tau != &tau_list.back()) {
                    summary_file << "|------------------|-------|---------|---------|---------|------------|----------|------------|----------|----------|------------|------------|------------|\n";
                }
            }
            summary_file << "========================================================================================================================================================\n";
            summary_file << "Note: Filter = ND_lb + EPT_lb, Verify = ND_astar + EPT_astar, NDC = unique nodes computed (deduplicated)\n";
            summary_file << "      ND(s) = NetDag phase total time, EPT(s) = EditPathTree phase total time\n\n";

            // Write Gisma detailed statistics if applicable
            if (has_gisma_methods) {
                summary_file << "==========================================================================================================================================\n";
                summary_file << "                                  GISMA DETAILED STATISTICS (ND vs EPT) [Dataset: " << config.dataset << "]\n";
                summary_file << "==========================================================================================================================================\n";
                snprintf(buf, sizeof(buf), "| %-16s | %5s | %10s | %12s | %10s | %12s | %10s | %12s | %10s | %12s |\n",
                       "Method", "Tau", "ND_Flt#", "ND_Flt(s)", "ND_Vfy#", "ND_Vfy(s)", "EPT_Flt#", "EPT_Flt(s)", "EPT_Vfy#", "EPT_Vfy(s)");
                summary_file << buf;
                summary_file << "|------------------|-------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|";

                for (const auto& tau : tau_list) {
                    summary_file << "\n";
                    for (const auto& method : methods_list) {
                        if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                            continue;
                        }

                        double total_nd_lb_time = 0.0, total_nd_astar_time = 0.0;
                        double total_ept_lb_time = 0.0, total_ept_astar_time = 0.0;
                        long long total_nd_lb_count = 0, total_nd_astar_count = 0;
                        long long total_ept_lb_count = 0, total_ept_astar_count = 0;
                        int total_queries = 0;

                        for (const auto& result : all_results) {
                            if (result.method == method && result.tau == tau) {
                                total_queries++;
                                total_nd_lb_time += result.details.nd_lb_time;
                                total_nd_astar_time += result.details.nd_astar_time;
                                total_ept_lb_time += result.details.ept_lb_time;
                                total_ept_astar_time += result.details.ept_astar_time;
                                total_nd_lb_count += result.details.nd_lb_count;
                                total_nd_astar_count += result.details.nd_astar_count;
                                total_ept_lb_count += result.details.ept_lb_count;
                                total_ept_astar_count += result.details.ept_astar_count;
                            }
                        }

                        double avg_nd_lb_count = total_queries > 0 ? (double)total_nd_lb_count / total_queries : 0.0;
                        double avg_nd_lb_time = total_queries > 0 ? total_nd_lb_time / total_queries : 0.0;
                        double avg_nd_astar_count = total_queries > 0 ? (double)total_nd_astar_count / total_queries : 0.0;
                        double avg_nd_astar_time = total_queries > 0 ? total_nd_astar_time / total_queries : 0.0;
                        double avg_ept_lb_count = total_queries > 0 ? (double)total_ept_lb_count / total_queries : 0.0;
                        double avg_ept_lb_time = total_queries > 0 ? total_ept_lb_time / total_queries : 0.0;
                        double avg_ept_astar_count = total_queries > 0 ? (double)total_ept_astar_count / total_queries : 0.0;
                        double avg_ept_astar_time = total_queries > 0 ? total_ept_astar_time / total_queries : 0.0;

                        snprintf(buf, sizeof(buf), "| %-16s | %5.1f | %10.1f | %12.6f | %10.1f | %12.6f | %10.1f | %12.6f | %10.1f | %12.6f |\n",
                               method.c_str(), tau, avg_nd_lb_count, avg_nd_lb_time,
                               avg_nd_astar_count, avg_nd_astar_time, avg_ept_lb_count, avg_ept_lb_time,
                               avg_ept_astar_count, avg_ept_astar_time);
                        summary_file << buf;
                    }
                    if (&tau != &tau_list.back()) {
                        summary_file << "|------------------|-------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|";
                    }
                }
                summary_file << "\n==========================================================================================================================================\n";
                summary_file << "Note: Flt = Filter (lower bound), Vfy = Verify (A*/App GED computation)\n";
                summary_file << "      ND = NetDag phase, EPT = EditPathTree phase\n\n";

                // Write Three Optimizations table
                summary_file << "===================================================================================================\n";
                summary_file << "                         THREE OPTIMIZATIONS STATISTICS [Dataset: " << config.dataset << "]\n";
                summary_file << "===================================================================================================\n";
                snprintf(buf, sizeof(buf), "| %-16s | %5s | %12s | %12s | %12s | %12s | %12s |\n",
                       "Method", "Tau", "SubtreePrn#", "SubtreePrnAv", "LB-Prop#", "LB-Prune#", "Reuse#");
                summary_file << buf;
                summary_file << "|------------------|-------|--------------|--------------|--------------|--------------|--------------|";

                for (const auto& tau : tau_list) {
                    summary_file << "\n";
                    for (const auto& method : methods_list) {
                        if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                            continue;
                        }

                        long long total_subtree_pruning_decisions = 0;
                        long long total_subtree_pruned = 0;
                        long long total_lb_propagation_count = 0;
                        long long total_lb_pruning_count = 0;
                        long long total_reuse_count = 0;
                        int total_queries = 0;

                        for (const auto& result : all_results) {
                            if (result.method == method && result.tau == tau) {
                                total_queries++;
                                total_subtree_pruning_decisions += result.details.subtree_pruning_decisions;
                                total_subtree_pruned += result.details.subtree_pruned;
                                total_lb_propagation_count += result.details.lb_propagation_count;
                                total_lb_pruning_count += result.details.lb_pruning_count;
                                total_reuse_count += result.details.reuse_count;
                            }
                        }

                        double avg_subtree_pruning_decisions = total_queries > 0 ? (double)total_subtree_pruning_decisions / total_queries : 0.0;
                        double avg_subtree_pruned = total_queries > 0 ? (double)total_subtree_pruned / total_queries : 0.0;
                        double avg_lb_propagation_count = total_queries > 0 ? (double)total_lb_propagation_count / total_queries : 0.0;
                        double avg_lb_pruning_count = total_queries > 0 ? (double)total_lb_pruning_count / total_queries : 0.0;
                        double avg_reuse_count = total_queries > 0 ? (double)total_reuse_count / total_queries : 0.0;

                        snprintf(buf, sizeof(buf), "| %-16s | %5.1f | %12.1f | %12.1f | %12.1f | %12.1f | %12.1f |\n",
                               method.c_str(), tau, avg_subtree_pruning_decisions, avg_subtree_pruned,
                               avg_lb_propagation_count, avg_lb_pruning_count, avg_reuse_count);
                        summary_file << buf;
                    }
                    if (&tau != &tau_list.back()) {
                        summary_file << "|------------------|-------|--------------|--------------|--------------|--------------|--------------|";
                    }
                }
                summary_file << "\n===================================================================================================\n";
                summary_file << "Note: SubtreePrn# = Subtree Pruning decisions, SubtreePrnAv = nodes avoided by pruning\n";
                summary_file << "      LB-Prop# = LB Propagation count, LB-Prune# = LB Pruning (GED computations skipped)\n";
                summary_file << "      Reuse# = Search Tree Reuse success count\n\n";

                // Write DB Graph vs Intermediate Graph Statistics table
                summary_file << "===============================================================================================================================\n";
                summary_file << "                           DB GRAPH vs INTERMEDIATE GRAPH STATISTICS [Dataset: " << config.dataset << "]\n";
                summary_file << "===============================================================================================================================\n";
                snprintf(buf, sizeof(buf), "| %-16s | %5s | %9s | %10s | %9s | %10s |\n",
                       "Method", "Tau", "DB_Flt#", "DB_Flt(s)", "Int_Flt#", "Int_Flt(s)");
                summary_file << buf;
                summary_file << "|------------------|-------|-----------|------------|-----------|------------|";

                for (const auto& tau : tau_list) {
                    summary_file << "\n";
                    for (const auto& method : methods_list) {
                        if (method != "Gisma" && method != "Gisma-default" && method != "Gisma-no-reuse" && method != "Gisma-only-dfs" && method != "Gisma-no-SP" && method != "Gisma-no-LP" && method != "Base+SS" && method != "Base+GS" && method != "Base_All_EPT" && method != "gisma-lsa" && method != "app-lsa" && method != "astar-lsa") {
                            continue;
                        }

                        long long total_db_lb_count = 0;
                        double total_db_lb_time = 0.0;
                        long long total_inter_lb_count = 0;
                        double total_inter_lb_time = 0.0;
                        int total_queries = 0;

                        for (const auto& result : all_results) {
                            if (result.method == method && result.tau == tau) {
                                total_queries++;
                                total_db_lb_count += result.details.db_graph_lb_count;
                                total_db_lb_time += result.details.db_graph_lb_time;
                                total_inter_lb_count += result.details.intermediate_graph_lb_count;
                                total_inter_lb_time += result.details.intermediate_graph_lb_time;
                            }
                        }

                        double avg_db_lb_count = total_queries > 0 ? (double)total_db_lb_count / total_queries : 0.0;
                        double avg_db_lb_time = total_queries > 0 ? total_db_lb_time / total_queries : 0.0;
                        double avg_inter_lb_count = total_queries > 0 ? (double)total_inter_lb_count / total_queries : 0.0;
                        double avg_inter_lb_time = total_queries > 0 ? total_inter_lb_time / total_queries : 0.0;

                        snprintf(buf, sizeof(buf), "| %-16s | %5.1f | %9.1f | %10.6f | %9.1f | %10.6f |\n",
                               method.c_str(), tau, avg_db_lb_count, avg_db_lb_time, avg_inter_lb_count, avg_inter_lb_time);
                        summary_file << buf;
                    }
                    if (&tau != &tau_list.back()) {
                        summary_file << "|------------------|-------|-----------|------------|-----------|------------|";
                    }
                }
                summary_file << "\n===============================================================================================================================\n";
                summary_file << "Note: DB = anchor/completed graphs, Int = intermediate graphs (edited)\n";
                summary_file << "      Flt# = avg filter count per query, Flt(s) = avg filter time per query (seconds)\n\n";
            }

            // Write EXP-5 table if applicable
            bool has_baseline_data_summary = false;
            for (const auto& result : all_results) {
                if (result.details.baseline_app_count > 0) {
                    has_baseline_data_summary = true;
                    break;
                }
            }
            if (has_baseline_data_summary) {
                summary_file << "=======================================================================================================================\n";
                summary_file << "                                EXP-5: Search Tree Reuse Effectiveness                                                 \n";
                summary_file << "=======================================================================================================================\n";
                snprintf(buf, sizeof(buf), "| %-5s | %-20s | %-10s | %-12s | %-10s | %-10s |\n",
                       "Tau", "Method", "Count", "Avg Time(ms)", "Speedup", "Accuracy%");
                summary_file << buf;
                summary_file << "|-------|----------------------|------------|--------------|------------|------------|\n";

                for (const auto& tau : tau_list) {
                    size_t tau_baseline_count = 0;
                    double tau_baseline_time = 0.0;
                    size_t tau_all_reuse_count = 0;
                    double tau_all_reuse_time = 0.0;
                    double tau_astar_reuse_time = 0.0;
                    size_t tau_correct = 0;
                    size_t tau_incorrect = 0;

                    for (const auto& result : all_results) {
                        if (result.tau == tau) {
                            tau_baseline_count += result.details.baseline_app_count;
                            tau_baseline_time += result.details.baseline_app_time;
                            tau_all_reuse_count += result.details.reuse_count;
                            tau_all_reuse_time += result.details.reuse_success_time;
                            tau_astar_reuse_time += result.details.baseline_reuse_time;
                            tau_correct += result.details.baseline_correct;
                            tau_incorrect += result.details.baseline_incorrect;
                        }
                    }

                    if (tau_baseline_count > 0 && tau_all_reuse_count > 0) {
                        double avg_all_reuse_ms = (tau_all_reuse_time / tau_all_reuse_count) * 1000.0;
                        double avg_astar_reuse_ms = (tau_astar_reuse_time / tau_baseline_count) * 1000.0;
                        double avg_baseline_ms = (tau_baseline_time / tau_baseline_count) * 1000.0;
                        double speedup_astar = avg_baseline_ms / avg_astar_reuse_ms;
                        double accuracy_pct = (tau_correct + tau_incorrect > 0)
                            ? 100.0 * tau_correct / (tau_correct + tau_incorrect) : 0.0;

                        snprintf(buf, sizeof(buf), "| %5.1f | %-20s | %10zu | %12.4f | %10s | %10s |\n",
                               tau, "Reuse (All)", tau_all_reuse_count, avg_all_reuse_ms, "-", "-");
                        summary_file << buf;
                        snprintf(buf, sizeof(buf), "| %5s | %-20s | %10zu | %12.4f | %9.2fx | %9.2f%% |\n",
                               "", "Reuse (AStar only)", tau_baseline_count, avg_astar_reuse_ms, speedup_astar, accuracy_pct);
                        summary_file << buf;
                        snprintf(buf, sizeof(buf), "| %5s | %-20s | %10zu | %12.4f | %10s | %10s |\n",
                               "", "Baseline (App)", tau_baseline_count, avg_baseline_ms, "(base)", "(ref)");
                        summary_file << buf;
                        summary_file << "|-------|----------------------|------------|--------------|------------|------------|\n";
                    }
                }
                summary_file << "=======================================================================================================================\n";
                summary_file << "Note: Reuse(All) includes dummy_fast_path; Reuse(AStar only) = same count as Baseline\n";
                summary_file << "      Accuracy% = percentage where Reuse and Baseline agree on whether GED <= tau\n";
                summary_file << "=======================================================================================================================\n\n";

                // Write sample pairs
                summary_file << "=======================================================================================================================\n";
                summary_file << "                              EXP-5: Sample Pairs (Reuse A* vs From-scratch A*, up to 10 per tau)                                \n";
                summary_file << "=======================================================================================================================\n";

                for (const auto& tau : tau_list) {
                    std::vector<std::pair<double, double>> tau_samples;
                    for (const auto& result : all_results) {
                        if (result.tau == tau) {
                            for (const auto& sample : result.details.baseline_samples) {
                                tau_samples.push_back(sample);
                                if (tau_samples.size() >= 10) break;
                            }
                            if (tau_samples.size() >= 10) break;
                        }
                    }

                    if (!tau_samples.empty()) {
                        snprintf(buf, sizeof(buf), "\nTau = %.1f (showing %zu samples):\n", tau, tau_samples.size());
                        summary_file << buf;
                        snprintf(buf, sizeof(buf), "| %-6s | %-14s | %-14s | %-10s |\n", "Sample", "Reuse A*(ms)", "From-scratch(ms)", "Speedup");
                        summary_file << buf;
                        summary_file << "|--------|----------------|----------------|------------|\n";

                        for (size_t i = 0; i < tau_samples.size(); ++i) {
                            double reuse_ms = tau_samples[i].first;
                            double baseline_ms = tau_samples[i].second;
                            double speedup = (reuse_ms > 0) ? baseline_ms / reuse_ms : 0.0;
                            snprintf(buf, sizeof(buf), "| %6zu | %14.6f | %14.6f | %9.2fx |\n",
                                   i + 1, reuse_ms, baseline_ms, speedup);
                            summary_file << buf;
                        }
                        summary_file << "|--------|----------------|----------------|------------|\n";
                    }
                }
                summary_file << "=======================================================================================================================\n";
            }

            summary_file.flush();
            if (summary_file.good()) {
                summary_file.close();
                printf("Summary saved to: %s\n", summary_path.string().c_str());
            } else {
                printf("[ERROR] Failed to write summary file (stream error)\n");
                summary_file.close();
            }
        } else {
            printf("[WARNING] Failed to open summary file: %s\n", summary_path.string().c_str());
        }

        // 12b. Write individual summary.txt for each (method, tau) combination in latest/
        printf("[experiment_mode] Writing individual summaries to latest/ directories...\n");
        for (const auto& method : methods_list) {
            for (const auto& tau : tau_list) {
                // 计算该方法+tau的统计数据
                int queries_with_gt = 0;
                int queries_with_results = 0;
                double sum_recall = 0.0, sum_precision = 0.0, sum_iou = 0.0;
                double total_time = 0.0;
                double total_nd_lb_time = 0.0, total_nd_astar_time = 0.0;
                double total_ept_lb_time = 0.0, total_ept_astar_time = 0.0;
                long long total_nd_lb_count = 0, total_nd_astar_count = 0;
                long long total_ept_lb_count = 0, total_ept_astar_count = 0;
                long long total_nd_ndc_count = 0, total_ept_ndc_count = 0;
                long long total_e7_ept_trees = 0, total_e7_answer_depth_sum = 0, total_e7_answer_count = 0;  // E7
                int total_queries = 0;

                for (const auto& result : all_results) {
                    if (result.method == method && result.tau == tau) {
                        total_queries++;
                        total_time += result.details.time;
                        total_nd_lb_time += result.details.nd_lb_time;
                        total_nd_astar_time += result.details.nd_astar_time;
                        total_ept_lb_time += result.details.ept_lb_time;
                        total_ept_astar_time += result.details.ept_astar_time;
                        total_nd_lb_count += result.details.nd_lb_count;
                        total_nd_astar_count += result.details.nd_astar_count;
                        total_nd_ndc_count += result.details.nd_ndc_count;
                        total_ept_lb_count += result.details.ept_lb_count;
                        total_ept_astar_count += result.details.ept_astar_count;
                        total_ept_ndc_count += result.details.ept_ndc_count;
                        total_e7_ept_trees += result.details.e7_ept_trees;
                        total_e7_answer_depth_sum += result.details.e7_answer_depth_sum;
                        total_e7_answer_count += result.details.e7_answer_count;
                        if (result.details.has_ground_truth) {
                            queries_with_gt++;
                            sum_recall += result.details.recall;
                            sum_iou += result.details.iou;
                        }
                        if (result.details.has_results) {
                            queries_with_results++;
                            sum_precision += result.details.precision;
                        }
                    }
                }

                double avg_recall = queries_with_gt > 0 ? sum_recall / queries_with_gt * 100 : 0.0;
                double avg_precision = queries_with_results > 0 ? sum_precision / queries_with_results * 100 : 0.0;
                double avg_iou = queries_with_gt > 0 ? sum_iou / queries_with_gt * 100 : 0.0;
                double avg_time = total_queries > 0 ? total_time / total_queries : 0.0;
                double avg_nd_lb = total_queries > 0 ? total_nd_lb_time / total_queries : 0.0;
                double avg_nd_astar = total_queries > 0 ? total_nd_astar_time / total_queries : 0.0;
                double avg_ept_lb = total_queries > 0 ? total_ept_lb_time / total_queries : 0.0;
                double avg_ept_astar = total_queries > 0 ? total_ept_astar_time / total_queries : 0.0;
                double avg_filter_count = total_queries > 0 ? (double)(total_nd_lb_count + total_ept_lb_count) / total_queries : 0.0;
                double avg_verify_count = total_queries > 0 ? (double)(total_nd_astar_count + total_ept_astar_count) / total_queries : 0.0;
                double avg_ndc_count = total_queries > 0 ? (double)(total_nd_ndc_count + total_ept_ndc_count) / total_queries : 0.0;
                // E7 search-time memory: 分别报「展开的 NetDag ball 数」(giant-step) 与「EPT 节点数」(small-step)
                double avg_nd_ndc = total_queries > 0 ? (double)total_nd_ndc_count / total_queries : 0.0;
                double avg_ept_ndc = total_queries > 0 ? (double)total_ept_ndc_count / total_queries : 0.0;
                double filter_time = avg_nd_lb + avg_ept_lb;
                double verify_time = avg_nd_astar + avg_ept_astar;
                double nd_time = avg_nd_lb + avg_nd_astar;
                double ept_time = avg_ept_lb + avg_ept_astar;

                // Write summary.txt
                std::filesystem::path latest_summary_path = latest_base_subdirs.at(method).at(tau) + "/summary.txt";
                std::ofstream latest_summary(latest_summary_path);
                if (latest_summary.is_open()) {
                    latest_summary << "Method:     " << method << "\n";
                    latest_summary << "Tau:        " << std::fixed << std::setprecision(1) << tau << "\n";
                    latest_summary << "Dataset:    " << config.dataset << "\n";
                    latest_summary << "Queries:    " << total_queries << " (range: [" << q_start << ", " << q_end << "])\n";
                    latest_summary << "Timestamp:  " << timestamp_str << "\n";
                    latest_summary << "Archive:    archive/" << config.dataset << "/" << timestamp_str << "/" << method << "_tau_" << std::fixed << std::setprecision(1) << tau << "/\n";
                    latest_summary << "\n";

                    latest_summary << "--- Parameters ---\n";
                    latest_summary << "Alpha:          " << config.alpha << "\n";
                    latest_summary << "Tau index:      " << config.tau_index << "\n";
                    latest_summary << "App max iter:   " << config.app_max_iter << "\n";
                    latest_summary << "ND mode:        " << config.nd_mode << "\n";
                    latest_summary << "DFS mode:       " << (config.dfs_mode.empty() ? "unified" : config.dfs_mode) << "\n";
                    latest_summary << "Fast down:      " << (config.disable_fast_down ? "disabled" : "enabled") << "\n";
                    latest_summary << "LSa pruning:    " << (config.disable_lsa_pruning ? "disabled" : "enabled") << "\n";
                    latest_summary << "Reuse LSa:      " << (config.disable_reuse_lsa ? "disabled" : "enabled") << "\n";
                    latest_summary << "\n";

                    latest_summary << "--- Results ---\n";
                    latest_summary << "Recall:     " << std::fixed << std::setprecision(2) << avg_recall << "%\n";
                    latest_summary << "Precision:  " << std::fixed << std::setprecision(2) << avg_precision << "%\n";
                    latest_summary << "IoU:        " << std::fixed << std::setprecision(2) << avg_iou << "%\n";
                    latest_summary << "Avg Time:   " << std::fixed << std::setprecision(6) << avg_time << "s\n";
                    latest_summary << "Filter:     " << std::fixed << std::setprecision(6) << filter_time << "s (count: " << std::fixed << std::setprecision(1) << avg_filter_count << ")\n";
                    latest_summary << "Verify:     " << std::fixed << std::setprecision(6) << verify_time << "s (count: " << std::fixed << std::setprecision(1) << avg_verify_count << ")\n";

                    bool has_nd_ept = (method.find("Gisma") != std::string::npos ||
                                       method.find("Base+SS") != std::string::npos ||
                                       method.find("Base+GS") != std::string::npos);
                    if (has_nd_ept) {
                        latest_summary << "ND time:    " << std::fixed << std::setprecision(6) << nd_time << "s\n";
                        latest_summary << "EPT time:   " << std::fixed << std::setprecision(6) << ept_time << "s\n";
                        latest_summary << "NDC count:  " << std::fixed << std::setprecision(1) << avg_ndc_count << "\n";
                        // E7: balls expanded (NetDag) + EPT nodes，单独列出供 search-time memory 分析
                        latest_summary << "ND NDC:     " << std::fixed << std::setprecision(1) << avg_nd_ndc << "\n";
                        latest_summary << "EPT NDC:    " << std::fixed << std::setprecision(1) << avg_ept_ndc << "\n";
                        if (config.e7_stats) {
                            double avg_e7_trees = total_queries > 0 ? (double)total_e7_ept_trees / total_queries : 0.0;
                            double avg_e7_depth = total_e7_answer_count > 0 ? (double)total_e7_answer_depth_sum / total_e7_answer_count : 0.0;
                            long e7_peak_kb = getPeakRSS_kb();
                            double e7_search_mem_mb = (e7_peak_kb > e7_rss_baseline_kb) ? (double)(e7_peak_kb - e7_rss_baseline_kb) / 1024.0 : 0.0;
                            latest_summary << "E7 EPT trees:     " << std::fixed << std::setprecision(2) << avg_e7_trees << "\n";
                            latest_summary << "E7 answer depth:  " << std::fixed << std::setprecision(2) << avg_e7_depth << "\n";
                            latest_summary << "E7 search mem MB: " << std::fixed << std::setprecision(2) << e7_search_mem_mb << "\n";
                        }
                    }

                    latest_summary.close();
                }

                // Write summary.json
                {
                    std::filesystem::path json_path = latest_base_subdirs.at(method).at(tau) + "/summary.json";
                    std::ofstream json_file(json_path);
                    if (json_file.is_open()) {
                        json_file << "{\n";
                        json_file << "  \"method\": \"" << method << "\",\n";
                        json_file << "  \"tau\": " << std::fixed << std::setprecision(1) << tau << ",\n";
                        json_file << "  \"dataset\": \"" << config.dataset << "\",\n";
                        json_file << "  \"queries\": " << total_queries << ",\n";
                        json_file << "  \"query_range\": [" << q_start << ", " << q_end << "],\n";
                        json_file << "  \"timestamp\": \"" << timestamp_str << "\",\n";
                        json_file << "  \"parameters\": {\n";
                        json_file << "    \"alpha\": " << config.alpha << ",\n";
                        json_file << "    \"tau_index\": " << config.tau_index << ",\n";
                        json_file << "    \"app_max_iter\": " << config.app_max_iter << ",\n";
                        json_file << "    \"nd_mode\": \"" << config.nd_mode << "\",\n";
                        json_file << "    \"dfs_mode\": \"" << (config.dfs_mode.empty() ? "unified" : config.dfs_mode) << "\",\n";
                        json_file << "    \"fast_down\": " << (config.disable_fast_down ? "false" : "true") << ",\n";
                        json_file << "    \"lsa_pruning\": " << (config.disable_lsa_pruning ? "false" : "true") << ",\n";
                        json_file << "    \"reuse_lsa\": " << (config.disable_reuse_lsa ? "false" : "true") << "\n";
                        json_file << "  },\n";
                        json_file << "  \"results\": {\n";
                        json_file << "    \"recall\": " << std::fixed << std::setprecision(4) << (avg_recall / 100.0) << ",\n";
                        json_file << "    \"precision\": " << std::fixed << std::setprecision(4) << (avg_precision / 100.0) << ",\n";
                        json_file << "    \"iou\": " << std::fixed << std::setprecision(4) << (avg_iou / 100.0) << ",\n";
                        json_file << "    \"avg_time\": " << std::fixed << std::setprecision(6) << avg_time << ",\n";
                        json_file << "    \"filter_time\": " << std::fixed << std::setprecision(6) << filter_time << ",\n";
                        json_file << "    \"filter_count\": " << std::fixed << std::setprecision(1) << avg_filter_count << ",\n";
                        json_file << "    \"verify_time\": " << std::fixed << std::setprecision(6) << verify_time << ",\n";
                        json_file << "    \"verify_count\": " << std::fixed << std::setprecision(1) << avg_verify_count << ",\n";
                        json_file << "    \"ndc_count\": " << std::fixed << std::setprecision(1) << avg_ndc_count << ",\n";
                        json_file << "    \"nd_time\": " << std::fixed << std::setprecision(6) << nd_time << ",\n";
                        json_file << "    \"ept_time\": " << std::fixed << std::setprecision(6) << ept_time << ",\n";
                        json_file << "    \"nd_lb_time\": " << std::fixed << std::setprecision(6) << avg_nd_lb << ",\n";
                        json_file << "    \"nd_astar_time\": " << std::fixed << std::setprecision(6) << avg_nd_astar << ",\n";
                        json_file << "    \"ept_lb_time\": " << std::fixed << std::setprecision(6) << avg_ept_lb << ",\n";
                        json_file << "    \"ept_astar_time\": " << std::fixed << std::setprecision(6) << avg_ept_astar << "\n";
                        json_file << "  }\n";
                        json_file << "}\n";
                        json_file.close();
                    }
                }
            }
        }
        printf("[experiment_mode] Individual summaries written to latest/ directories\n");

        // 12c. Generate latest/{dataset}/summary.txt (aggregated summary for all methods and tau values)
        std::filesystem::path latest_dataset_summary_path = std::filesystem::path(latest_dir) / "summary.txt";
        std::ofstream latest_dataset_summary(latest_dataset_summary_path);
        if (latest_dataset_summary.is_open()) {
            // Get current time for "Generated" field
            auto gen_now = std::chrono::system_clock::now();
            auto gen_time_t = std::chrono::system_clock::to_time_t(gen_now);
            std::tm gen_tm;
            #ifdef _WIN32
            localtime_s(&gen_tm, &gen_time_t);
            #else
            localtime_r(&gen_time_t, &gen_tm);
            #endif
            char gen_time_str[64];
            std::strftime(gen_time_str, sizeof(gen_time_str), "%Y-%m-%d %H:%M:%S", &gen_tm);

            latest_dataset_summary << std::string(120, '=') << "\n";
            latest_dataset_summary << "                           LATEST RESULTS SUMMARY [" << config.dataset << "]\n";
            latest_dataset_summary << std::string(120, '=') << "\n";
            latest_dataset_summary << "Generated: " << gen_time_str << "\n";
            latest_dataset_summary << "Dataset:   " << config.dataset << "\n";
            latest_dataset_summary << "\n";

            // Main comparison table
            latest_dataset_summary << std::string(120, '-') << "\n";
            char hdr[256];
            snprintf(hdr, sizeof(hdr), "| %-16s | %5s | %7s | %7s | %7s | %10s | %8s | %10s | %8s | %10s |\n",
                   "Method", "Tau", "Recall%", "Prec%", "IoU%", "Filter(s)", "Filter#", "Verify(s)", "Verify#", "Total(s)");
            latest_dataset_summary << hdr;
            latest_dataset_summary << "|------------------|-------|---------|---------|---------|------------|----------|------------|----------|------------|\n";

            double prev_tau = -1;
            for (const auto& tau : tau_list) {
                if (prev_tau >= 0 && tau != prev_tau) {
                    latest_dataset_summary << "|------------------|-------|---------|---------|---------|------------|----------|------------|----------|------------|\n";
                }
                prev_tau = tau;

                for (const auto& method : methods_list) {
                    int queries_with_gt = 0;
                    int queries_with_results = 0;
                    double sum_recall = 0.0, sum_precision = 0.0, sum_iou = 0.0;
                    double total_time = 0.0;
                    double total_filter_time = 0.0, total_verify_time = 0.0;
                    long long total_filter_count = 0, total_verify_count = 0;
                    int total_queries = 0;

                    for (const auto& result : all_results) {
                        if (result.method == method && result.tau == tau) {
                            total_queries++;
                            total_time += result.details.time;
                            total_filter_time += result.details.nd_lb_time + result.details.ept_lb_time;
                            total_verify_time += result.details.nd_astar_time + result.details.ept_astar_time;
                            total_filter_count += result.details.nd_lb_count + result.details.ept_lb_count;
                            total_verify_count += result.details.nd_astar_count + result.details.ept_astar_count;
                            if (result.details.has_ground_truth) {
                                queries_with_gt++;
                                sum_recall += result.details.recall;
                                sum_iou += result.details.iou;
                            }
                            if (result.details.has_results) {
                                queries_with_results++;
                                sum_precision += result.details.precision;
                            }
                        }
                    }

                    if (total_queries == 0) continue;

                    double avg_recall = queries_with_gt > 0 ? sum_recall / queries_with_gt * 100 : 0.0;
                    double avg_precision = queries_with_results > 0 ? sum_precision / queries_with_results * 100 : 0.0;
                    double avg_iou = queries_with_gt > 0 ? sum_iou / queries_with_gt * 100 : 0.0;
                    double avg_time = total_time / total_queries;
                    double avg_filter_time = total_filter_time / total_queries;
                    double avg_verify_time = total_verify_time / total_queries;
                    double avg_filter_count = (double)total_filter_count / total_queries;
                    double avg_verify_count = (double)total_verify_count / total_queries;

                    char row[256];
                    snprintf(row, sizeof(row), "| %-16s | %5.1f | %7.2f | %7.2f | %7.2f | %10.6f | %8.1f | %10.6f | %8.1f | %10.6f |\n",
                           method.c_str(), tau, avg_recall, avg_precision, avg_iou,
                           avg_filter_time, avg_filter_count, avg_verify_time, avg_verify_count, avg_time);
                    latest_dataset_summary << row;
                }
            }
            latest_dataset_summary << std::string(120, '-') << "\n";
            latest_dataset_summary << "Note: Filter = ND_lb + EPT_lb, Verify = ND_astar + EPT_astar\n";
            latest_dataset_summary << "\n";

            // Gisma detailed stats (ND vs EPT)
            bool has_gisma = false;
            for (const auto& m : methods_list) {
                if (m.find("Gisma") != std::string::npos) { has_gisma = true; break; }
            }

            if (has_gisma) {
                latest_dataset_summary << std::string(100, '-') << "\n";
                latest_dataset_summary << "                           GISMA DETAILED (ND vs EPT)\n";
                latest_dataset_summary << std::string(100, '-') << "\n";
                snprintf(hdr, sizeof(hdr), "| %-16s | %5s | %10s | %10s | %10s | %8s |\n",
                       "Method", "Tau", "ND(s)", "EPT(s)", "NDC#", "Queries");
                latest_dataset_summary << hdr;
                latest_dataset_summary << "|------------------|-------|------------|------------|------------|----------|\n";

                prev_tau = -1;
                for (const auto& tau : tau_list) {
                    if (prev_tau >= 0 && tau != prev_tau) {
                        latest_dataset_summary << "|------------------|-------|------------|------------|------------|----------|\n";
                    }
                    prev_tau = tau;

                    for (const auto& method : methods_list) {
                        if (method.find("Gisma") == std::string::npos) continue;

                        double total_nd_time = 0.0, total_ept_time = 0.0;
                        long long total_ndc_count = 0;
                        int total_queries = 0;

                        for (const auto& result : all_results) {
                            if (result.method == method && result.tau == tau) {
                                total_queries++;
                                total_nd_time += result.details.nd_lb_time + result.details.nd_astar_time;
                                total_ept_time += result.details.ept_lb_time + result.details.ept_astar_time;
                                total_ndc_count += result.details.nd_ndc_count + result.details.ept_ndc_count;
                            }
                        }

                        if (total_queries == 0) continue;

                        char row[256];
                        snprintf(row, sizeof(row), "| %-16s | %5.1f | %10.6f | %10.6f | %10.1f | %8d |\n",
                               method.c_str(), tau,
                               total_nd_time / total_queries,
                               total_ept_time / total_queries,
                               (double)total_ndc_count / total_queries,
                               total_queries);
                        latest_dataset_summary << row;
                    }
                }
                latest_dataset_summary << std::string(100, '-') << "\n";
            }

            latest_dataset_summary << "\n";
            latest_dataset_summary << std::string(120, '=') << "\n";
            latest_dataset_summary.close();
            printf("[experiment_mode] Dataset summary written to: %s\n", latest_dataset_summary_path.string().c_str());
        }

        // 12d. Generate latest/{dataset}/summary.json (aggregated JSON summary)
        {
            std::filesystem::path json_path = std::filesystem::path(latest_dir) / "summary.json";
            std::ofstream jf(json_path);
            if (jf.is_open()) {
                auto gen_now2 = std::chrono::system_clock::now();
                auto gen_time_t2 = std::chrono::system_clock::to_time_t(gen_now2);
                std::tm gen_tm2;
                #ifdef _WIN32
                localtime_s(&gen_tm2, &gen_time_t2);
                #else
                localtime_r(&gen_time_t2, &gen_tm2);
                #endif
                char gen_time_str2[64];
                std::strftime(gen_time_str2, sizeof(gen_time_str2), "%Y-%m-%d %H:%M:%S", &gen_tm2);

                jf << "{\n";
                jf << "  \"dataset\": \"" << config.dataset << "\",\n";
                jf << "  \"generated\": \"" << gen_time_str2 << "\",\n";
                jf << "  \"parameters\": {\n";
                jf << "    \"alpha\": " << config.alpha << ",\n";
                jf << "    \"tau_index\": " << config.tau_index << ",\n";
                jf << "    \"app_max_iter\": " << config.app_max_iter << ",\n";
                jf << "    \"nd_mode\": \"" << config.nd_mode << "\",\n";
                jf << "    \"dfs_mode\": \"" << (config.dfs_mode.empty() ? "unified" : config.dfs_mode) << "\",\n";
                jf << "    \"db_size\": " << db.size() << ",\n";
                jf << "    \"query_range\": [" << q_start << ", " << q_end << "]\n";
                jf << "  },\n";
                jf << "  \"results\": [\n";

                bool first_entry = true;
                for (const auto& tau : tau_list) {
                    for (const auto& method : methods_list) {
                        int queries_with_gt = 0;
                        int queries_with_results = 0;
                        double sum_recall = 0.0, sum_precision = 0.0, sum_iou = 0.0;
                        double total_time = 0.0;
                        double total_nd_lb_t = 0.0, total_nd_astar_t = 0.0;
                        double total_ept_lb_t = 0.0, total_ept_astar_t = 0.0;
                        long long total_nd_lb_c = 0, total_nd_astar_c = 0;
                        long long total_ept_lb_c = 0, total_ept_astar_c = 0;
                        long long total_nd_ndc = 0, total_ept_ndc = 0;
                        int total_q = 0;

                        for (const auto& result : all_results) {
                            if (result.method == method && result.tau == tau) {
                                total_q++;
                                total_time += result.details.time;
                                total_nd_lb_t += result.details.nd_lb_time;
                                total_nd_astar_t += result.details.nd_astar_time;
                                total_ept_lb_t += result.details.ept_lb_time;
                                total_ept_astar_t += result.details.ept_astar_time;
                                total_nd_lb_c += result.details.nd_lb_count;
                                total_nd_astar_c += result.details.nd_astar_count;
                                total_ept_lb_c += result.details.ept_lb_count;
                                total_ept_astar_c += result.details.ept_astar_count;
                                total_nd_ndc += result.details.nd_ndc_count;
                                total_ept_ndc += result.details.ept_ndc_count;
                                if (result.details.has_ground_truth) {
                                    queries_with_gt++;
                                    sum_recall += result.details.recall;
                                    sum_iou += result.details.iou;
                                }
                                if (result.details.has_results) {
                                    queries_with_results++;
                                    sum_precision += result.details.precision;
                                }
                            }
                        }

                        if (total_q == 0) continue;

                        double ar = queries_with_gt > 0 ? sum_recall / queries_with_gt : 0.0;
                        double ap = queries_with_results > 0 ? sum_precision / queries_with_results : 0.0;
                        double ai = queries_with_gt > 0 ? sum_iou / queries_with_gt : 0.0;
                        double at = total_time / total_q;
                        double nd_lb = total_nd_lb_t / total_q;
                        double nd_as = total_nd_astar_t / total_q;
                        double ept_lb = total_ept_lb_t / total_q;
                        double ept_as = total_ept_astar_t / total_q;

                        if (!first_entry) jf << ",\n";
                        first_entry = false;

                        jf << "    {\n";
                        jf << "      \"method\": \"" << method << "\",\n";
                        jf << "      \"tau\": " << std::fixed << std::setprecision(1) << tau << ",\n";
                        jf << "      \"queries\": " << total_q << ",\n";
                        jf << "      \"recall\": " << std::fixed << std::setprecision(4) << ar << ",\n";
                        jf << "      \"precision\": " << std::fixed << std::setprecision(4) << ap << ",\n";
                        jf << "      \"iou\": " << std::fixed << std::setprecision(4) << ai << ",\n";
                        jf << "      \"avg_time\": " << std::fixed << std::setprecision(6) << at << ",\n";
                        jf << "      \"filter_time\": " << std::fixed << std::setprecision(6) << (nd_lb + ept_lb) << ",\n";
                        jf << "      \"filter_count\": " << std::fixed << std::setprecision(1) << (double)(total_nd_lb_c + total_ept_lb_c) / total_q << ",\n";
                        jf << "      \"verify_time\": " << std::fixed << std::setprecision(6) << (nd_as + ept_as) << ",\n";
                        jf << "      \"verify_count\": " << std::fixed << std::setprecision(1) << (double)(total_nd_astar_c + total_ept_astar_c) / total_q << ",\n";
                        jf << "      \"ndc_count\": " << std::fixed << std::setprecision(1) << (double)(total_nd_ndc + total_ept_ndc) / total_q << ",\n";
                        jf << "      \"nd_time\": " << std::fixed << std::setprecision(6) << (nd_lb + nd_as) << ",\n";
                        jf << "      \"ept_time\": " << std::fixed << std::setprecision(6) << (ept_lb + ept_as) << ",\n";
                        jf << "      \"nd_lb_time\": " << std::fixed << std::setprecision(6) << nd_lb << ",\n";
                        jf << "      \"nd_astar_time\": " << std::fixed << std::setprecision(6) << nd_as << ",\n";
                        jf << "      \"ept_lb_time\": " << std::fixed << std::setprecision(6) << ept_lb << ",\n";
                        jf << "      \"ept_astar_time\": " << std::fixed << std::setprecision(6) << ept_as << "\n";
                        jf << "    }";
                    }
                }
                jf << "\n  ]\n";
                jf << "}\n";
                jf.close();
                printf("[experiment_mode] Dataset JSON summary written to: %s\n", json_path.string().c_str());
            }
        }

    } else {
        printf("[experiment_mode] No files saved (--save not specified)\n");
    }

    // 13. Cleanup
    printf("[experiment_mode] Cleaning up...\n");
    printf("[experiment_mode] Experiment mode completed successfully\n");
}
