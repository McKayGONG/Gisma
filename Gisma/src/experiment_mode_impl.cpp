// Experiment mode implementation
// This will be included in main.cpp

// Global coverage sets for GismaSearchEngine use
std::unordered_set<int> g_ept_coverage;
std::unordered_set<int> g_extra_coverage;

// Task structure for mixed scheduling
struct ExperimentTask {
    std::string method;
    double tau;
    int query_idx;
    int query_local_idx;
};

void experiment_mode(const Config& config) {
    printf("[experiment_mode] Starting experiment mode...\n");

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

    // 4. Load ground truth (shared by all methods)
    printf("[experiment_mode] Loading ground truth...\n");
    std::map<int, std::map<double, std::vector<int>>> ground_truth;
    if (!Utility::load_exact_ground_truth(config.ground_truth_path, ground_truth)) {
        std::cerr << "[ERROR] Failed to load ground truth from " << config.ground_truth_path << std::endl;
        return;
    }
    printf("[experiment_mode] Ground truth loaded successfully: %zu queries\n", ground_truth.size());

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
        if (method == "Gisma" || method == "Base+GS" || method == "Base+SS" || method == "Base_All_EPT") {
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

        std::string index_name = Utility::get_index_name(config.dataset, alpha, tau_index, error_tolerance_index, db.size());
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

        // Collect EPT and Extra coverage for the entire database, store in global variables
        g_ept_coverage.clear();
        g_extra_coverage.clear();
        for (const auto& anchor : netdag_ptr->anchors) {
            // EPT coverage: anchor itself + all tree_node completed_db_graph_ids
            g_ept_coverage.insert(anchor->node_id);
            EditPathTree* ept = ept_manager->get_ept(anchor->node_id);
            if (ept) {
                for (const auto& tree_node : ept->tree_nodes) {
                    for (int id : tree_node.completed_db_graph_ids) {
                        g_ept_coverage.insert(id);
                    }
                }
            }
            // Extra coverage: nodes_in_cluster
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

    // 7. Create GismaSearchEngine for each method
    std::vector<double> ged_matrix;
    bool has_ged_matrix = false;

    std::map<std::string, std::shared_ptr<GismaSearchEngine>> searchers;
    for (const auto& method : methods_list) {
        printf("[experiment_mode] Creating GismaSearchEngine for %s...\n", method.c_str());
        auto searcher = std::make_shared<GismaSearchEngine>(
            netdag_ptr,
            config.tau_index,
            config.error_tolerance_search,
            q_start,
            q_end,
            has_ged_matrix,
            ged_matrix,
            method,  // Use current method
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
            config.nd_mode,
            config.dfs_mode,
            !config.disable_ept_filters,  // enabled by default, --disable_ept_filters to disable
            config.only_compute_db_graph,
            config.app_max_iter,
            !config.disable_fast_down,  // fast-down strategy (inverted)
            config.exact_max_iter,
            config.nd_filter_ratio,
            config.disable_lsa_pruning,
            config.disable_reuse_lsa,
            (method == "Gisma") ? config.verify_reuse_baseline : false,  // only collect reuse stats in Gisma method
            config.chain_reuse  // chain reuse
        );
        searchers[method] = searcher;
    }
    printf("[experiment_mode] All GismaSearchEngines created successfully\n");

    // 8. Create experiment directories (skip if --save not specified)
    // Structure: experiment_results/{dataset}/{timestamp}/ (with summary.txt)
    // Directory structure:
    //   experiment_results/archive/{dataset}/{timestamp}/  (archive, with summary)
    //   experiment_results/latest/{dataset}/               (overwrite, without summary)
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

        // Timestamp directory (archive)：experiment_results/archive/{dataset}/{timestamp}/
        exp_dir = base_exp_dir + "/archive/" + config.dataset + "/" + timestamp;
        std::filesystem::create_directories(exp_dir);

        // latest directory：experiment_results/latest/{dataset}/
        latest_dir = base_exp_dir + "/latest/" + config.dataset;
        std::filesystem::create_directories(latest_dir);

        printf("[experiment_mode] Results directory: %s\n", exp_dir.c_str());
        printf("[experiment_mode] Latest directory:  %s\n", latest_dir.c_str());
    } else {
        printf("[experiment_mode] --save not specified, skipping directory creation\n");
    }

    // 8b. Print experiment parameters summary to console

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
    std::map<std::string, std::map<double, std::string>> subdirs;             // timestamp directory
    std::map<std::string, std::map<double, std::string>> latest_subdirs;      // latest directory (results/ subdirectory)
    std::map<std::string, std::map<double, std::string>> latest_base_subdirs; // latest directory (base directory, for writing summary)
    std::string timestamp_str;  // save timestamp string for writing latest summary
    if (config.save) {
        // extract timestamp from exp_dir (last directory name)
        size_t last_slash = exp_dir.find_last_of("/\\");
        timestamp_str = (last_slash != std::string::npos) ? exp_dir.substr(last_slash + 1) : "";

        for (const auto& method : methods_list) {
            for (const auto& tau : tau_list) {
                std::ostringstream subdir_name;
                subdir_name << method << "_tau_" << std::fixed << std::setprecision(1) << tau;

                // timestamp directory
                std::string subdir_path = exp_dir + "/" + subdir_name.str();
                std::filesystem::create_directories(subdir_path);
                subdirs[method][tau] = subdir_path;

                // latest directory：base directory + results/ subdirectory
                std::string latest_base_path = latest_dir + "/" + subdir_name.str();
                std::string latest_results_path = latest_base_path + "/results";
                std::filesystem::create_directories(latest_results_path);
                latest_base_subdirs[method][tau] = latest_base_path;
                latest_subdirs[method][tau] = latest_results_path;  // json saved to results/ subdirectory
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
            // Use user-specified thread count, but cap at 70% of available cores
            num_threads = std::min(static_cast<unsigned int>(config.num_workers), max_allowed);
        } else {
            // Automatically use 70% of available cores
            num_threads = max_allowed;
        }

        printf("\n[experiment_mode] Using PARALLEL execution with DYNAMIC TASK QUEUE for %d threads and %d mixed tasks...\n",
               num_threads, total_tasks);

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
                 &all_results, &results_mutex, &next_task_index]() {

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

                        if (task.method == "Gisma") {
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
                            results = searcher->App_BMao_search(query_node, task.tau, local_stats);
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
                        } else {
                            results = searcher->Gisma_search(query_node, task.tau, local_stats);
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
                        // ND and EPT time separately
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
                        // EXP-5: Baseline statistics
                        query_detail.baseline_app_count = local_stats.EPT_baseline_app_count;
                        query_detail.baseline_app_time = local_stats.EPT_baseline_app_time;
                        query_detail.baseline_reuse_time = local_stats.EPT_baseline_reuse_time;
                        query_detail.baseline_correct = local_stats.EPT_reuse_correct;
                        query_detail.baseline_incorrect = local_stats.EPT_reuse_incorrect;
                        query_detail.baseline_samples = local_stats.baseline_samples;
                        // db graph vs intermediate graph statistics
                        query_detail.db_graph_lb_count = local_stats.EPT_db_graph_lb_count;
                        query_detail.db_graph_lb_time = local_stats.EPT_db_graph_lb_time;
                        query_detail.db_graph_astar_count = local_stats.EPT_db_graph_astar_count;
                        query_detail.db_graph_astar_time = local_stats.EPT_db_graph_astar_time;
                        query_detail.intermediate_graph_lb_count = local_stats.EPT_intermediate_graph_lb_count;
                        query_detail.intermediate_graph_lb_time = local_stats.EPT_intermediate_graph_lb_time;
                        query_detail.intermediate_graph_astar_count = local_stats.EPT_intermediate_graph_astar_count;
                        query_detail.intermediate_graph_astar_time = local_stats.EPT_intermediate_graph_astar_time;

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

        // Batch save all results (skip if --save not specified)
        if (config.save) {
            printf("[experiment_mode] All computation completed. Now saving %zu results to files...\n", all_results.size());
            auto save_start = std::chrono::high_resolution_clock::now();
            for (const auto& result : all_results) {
                // Save to timestamp directory
                const std::string& subdir = subdirs.at(result.method).at(result.tau);
                searchers.at(result.method)->save_single_query_json(result.details, result.tau, subdir);
                // Save to latest directory
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

            if (task.method == "Gisma") {
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
                results = searcher->App_BMao_search(query_node, task.tau, local_stats);
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
            } else {
                results = searcher->Gisma_search(query_node, task.tau, local_stats);
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
            // ND and EPT time separately
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
            // EXP-5: Baseline statistics
            query_detail.baseline_app_count = local_stats.EPT_baseline_app_count;
            query_detail.baseline_app_time = local_stats.EPT_baseline_app_time;
            query_detail.baseline_reuse_time = local_stats.EPT_baseline_reuse_time;
            query_detail.baseline_correct = local_stats.EPT_reuse_correct;
            query_detail.baseline_incorrect = local_stats.EPT_reuse_incorrect;
            query_detail.baseline_samples = local_stats.baseline_samples;
            // db graph vs intermediate graph statistics
            query_detail.db_graph_lb_count = local_stats.EPT_db_graph_lb_count;
            query_detail.db_graph_lb_time = local_stats.EPT_db_graph_lb_time;
            query_detail.db_graph_astar_count = local_stats.EPT_db_graph_astar_count;
            query_detail.db_graph_astar_time = local_stats.EPT_db_graph_astar_time;
            query_detail.intermediate_graph_lb_count = local_stats.EPT_intermediate_graph_lb_count;
            query_detail.intermediate_graph_lb_time = local_stats.EPT_intermediate_graph_lb_time;
            query_detail.intermediate_graph_astar_count = local_stats.EPT_intermediate_graph_astar_count;
            query_detail.intermediate_graph_astar_time = local_stats.EPT_intermediate_graph_astar_time;

            // Collect result for summary
            TaskResult result{task.method, task.tau, query_detail};
            all_results.push_back(result);

            // Save immediately (skip if --save not specified)
            if (config.save) {
                // Save to timestamp directory
                const std::string& subdir = subdirs.at(task.method).at(task.tau);
                searcher->save_single_query_json(query_detail, task.tau, subdir);
                // Save to latest directory
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
    printf("| %-16s | %5s | %7s | %7s | %7s | %8s | %10s |\n",
           "Method", "Tau", "Recall%", "Prec%", "IoU%", "NDC#", "Total(s)");
    printf("|------------------|-------|---------|---------|---------|----------|------------|\n");

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
            double avg_ndc_count = total_queries > 0 ? (double)(total_nd_ndc_count + total_ept_ndc_count) / total_queries : 0.0;  // NDC = unique node count (deduplicated)

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

            printf("| %-16s | %5.1f | %7.2f | %7.2f | %7.2f | %8s | %10.6f |\n",
                   method.c_str(), tau, avg_recall, avg_precision, avg_iou,
                   fmt_count(avg_ndc_count).c_str(),
                   avg_time);
        }
        // Separator between tau groups
        if (&tau != &tau_list.back()) {
            printf("|------------------|-------|---------|---------|---------|----------|------------|\n");
        }
    }
    printf("========================================================================================================================================================\n");
    printf("Note: Filter = ND_lb + EPT_lb, Verify = ND_astar + EPT_astar, NDC = unique nodes computed (deduplicated)\n");
    printf("      ND(s) = NetDag phase total time, EPT(s) = EditPathTree phase total time\n");

    // Print Gisma detailed statistics table (for methods with ND/EPT breakdown)
    bool has_gisma_methods = false;
    for (const auto& method : methods_list) {
        if (method == "Gisma" || method == "Gisma-no-reuse" || method == "Gisma-only-dfs" || method == "Gisma-no-SP" || method == "Gisma-no-LP" || method == "Base+SS" || method == "Base+GS" || method == "Base_All_EPT") {
            has_gisma_methods = true;
            break;
        }
    }

    if (has_gisma_methods) {
        // Detail statistics tables removed
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
            // Write parameters section

            // Write main comparison table
            summary_file << "========================================================================================================================================================\n";
            summary_file << "                                               COMPARISON TABLE [Dataset: " << config.dataset << "]\n";
            summary_file << "========================================================================================================================================================\n";

            char buf[512];
            snprintf(buf, sizeof(buf), "| %-16s | %5s | %7s | %7s | %7s | %8s | %10s |\n",
                   "Method", "Tau", "Recall%", "Prec%", "IoU%", "NDC#", "Total(s)");
            summary_file << buf;
            summary_file << "|------------------|-------|---------|---------|---------|----------|------------|\n";

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

                    snprintf(buf, sizeof(buf), "| %-16s | %5.1f | %7.2f | %7.2f | %7.2f | %8s | %10.6f |\n",
                           method.c_str(), tau, avg_recall, avg_precision, avg_iou,
                           
                           
                           fmt_count_f(avg_ndc_count).c_str(),  avg_time);
                    summary_file << buf;
                }
                if (&tau != &tau_list.back()) {
                    summary_file << "|------------------|-------|---------|---------|---------|----------|------------|\n";
                }
            }
            summary_file << "========================================================================================================================================================\n";
            summary_file << "Note: Filter = ND_lb + EPT_lb, Verify = ND_astar + EPT_astar, NDC = unique nodes computed (deduplicated)\n";
            summary_file << "      ND(s) = NetDag phase total time, EPT(s) = EditPathTree phase total time\n\n";

            // Write Gisma detailed statistics if applicable
            if (has_gisma_methods) {
                // Detail statistics tables removed
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
                summary_file << "                              EXP-5: Sample Pairs (Reuse vs Baseline, up to 10 per tau)                                \n";
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
                        snprintf(buf, sizeof(buf), "| %-6s | %-14s | %-14s | %-10s |\n", "Sample", "Reuse(ms)", "Baseline(ms)", "Speedup");
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
                // Compute statistics for this method+tau
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
