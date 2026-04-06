#ifndef CONFIG_H
#define CONFIG_H

#include <string>

class Config {
public:
    // Common parameters
    std::string mode;
    std::string dataset;       // dataset name
    std::string db_name;
    std::string query_name;
    std::string ground_truth_path;

    std::string search_method;

    // Compute GED mode parameters
    std::string query_file;
    std::string target_file;
    std::string ged_algorithm;  // App, AStar, or default

    // Batch GED mode parameters (for GHash integration)
    std::string candidate_ids_file;  // File containing candidate IDs, one per line
    int query_id;                    // Query graph ID

    // Query range parameters
    int q_start;
    int q_end;
    int root_ind;              // added: root node index

    // Construct mode parameters
    double alpha;
    double tau_index;
    double error_tolerance_index;
    double old_tau_index;  // upgrade_tau mode: old tau_index value
    double old_alpha;      // construct_from mode: alpha value of existing NetDag

    // Search mode parameters
    double tau_search;
    double error_tolerance_search;
    double include_compute_dist;
    double max_exact_ged_for_EPT;
    bool has_ged_matrix;
    int server_id;
    int total_servers;
    bool use_parallel;
    int num_workers;               // number of parallel threads, 0 means auto-detect (use CPU core count)
    bool enable_friends_reassign;  // toggle for friends reassignment feature
    bool save_logs;  // whether to save detailed logs to file
    int feature_dim;
    std::string nd_mode;   // NetDag mode: "filters" / "astar" / "filters_astar"
    std::string dfs_mode;  // DFS traversal mode: default fully optimized, options: "no_reuse" / "no_SP" / "no_LP" / "only_dfs"
    bool disable_ept_filters;  // disable EPT lower bound filtering (default: enabled)
    bool only_compute_db_graph;   // only compute GED for EPT nodes with completed_db_graph_ids (i.e., DB graph nodes) (default: false)
    bool disable_fast_down;  // disable fast-down (default: fast-down enabled, descend on first match)
    int app_max_iter;      // A* max iterations, controls APP_CNT/ASTAR_CNT/REUSE_CNT (default: 2300)
    int exact_max_iter;    // exact computation max iterations, for training data generation (default: 1000000)
    double nd_filter_ratio;  // NetDag filter tightening coefficient, condition: lb <= (alpha + tau) * ratio (default: 1.0, no tightening)
    bool disable_all_lsa;      // master switch: disable all LSA (auto-sets the following two to true)
    bool disable_lsa_pruning;  // disable LSa pruning (still save lsa_lb for reuse)
    bool disable_reuse_lsa;    // disable LSa recomputation in reuse (use simple ged_gap subtraction)
    bool verify_reuse_baseline;  // verify reuse effectiveness: compute baseline time with AppForComputation on each reuse
    bool chain_reuse;            // chain reuse: nodes using reuse also save snapshots for subsequent nodes
    bool skip_hierarchy;         // construct_from mode: skip children and parent_by_phase_dict (keep only anchor+cluster for Base+SS)

    // experiment mode parameters
    std::string tau_values;  // comma-separated tau value list, e.g. "2,4,6,8,10,12"
    std::string methods;     // comma-separated method list, e.g. "Gisma,App-BMao,AStar-BMao"
    bool save_query_logs;    // whether to save detailed logs per query
    bool save;               // save results locally (default: false, enable with --save)

    // select-alpha mode parameters
    double alpha_min;      // minimum alpha value
    double alpha_max;      // maximum alpha value
    double alpha_step;     // alpha step size

    // Constructor with default values
    Config()
        : mode("construct"),
          dataset("AIDS"),
          db_name(""),
          query_name(""),
          ground_truth_path(""),
          search_method("Gisma"),
          query_file(""),
          target_file(""),
          ged_algorithm("App"),  // Default to AppForComputation
          candidate_ids_file(""),  // Batch GED mode: candidate IDs file
          query_id(-1),            // Batch GED mode: query graph ID
          q_start(-1),  // default -1 means no start position restriction
          q_end(-1),    // default -1 means no end position restriction
          root_ind(0),  // default 0 means use the first graph as root node
          alpha(12.0),
          tau_index(8.0),
          error_tolerance_index(0.0),
          old_tau_index(0.0),
          old_alpha(0.0),
          tau_search(2.0),
          error_tolerance_search(0.0),
          include_compute_dist(35.0),
          max_exact_ged_for_EPT(-1.0), // -1 means use alpha + 4
          has_ged_matrix(false),
          server_id(0),
          total_servers(1),
          use_parallel(false),
          num_workers(0),                  // default 0 means auto-detect CPU core count
          enable_friends_reassign(false),  // default disabled (enable with --enable_friends_reassign)
          save_logs(false),  // default: do not save logs
          feature_dim(62),
          nd_mode("filters"),      // default: use filters mode
          dfs_mode(""),            // default: use fully optimized DFS mode
          disable_ept_filters(false),  // default: EPT lower bound filtering enabled
          only_compute_db_graph(false),   // default: do not restrict to only DB graph nodes
          disable_fast_down(false),  // default: fast-down enabled
          app_max_iter(2300),      // default 2300 (AIDS: 2300; PubChem: 3000; SYN: 200)
          exact_max_iter(1000000), // default 1000000
          nd_filter_ratio(1.0),    // default 1.0, no tightening
          disable_all_lsa(false),      // default: do not disable all LSA
          disable_lsa_pruning(false),  // default: do not disable LSa pruning
          disable_reuse_lsa(false),    // default: do not disable LSa recomputation in reuse
          verify_reuse_baseline(false), // default: do not verify reuse baseline
          chain_reuse(false),      // default: chain reuse disabled
          skip_hierarchy(false),   // default: do not skip hierarchy
          tau_values(""),          // default: empty
          methods(""),             // default: empty (use search_method)
          save_query_logs(false),  // default: do not save per-query logs
          save(false),             // default: do not save results (enable with --save)
          alpha_min(2.0),          // default minimum alpha
          alpha_max(20.0),         // default maximum alpha
          alpha_step(1.0)          // default step size
    {
        this->update_paths();
    }

    // Helper function: construct paths from dataset name
    void update_paths() {
        db_name = "./datasets/" + dataset + "/db.txt";
        query_name = "./datasets/" + dataset + "/queries.txt";
        ground_truth_path = "./datasets/" + dataset + "/ground_truth.txt";
    }

    // Setter for modifying dataset at runtime
    void setDataset(const std::string &ds) {
        dataset = ds;
        update_paths(); 
    }
};

#endif // CONFIG_H
