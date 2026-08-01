#ifndef CONFIG_H
#define CONFIG_H

#include <string>

class Config {
public:
    // Common parameters
    std::string mode;
    std::string dataset;       // 数据集名称
    std::string db_name;
    std::string query_name;
    std::string ground_truth_path;

    std::string search_method;

    // Compute GED mode parameters
    std::string query_file;
    std::string target_file;
    std::string index_name;     // 显式指定索引名（覆盖自动命名）；空=用 get_index_name 自动生成
    std::string timing_log;     // E1 construction 计时输出文件；空=不统计（默认）。设了才写 construct_timing.tsv 格式

    // Batch GED mode parameters (for GHash integration)
    std::string candidate_ids_file;  // File containing candidate IDs, one per line
    int query_id;                    // Query graph ID

    // 添加查询范围参数
    int q_start;
    int q_end;
    int root_ind;              // 新增：根节点索引

    // Construct mode parameters
    double alpha;
    double tau_index;
    double error_tolerance_index;
    double old_tau_index;  // upgrade_tau模式：旧的tau_index值
    double old_alpha;      // construct_from模式：已有NetDag的alpha值

    // Search mode parameters
    double tau_search;
    double error_tolerance_search;
    double include_compute_dist;
    double max_exact_ged_for_EPT;
    bool has_ged_matrix;
    int server_id;
    int total_servers;
    bool use_parallel;
    int num_workers;               // 并行线程数，0表示自动检测（使用CPU核心数）
    bool enable_friends_reassign;  // 控制friends收留功能的开关
    bool save_logs;  // 是否保存详细日志到文件
    int feature_dim;
    std::string nd_mode;   // NetDag模式: "filters" / "astar" / "filters_astar"
    std::string dfs_mode;  // DFS遍历模式: 默认全优化，可选 "no_reuse" / "no_SP" / "no_LP" / "only_dfs"
    bool disable_ept_filters;  // 禁用EPT下界过滤（默认启用）
    bool only_compute_db_graph;   // 只对有completed_db_graph_ids的EPT节点（即对应db图的节点）计算GED（默认false）
    bool disable_fast_down;  // 禁用快速下降（默认启用快速下降：找到第一个满足条件的就往下走）
    int app_max_iter;      // A*最大迭代次数，控制APP_CNT/ASTAR_CNT/REUSE_CNT（默认2300）
    bool use_afc = false;  // compute_ged 用 AppForComputation(诊断)
    int exact_max_iter;    // 精确计算最大迭代次数，用于训练数据生成（默认1000000）
    double nd_filter_ratio;  // NetDag筛选收紧系数，条件改为 lb ≤ (alpha + tau) * ratio（默认1.0，不收紧）
    bool disable_all_lsa;      // 总开关：禁用所有LSA（自动设置下面两个为true）
    bool disable_lsa_pruning;  // 禁用LSa剪枝（仍保存lsa_lb供reuse用）
    int  lsa_layer;            // App_test 的 LSa-as-layer 额外剪枝层: 0=关(默认, 实测为死重量) / 1=开
    bool orig_verifier;        // index 路径用原作者引擎(origbmao)当验证器（干净对照实验）
    bool disable_reuse_lsa;    // 禁用reuse中的LSa重计算（使用简单ged_gap减法）
    bool exact_value_mode;       // App() / app_reuse() 算精確 GED（停用 ≤tau early-exit）。默認 false。
    bool early_stop_at_tau;      // parent sibling loop break at lb>tau + child reuse 跳過 intersection。默認 false。
    bool verify_reuse;  // 验证reuse效果：每次reuse时用AppForComputation计算baseline时间
    bool chain_reuse;            // 链式复用：使用reuse的节点也保存snapshot供后续节点复用
    int max_ged_gap;             // 最大GED gap：EPT中距离超过此值的子节点不使用reuse（默认3）
    int max_margin;              // 最大margin：A*搜索时保留边界节点的范围（默认3）
    bool all_edge_labels_same;   // 跳過 edge label 比較（AIDS 等优化，默认关）
    bool skip_hierarchy;         // construct_from模式：跳过children和parent_by_phase_dict（只保留anchor+cluster用于Base+SS）

    // experiment mode parameters
    std::string tau_values;  // 逗号分隔的tau值列表，如 "2,4,6,8,10,12"
    std::string methods;     // 逗号分隔的方法列表，如 "Gisma,App-BMao,AStar-BMao"
    bool save_query_logs;    // 是否保存每个query的详细日志
    bool save;               // 保存结果到本地（默认false，即不保存，需要--save开启）

    // select-alpha mode parameters
    double alpha_min;      // 最小alpha值
    double alpha_max;      // 最大alpha值
    double alpha_step;     // alpha步长

    // E11 update cost: masked (lazy) deletion —— 标记 deleted_frac 比例的 db 图为已删除，
    // 搜索结果中过滤掉(FreshDiskANN lazy deletion: 导航仍用、结果排除)。0=不删。
    double deleted_frac;

    // E11 insert 性能：挑 insert_count 个 EPT 叶子图(代表真实可插入的图)，用 GS_search 定位(=我们的 insert)
    // 逐个插入并统计 avg 时间+NDC(insert=search [HNSW])。measure_insert=true 时触发。
    bool measure_insert;
    int insert_count;

    // E11 insert 稳定性(round-trip): 把 insert_count 个 EPT 叶子图当作"还没插入"从结果中排除，
    // 但【不】剔除 ground truth → recall 反映"插入前"(这些图缺失)。baseline(本 flag 关) = "插入后"。
    // insert=GS_search 把图放回正确 ball，故插入后 recall 回到 baseline。
    bool insert_stability;

    // E11 真·增量插入(insert_rebuild): 把 insert_count 个 EPT 叶子图从其所在 EPT 节点移除，
    // 再用插入算法(算 anchor→g 的精确 GED 作距离, 作为 root 的一个分支挂回)真正改写索引结构，
    // 然后在【修改后的增量索引】上跑搜索测 recall。回答"增量索引 != 从头构建的 paper 索引"。
    bool insert_rebuild;

    // E11 insert 探测: 仅对每个被删图跑 GS_search + 检查"家"anchor 在不在候选(不算 GED/不插入)，
    // 统计 home_in_cand=0 的比例(= GS_search 漏召回家 anchor 的比例)。并行、快。
    bool insert_probe;
    bool e7_stats;  // E7: gated search-time stats (working-set RSS / #EPT trees / answer depth). Default off = zero production overhead.

    // E11 insert #2(插入前 recall): 移除 insert_count 个叶子图后【不】插回, 直接测 recall(over 完整 GT)。
    bool insert_remove_only;

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
          index_name(""),        // 默认空：自动命名
          timing_log(""),        // 默认空：不统计 construction 计时
          candidate_ids_file(""),  // Batch GED mode: candidate IDs file
          query_id(-1),            // Batch GED mode: query graph ID
          q_start(-1),  // 默认值-1表示不限制起始位置
          q_end(-1),    // 默认值-1表示不限制结束位置
          root_ind(0),  // 默认值0表示使用第一个图作为根节点
          alpha(12.0),
          tau_index(8.0),
          error_tolerance_index(0.0),
          old_tau_index(0.0),
          old_alpha(0.0),
          tau_search(2.0),
          error_tolerance_search(0.0),
          include_compute_dist(35.0),
          max_exact_ged_for_EPT(-1.0), // -1 表示使用 alpha + 4
          has_ged_matrix(false),
          server_id(0),
          total_servers(1),
          use_parallel(false),
          num_workers(0),                  // 默认0表示自动检测CPU核心数
          enable_friends_reassign(false),  // 默认禁用friends收留功能（使用--enable_friends_reassign参数启用）
          save_logs(false),  // 默认不保存日志
          feature_dim(62),
          nd_mode("filters"),      // 默认使用filters模式
          dfs_mode(""),            // 默认使用全优化DFS模式
          disable_ept_filters(false),  // 默认启用EPT下界过滤
          only_compute_db_graph(false),   // 默认不限制只计算db图节点
          disable_fast_down(false),  // 默认启用快速下降
          app_max_iter(2300),      // 默认2300（AIDS: 2300; PubChem: 3000; SYN: 200）
          exact_max_iter(1000000), // 默认1000000
          nd_filter_ratio(1.0),    // 默认1.0，不收紧
          disable_all_lsa(false),      // 默认不禁用所有LSA
          disable_lsa_pruning(false),  // 默认不禁用LSa剪枝
          lsa_layer(0),                // 默认关: 实测该层剪不掉节点, LSa 侧白耗 22-27%
          orig_verifier(false),        // 默认用 Gisma 自己的验证器
          disable_reuse_lsa(false),    // 默认不禁用reuse中的LSa重计算
          exact_value_mode(false),  // 默认 false：保留 ≤tau early-exit
          early_stop_at_tau(true),  // 默认 true：production -5% time, recall 損失 < 0.2pp（噪音範圍）
          verify_reuse(false), // 默认不验证reuse baseline
          chain_reuse(false),      // 默认不启用链式复用
          max_ged_gap(3),          // 默认3
          max_margin(3),           // 默认3
          all_edge_labels_same(true), // 默认开（Gisma GED 模型无边标签，跳过边标签比较恒等价）；--disable_all_edge_labels_same 关
          skip_hierarchy(false),   // 默认不跳过hierarchy
          tau_values(""),          // 默认为空
          methods(""),             // 默认为空（使用search_method）
          save_query_logs(false),  // 默认不保存每条query日志
          save(false),             // 默认不保存结果（需要--save开启）
          alpha_min(2.0),          // 默认最小alpha
          alpha_max(20.0),         // 默认最大alpha
          alpha_step(1.0),         // 默认步长
          deleted_frac(0.0),       // 默认不删除（E11 masked deletion）
          measure_insert(false),   // 默认不测 insert
          insert_count(100),       // E11 insert 默认挑 100 个叶子图
          insert_stability(false), // 默认不测 insert 稳定性 round-trip
          insert_rebuild(false),   // 默认不做真增量插入
          insert_probe(false),     // 默认不做探测
          e7_stats(false),         // 默认不采集 E7 stats
          insert_remove_only(false)// 默认不做"只移除"
    {
        this->update_paths();
    }

    // 一个辅助函数：根据 dataset 拼出路径
    void update_paths() {
        db_name = "./datasets/" + dataset + "/db.txt";
        query_name = "./datasets/" + dataset + "/queries.txt";
        ground_truth_path = "./datasets/" + dataset + "/ground_truth.txt";
    }

    // 如果想在程序运行时修改 dataset，也可以提供一个 setter
    void setDataset(const std::string &ds) {
        dataset = ds;
        update_paths(); 
    }
};

#endif // CONFIG_H
