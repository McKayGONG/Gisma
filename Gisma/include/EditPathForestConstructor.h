#ifndef EDITPATHFORESTCONSTRUCTOR_H
#define EDITPATHFORESTCONSTRUCTOR_H

#include "Utility.h"
#include "Graph.h"
#include "Anchor.h"
#include "Node.h"
#include "EditPathTree.h"
#include <string>
class GismaIndexBuilder;
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>

// Forward declaration
class NetDagConst;

class EditPathForestConstructor {
public:
    // Constructor
    EditPathForestConstructor(GismaIndexBuilder* parent);
    
    // EPT construction methods
    void construct_EPT();
    void construct_EPT_parallel();
    
    // Edit path computation and analysis
    void compute_edit_path_and_save_to_csv();
    void find_within_tau_and_save_csv(const std::string &csv_filename);
    void build_one_ball_EPT_from_csv(int EPT_root_ind,
                                     double alpha,
                                     const std::string &csv_path,
                                     const std::string &ept_filename);
    
    // Exact anchor updates
    void update_one_exact_anchor();
    void update_one_exact_anchor_parallel();
    
    // Query selection and filtering
    int select_queries_from_csv(
        int root_ind,
        int need_count,
        const std::string &csv_path,
        const std::string &out_queries_txt,
        std::string &out_filtered_csv
    );
    
    int select_queries_by_ged_ranges(
        int root_ind,
        const std::string &csv_path,
        const std::string &output_dir
    );
    
private:
    GismaIndexBuilder* parent_;  // Reference to parent GismaIndexBuilder object

    // Helper functions for EPT construction
    void build_edit_path_tree(
        EditPathTree& ept,
        const std::vector<std::vector<EditOperation>>& edit_operations_list,
        const std::vector<ui>& db_graph_ids,
        ui anchor_id,
        const Graph& anchor_graph
    );
};

#endif // EDITPATHFORESTCONSTRUCTOR_H