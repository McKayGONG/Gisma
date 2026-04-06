#include "GismaIndexBuilder.h"
#include "NetDagConstructor.h"
#include "EditPathForestConstructor.h"
#include "Application.h"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <atomic>
#include <future>
#include <thread>
#include <mutex>
#include <iomanip>
#include <random>
#include <algorithm>
#include <sstream>
#include <set>
#include <vector>
#include <utility>

namespace fs = std::filesystem;

GismaIndexBuilder::GismaIndexBuilder(const std::string& index_name,
                         const std::vector<std::shared_ptr<Graph>>& graphs,
                         const std::vector<std::string>& nameList,
                         int root_ind,
                         double alpha,
                         double tau,
                         double error_tolerance_index,
                         std::shared_ptr<NetDag> net_dag,
                         bool use_parallel,
                         bool has_ged_matrix,
                         const std::vector<double>& ged_matrix,
                         int N,
                         double max_exact_ged_for_EPT,
                         const std::string& folder_name,
                         const std::map<std::string, ui>& vM,
                         const std::map<std::string, ui>& eM,
                         ui max_n,
                         const std::vector<std::vector<float>>& embeddings,
                         bool enable_friends_reassign)
    : index_name(index_name),
      dataset(index_name.substr(0, index_name.find('_'))), // Extract dataset from index_name
      graphs(graphs),
      nameList(nameList),
      root_ind(root_ind),
      alpha(alpha),
      tau(tau),
      error_tolerance_index(error_tolerance_index),
      net_dag(net_dag),
      use_parallel(use_parallel),
      has_ged_matrix(has_ged_matrix),
      ged_matrix(ged_matrix),
      N(N),
      max_exact_ged_for_EPT(max_exact_ged_for_EPT),
      folder_name(folder_name),
      vM(vM),
      eM(eM),
      max_n(max_n),
      embeddings(embeddings),
      enable_friends_reassign(enable_friends_reassign),
      NDC(0) {
    brute_range = 2 * tau + 1;
    r_k = INF; // use INF from Utility.h
    child_range = 3 * r_k + 2 * tau + 4 * error_tolerance_index;
    max_dist = -1.0;
    max_anchor_node_id = -1;
    phase = 0;
    
    // Initialize constructor instances
    netdag_constructor_ = std::make_unique<NetDagConstructor>(this);
    ept_constructor_ = std::make_unique<EditPathForestConstructor>(this);
}

GismaIndexBuilder::~GismaIndexBuilder() {
    // Release all EPT memory
    for (auto& pair : anchor_ept_map) {
        if (pair.second != nullptr) {
            delete pair.second;
            pair.second = nullptr;
        }
    }
}

// Utility methods
const std::vector<std::shared_ptr<Anchor>>& GismaIndexBuilder::get_anchors() const {
    return anchors;
}

const std::vector<std::shared_ptr<Node>>& GismaIndexBuilder::get_nodes() const {
    return nodes;
}

void GismaIndexBuilder::setNetDag(std::shared_ptr<NetDag> nd) {
    this->net_dag = std::move(nd);
}

void GismaIndexBuilder::setAnchors(const std::vector<std::shared_ptr<Anchor>>& anchors) {
    this->anchors = anchors;
}

void GismaIndexBuilder::setNodes(const std::vector<std::shared_ptr<Node>>& nodes) {
    this->nodes = nodes;
}

// Callback methods for statistics collection
void GismaIndexBuilder::set_cluster_stats_callback(ClusterStatsCallback callback) {
    cluster_stats_callback_ = callback;
}

const GismaIndexBuilder::ClusterStatsCallback& GismaIndexBuilder::get_cluster_stats_callback() const {
    return cluster_stats_callback_;
}

double GismaIndexBuilder::get_ged_from_matrix(int i, int j) {
    NDC++;
    if (i < 0 || j < 0 || i >= N || j >= N) {
        throw std::out_of_range("Index out of bounds in get_ged_from_matrix");
    }
    if (i > j) std::swap(i, j);
    size_t index = static_cast<size_t>(i * N - (i * (i - 1)) / 2 + (j - i));
    return ged_matrix[index];
}

// Delegated NetDag construction methods
void GismaIndexBuilder::overall_NetDag_const() {
    netdag_constructor_->overall_NetDag_const();
}

void GismaIndexBuilder::net_tree_const() {
    netdag_constructor_->net_tree_const();
}

void GismaIndexBuilder::initialize() {
    netdag_constructor_->initialize();
}

std::vector<double> GismaIndexBuilder::getStateList(double maxRadius, double alpha) {
    return netdag_constructor_->getStateList(maxRadius, alpha);
}

void GismaIndexBuilder::update_phase() {
    netdag_constructor_->update_phase();
}

void GismaIndexBuilder::find_max_dist() {
    netdag_constructor_->find_max_dist();
}

void GismaIndexBuilder::update_friends() {
    netdag_constructor_->update_friends();
}

void GismaIndexBuilder::update_child_and_parent_by_phase_dict() {
    netdag_constructor_->update_child_and_parent_by_phase_dict();
}

void GismaIndexBuilder::link_parent_and_child() {
    netdag_constructor_->link_parent_and_child();
}

void GismaIndexBuilder::update_one_anchor() {
    netdag_constructor_->update_one_anchor();
}

void GismaIndexBuilder::get_nodes_in_cover_range() {
    netdag_constructor_->get_nodes_in_cover_range();
}

double GismaIndexBuilder::compute_ged_online(const std::shared_ptr<Node> &node1, const std::shared_ptr<Node> &node2) {
    return netdag_constructor_->compute_ged_online(node1, node2);
}

void GismaIndexBuilder::make_friends(std::shared_ptr<Anchor> anchor1, std::shared_ptr<Anchor> anchor2, double distance) {
    netdag_constructor_->make_friends(anchor1, anchor2, distance);
}

void GismaIndexBuilder::add_edge(std::shared_ptr<Anchor> parent, std::shared_ptr<Node> child, int child_phase, double child_dist) {
    netdag_constructor_->add_edge(parent, child, child_phase, child_dist);
}

void GismaIndexBuilder::compute_ged_to_csv() {
    netdag_constructor_->compute_ged_to_csv();
}

void GismaIndexBuilder::reassign_nodes_in_cluster() {
    netdag_constructor_->reassign_nodes_in_cluster();
}

void GismaIndexBuilder::reassign_nodes_in_cluster_with_csv() {
    netdag_constructor_->reassign_nodes_in_cluster_with_csv();
}

// Delegated EPT construction methods
void GismaIndexBuilder::construct_EPT() {
    ept_constructor_->construct_EPT();
}

void GismaIndexBuilder::construct_EPT_parallel() {
    ept_constructor_->construct_EPT_parallel();
}

void GismaIndexBuilder::compute_edit_path_and_save_to_csv() {
    ept_constructor_->compute_edit_path_and_save_to_csv();
}

void GismaIndexBuilder::find_within_tau_and_save_csv(const std::string &csv_filename) {
    ept_constructor_->find_within_tau_and_save_csv(csv_filename);
}

void GismaIndexBuilder::build_one_ball_EPT_from_csv(int EPT_root_ind,
                                             double alpha,
                                             const std::string &csv_path,
                                             const std::string &ept_filename) {
    ept_constructor_->build_one_ball_EPT_from_csv(EPT_root_ind, alpha, csv_path, ept_filename);
}

void GismaIndexBuilder::update_one_exact_anchor() {
    ept_constructor_->update_one_exact_anchor();
}

void GismaIndexBuilder::update_one_exact_anchor_parallel() {
    ept_constructor_->update_one_exact_anchor_parallel();
}

int GismaIndexBuilder::select_queries_from_csv(
    int root_ind,
    int need_count,
    const std::string &csv_path,
    const std::string &out_queries_txt,
    std::string &out_filtered_csv
) {
    return ept_constructor_->select_queries_from_csv(root_ind, need_count, csv_path, out_queries_txt, out_filtered_csv);
}

int GismaIndexBuilder::select_queries_by_ged_ranges(
    int root_ind,
    const std::string &csv_path,
    const std::string &output_dir
) {
    return ept_constructor_->select_queries_by_ged_ranges(root_ind, csv_path, output_dir);
}