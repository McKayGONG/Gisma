#pragma once
#include <vector>
#include <string>
#include <cstdint>

// Generic lightweight search node
struct SearchNodeLite {
    int16_t  level;     // node depth
    int16_t  image;     // mapped G vertex
    uint16_t lb;        // lower_bound
    uint16_t mc;        // mapped_cost
    uint8_t  is_open;   // whether active node: 0=inactive, 1=active, 3=full-mapping
    uint8_t  pad;       // alignment padding
    uint16_t vl_lb;     // vertex label lower bound
    uint16_t vl_common; // common label count
    uint16_t mc_cross;  // cross-edge cost
    uint8_t  vlabel_same; // whether vertex labels are the same
    uint8_t  _pad2;     // alignment padding

    
#ifdef _EXPAND_ALL_
    uint16_t siblings_n;      // number of siblings
    uint32_t siblings_offset; // offset of siblings in data array
#endif

#ifdef _USE_LSa_ESTIMATE_BMao_
    ui lsa_lb = INF;
#endif
};



/* ---------------- Generic Search Tree Snapshot ----------------
 *   . mo  -- mapping order of query graph (|Q| vertex IDs)
 *   . v   -- level-order node sequence
 *   . ub  -- upper bound at snapshot time
 *   . siblings_data -- siblings data for all nodes (only with _EXPAND_ALL_)
 * Only circulated in memory; not serialized to disk
 */
struct SearchSnapshot {
    std::vector<uint32_t>        mo;    // mapping order
    std::vector<SearchNodeLite>  v;     // lightweight node sequence
    uint32_t                     ub{0}; // upper-bound at termination
    uint32_t                     overall_lb{0};
    int                          margin{0}; // margin used when generating the snapshot
#ifdef _EXPAND_ALL_
    std::vector<uint16_t>        siblings_data; // flattened storage of all siblings data
#endif
};