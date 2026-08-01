#pragma once
#include <vector>
#include <string>
#include <cstdint>

// 通用的轻量级搜索节点
struct SearchNodeLite {
    int16_t  level;     // 节点深度
    int16_t  image;     // 映射的G顶点
    uint16_t lb;        // lower_bound（下界）
    uint16_t mc;        // mapped_cost（映射代价）
    uint8_t  is_open;   // 是否为活跃节点：0=非活跃, 1=活跃, 3=full-mapping
    uint8_t  pad;       // 对齐填充

#ifdef _EXPAND_ALL_
    uint16_t siblings_n;      // siblings数量
    uint32_t siblings_offset; // siblings在数据数组中的偏移
#endif

#ifdef _USE_LSa_ESTIMATE_BMao_
    ui lsa_lb = INF;
#endif

};



/* ---------------- 通用搜索树快照 ---------------- 
 *   · mo  —— 查询图的映射顺序（|Q| 个顶点 ID）
 *   · v   —— level-order 结点序列
 *   · ub  —— 快照时的上界
 *   · siblings_data —— 所有节点的siblings数据（仅在_EXPAND_ALL_时）
 * 只在内存中流转，不做磁盘序列化
 */
struct SearchSnapshot {
    std::vector<uint32_t>        mo;    // 映射顺序（mapping order）
    std::vector<SearchNodeLite>  v;     // 轻量结点序列
    uint32_t                     ub{0}; // 终止时的 upper-bound
    uint32_t                     overall_lb{0};
    int                          margin{0}; // 生成snapshot时使用的margin
    int                          chain_depth{0}; // chain reuse depth: 0=fresh A*, N=Nth reuse in chain
#ifdef _EXPAND_ALL_
    std::vector<uint16_t>        siblings_data; // 扁平化存储的所有siblings数据
#endif
};