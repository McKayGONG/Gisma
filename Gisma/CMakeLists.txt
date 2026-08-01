cmake_minimum_required(VERSION 3.10)

project(GismaProject CXX)

# ---------------------------
# MSVC UTF-8 编码支持（解决中文注释编译问题）
# ---------------------------
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ---------------------------
# 若用户没传 -DCMAKE_BUILD_TYPE，默认用 Release
# ---------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---------------------------
# 在 Release 模式时附加 -O3 -DNDEBUG
# ---------------------------
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# ---------------------------
# 设置 C++17
# ---------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------
# ONNX Runtime dependencies removed (using embedding vectors instead)
# ---------------------------
# set(ONNXRUNTIME_DIR
#     "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.19.0"
# )
# 
# include_directories("${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")

# ---------------------------
# Torch dependencies removed (using embedding vectors instead)
# ---------------------------
# set(CMAKE_PREFIX_PATH
#     "/home/comp/ykgong/usr;"
#     "/home/comp/ykgong/torch_sparse_install;"
#     "/home/comp/ykgong/PGPG/PG/NetDag_code_in_cpp/libtorch"
# )
# 
# find_package(Torch REQUIRED)
# find_package(TorchScatter REQUIRED)
# find_package(TorchSparse REQUIRED)

# =========================================================
#                 可执行文件与源码
# =========================================================
add_executable(GismaProject
    src/main.cpp
    src/GismaIndexBuilder.cpp
    src/NetDagConstructor.cpp
    src/EditPathForestConstructor.cpp
    src/NetDag.cpp
    src/Graph.cpp
    src/Utility.cpp
    src/GismaSearchEngine.cpp
    src/Application.cpp
    src/OrigApp.cpp             # ★ 原版 App-BMao (namespace origbmao, baseline 对照)
    src/ReuseSearchTree.cpp      # ★ 新增
    src/EditPathTree.cpp
    # src/ML_graph.cpp        # Removed: Using embedding vectors instead
)

add_executable(GreedTrainingDataGenerator
    src/GreedTrainingDataGenerator.cpp
    src/Application.cpp
    src/Graph.cpp
    src/Utility.cpp
    src/ReuseSearchTree.cpp      # ★ 新增
    # src/ML_graph.cpp        # Removed: Using embedding vectors instead
)

add_executable(get_ground_truth
    src/get_ground_truth.cpp
    src/Application.cpp
    src/Graph.cpp
    src/Utility.cpp
    src/ReuseSearchTree.cpp      # ★ 新增
    # src/ML_graph.cpp        # Removed: Using embedding vectors instead
)

# Test executables have been removed during cleanup
# ---------------------------
# 公共头文件目录
# ---------------------------
target_include_directories(GismaProject PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(GreedTrainingDataGenerator PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(get_ground_truth PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------
# 链接库
# ---------------------------
find_package(Threads REQUIRED)
# find_package(OpenMP)  # 不使用 OpenMP，并行用 std::async

foreach(tgt GismaProject GreedTrainingDataGenerator get_ground_truth)
    target_link_libraries(${tgt}
        PRIVATE
        # "${TORCH_LIBRARIES}"      # Removed: Using embedding vectors instead
        # TorchScatter::TorchScatter # Removed: Using embedding vectors instead
        # TorchSparse::TorchSparse   # Removed: Using embedding vectors instead
        Threads::Threads
        # onnxruntime  # Removed: Using embedding vectors instead
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    )
    # if(OpenMP_CXX_FOUND)
    #     target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    # endif()
endforeach()

# ---------------------------
# 生成的可执行文件输出到项目根
# ---------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# ---------------------------
# 打印调试信息
# ---------------------------
message(STATUS ">>> Build type          = ${CMAKE_BUILD_TYPE}")
message(STATUS ">>> CMAKE_CXX_FLAGS     = ${CMAKE_CXX_FLAGS}")
message(STATUS ">>> CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
# message(STATUS ">>> Torch include dirs  = ${TORCH_INCLUDE_DIRS}")  # Removed: Using embedding vectors instead
# message(STATUS ">>> Torch libraries     = ${TORCH_LIBRARIES}")     # Removed: Using embedding vectors instead
# message(STATUS ">>> ONNXRUNTIME_DIR     = ${ONNXRUNTIME_DIR}")      # Removed: Using embedding vectors instead

# 导出编译命令 (便于 clangd / VSCode 等工具)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
