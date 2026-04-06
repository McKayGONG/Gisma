cmake_minimum_required(VERSION 3.10)

project(GismaProject CXX)

if(MSVC)
    add_compile_options(/utf-8)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/ReuseSearchTree.cpp
    src/EditPathTree.cpp
)

target_include_directories(GismaProject PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads REQUIRED)
target_link_libraries(GismaProject
    PRIVATE
    Threads::Threads
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS ">>> Build type          = ${CMAKE_BUILD_TYPE}")
message(STATUS ">>> CMAKE_CXX_FLAGS     = ${CMAKE_CXX_FLAGS}")
message(STATUS ">>> CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
