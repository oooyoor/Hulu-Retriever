cmake_minimum_required(VERSION 3.28)
project(FS_COMPARE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_library(PTHREAD pthread)
find_library(URING uring)
message (STATUS "PTHREAD: ${PTHREAD}")
message (STATUS "URING: ${URING}")

set(THIRD_PARTY_DIR ../external)
set(INCLUDE_DIR ../include)
include_directories(${THIRD_PARTY_DIR} ${INCLUDE_DIR} ${THIRD_PARTY_DIR}/hnswlib ${THIRD_PARTY_DIR}/spdlog/include)

# 如果从源码编译 liburing，可以通过环境变量 LIBURING 指定路径
# 系统安装的 liburing 头文件在 /usr/include，编译器会自动找到，无需额外配置
if(DEFINED ENV{LIBURING})
    include_directories($ENV{LIBURING}/src/include)
    message(STATUS "Using LIBURING from environment: $ENV{LIBURING}/src/include")
endif()

set(SRC    
    ./HnswFsIO.cpp
    # ./HnswOffsetIO.cpp
    # ./HnswMultiLayerFsIO.cpp
    ${INCLUDE_DIR}/param_config.cpp
)

add_executable(lab ${SRC})
message(STATUS "Lab directory: ${CMAKE_SOURCE_DIR}/execs")
set_target_properties(lab PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/execs)
# ---------------------------------------------------------------------------------------
# Example of using header-only library
# ---------------------------------------------------------------------------------------
if(SPDLOG_BUILD_EXAMPLE_HO)
    target_link_libraries(lab PRIVATE spdlog::spdlog_header_only)
endif()

target_link_libraries(lab PRIVATE
    pthread
    uring  
    rt
    numa 
)

