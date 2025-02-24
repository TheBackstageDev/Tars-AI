cmake_minimum_required(VERSION 3.10)
project(TARS)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/OUTPUT/${PROJECT_NAME}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(deps)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.hpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC src/ src/deps/)
