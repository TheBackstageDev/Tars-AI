cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TARS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/OUTPUT/${PROJECT_NAME}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(deps)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.hpp src/*.cu)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_options(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_include_directories(${PROJECT_NAME} PUBLIC
    src/
    src/deps/
    deps/
    deps/glfw/include/
    deps/imgui/imgui/
    deps/miniaudio/
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    glad
    opengl32
    imgui
    miniaudio
    ${CUDA_LIBRARIES}
    cublas                     
)

set(MNIST_DATA_LOCATION ${CMAKE_SOURCE_DIR}/data/mnist)
set(MNIST_FOUND TRUE)

configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h)
