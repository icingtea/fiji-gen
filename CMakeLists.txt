cmake_minimum_required(VERSION 3.20)
project(fiji-gen)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(include)

# LibTorch setup
set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
find_package(Torch REQUIRED)

# nlohmann::json setup
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Deceptive Royal Road
add_executable(deceptive_royal_road_problem src/deceptive_royal_road_problem.cpp)
target_compile_options(deceptive_royal_road_problem PRIVATE -O2 -Wall -Wextra)
target_link_libraries(deceptive_royal_road_problem pthread)

add_executable(deceptive_royal_road_problem_rl src/deceptive_royal_road_problem_rl.cpp)
target_compile_options(deceptive_royal_road_problem_rl PRIVATE -O2 -Wall -Wextra)
target_link_libraries(deceptive_royal_road_problem_rl pthread "${TORCH_LIBRARIES}")

# Jump Problem
add_executable(jump_problem src/jump_problem.cpp)
target_compile_options(jump_problem PRIVATE -O2 -Wall -Wextra)
target_link_libraries(jump_problem pthread nlohmann_json::nlohmann_json)

add_executable(jump_problem_rl src/jump_problem_rl.cpp)
target_compile_options(jump_problem_rl PRIVATE -O2 -Wall -Wextra)
target_link_libraries(jump_problem_rl pthread "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
