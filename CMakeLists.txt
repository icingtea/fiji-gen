cmake_minimum_required(VERSION 3.20)
project(fiji-gen)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output directory globally for all executables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(include)

# --- LibTorch Setup ---
set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
find_package(Torch REQUIRED)

# --- Baseline royal road (sequential + parallel island GA) ---
# add_executable(royal_road_problem src/royal_road_problem.cpp)
# target_compile_options(royal_road_problem PRIVATE -O2 -Wall -Wextra)
# target_link_libraries(royal_road_problem pthread)

# --- Deceptive RL royal road (Q-learning agent + parallel RL island model) ---
add_executable(deceptive_royal_road_problem src/deceptive_royal_road_problem.cpp)
target_compile_options(deceptive_royal_road_problem PRIVATE -O2 -Wall -Wextra)
target_link_libraries(deceptive_royal_road_problem pthread)

add_executable(deceptive_royal_road_problem_rl src/deceptive_royal_road_problem_rl.cpp)
target_compile_options(deceptive_royal_road_problem_rl PRIVATE -O2 -Wall -Wextra)
target_link_libraries(deceptive_royal_road_problem_rl pthread "${TORCH_LIBRARIES}")

# --- Baseline flow shop (sequential + parallel island GA) ---
# add_executable(flow_shop_problem src/flow_shop_problem.cpp)
# target_compile_options(flow_shop_problem PRIVATE -O2 -Wall -Wextra)
# target_link_libraries(flow_shop_problem pthread)

# --- RL royal road (Q-learning agent + parallel RL island model) ---
# add_executable(royal_road_problem_rl src/royal_road_problem_rl.cpp)
# target_compile_options(royal_road_problem_rl PRIVATE -O2 -Wall -Wextra)
# target_link_libraries(royal_road_problem_rl pthread "${TORCH_LIBRARIES}")

# --- RL flow shop (DQN agent + parallel RL island model) ---
# add_executable(flow_shop_problem_rl src/flow_shop_problem_rl.cpp)
# target_compile_options(flow_shop_problem_rl PRIVATE -O2 -Wall -Wextra)
# target_link_libraries(flow_shop_problem_rl pthread "${TORCH_LIBRARIES}")

# --- XOR Test ---
# add_executable(xor_nn src/xor_nn.cpp)
# target_compile_options(xor_nn PRIVATE -O2 -Wall -Wextra)
# target_link_libraries(xor_nn "${TORCH_LIBRARIES}")
