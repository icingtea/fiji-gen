cmake_minimum_required(VERSION 4.2.3)
project(fiji-gen)

set(SOURCES
  src/cu_rl.cpp
  src/cu_vanilla.cpp
  src/main.cpp
  src/par_rl.cpp
  src/par_vanilla.cpp
  src/seq_rl.cpp
  src/seq_vanilla.cpp
)

add_executable(fiji_gen ${SOURCES})
set_target_properties(fiji_gen PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
