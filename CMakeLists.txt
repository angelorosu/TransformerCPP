cmake_minimum_required(VERSION 3.15)
project(TransformerInC VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add debug symbols even in release for profiling
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Library: Core autograd engine
add_library(autograd STATIC
    src/graph.cpp
    src/tensor.cpp
    src/optimizer.cpp
    src/layers.cpp
    src/init.cpp
    src/attention.cpp
    src/transformer_block.cpp
)

# Link math library if needed (for some systems)
if(UNIX AND NOT APPLE)
    target_link_libraries(autograd PRIVATE m)
endif()

# Executable: Main program
add_executable(transformer_main
    src/main.cpp
)
target_link_libraries(transformer_main PRIVATE autograd)


add_executable( xor_example
    examples/xor.cpp)
target_link_libraries(xor_example PRIVATE autograd)


# Optional: Enable testing
enable_testing()

# Optional: Add tests subdirectory if you create one
# add_subdirectory(tests)

# Installation rules (optional)
install(TARGETS transformer_main DESTINATION bin)
install(TARGETS autograd DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)