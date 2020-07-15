cmake_minimum_required(VERSION 3.0)
project(libwavae)

find_package(Torch REQUIRED)

file(GLOB SRC src/*.cpp)

# TEST EXECUTABLE
add_executable(test ${SRC})
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)

# ACTUAL LIBRARY
add_library(wavae SHARED ${SRC})
target_link_libraries(wavae "${TORCH_LIBRARIES}")
set_property(TARGET wavae PROPERTY CXX_STANDARD 14)