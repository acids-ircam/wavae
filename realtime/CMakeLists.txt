cmake_minimum_required(VERSION 3.0)

project(WAVAE)

add_subdirectory("./libwavae")

add_library(encoder SHARED src/encoder.cpp)
set_target_properties(encoder PROPERTIES PREFIX "" SUFFIX "~.pd_linux")

add_library(decoder SHARED src/decoder.cpp)
set_target_properties(decoder PROPERTIES PREFIX "" SUFFIX "~.pd_linux")

