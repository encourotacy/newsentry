cmake_minimum_required(VERSION 3.16)

add_library(auto_outpost OBJECT
    outpost_solver.cpp
)

target_link_libraries(auto_outpost ${OpenCV_LIBS} yaml-cpp)
