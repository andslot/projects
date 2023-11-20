cmake_minimum_required(VERSION 3.0)
project(FasterNN)

set(CMAKE_CXX_STANDARD 23)

add_executable(FasterNN main.cpp
        Activation.h
        Activation.cpp)
