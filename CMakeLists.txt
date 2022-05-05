cmake_minimum_required(VERSION 3.22)
project(twoPhase)

set(CMAKE_CXX_STANDARD 14)

add_executable(twoPhase main.cpp newton_krylov.cpp lgmres.cpp)
