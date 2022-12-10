cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME TinyTorch)
project (${LIB_NAME} VERSION 1.0.0 LANGUAGES CXX)



############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



############# SOURCE ###############

add_subdirectory(src)







