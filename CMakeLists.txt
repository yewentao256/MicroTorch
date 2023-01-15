# check mimimum cmake version
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# global LIB_NAME and setup project
set(LIB_NAME tinytorch)
project (${LIB_NAME} VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DEBUG mode or release mode, passing from setup.py
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# find all the src files
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB HEADER_FILES "src/*.hpp")
file (GLOB PYTHON_FILES "python/*.cpp")
file (GLOB CUDA_FILES "src/*.cu")

MESSAGE(WARNING "here is the cuda files: ${CUDA_FILES}")

# cuda
# TODO: 加到编译配置项
find_package(CUDA)
if (NOT CUDA_FOUND)
message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# register to pybind11
add_subdirectory(pybind11)
pybind11_add_module(${LIB_NAME}  
	${SRC_FILES}
	${HEADER_FILES}
    ${PYTHON_FILES}
)

cuda_add_library(LIB_NAME SHARED ${CUDA_FILES})



