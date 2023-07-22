# check mimimum cmake version
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# global LIB_NAME and setup project
set(LIB_NAME _tinytorch)
project (${LIB_NAME} VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DEBUG mode or release mode, passing from setup.py
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -fPIC")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -fPIC /MTd")# for msvc
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC /MT")  # for msvc

# all the src files
set(HEADER_FILES
    src/backward.hpp
	src/context.hpp
	src/engine.hpp
	src/graph.hpp
	src/ops.hpp
	src/optimizer.hpp
	src/tensor.hpp
	src/allocator.hpp
	src/exception.hpp
	src/storage.hpp
	src/unit_test.hpp
	src/device.hpp
)
file (GLOB SRC_FILES "src/*.cpp")
file (GLOB BINDING_FILES "binding/*.cpp")
file (GLOB CUDA_FILES "src/*.cu")

# register to pybind11
add_subdirectory(pybind11-2.10)
pybind11_add_module(${LIB_NAME}  
	${SRC_FILES}
	${HEADER_FILES}
    ${BINDING_FILES}
)

# cuda
option(CUDA "Whether to use CUDA when available" OFF)
if (CUDA)
	set(CUDA_LIB gpu)
	message(STATUS "Using cuda mode to compile the project...")
	find_package(CUDA)
	if (CUDA_FOUND)
		# compile cuda and link it to our project
		cuda_add_library(${CUDA_LIB} STATIC ${CUDA_FILES})
		target_link_libraries (${LIB_NAME} PRIVATE ${CUDA_LIB})
		add_definitions(-DUSE_CUDA)
	else()
		message(FATAL_ERROR "CUDA not found, please turn off CUDA mode")
	endif()
endif()
