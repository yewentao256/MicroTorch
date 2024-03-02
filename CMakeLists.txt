# check mimimum cmake version
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# global MICROTORCH and setup project
set(MICROTORCH _microtorch)
project (${MICROTORCH} VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DEBUG mode or release mode, passing from setup.py
add_compile_options("$<$<CONFIG:DEBUG>:-O0;-Wall;-g;-fPIC>")
add_compile_options("$<$<CONFIG:RELEASE>:-O3;-Wall;-fPIC>")

# MSVC specific flags, if needed, cmake auto detection variable
if (MSVC)
    add_compile_options("$<$<CONFIG:DEBUG>:/MTd;/Zi>")
    add_compile_options("$<$<CONFIG:RELEASE>:/MT>")
endif()

# all the src files
file (GLOB CORE_FILES "src/core/*.cpp")
file (GLOB CPU_FILES "src/cpu/*.cpp")
file (GLOB BINDING_FILES "src/binding/*.cpp")

# register to pybind11
add_subdirectory(pybind11-2.10)
pybind11_add_module(${MICROTORCH}  
	${CORE_FILES}
	${CPU_FILES}
    ${BINDING_FILES}
)

# cuda
option(CUDA "Whether to use CUDA when available" OFF)
if(CUDA)
    message("Configuring Cuda mode.")
    # compile for all architecture
    cmake_policy(SET CMP0104 OLD)

	set(CUDA_LIB cuda_lib)
    enable_language(CUDA)
    file (GLOB CUDA_FILES "src/cuda/*.cu")

    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -Xcompiler -Wall -fPIC --extended-lambda")
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xcompiler -Wall -fPIC --extended-lambda")
    endif()
    add_library(${CUDA_LIB} STATIC ${CUDA_FILES})

    set_property(TARGET ${CUDA_LIB} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${CUDA_LIB} PROPERTY CUDA_STANDARD_REQUIRED ON)

    # For including header files
    target_include_directories(${CUDA_LIB} PRIVATE include)
    target_include_directories(${MICROTORCH} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # link cuda lib
    target_link_libraries(${MICROTORCH} PRIVATE ${CUDA_LIB})

    # USE_CUDA micro
    target_compile_definitions(${MICROTORCH} PRIVATE USE_CUDA)
    target_compile_definitions(${CUDA_LIB} PRIVATE USE_CUDA)

endif()


# For including header files
target_include_directories(${MICROTORCH} PRIVATE include)

# print symbol info
message("Using ${CMAKE_BUILD_TYPE} mode.")
