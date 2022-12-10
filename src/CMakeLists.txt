
set(APP_NAME smallTorchExample)

# using these to compile if you want to gdb or lldb
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# ser(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_executable(${APP_NAME} "main.cpp")
target_include_directories(${APP_NAME} PUBLIC ".")


#############  source #############

FILE(GLOB_RECURSE ALL_SRC  *.cpp)
FILE(GLOB_RECURSE ALL_HEADER  *.h)

target_sources(${APP_NAME} PRIVATE ${ALL_SRC} PUBLIC ${ALL_HEADER})


#############  output dir ############# 

set_target_properties( ${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 17)


