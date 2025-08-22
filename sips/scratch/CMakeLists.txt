# minimum version of CMake
cmake_minimum_required(VERSION 3.10)
# set project name
project(scratch)
# set compilor options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable warnings and optimizations
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -O3 -fPIC -march=native -flto")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
# mode dependent flags(debug or release) in addition to CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -march=native -flto")

# include director
get_filename_component(PARENT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)
include_directories(${PARENT_DIR}/include)

# add_executable should always located bofore liking libraries
add_executable(scratch ${PARENT_DIR}/scratch/scratch_1.cpp)

# Find and link OpenMP
# see https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
# for lnking OpenMP with modern cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(scratch PUBLIC OpenMP::OpenMP_CXX)
endif()









