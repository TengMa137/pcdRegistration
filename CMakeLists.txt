cmake_minimum_required(VERSION 3.0)

project(pcdregister)

add_executable(pcdregister pcdregister.cpp)

install(TARGETS pcdregister RUNTIME DESTINATION bin)
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries (pcdregister ${PCL_LIBRARIES})
