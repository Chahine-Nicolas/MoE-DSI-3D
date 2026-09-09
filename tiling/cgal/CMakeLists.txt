cmake_minimum_required(VERSION 3.1...3.15)

project( main )


# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )
find_package (Eigen3)

INCLUDE(${main_SOURCE_DIR}/FindLAS.cmake)

if ( NOT CGAL_FOUND )
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  
endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )
  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()  
endif()

message(STATUS "laslib : ${LAS_LIBRARIES}")

# include for local directory
include_directories(${main_SOURCE_DIR}/extern/LAStools/LASlib/inc/)
include_directories(${main_SOURCE_DIR}/extern/LAStools/LASzip/src/)
include_directories(/usr/include/eigen3/)
#include(${CGAL_USE_FILE})

# Creating entries for target: main
# ############################

add_executable( compute_normals  ${main_SOURCE_DIR}/src/compute_normals.cpp )
target_link_libraries(compute_normals CGAL::CGAL ${LAS_LIBRARIES})

add_to_cached_list( CGAL_EXECUTABLE_TARGETS compute_normals )



