cmake_minimum_required(VERSION 2.8.12.1)

PROJECT(project1D)
SET(VTK_DIR /home/walton/Programming/Classes/src/VTK-8.1.2/build)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#file(GLOB sources "*.cxx")
add_executable(project1D project1D)
#add_executable(project1D ${sources})

if(VTK_LIBRARIES)
  target_link_libraries(project1D ${VTK_LIBRARIES})
else()
  target_link_libraries(project1D vtkHybrid)
endif()

