cmake_minimum_required(VERSION 2.8.12.1)

PROJECT(project1F)
SET(VTK_DIR /home/walton/Programming/Classes/src/VTK-8.1.2/build)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#file(GLOB sources "*.cxx")
add_executable(project1F project1F)
#add_executable(project1F ${sources})

if(VTK_LIBRARIES)
  target_link_libraries(project1F ${VTK_LIBRARIES})
else()
  target_link_libraries(project1F vtkHybrid)
endif()

