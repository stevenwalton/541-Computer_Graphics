cmake_minimum_required(VERSION 2.8.12.1)

PROJECT(project1C)
SET(VTK_DIR /home/walton/Programming/Classes/src/VTK-8.1.2/build)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#file(GLOB sources "*.cxx")
add_executable(project1C project1C)
#add_executable(project1C ${sources})

if(VTK_LIBRARIES)
  target_link_libraries(project1C ${VTK_LIBRARIES})
else()
  target_link_libraries(project1C vtkHybrid)
endif()

