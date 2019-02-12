cmake_minimum_required(VERSION 2.8.12.1)

PROJECT(project2A)
SET(VTK_DIR /home/walton/Programming/Classes/src/VTK-6.3.0/build)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#file(GLOB sources "*.cxx")
add_executable(project2A project2A)
#add_executable(project2A ${sources})

if(VTK_LIBRARIES)
  target_link_libraries(project2A ${VTK_LIBRARIES})
else()
  target_link_libraries(project2A vtkHybrid)
endif()

