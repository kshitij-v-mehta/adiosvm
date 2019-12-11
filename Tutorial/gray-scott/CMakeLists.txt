cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
# ENABLE ADIOS2_ROOT
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(gray-scott LANGUAGES C CUDA CXX)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)

option(USE_TIMERS "Use profiling timers")
if(USE_TIMERS)
  message(STATUS "Enabling profiling timers")
  add_definitions(-DENABLE_TIMERS)
endif()

# We are not using the C++ API of MPI, this will stop the compiler look for it
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

# add_definitions(-Xcompiler -acc -fast -Minfo=accel -fPIC -g -ta=tesla:managed CUDA_HOME=/sw/summit/cuda/10.1.168/)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
# set(CMAKE_CXX_FLAGS "-acc -fast -Minfo=accel -g -ta=tesla:managed CUDA_HOME=/sw/summit/cuda/10.1.168/")
set(CMAKE_CXX_FLAGS "-acc -fast -Minfo=accel -g -ta=tesla:managed")
# SET(CMAKE_EXE_LINKER_FLAGS "-acc -fast -g -Mcuda -ta=tesla:managed CUDA_HOME=/sw/summit/cuda/10.1.168/ -lcurand")
SET(CMAKE_EXE_LINKER_FLAGS "-acc -fast -g -Mcuda -ta=tesla:managed -lcurand")

add_executable(gray-scott
  simulation/main.cpp
#  simulation/random_gen.h
#  simulation/random_gen.cu
  simulation/gray-scott.cpp
  simulation/settings.cpp
  simulation/writer.cpp
)
# set_target_properties(gray-scott PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(gray-scott PROPERTIES LINK_FLAGS "-acc -fast -g -Mcuda -ta=tesla:managed CUDA_HOME=/sw/summit/cuda/10.1.168/ -lcurand")
target_link_libraries(gray-scott adios2::adios2 MPI::MPI_C)

add_executable(pdf_calc analysis/pdf_calc.cpp)
target_link_libraries(pdf_calc adios2::adios2 MPI::MPI_C)

option(VTK "Build VTK apps")
if (VTK_ROOT)
  set(VTK ON)
endif(VTK_ROOT)

if (VTK)
  message(STATUS "Configuring VTK apps")

  find_package(VTK COMPONENTS
    vtkFiltersCore
    vtkIOImage
    vtkIOXML
  )

  if(VTK_FOUND)
    add_executable(isosurface analysis/isosurface.cpp)
    target_link_libraries(isosurface adios2::adios2 ${VTK_LIBRARIES}
      MPI::MPI_C)
  endif(VTK_FOUND)

  find_package(VTK COMPONENTS
    vtkFiltersCore
    vtkFiltersGeometry
  )

  if(VTK_FOUND)
    add_executable(find_blobs analysis/find_blobs.cpp)
    target_link_libraries(find_blobs adios2::adios2 ${VTK_LIBRARIES}
      MPI::MPI_C)
  endif(VTK_FOUND)

  find_package(VTK COMPONENTS
    vtkFiltersGeneral
  )

  if(VTK_FOUND)
    add_executable(compute_curvature analysis/curvature.cpp)
    target_link_libraries(compute_curvature adios2::adios2 ${VTK_LIBRARIES}
      MPI::MPI_C)
  endif(VTK_FOUND)


  find_package(VTK COMPONENTS
    vtkRenderingOpenGL2
    vtkViewsInfovis
  )

  if(VTK_FOUND)
    add_executable(render_isosurface plot/render_isosurface.cpp)
    target_link_libraries(render_isosurface adios2::adios2 ${VTK_LIBRARIES}
      MPI::MPI_C)
  endif(VTK_FOUND)
endif(VTK)
