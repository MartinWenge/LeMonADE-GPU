cmake_minimum_required(VERSION 3.1)

FIND_PACKAGE(CUDA REQUIRED)
if( ${CUDA_VERSION_MAJOR} GREATER 7 )
    set( NVCC_EXPERIMENTAL_FLAG "--expt" )
else()
    set( NVCC_EXPERIMENTAL_FLAG "-" )
endif()

set( CUDA_ARCH "50")

set( MINIMAL_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -Xcompiler -fPIC,-Wno-unused-parameter
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
    ${NVCC_EXPERIMENTAL_FLAG}-relaxed-constexpr
)
set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} " --compiler-options -Wall,-Wextra" )
# sauron
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
#        -lineinfo -gencode arch=compute_20,code=sm_20 --compiler-bindir ${CXX} -I${INCLUDED})

# samweis
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_50,code=sm_50)
        
#for taurus
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -gencode arch=compute_35,code=sm_35)


if (NOT DEFINED LEMONADE_INCLUDE_DIR)
message("LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location")
endif()

if (NOT DEFINED LEMONADE_LIBRARY_DIR)
message("LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location")
endif()

include_directories (${LEMONADE_INCLUDE_DIR})
link_directories (${LEMONADE_LIBRARY_DIR})

# cuda_add_library( pscbfmUpdater
#     src/pscbfm/UpdaterGPUScBFM_AB_Type.cu
# )
# target_link_libraries( pscbfmUpdater ${CUDA_curand_LIBRARY} cuda )

add_library( mainCUDASimulator mainCUDASimulatorGPUScBFM_AB_Type.cpp)
# cuda_add_executable( mainCUDASimulator
#     src/pscbfm/mainCUDASimulatorGPUScBFM_AB_Type.cpp
# )
cuda_add_executable(SimulatorCUDA UpdaterGPUScBFM_AB_Type.cu)
target_link_libraries(SimulatorCUDA mainCUDASimulator LeMonADE ${CUDA_curand_LIBRARY} cuda )
# target_link_libraries(SimulatorCUDA mainCUDASimulator pscbfmUpdater LeMonADE ${CUDA_curand_LIBRARY} )
