FIND_PACKAGE(CUDA REQUIRED)
if( ${CUDA_VERSION_MAJOR} GREATER 7 )
    set( NVCC_EXPERIMENTAL_FLAG "--expt" )
else()
    set( NVCC_EXPERIMENTAL_FLAG "-" )
endif()


# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# include the directory to be tested
include_directories(
    ${CMAKE_SOURCE_DIR}/src/ 
    ${CMAKE_SOURCE_DIR}/src/pscbfm 
    ${CMAKE_SOURCE_DIR}/src/pscbfm/rngs/
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library( pscbfmUpdater
    ${CMAKE_SOURCE_DIR}/src/pscbfm/UpdaterGPUScBFM_AB_Type.cu
)

# Make test executable
# here we use a hard coded list of test file to avoid the pitfalls of file(GLOB ... )
# we could organize the test cases in subgroups if we have enough of them
set(TEST_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/lemonadeGPUTest-main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/testGPUScBFM_AB_Type.cpp
)

cuda_add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests Catch LeMonADE pscbfmUpdater ${CUDA_curand_LIBRARY} cuda)

MESSAGE("Run test with: make runtest")
ADD_CUSTOM_TARGET(
    runtest
    
    COMMENT "Running the tests." VERBATIM
    COMMAND ${CMAKE_BINARY_DIR}/tests/bin/tests
)