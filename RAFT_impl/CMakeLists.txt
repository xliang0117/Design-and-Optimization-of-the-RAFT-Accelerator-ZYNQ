cmake_minimum_required(VERSION 3.0.0)
project(RAFT_impl VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
   message(STATUS "Optimization level: 3")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
else()
   message(STATUS "Optimization: off")
   set(CMAKE_BUILD_TYPE "Debug")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -Wno-unknown-pragmas")
endif()


set(VIVADO_DIRECTORY "/media/lyx/software/Xilinx2021/Vitis_HLS/2021.2/")


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${VIVADO_DIRECTORY}/include
    ${VIVADO_DIRECTORY}/lnx64/tools/
    ${VIVADO_DIRECTORY}/tps/lnx64/gcc-6.2.0/include
    ${VIVADO_DIRECTORY}/tps/lnx64/gcc-6.2.0/lib/gcc/x86_64-pc-linux-gnu/6.2.0
)


link_directories(
    ${VIVADO_DIRECTORY}/lnx64/tools/fpo_v7_0
    ${VIVADO_DIRECTORY}/lnx64/tools/fft_v9_1
    ${VIVADO_DIRECTORY}/lnx64/tools/fir_v7_0
    ${VIVADO_DIRECTORY}/lnx64/tools/dds_v6_0
    ${VIVADO_DIRECTORY}/lnx64/csim
    /usr/lib/x86_64-linux-gnu
)

aux_source_directory(./ ALL_SRC)

add_executable( main
    main.cpp
    # ${ALL_SRC}
    fsm.cpp
    convolution.cpp
    corr.cpp
    corr_index.cpp
    log.cpp
)

target_link_libraries( main
    Ip_floating_point_v7_0_bitacc_cmodel
    mpfr
    gmp
)