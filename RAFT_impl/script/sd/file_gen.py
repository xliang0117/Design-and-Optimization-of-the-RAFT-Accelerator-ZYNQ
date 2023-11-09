import sys

fileTemplate = """
#include "include/fsm.hpp"
#include "include/corr.hpp"
#include "hls_print.h"

void conv_hw(ADT4* IMAGE1, ADT4* IMAGE2, ADT32* fm, WDT32* weight, ADT* corr, FDT flow[2304])
{
    #pragma hls interface m_axi port=IMAGE1 depth=148000 offset=slave
    #pragma hls interface m_axi port=IMAGE2 depth=148000 offset=slave
    #pragma hls interface m_axi port=fm depth=360000 offset=slave
    #pragma hls interface m_axi port=weight depth=6400 offset=slave
    #pragma hls interface m_axi port=corr depth=517000 offset=slave
	// #pragma hls interface ap_memory port=flow storage_type=ram_1p
    #pragma hls interface m_axi port=flow depth=2500 offset=slave bundle=fbram
    #pragma hls interface s_axilite register port=return


    static hls::vector<ADT, 32> afm_1[50][50];  // [col][row]
    static hls::vector<ADT, 32> afm_2[50][50];
    static hls::vector<ADT, 32> afm_3[50][50];
    static hls::vector<ADT, 32> afm_4[50][50];
    static hls::vector<RDT, 32> rfm[3][50][50]; // [outpart][col][row]
    #pragma HLS bind_storage variable=rfm type=RAM_2P impl=URAM

    static hls::vector<WDT, 32> wBUF3x3_A[3][3];   // [pingpong][k][k]
    static hls::vector<WDT, 32> wBUF3x3_B[3][3];   // [pingpong][k][k]
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_A dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_A dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_B dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_B dim=2 complete

    static hls::vector<WDT, 32> wBUF1x1_A[3][32];  // [outpart][output channel]
    static hls::vector<WDT, 32> wBUF1x1_B[3][32];  // [outpart][output channel]
    #pragma HLS ARRAY_PARTITION variable=wBUF1x1_A dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF1x1_B dim=2 complete

    static BDT dwb_buf_A[32];
    static BDT dwb_buf_B[32];
    #pragma HLS ARRAY_PARTITION variable=dwb_buf_A dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=dwb_buf_B dim=1 complete

    static BDT pwb_buf[3][32];
    #pragma HLS ARRAY_PARTITION variable=pwb_buf dim=2 complete

// usercode here
    
}
"""

def fileWrapper(filename="./fsm_gen.cpp", inText="", fileTemplate=fileTemplate):
    fileHead, fileTail = fileTemplate.split("// usercode here\n", 2)
    with open(filename, "w") as f:
        f.write(fileHead + inText  + fileTail)
    print("file successfully gen!")