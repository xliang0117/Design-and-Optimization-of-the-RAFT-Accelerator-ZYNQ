
#include "include/fsm.hpp"
#include "include/corr.hpp"
#include "hls_print.h"

void gs_1_stage_predict(ADT4* IMAGE1, ADT4* IMAGE2, ADT32* fm, WDT32* weight, ADT8* corr, FDT flow[2304], ADT8* corr_buffer, 
                        bool update[2304])
{
    #pragma hls interface m_axi port=IMAGE1 depth=148000 offset=slave
    #pragma hls interface m_axi port=IMAGE2 depth=148000 offset=slave
    #pragma hls interface m_axi port=fm depth=360000 offset=slave
    #pragma hls interface m_axi port=weight depth=6400 offset=slave
    #pragma hls interface m_axi port=corr depth=517000 offset=slave
    #pragma hls interface m_axi port=corr_buffer depth=80000 offset=slave bundle=corr_bf
	// #pragma hls interface ap_memory port=flow storage_type=ram_1p
    #pragma hls interface m_axi port=flow depth=2500 offset=slave bundle=fbram
    #pragma hls interface bram port=update depth=2304
    #pragma hls interface s_axilite register port=return

    CORR:for (short layerNum = 0; layerNum < 4; layerNum++)
    {
        grid_sample_UPDATE(fm+10000, flow, update, corr_buffer+layerNum*2304*8, layerNum);
        grid_sample_ING(corr, flow, corr_buffer+layerNum*2304*8, layerNum);
    }
    
}

void gs_2_stage_predict(ADT4* IMAGE1, ADT4* IMAGE2, ADT32* fm, WDT32* weight, ADT8* corr, FDT flow[2304], ADT16* corr_buffer, 
                        bool update[2304], bool des[2304])
{
    #pragma hls interface m_axi port=IMAGE1 depth=148000 offset=slave
    #pragma hls interface m_axi port=IMAGE2 depth=148000 offset=slave
    #pragma hls interface m_axi port=fm depth=360000 offset=slave
    #pragma hls interface m_axi port=weight depth=6400 offset=slave
    #pragma hls interface m_axi port=corr depth=517000 offset=slave
    #pragma hls interface m_axi port=corr_buffer depth=80000 offset=slave bundle=corr_bf
	// #pragma hls interface ap_memory port=flow storage_type=ram_1p
    #pragma hls interface m_axi port=flow depth=2500 offset=slave bundle=fbram
    #pragma hls interface bram port=update depth=2304
    #pragma hls interface bram port=des depth=2304
    #pragma hls interface s_axilite register port=return
    

    CORR:for (short layerNum = 0; layerNum < 4; layerNum++)
    {
        grid_predict_UPDATE(fm+10000, flow, update, des, corr_buffer+layerNum*2304*8, layerNum);
        grid_predict_ING(corr, flow, des, corr_buffer+layerNum*2304*8, layerNum);
    }
    
}
