#include "include/corr.hpp"

void load_feature_vector(ADT32 *fm, hls::vector<WDT, 32> feature[3][32], short group)  //[input part][output part][outc]
{
    uint offset = group * 96;
    for (short op = 0; op < 3; op++)
    {
        for (short oc = 0; oc < 32; oc++)
        {
            #pragma HLS PIPELINE II=1
            WDT32 ddr_data = fm[offset + op*32 + oc];
            hls::vector<WDT, 32> temp_feature;
            for (short ic = 0; ic < 32; ic++)
            {
                #pragma HLS UNROLL
                temp_feature[ic] = ddr_data[ic];
            }   
            feature[op][oc] = temp_feature;
        }
    }
}


void export_corr(ADT32 *fm, hls::vector<RDT, 32> rfm[50][50], short outx, short group)  //one group is correspondant to 96 output channels
{
    uint offset = group*3*2304 + outx*2304;

    for (short row = 1; row < 49; row++)
    {
        for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 out_data;
            for (short c = 0; c < 32; c++)
            {
                #pragma HLS unroll

                ap_int<9> qy = rfm[row][col][c] >> 10; // div 8 = >> 3, scale = 7, so >> (3+7)

                ADT ddr_data = qy < amin ? ADT(amin) : qy > amax ? ADT(amax) : ADT(qy);

                out_data[c] = ddr_data;
                rfm[row][col][c] = 0;
            }
            fm[offset + (row-1)*48 + col - 1] = out_data;
        }
    }
    
}

//corr pool
ADT32 avg_pool(ADT32 op1, ADT32 op2, ADT32 op3, ADT32 op4)
{
    ADT32 res;
    for (short i = 0; i < 32; i++)
    {
        #pragma hls unroll
        #pragma hls pipeline II=1
        ap_int<9> add1 = op1[i] + op2[i];
        ap_int<9> add2 = op3[i] + op4[i];
        res[i] = (add1 + add2) >> 2;
    } 
    return res;
}

void corr_pool_load(ADT32 *ifm_ddr, hls::stream<ADT32> &poolin_flow, short layerNum)
{
    short dims;
    uint offset;
    switch(layerNum)
    {
        case 1: dims = 48; offset = 0; break;
        case 2: dims = 24; offset = 165888; break; // offset = 48*48*2304/32
        case 3: dims = 12; offset = 165888 + 41472; break; // offset = 48*48*2304/32 +  24*24*2304/32
        default: dims = 0; break;
    }

    for (int i = 0; i < 72*dims*dims; i++)
    {
        #pragma HLS pipeline II=1
        ADT32 ddr_data = ifm_ddr[offset + i];
        poolin_flow.write(ddr_data);
    }

}

void corr_pool_cal(hls::stream<ADT32> &poolin_flow, hls::stream<ADT32> &poolout_flow, short layerNum)
{
    short dims;
    switch(layerNum)
    {
        case 1: dims = 48; break;
        case 2: dims = 24; break; // offset = 24*24*2304/32
        case 3: dims = 12; break; // offset = 12*12*2304/32
        default: dims = 0; break;
    }
    ADT32 line_buffer[48];
    ADT32 temp_data;
    
    for (int ch_part = 0; ch_part < 72; ch_part++)
    {
        for (int row = 0; row < dims; row++)
        {
            for (int col = 0; col < dims; col++)
            {
                #pragma HLS pipeline II=1
                ADT32 ddr_data = poolin_flow.read();
                ADT32 res;

                ADT32 op1 = (col % 2 == 1)? line_buffer[col-1] : line_buffer[col];
                ADT32 op2 = line_buffer[col];
                for (short i = 0; i < 32; i++)
                {
                    #pragma hls unroll
                    ap_int<9>add1 = op1[i] + op2[i];
                    ap_int<9>add2 = temp_data[i] + ddr_data[i];
                    ap_int<10>add3 = (add1 + add2);
                    res[i] = add3(9,2);
                }

                if(row % 2 == 0)
                {
                    line_buffer[col] = ddr_data;
                }
                else if(col % 2 == 0)
                {
                    temp_data = ddr_data;
                }
                else
                {
                    poolout_flow.write(res);
                }
            } 
        }
    }
}


void corr_pool_export(ADT32 *ifm_ddr, hls::stream<ADT32> &poolout_flow, short layerNum)
{
    short dims;
    uint offset;
    switch(layerNum)
    {
        case 1: dims = 24; offset = 165888; break;// offset = 48*48*2304/32
        case 2: dims = 12; offset = 41472 + 165888; break; // offset = 24*24*2304/32 +  12*12*2304/32
        case 3: dims = 6; offset = 10368 + 41472 + 165888; break; // offset = 12*12*2304/32
        default: dims = 0; offset = 0; break;
    }

    for (int i = 0; i < 72*dims*dims; i++)
    {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=864 max=13824
        ADT32 ddr_data = poolout_flow.read();
        ifm_ddr[offset + i] = ddr_data;
    }
}

void corr_pool(ADT32 *fm, short layerNum)
{
    #pragma HLS dataflow
    hls::stream<ADT32> poolin_flow;
    hls::stream<ADT32> poolout_flow;
    
    corr_pool_load(fm, poolin_flow, layerNum);
    corr_pool_cal(poolin_flow, poolout_flow, layerNum);
    corr_pool_export(fm, poolout_flow, layerNum);
}