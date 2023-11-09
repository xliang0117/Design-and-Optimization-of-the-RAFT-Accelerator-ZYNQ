#include "include/corr.hpp"


void load_corrmul_op(HALF8 *ifm_ddr, half wbuf1x1[32][4][8][32])
{
    #pragma HLS inline off
    ME_CPART: for (short c_part = 0; c_part < 3; c_part++)
    {
        ME_CID: for (short c_id = 0; c_id < 4; c_id++)
        {
            ME_COLPART: for (short col_part = 0; col_part < 8; col_part++)
            {
                #pragma HLS pipeline II=8
                ME_COL:for (short col = 0; col < 8; col++)
                {
                    #pragma HLS pipeline II=1
                    HALF8 ddr_data = ifm_ddr[(4*c_part+c_id)*3072 + (8*col_part+col)];

                    for (short c = 0; c < 8; c++)
                    {
                        #pragma HLS unroll
                        wbuf1x1[col_part][c_part][col][8*c_id+c] = rawBitsToHalf(ddr_data(c*16+15, c*16));
                    }
                    
                }
            }    
        }
    }
}

void export_corrmul(HALF8 *ofm_ddr, half ofm[4][32][18][18])
{
    #pragma HLS inline off
    ME_ROW:for (short row = 1; row < 17; row++)
    {
        ME_COL:for (short col = 1; col < 17; col++)
        {
            ME_CPART:for (short c_part = 0; c_part < 3; c_part++)
            {
                #pragma HLS pipeline II=4
                ME_CID:for (short c_id = 0; c_id < 4; c_id++)
                {
                    #pragma HLS pipeline II=1
                    HALF8 out_data;
                    for (short c = 0; c < 8; c++)
                    {
                        #pragma HLS unroll
                        half out_div8 = ofm[c_part][c_id*8+c][col][row];
                        out_data(c*16+15,c*16) = halfToRawBits(out_div8);
                        ofm[c_part][c_id*8+c][col][row] = 0;
                    }
                    ofm_ddr[((row-1)*16+(col-1))*3072/8 + (c_part*4+c_id)] = out_data;
                }
                
            }
        }
    }
    
}

// corr pool------------------------------------------------------------------------------
ap_uint<16> POOL4(half op1, half op2, half op3, half op4)
{
    #pragma HLS inline off
    half A,B;
    A = op1 + op2;
    B = op3 + op4;

    half C;
    C = A + B;

    half res;
    res = C/4;
    return halfToRawBits(res);
}

void corr_pool_load(HALF8 *ifm_ddr, hls::stream<HALF8> &corr_in_flow, short scale)
{
    short row_limit, col_limit;
    switch(scale)
    {
        case 0: row_limit = 48; col_limit = 8; break;
        case 1: row_limit = 24; col_limit = 4; break;
        case 2: row_limit = 12; col_limit = 2; break;
        case 3: row_limit = 6; col_limit = 1; break;
        default: row_limit = 0; col_limit = 0; break;
    }

    for (int i = 0; i < 3072*row_limit*col_limit; i++)
    {
        #pragma HLS pipeline II=1
        HALF8 ddr_data = ifm_ddr[i];
        corr_in_flow.write(ddr_data);
    }

}

void corr_pool_cal(hls::stream<HALF8> &corr_in_flow, hls::stream<HALF8> &corr_out_flow, short scale)
{
    short row_limit, col_limit;
    switch(scale)
    {
        case 0: row_limit = 48; col_limit = 8; break;
        case 1: row_limit = 24; col_limit = 4; break;
        case 2: row_limit = 12; col_limit = 2; break;
        case 3: row_limit = 6; col_limit = 1; break;
        default: row_limit = 0; col_limit = 0; break;
    }

    HALF8 line_buffer[8];
    
    for (int channel = 0; channel < 3072; channel++)
    {
        for (int row = 0; row < row_limit; row++)
        {
            for (int col = 0; col < col_limit; col++)
            {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=18432 max=1179648
                HALF8 ddr_data = corr_in_flow.read();
                if(row % 2 == 0)
                {
                    line_buffer[col] = ddr_data;
                }
                else
                {
                    HALF4 res;
                    HALF8 ddr_out_data;
                    for (int s = 0; s < 4; s++)
                    {   
                        #pragma HLS unroll
                        res(s*16+15, s*16) = POOL4(rawBitsToHalf(line_buffer[col](s*32+15,s*32)),
                                        rawBitsToHalf(line_buffer[col](s*32+31,s*32+16)),
                                        rawBitsToHalf(ddr_data(s*32+15,s*32)),
                                        rawBitsToHalf(ddr_data(s*32+31,s*32+16)));
                    }
                    if (col % 2 == 0)
                    {
                        ddr_out_data(63,0) = res;
                    }
                    else
                    {
                        ddr_out_data(127,64) = res;
                        corr_out_flow.write(ddr_out_data);
                    }
                    
                }
            } 
        }
    }
}

void corr_pool_export(HALF8 *ofm_ddr, hls::stream<HALF8> &corr_out_flow, short scale)
{
    short row_limit, col_limit;
    switch(scale)
    {
        case 0: row_limit = 24; col_limit = 4; break;
        case 1: row_limit = 12; col_limit = 2; break;
        case 2: row_limit = 6; col_limit = 1; break;
        default: row_limit = 0; col_limit = 0; break;
    }

    for (int i = 0; i < 3072*row_limit*col_limit; i++)
    {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=18432 max=294912
        HALF8 ddr_out_data = corr_out_flow.read();
        ofm_ddr[i] = ddr_out_data;
    }

}


void corr_pool(HALF8 *ifm_ddr, HALF8 *ofm_ddr, short scale)
{
    #pragma HLS dataflow
    hls::stream<HALF8> corr_in_flow;
    hls::stream<HALF8> corr_out_flow;
    
    corr_pool_load(ifm_ddr, corr_in_flow, scale);
    corr_pool_cal(corr_in_flow, corr_out_flow, scale);
    corr_pool_export(ofm_ddr, corr_out_flow, scale);
}

void corr_pool_test(HALF8 *ifm_ddr, HALF8 *ofm_ddr)
{

    #pragma HLS INTERFACE m_axi port=ifm_ddr depth=80000 offset=slave bundle=fm
    #pragma HLS INTERFACE m_axi port=ofm_ddr depth=61500 offset=slave bundle=fm
    for (short scale = 0; scale < 3; scale++)
    {
        corr_pool(ifm_ddr, ofm_ddr, scale);
    }

}
