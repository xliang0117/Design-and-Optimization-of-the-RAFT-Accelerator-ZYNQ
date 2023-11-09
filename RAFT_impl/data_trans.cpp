#include "include/convolution.hpp"
#include "include/log.hpp"
#define LOCAL_LOG_LEVEL LOG_LEVEL_DEBUG


void load_fm_IMAGE(ADT4 *ifm_ddr, hls::vector<ADT, 32> ifm[50][50], short rowx, short colx) //all ex addr should before one row and one col
{
    int offset = 385 + rowx*48*384 + colx*48; // 384+1 for start offset
    short row_padding = (rowx == 0)? TOP_ROW : (rowx == 8)? BOTTOM_ROW : NO_PADDING;
    short col_padding = (colx == 0)? TOP_COL : (colx == 8)? BOTTOM_COL : NO_PADDING;

    for (short row = 0; row < 50; row++)
    {
        for (short col = 0; col < 50; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT4 in_ddr_data;
            hls::vector<ADT, 32> temp_ifm;
            in_ddr_data = ifm_ddr[offset + (row-1) * 384 + col-1];

            for (short c = 0; c < 3; c++)
            {
                #pragma hls unroll
                ap_uint<8> ddr_data;
                ADT valid_data;
                ADT zero_data = 0;
                
                ddr_data(7,0) = in_ddr_data[c](7,0);
                if ( (row == 0 && row_padding == TOP_ROW) || (row == 49 && row_padding == BOTTOM_ROW) 
                    || (col == 0 && col_padding == TOP_COL) || (col == 49 && col_padding == BOTTOM_COL) )
                {
                    valid_data = zero_data;
                }
                else
                {
                    valid_data = ddr_data - 128;
                }
                temp_ifm[c] = valid_data;
            }
            ifm[row][col] = temp_ifm;
        }
    }
}



void load_fm_tile(ADT32 *ifm_ddr, hls::vector<ADT, 32> ifm[50][50], short rowx, short colx, short chx, short width, short height) //all ex addr should before one row and one col
{
    int offset = chx*width*height + rowx*48*width + colx*48 +width+1; //+width+1 for padding
    short row_padding = (rowx == 0)? TOP_ROW : ((rowx+1)*48 == height)? BOTTOM_ROW : NO_PADDING;
    short col_padding = (colx == 0)? TOP_COL : ((colx+1)*48 == width)? BOTTOM_COL : NO_PADDING;

    for (short row = 0; row < 50; row++)
    {
        for (short col = 0; col < 50; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 in_ddr_data;
            hls::vector<ADT, 32> temp_ifm;
            in_ddr_data = ifm_ddr[offset + (row-1) * width + (col-1)];

            for (short c = 0; c < 32; c++)
            {
                ADT valid_data;
                ADT ddr_data = in_ddr_data[c];
                ADT zero_data = 0;
                
                if ( (row == 0 && row_padding == TOP_ROW) || (row == 49 && row_padding == BOTTOM_ROW) 
                    || (col == 0 && col_padding == TOP_COL) || (col == 49 && col_padding == BOTTOM_COL) )
                {
                    valid_data = zero_data;
                }
                else
                {
                    valid_data = ddr_data;
                }
                temp_ifm[c] = valid_data;
            }
            ifm[row][col] = temp_ifm;
        }
    }
}



void load_fm_s2(ADT32 *ifm_ddr, hls::vector<ADT, 32> ifm[50][50], short rowx, short colx, short chx, short width, short height) //all ex addr should before one row and one col
{
    int offset = chx*width*height + rowx*96*width + colx*96;

    for (short row = 1; row < 49; row++)
    {
        for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 in_ddr_data;
            hls::vector<ADT, 32> temp_ifm;
            in_ddr_data = ifm_ddr[offset + 2*(row-1)*width + 2*(col-1)];
            for (short c = 0; c < 32; c++)
            {
                temp_ifm[c] = in_ddr_data[c];
            }
            ifm[row][col] = temp_ifm;
        }
    }
}

void load_fm_full(ADT32 *ifm_ddr, hls::vector<ADT, 32> ifm[50][50], short chx) //all ex addr should before one row and one col
{
    int offset = chx*48*48 +48+1; //+48+1 for padding

    for (short row = 0; row < 50; row++)
    {
        for (short col = 0; col < 50; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 in_ddr_data;
            hls::vector<ADT, 32> temp_ifm;
            in_ddr_data = ifm_ddr[offset + (row-1)*48 + (col-1)];
            for (short c = 0; c < 32; c++)
            {
                ADT zero_data = 0;
                ADT valid_data;
                if(row == 0 || row == 49 || col == 0 || col == 49 ){
                    valid_data = zero_data;
                }
                else{
                    valid_data = in_ddr_data[c];
                }
                temp_ifm[c] = valid_data;
            }
            ifm[row][col] = temp_ifm;
        }
    }
}

void load_fm_flow(FDT flow[2304], hls::vector<ADT, 32> ifm[50][50])
{
    for (short row = 1; row < 49; row++)
    {
        for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            FDT temp_flow;
            temp_flow = flow[(row-1)*48 + (col-1)];
            ifm[row][col][0] = temp_flow[0];
            ifm[row][col][1] = temp_flow[1];
        }
    }
}



void load_w3x3(WDT32 *weight_ddr, hls::vector<WDT, 32> wBUF3x3[3][3], short inx)
{
    int offset = inx * 9;
    
    for (short row = 0; row < 3; row++)
    {
        for (short col = 0; col < 3; col++)
        {
            #pragma HLS PIPELINE II=1
            WDT32 weight_ddr_data = weight_ddr[offset + row*3 + col];
            for (short c = 0; c < 32; c++)
            {
                #pragma HLS UNROLL
                wBUF3x3[row][col][c] = weight_ddr_data[c];
            }
        }
    }
}

void load_w1x1(WDT32 *weight_ddr, hls::vector<WDT, 32> wBUF1x1[3][32], short inx, short outPart)
{
    int offset = (inx)*outPart*32;

    for (short op = 0; op < outPart; op++)
    {
        for (short c_o = 0; c_o < 32; c_o++)
        {
            #pragma HLS PIPELINE II=1
            WDT32 ddr_data = weight_ddr[offset + op*32 + c_o];
            for (short c_i = 0; c_i < 32; c_i++)
            {
                wBUF1x1[op][c_o][c_i] = ddr_data[c_i];
            }
        }
    }
}

void load_dwbbuf(WDT32 *weight_ddr, BDT bbuf[32], short inx)
{
    #pragma hls inline off

    for (int iter = 0; iter < 2; iter++)
    {
        #pragma hls pipeline II=1
        WDT32 in_ddr_data = weight_ddr[2*inx+iter];
        for (int c = 0; c < 16; c++)
        {
            bbuf[iter*16+c] = (in_ddr_data[2*c+1], in_ddr_data[2*c]);
        }
    }
}

void load_pwbbuf(WDT32 *weight_ddr, BDT bbuf[3][32], short outPart)
{
    #pragma hls inline off

    for (int op = 0; op < outPart; op++)
    {
        for (int iter = 0; iter < 2; iter++)
        {
            #pragma hls pipeline II=1
            WDT32 in_ddr_data = weight_ddr[2*op+iter];
            for (int c = 0; c < 16; c++)
            {
                bbuf[op][iter*16+c] = (in_ddr_data[2*c+1], in_ddr_data[2*c]);
            }
        }
    }
}


void out_with_flow(hls::vector<ADT, 32> ifm[50][50], FDT flow[2304], short chx)
{
    for (int row = 1; row < 49; row++)
    {
        for (int col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            if (chx == 2)
            {
                FDT temp_flow;
                temp_flow = flow[(row-1)*48 + col-1];
                ifm[row][col][16] = temp_flow[0];
                ifm[row][col][17] = temp_flow[1];
            }
        }
    }
}

void export_fm_tile(ADT32 *ofm_ddr, hls::vector<ADT, 32> ofm[50][50], short rowx, short colx, short chx, short width, short height)
{
    int offset = chx*width*height + rowx*48*width + colx*48;

    for (short row = 1; row < 49; row++)
    {
        for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 out_data;
            for (short c = 0; c < 32; c++)
            {
                #pragma HLS unroll
                ADT ddr_data = ofm[row][col][c];
                out_data[c] = ddr_data;
            }
            ofm_ddr[offset + (row-1)*width + col - 1] = out_data;
        }
    }
}


void export_fm_s2(ADT32 *ofm_ddr, hls::vector<ADT, 32> ofm[50][50], short rowx, short colx, short chx, short width, short height)
{
    int offset = chx*width*height + rowx*24*width + colx*24;

    for (short row = 0; row < 24; row++)
    {
        for (short col = 0; col < 24; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 out_data;
            for (short c = 0; c < 32; c++)
            {
                #pragma HLS unroll
                ADT ddr_data = ofm[2*row+1][2*col+1][c];
                out_data[c] = ddr_data;
            }
            ofm_ddr[offset + (row)*width + col] = out_data;
        }
    }

}

void export_fm_full(ADT32 *ofm_ddr, hls::vector<ADT, 32> ofm[50][50], short chx)
{
    int offset = chx*48*48;
    for (short row = 1; row < 49; row++)
    {
        for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 out_data;
            for (short c = 0; c < 32; c++)
            {
                #pragma HLS unroll
                ADT ddr_data = ofm[row][col][c];
                out_data[c] = ddr_data;
            }
            ofm_ddr[offset + (row-1)*48 + col - 1] = out_data;
        }
    }
}

void export_fm_full_withact(ADT32 *ofm_ddr, hls::vector<RDT, 32> ofm[50][50], short chx, BDT bbuf[32], SDT scale, bool relu)
{
    int offset = chx*48*48;
    for (short row = 1; row < 49; row++)
    {
        for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            ADT32 out_data;
            for (short c = 0; c < 32; c++)
            {
                #pragma HLS unroll
                ap_int<20> qy = ofm[row][col][c] + bbuf[c];

                if (qy < 0 && relu)
                {
                    qy = 0;
                }

                qy = qy >> scale;

                out_data[c] = qy < amin ? ADT(amin) : qy > amax ? ADT(amax) : ADT(qy);
                ofm[row][col][c] = 0;
            }
            ofm_ddr[offset + (row-1)*48 + col - 1] = out_data;
        }
    }
}

void export_flow(hls::vector<ADT, 32> ofm[50][50], FDT flow[2304], bool update[2304])
{
    #pragma hls inline off
    FLOW_ROW:for (short row = 1; row < 49; row++)
    {
        FLOW_COL:for (short col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            ap_int<16> x_flow, y_flow;
            FDT temp_flow, out_flow;
            temp_flow = flow[(row-1)*48+col-1];
            x_flow = ofm[row][col][0] + temp_flow[0];
            y_flow = ofm[row][col][1] + temp_flow[1];

            out_flow[0] = x_flow < amin ? ADT(amin) : x_flow > amax ? ADT(amax) : ADT(x_flow);
            out_flow[1] = y_flow < amin ? ADT(amin) : y_flow > amax ? ADT(amax) : ADT(y_flow);
            update[(row-1)*48+col-1] = (temp_flow[0](7,5) == out_flow[0](7,5) && temp_flow[1](7,5) == out_flow[1](7,5))? false : true;

            flow[(row-1)*48+col-1] = out_flow;
        }
    }
}