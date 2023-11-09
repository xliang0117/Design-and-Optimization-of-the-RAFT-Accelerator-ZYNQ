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

void export_flow(hls::vector<ADT, 32> ofm[50][50], FDT flow[2304])
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

            flow[(row-1)*48+col-1] = out_flow;
        }
    }
}

void export_interpolate_RGBA_flow(ADT32 *ofm_ddr, FDT flow[2304])
{
    FDT lineBuffer[48];
    FDT nextLineFirst, nextLineSecond;

    // printf("firstLine:x %d, y %d, x %d, y %d\n", flow[0][0], flow[0][1], flow[1][0], flow[1][1]);
    // printf("secondLine:x %d, y %d, x %d, y %d\n", flow[48][0], flow[48][1], flow[49][0], flow[49][1]);
    // printf("thirdLine:x %d, y %d, x %d, y %d\n", flow[96][0], flow[96][1], flow[97][0], flow[97][1]);
    
    FLOW_ROW:for (short row = 0; row < 48; row++)
    {
        FLOW_COL:for (short col = 0; col < 48; col++)
        {
            #pragma HLS pipeline II=8
            if(col < 2) 
            {
                lineBuffer[col + 46] = nextLineFirst;
            } 
            else 
            {
                lineBuffer[col - 2] = nextLineFirst;
            }
            nextLineFirst = nextLineSecond;
            nextLineSecond = flow[row*48+col];
            flow[row*48+col][0] = 0;
            flow[row*48+col][1] = 0;
            for(short line = 0; line < 8; line++) 
            {
                #pragma HLS pipeline II=1
                ADT32 pixelLineBuffer;
                ADT4 pixelBuffer[8]; //RGBA
                for (short pixel = 0; pixel < 8; pixel++)
                {
                    #pragma HLS unroll
                    ap_int<12> inter_flow[2];
                    ap_int<12> inter_flow_2_pixel[2];
                    FDT LU, LD, RU, RD;
                    LU = col > 0 ? lineBuffer[col-1] : lineBuffer[0];
                    LD = nextLineFirst;
                    RU = lineBuffer[col];
                    RD = nextLineSecond;
                    inter_flow[0] = ((8-pixel)*(8-line)*LU[0] +
                                    (8-pixel)*(line)*LD[0]+
                                    (pixel)*(8-line)*RU[0]+
                                    (pixel)*(line)*RD[0]) >> 3;

                    inter_flow[1] = ((8-pixel)*(8-line)*LU[1] +
                                    (8-pixel)*(line)*LD[1]+
                                    (pixel)*(8-line)*RU[1]+
                                    (pixel)*(line)*RD[1]) >> 3;
                    // if(row == 1 && col == 1 && line == 7 && pixel == 7){
                    //     printf("row1:x %d, y %d\n", inter_flow[0], inter_flow[1] );
                    //     printf("LU:x %d, y %d, LD:x %d, y %d, RU:x %d, y %d, RD:x %d, y %d\n", LU[0], LU[1], LD[0], LD[1], RU[0], RU[1], RD[0], RD[1]);
                    //     printf("next line second :x %d, y %d \n", nextLineSecond[0], nextLineSecond[1]);
                    // }
                    // if(row == 2 && col == 1 && line == 7 && pixel == 7){
                    //     printf("row2:x %d, y %d\n", inter_flow[0], inter_flow[1] );
                    //     printf("LU:x %d, y %d, LD:x %d, y %d, RU:x %d, y %d, RD:x %d, y %d\n", LU[0], LU[1], LD[0], LD[1], RU[0], RU[1], RD[0], RD[1]);
                    //     printf("next line second :x %d, y %d \n", nextLineSecond[0], nextLineSecond[1]);
                    // }

                    inter_flow_2_pixel[0] = inter_flow[0] + 255;
                    inter_flow_2_pixel[1] = inter_flow[1] + 255;
                    ap_int<12> average_flow = 255 - ((inter_flow[0] + inter_flow[1]) >> 1);

                    pixelLineBuffer[4*pixel](7,0) = inter_flow_2_pixel[0] > 255? 255 : inter_flow_2_pixel[0](7,0); //red
                    pixelLineBuffer[4*pixel+1](7,0) = average_flow > 255? 255 : average_flow(7,0);  //blue
                    pixelLineBuffer[4*pixel+2](7,0) = inter_flow_2_pixel[1] > 255? 255 : inter_flow_2_pixel[1](7,0);    //green
                    pixelLineBuffer[4*pixel+3](7,0) = 0xff;

                }
                if(row > 0 && col > 0 ){
                    ofm_ddr[(row-1)*240*8 + line*240 + col-1] = pixelLineBuffer;
                }

            }
        }
    }
}