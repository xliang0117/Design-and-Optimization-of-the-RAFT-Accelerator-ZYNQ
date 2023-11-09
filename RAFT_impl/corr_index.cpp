#include "include/corr.hpp"

//flow fixedpoint 5 frac
void load_corr(ADT32 *fm, FDT flow[2304], hls::stream<ADT> &corr_flow, hls::stream<hls::vector<ap_uint<11>, 4>> &factor_flow, short layerNum)
{
    short dim_limit = 48 >> layerNum;
    int offset;
    switch(layerNum)
    {
        case 0: dim_limit = 48; offset = 0; break;
        case 1: dim_limit = 24; offset = 165888; break;// offset = 48*48*2304/32
        case 2: dim_limit = 12; offset = 41472 + 165888; break; // offset = 24*24*2304/32 +  12*12*2304/32
        case 3: dim_limit = 6; offset = 10368 + 41472 + 165888; break; // offset = 12*12*2304/32
        default: dim_limit = 0; offset = 0; break;
    }

    for (CDT row = 0; row < 48; row++)
    {
        for (CDT col = 0; col < 48; col++)
        {
            short cur_ch = (row*48 + col) % 32;
            short cur_part = (row*48 + col) >> 5;

            CDT coords[2];
            coords[0] = ((col << 5) + flow[row*48+col][0]) >> layerNum;
            coords[1] = ((row << 5) + flow[row*48+col][1]) >> layerNum;

            // integer part
            ap_int<7> centoid_x = coords[0](11,5);
            ap_int<7> centoid_y = coords[1](11,5);

            //frac part
            ap_uint<6> factor_x_left = (1 << 5) - coords[0](4,0);
            ap_uint<6> factor_y_up = (1 << 5) - coords[1](4,0);
            ap_uint<6> factor_x_right = coords[0](4,0);
            ap_uint<6> factor_y_down = coords[1](4,0);

            hls::vector<ap_uint<11>, 4> factor; // scale 2^-10
            factor[0] = factor_x_left * factor_y_up;
            factor[1] = factor_x_right * factor_y_up;
            factor[2] = factor_x_left * factor_y_down;
            factor[3] = factor_x_right * factor_y_down;

            for (int n_row = 0; n_row < 8; n_row++)
            {
                #pragma HLS pipeline II=8
                for (int n_col = 0; n_col < 8; n_col++)
                {
                    #pragma HLS pipeline II=1
                    ADT32 ddr_data = fm[offset + cur_part*dim_limit*dim_limit + (centoid_y-3+n_row)*dim_limit + centoid_x-3+n_col];
                    ADT zero_num = 0;

                    ADT corr_data = ((centoid_x + n_col - 3) >= 0 && (centoid_x + n_col - 3) < dim_limit &&
                                (centoid_y + n_row - 3) >= 0 && (centoid_y + n_row - 3) < dim_limit)? ddr_data[cur_ch] : zero_num;
                    corr_flow.write(corr_data);
                    factor_flow.write(factor);

                }            
            }
        }
    }
}

void cal_corr(hls::stream<ADT> &corr_flow, hls::stream<hls::vector<ap_uint<11>, 4>> &factor_flow, hls::stream<ADT> &corr_out_flow)
{
    ADT line_buffer1[8];
    ADT line_buffer2[8];

    for (int img = 0; img < 2304; img++)
    {
        for (int row = 0; row < 8; row++) 
        {
            for (int col = 0; col < 8; col++)
            {
                #pragma HLS pipeline II=1
                ADT corr_data = corr_flow.read();
                hls::vector<ap_uint<11>, 4> factor = factor_flow.read();

                line_buffer2[col] = corr_data;                
                if (row == 0)
                {
                    line_buffer1[col] = corr_data;
                }
                else if (col == 0)
                {
                    line_buffer1[7] = line_buffer2[7];
                }
                else
                {
                    ADT corr_out_data = (factor[0] * line_buffer1[col-1] 
                                        + factor[1] * line_buffer1[col] 
                                        + factor[2] * line_buffer2[col-1] 
                                        + factor[3] * corr_data) >> 10;
                    corr_out_flow.write(corr_out_data);
                    line_buffer1[col-1] = line_buffer2[col-1];
                }
            }
        }
    }
}

void save_corr(hls::stream<ADT> &corr_out_flow, ADT *corr_out, short layerNum)
{
    int offset1, offset2, offset3;
    switch(layerNum)
    {
        case 0: offset1 = 0; offset2 = 32*4*2304-32; offset3 = 32*6*2304-48; break;
        case 1: offset1 = 32*1*2304; offset2 = 32*4*2304+16-32; offset3 = 32*6*2304+1-48; break;// offset = 48*48*2304/32
        case 2: offset1 = 32*2*2304; offset2 = 32*5*2304-32;    offset3 = 32*6*2304+2-48; break; // offset = 24*24*2304/32 +  12*12*2304/32
        case 3: offset1 = 32*3*2304; offset2 = 32*5*2304+16-32; offset3 = 32*6*2304+3-48; break; // offset = 12*12*2304/32
        default: offset1 = 0; offset2 = 0; offset3 = 0; break;
    }
    for (int i = 0; i < 2304; i++)
    {
        for (int c = 0; c < 49; c++)
        {
            #pragma HLS pipeline II=1
            int offset;
            offset = c < 32 ? offset1 : c == 48 ? offset3 : offset2;
            ADT corr_out_data = corr_out_flow.read();
            corr_out[offset + i*32 + c] = corr_out_data;
        }
    }
}

void grid_sample(ADT *corr, ADT32 *fm, FDT flow[2304], int layerNum)
{
    hls::stream<ADT, 32> corr_flow;
    hls::stream<ADT, 32> corr_out_flow;
    hls::stream<hls::vector<ap_uint<11>, 4>, 32> factor_flow;

    #pragma HLS dataflow
    load_corr(fm, flow, corr_flow, factor_flow, layerNum);
    cal_corr(corr_flow, factor_flow, corr_out_flow);
    save_corr(corr_out_flow, corr, layerNum);
}
