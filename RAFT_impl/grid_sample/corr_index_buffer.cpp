#include "include/corr.hpp"

void grid_sample_UPDATE_LOAD(ADT32 *fm, FDT flow[2304], bool update[2304], hls::stream<bool> &update_fifo, hls::stream<ADT8> &cb_fifo, short layerNum)
{
    short dim_limit;
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
            bool update_flag = update[row*48+col];
            update_fifo.write(update_flag);
            if(update_flag)
            {
                short cur_ch = (row*48 + col) % 32;
                short cur_part = (row*48 + col) >> 5;

                CDT coords[2];
                coords[0] = ((col << 5) + flow[row*48+col][0]) >> layerNum;
                coords[1] = ((row << 5) + flow[row*48+col][1]) >> layerNum;

                // integer part
                ap_int<7> centoid_x = coords[0](11,5);
                ap_int<7> centoid_y = coords[1](11,5);

                for (int n_row = 0; n_row < 8; n_row++)
                {
                    #pragma HLS pipeline II=8
                    ADT8 buffer_data;
                    for (int n_col = 0; n_col < 8; n_col++)
                    {
                        #pragma HLS pipeline II=1
                        ADT32 ddr_data = fm[offset + cur_part*dim_limit*dim_limit + (centoid_y-3+n_row)*dim_limit + centoid_x-3+n_col];
                        ADT zero_num = 0;

                        ADT corr_data = ((centoid_x + n_col - 3) >= 0 && (centoid_x + n_col - 3) < dim_limit &&
                                    (centoid_y + n_row - 3) >= 0 && (centoid_y + n_row - 3) < dim_limit)? ddr_data[cur_ch] : zero_num;
                        
                        buffer_data[n_col] = corr_data;
                    }   
                    cb_fifo.write(buffer_data);     
                }
            }
        }
    }
}

void grid_sample_UPDATE_EXPORT(ADT8* corr_buffer, hls::stream<bool> &update_fifo, hls::stream<ADT8> &cb_fifo)
{
    for (CDT row = 0; row < 48; row++)
    {
        for (CDT col = 0; col < 48; col++)
        {
            bool update_flag;
            update_flag = update_fifo.read();
            if(update_flag)
            {
                for (int n_row = 0; n_row < 8; n_row++)
                {
                    #pragma HLS pipeline II=8
                    ADT8 buffer_data;
                    buffer_data = cb_fifo.read();
                    corr_buffer[(row*48+col)*8 + n_row] = buffer_data;
                }
            }
        }
    }

}


void grid_sample_UPDATE(ADT32 *fm, FDT flow[2304], bool update[2304], ADT8* corr_buffer, short layerNum)
{
    #pragma hls dataflow
    hls::stream<bool, 20> update_fifo;
    hls::stream<ADT8, 20> cb_fifo;
    grid_sample_UPDATE_LOAD(fm, flow, update, update_fifo, cb_fifo, layerNum);
    grid_sample_UPDATE_EXPORT(corr_buffer, update_fifo, cb_fifo);
}



void grid_sample_LOAD(hls::stream<ADT8> &corr_flow, FDT flow[2304], ADT8 corr_buffer[2304*8], hls::stream<hls::vector<ap_uint<11>, 4>> &factor_flow, short layerNum)
{
    for (CDT row = 0; row < 48; row++)
    {
        for (CDT col = 0; col < 48; col++)
        {
            #pragma hls pipeline II=8
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
            for (short n_row = 0; n_row < 8; n_row++)
            {
                #pragma hls pipeline II=1
                ADT8 corr_data = corr_buffer[(row*48+col)*8+n_row];
                factor_flow.write(factor);
                corr_flow.write(corr_data);
            }
        }
    }

}


void grid_sample_CAL(hls::stream<ADT8> &corr_flow, hls::stream<hls::vector<ap_uint<11>, 4>> &factor_flow, hls::stream<ADT8> &corr_out_flow, short layerNum)
{
    for (CDT row = 0; row < 48; row++)
    {
        for (CDT col = 0; col < 48; col++)
        {
            for (short n_row = 0; n_row < 8; n_row++)
            {
                #pragma hls pipeline II=1

                hls::vector<ap_uint<11>, 4> factor; // scale 2^-10
                ADT8 line_buffer1, line_buffer2;
                ADT8 corr_out;

                factor = factor_flow.read();
                line_buffer1 = line_buffer2;
                line_buffer2 = corr_flow.read();
                for (short n_col = 0; n_col < 7; n_col++)
                {
                    #pragma hls unroll
                    corr_out[n_col] = (factor[0] * line_buffer1[n_col] 
                                        + factor[1] * line_buffer1[n_col+1] 
                                        + factor[2] * line_buffer2[n_col] 
                                        + factor[3] * line_buffer2[n_col+1] ) >> 10;
                }
                if(n_row != 0)
                {
                   corr_out_flow.write(corr_out);
                }
            }
        }
    }
}

void grid_sample_OUT(ADT8 *corr, hls::stream<ADT8> &corr_out_flow, short layerNum)
{
    int offset1, offset2, offset3;
    switch(layerNum)
    {
        case 0: offset1 = 0; offset2 = 4*4*2304-4; offset3 = 4*6*2304-6; break;
        case 1: offset1 = 4*1*2304; offset2 = 4*4*2304+16-4; offset3 = 4*6*2304+1-6; break;// offset = 48*48*2304/32
        case 2: offset1 = 4*2*2304; offset2 = 4*5*2304-4;    offset3 = 4*6*2304+2-6; break; // offset = 24*24*2304/32 +  12*12*2304/32
        case 3: offset1 = 4*3*2304; offset2 = 4*5*2304+16-4; offset3 = 4*6*2304+3-6; break; // offset = 12*12*2304/32
        default: offset1 = 0; offset2 = 0; offset3 = 0; break;
    }
    for (CDT row = 0; row < 48; row++)
    {
        for (CDT col = 0; col < 48; col++)
        {
            for (short n_row = 0; n_row < 7; n_row++)
            {
                #pragma hls pipeline II=1
                ADT8 corr_out = corr_out_flow.read();
                int offset = n_row < 4 ? offset1 : n_row == 6 ? offset3 : offset2;
                corr[offset+(row*48+col)*4+n_row] = corr_out;

            }
        }
    }
}

void grid_sample_ING(ADT8 *corr, FDT flow[2304], ADT8* corr_buffer, short layerNum)
{
    #pragma hls dataflow
    hls::stream<ADT8, 32> corr_flow;
    hls::stream<ADT8, 32> corr_out_flow;
    hls::stream<hls::vector<ap_uint<11>, 4>, 32> factor_flow;
    grid_sample_LOAD(corr_flow, flow, corr_buffer, factor_flow, layerNum);
    grid_sample_CAL(corr_flow, factor_flow, corr_out_flow, layerNum);
    grid_sample_OUT(corr, corr_out_flow, layerNum);

}