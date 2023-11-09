#include "include/corr.hpp"

void load_coords(HALF8 *ifm_ddr, half coords[3072][2], bool coords_init)
{
    #pragma HLS ARRAY_PARTITION variable=coords dim=2 complete
    if (coords_init)
    {
        for (int row = 0; row < 48; row++)
        {
            for (short col = 0; col < 64; col++)
            {
                #pragma HLS pipeline II=1
                coords[row*64+col][0] = 31;
                coords[row*64+col][1] = 23;
            }
            
        }
        
    }

    for (short i = 0; i < 3072; i++)
    {
        #pragma HLS pipeline II=1
        HALF8 ddr_data = ifm_ddr[i];
        half x_half = rawBitsToHalf(ddr_data(15, 0));
        half y_half = rawBitsToHalf(ddr_data(31, 16));

        coords[i][0] += x_half;
        coords[i][1] += y_half;
    }
    
}


void load_corr(half *corr, half coords[3072][2], hls::stream<half> &corr_flow, hls::stream<fourhalf> &factor_flow, short scale)
{
    short x_limit = 64 >> scale;
    short y_limit = 48 >> scale;
    short scale_factor = 1 << scale;
    short corr_size = 3072 >> (2*scale); 

    for (int i = 0; i < 3072; i++)
    {
        half centroid_x = coords[i][0] / scale_factor;
        half centroid_y = coords[i][1] / scale_factor;

        short x_id = centroid_x;
        short y_id = centroid_y;


        fourhalf factor_pre, factor;
        factor_pre.part[0] = 1 - centroid_x + x_id;
        factor_pre.part[1] = 1 - centroid_y + y_id;
        factor_pre.part[2] = centroid_x - x_id;
        factor_pre.part[3] = centroid_y - y_id;

        factor.part[0] = factor_pre.part[0] * factor_pre.part[1];
        factor.part[1] = factor_pre.part[2] * factor_pre.part[1];
        factor.part[2] = factor_pre.part[0] * factor_pre.part[3];
        factor.part[3] = factor_pre.part[2] * factor_pre.part[3];     

        for (int row = 0; row < 8; row++)
        {
            #pragma HLS pipeline II=8
            
            half ddr_data[8];
            memcpy(ddr_data, corr + i*x_limit*y_limit + (y_id-3+row)*x_limit+x_id-3, 16);
            for (int c = 0; c < 8; c++)
            {
                half zero_num = 0;

                half corr_data = ((x_id + c - 3) >= 0 && (x_id + c - 3) < x_limit &&
                            (y_id + row - 3) >= 0 && (y_id + row - 3) < y_limit)? ddr_data[c] : zero_num;
                corr_flow.write(corr_data);
                factor_flow.write(factor);

            }            
        }
        
    }
}

void cal_corr( hls::stream<half> &corr_flow, hls::stream<fourhalf> &factor_flow, hls::stream<half> &corr_out_flow)
{
    half line_buffer1[8];
    half line_buffer2[8];

    for (int img = 0; img < 3072; img++)
    {
        for (int row = 0; row < 8; row++) 
        {
            for (int col = 0; col < 8; col++)
            {
                #pragma HLS pipeline II=1
                half corr_data = corr_flow.read();
                fourhalf factor = factor_flow.read();

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
                    half corr_out_data = factor.part[0] * line_buffer1[col-1] + factor.part[1] * line_buffer1[col] + factor.part[2] * line_buffer2[col-1] + factor.part[2] * corr_data;
                    corr_out_flow.write(corr_out_data);
                    line_buffer1[col-1] = line_buffer2[col-1];
                }
                
            }
            
        }
    }
    
}

void save_corr(hls::stream<half> &corr_out_flow, half *corr_out, short scale)
{
    for (int i = 0; i < 3072; i++)
    {
        for (int c = 0; c < 7; c++)
        {
            for (int burst = 0; burst < 7; burst++)
            {
                #pragma HLS pipeline II=1
                half corr_out_data = corr_out_flow.read();
                corr_out[i*8 + c*3072*8 + burst] = corr_out_data;
            }
        }
    }
}


void grid_sample(half *corr, half coords[3072][2], half *corr_out, int scale)
{
    hls::stream<half> corr_flow;
    hls::stream<half> corr_out_flow;
    hls::stream<fourhalf> factor_flow;

    #pragma HLS dataflow
    load_corr(corr, coords, corr_flow, factor_flow, scale);
    cal_corr(corr_flow, factor_flow, corr_out_flow);
    save_corr(corr_out_flow, corr_out, scale);
}
