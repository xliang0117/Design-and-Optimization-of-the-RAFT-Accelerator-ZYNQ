#include <stdlib.h>
#include <stdio.h>
#include "include/corr.hpp"
#include "include/convolution.hpp"
#include "include/fsm.hpp"
#include "include/log.hpp"
#define LOCAL_LOG_LEVEL 0

void validate_res(HALF8 *res, int layer_index, int c_limit, int height, int width, half corner, half edge, half normal)
{
    LOG_DEBUG("debug: %s", config_ext[0][layer_index].name);
    for (int c_id = 0; c_id < c_limit; c_id++)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                HALF8 temp_num = res[config_ext[0][layer_index].ex_addr + c_id*height*width + row*width + col];
                for (int c = 0; c < 8; c++)
                {
                    if ((row == 0) && (col == 0))
                    {
                        if (rawBitsToHalf(temp_num(c*16+15, c*16)) != corner)
                        {
                            LOG_ERROR("layer-1: Error: c_id:%d, row: %d, col: %d, c:%d, value:%f", c_id, row, col, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                            goto error_label;
                        }
                    }
                    else if((row == 0) || (col == 0))
                    {
                        if (rawBitsToHalf(temp_num(c*16+15, c*16)) != edge)
                        {
                            LOG_ERROR("layer-2: Error: c_id:%d, row: %d, col: %d, c:%d, value:%f", c_id, row, col, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                            goto error_label;
                        }
                    }
                    else
                    {
                        if (rawBitsToHalf(temp_num(c*16+15, c*16)) != normal)
                        {
                            LOG_ERROR("layer-3: Error: c_id:%d, row: %d, col: %d, c:%d, value:%f", c_id, row, col, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                            goto error_label;
                        }
                    }

                }
                
            }
            
        }
    }

error_label:
    return;
}

void validate_corr(HALF8 *res, int layer_index, half corner, half edge, half normal, int height, int width)
{
    LOG_DEBUG("debug: %s", config_corr[layer_index].name);

    for (int row_id = 0; row_id < 3; row_id++)
    {
        for (int col_id = 0; col_id < 4; col_id++)
        {
            for (int row = 0; row < 16; row++)
            {
                for (int col = 0; col < 16; col++)
                {
                    for (int pixel = 0; pixel < height*width/8; pixel++)
                    {
                        HALF8 temp_num = res[config_corr[layer_index].ex_addr + ((row_id*4+col_id)*256 + row*16 + col)*height*width/8 + pixel];
                        for (int c = 0; c < 8; c++)
                        {
                            if ((row == 0) && (col == 0) && (row_id == 0) && (col_id == 0))
                            {
                                if (rawBitsToHalf(temp_num(c*16+15, c*16)) != corner)
                                {
                                    LOG_ERROR("layer-1: Error: row_id: %d, col_id: %d, row: %d, col: %d, pixel: %d, c:%d, value:%f", row_id, col_id, row, col, pixel*8, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                                    goto error_label;
                                }
                            }
                            else if((row == 0 && row_id == 0) || (col == 0 && col_id == 0))
                            {
                                if (rawBitsToHalf(temp_num(c*16+15, c*16)) != edge)
                                {
                                    LOG_ERROR("layer-2: Error: row_id: %d, col_id: %d, row: %d, col: %d, pixel: %d, c:%d, value:%f", row_id, col_id, row, col, pixel*8, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                                    goto error_label;
                                }
                            }
                            else
                            {
                                if (rawBitsToHalf(temp_num(c*16+15, c*16)) != normal)
                                {
                                    LOG_ERROR("layer-3: Error: row_id: %d, col_id: %d, row: %d, col: %d, pixel: %d, c:%d, value:%f", row_id, col_id, row, col, pixel*8, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                                    goto error_label;
                                }
                            }

                        }
                    }
                    
                }
                
            }
        }
        
    }
    

error_label:
    return;
}

void validate_iter(HALF8 *res, int layer_index, int c_limit, int height, int width, half corner, half edge, half normal)
{
    LOG_DEBUG("debug: %s", config_iter[layer_index].name);
    for (int c_id = 0; c_id < c_limit; c_id++)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                HALF8 temp_num = res[config_iter[layer_index].ex_addr + c_id*height*width + row*width + col];
                for (int c = 0; c < 8; c++)
                {
                    if (((row == 0 || row == height-1)  && (col == 0 || col == width-1)))
                    {
                        if (rawBitsToHalf(temp_num(c*16+15, c*16)) != corner)
                        {
                            LOG_ERROR("layer-1: Error: c_id:%d, row: %d, col: %d, c:%d, value:%f", c_id, row, col, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                            goto error_label;
                        }
                    }
                    else if((row == 0) || (col == 0))
                    {
                        if (rawBitsToHalf(temp_num(c*16+15, c*16)) != edge)
                        {
                            LOG_ERROR("layer-2: Error: c_id:%d, row: %d, col: %d, c:%d, value:%f", c_id, row, col, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                            goto error_label;
                        }
                    }
                    else
                    {
                        if (rawBitsToHalf(temp_num(c*16+15, c*16)) != normal)
                        {
                            LOG_ERROR("layer-3: Error: c_id:%d, row: %d, col: %d, c:%d, value:%f", c_id, row, col, c, (float)rawBitsToHalf(temp_num(c*16+15, c*16)) );
                            goto error_label;
                        }
                    }

                }
                
            }
            
        }
    }

error_label:
    return;
}


void tb_conv()
{
    HALF8 *fm = new HALF8[3000000];
    HALF16 *weight = new HALF16[20000]; // (200*96 + 8*9*9 + 8*64) / 16
    
    memset(weight, 0, 20000*16*2);

    LOG_INFO_S("start testing!");
    
    conv_hw(fm, weight, (half*)fm);

    LOG_INFO_S("success!");

    delete[] fm;
    delete[] weight;

}


int main()
{
    tb_conv();


    return 0;
    
}
