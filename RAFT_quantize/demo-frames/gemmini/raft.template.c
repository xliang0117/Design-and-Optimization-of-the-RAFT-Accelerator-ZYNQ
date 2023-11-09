#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "raft_params.h"

void grid_sample(elem_t *flow, elem_t *corr_whole, elem_t *sample_res, acc_scale_t flow_scale, acc_scale_t sample_res_scale, int in_dim, int neighbourhood, int corr_layer)
{

    for (int y = 0; y < in_dim; y++)
    {
        for (int x = 0; x < in_dim; x++)
        {
            float x_f = (x + flow[2*(y*in_dim + x)] * flow_scale) / (1 << corr_layer);
            float y_f = (y + flow[2*(y*in_dim + x) + 1] * flow_scale) / (1 << corr_layer);
            float bi_x_factor = abs(x_f - x);
            float bi_y_factor = abs(y_f - y);
            float bi_factor[4];
            bi_factor[0] = (1-bi_x_factor) * (1-bi_y_factor);
            bi_factor[1] = bi_x_factor * (1-bi_y_factor);
            bi_factor[2] = (1-bi_x_factor) * bi_y_factor; 
            bi_factor[2] = bi_x_factor * bi_y_factor;

            int current_dim = in_dim / (1 << corr_layer);
            int current_channle = y*in_dim + x;

            int center_x_left_index = (int)x_f;
            int center_y_down_index = (int)y_f;

            int n_area = (neighbourhood*2+1) * (neighbourhood*2+1);

            elem_t corr_data[8][8];
            for (int n_y = 0; n_y < neighbourhood*2+1+1; n_y++) // first add 1 for neighbourhood, another add 1 for right and up pixel
            {
                for (int n_x = 0; n_x < neighbourhood*2+1+1; n_x++)
                {
                    int x_index = center_x_left_index  - neighbourhood + n_x;
                    int y_index = center_y_down_index  - neighbourhood + n_y;
                    if (x_index > 0 && y_index > 0 && x_index < current_dim && y_index < current_dim)
                    {
                        corr_data[n_y][n_x] = corr_whole[current_channle*current_dim*current_dim + y_index*current_dim + x_index];
                    }
                    else
                    {
                        corr_data[n_y][n_x] = 0;
                    }

                    if (n_x > 0 && n_y > 0)
                    {
                        float sample_out = bi_factor[0] * corr_data[n_y-1][n_x-1] + bi_factor[1] * corr_data[n_y-1][n_x]
                                            + bi_factor[1] * corr_data[n_y][n_x-1] + bi_factor[2] * corr_data[n_y][n_x];

                        sample_res[current_channle*n_area + (n_y-1)*(neighbourhood*2+1) + (n_x-1) ] = sample_out / sample_res_scale + 0.5; //quantize
                    }
                    
                }
            }            

        }        
    }    
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool conv = true;
    
    if (argc < 3) {
        conv = true;
    } else if (strcmp(argv[2], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[2], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check = false;

    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;
    uint64_t fe_cycles = 0, corrgen_cycles = 0, sample_cycles = 0, update_cycles = 0;

    //generate code here

    uint64_t total_cycles = fe_cycles + corrgen_cycles + sample_cycles + update_cycles;

    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);

    printf("fe cycles: %llu (%d%%)\n", fe_cycles, (fe_cycles * 100) / total_cycles);
    printf("corrgen cycles: %llu (%d%%)\n", corrgen_cycles, (corrgen_cycles * 100) / total_cycles);
    printf("sample cycles: %llu (%d%%)\n", sample_cycles, (sample_cycles * 100) / total_cycles);
    printf("update cycles: %llu (%d%%)\n", update_cycles, (update_cycles * 100) / total_cycles);

    printf("PASS\n");
    exit(0);
}