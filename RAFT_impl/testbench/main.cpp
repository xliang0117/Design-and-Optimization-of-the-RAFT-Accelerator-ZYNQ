#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "include/corr.hpp"
#include "include/convolution.hpp"
#include "include/fsm.hpp"
#include "include/log.hpp"
#define LOCAL_LOG_LEVEL 0

void load_image(ADT4 *IMAGE, const char* filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file){ std::cout << "error" << std::endl; return ;}
    for (int p = 0; p < 384*384; p++){
        int8_t temp_pixel[3];
        file.read((char*)&temp_pixel[0], 3);
        for (int c = 0; c < 3; c++){
            IMAGE[385+p][c] = temp_pixel[c];    //385 for start index
        }
    }
    file.close();
}

void load_weight(ADT32 *weight, const char* filename)
{
    int w_idx = 0;
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file){ std::cout << "error" << std::endl; return ;}
    while (!file.eof()){
        int8_t temp_weight[32];
        file.read((char*)&temp_weight[0], 32);
        for (int c = 0; c < 32; c++){
            weight[w_idx][c] = temp_weight[c];
        }
        w_idx++;
    }
    file.close();
}

void load_fm(ADT32 *fm, const char* filename)
{
    int idx = 0;
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file){ std::cout << "error" << std::endl; return ;}
    while (!file.eof()){
        int8_t temp_pixel[32];
        file.read((char*)&temp_pixel[0], 32);
        for (int c = 0; c < 32; c++){
            fm[idx][c] = temp_pixel[c];
        }
        idx++;
    }
    file.close();
}

void load_flow(FDT* flow, const char* filename)
{
    int idx = 0;
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file){ std::cout << "error" << std::endl; return ;}
    while (!file.eof()){
        int8_t temp_pixel[32];
        file.read((char*)&temp_pixel[0], 32);
        for (int c = 0; c < 2; c++){
            flow[idx][c] = temp_pixel[c];
        }
        idx++;
    }
    file.close();
}

void save_flow(FDT* flow, const char* filename)
{
    int idx = 0;
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file){ std::cout << "error" << std::endl; return ;}
    for(int idx=0; idx < 2304; idx++){
        int8_t temp_flow[2];
        for (int c = 0; c < 2; c++){
            temp_flow[c] = flow[idx][c];
        }
        file.write((char*)&temp_flow[0], 2);
    }
    file.close();
}

void validate_flow(FDT *flow, const char* filename, int dims)
{
    FDT* flow_compare = new FDT[2304];
    load_flow(flow_compare, filename);
    for (int row = 0; row < dims; row++){
        for (int col = 0; col < dims; col++){
            FDT out = flow[row*dims + col];
            FDT real = flow_compare[row*dims + col];
            for (int c = 0; c < 2; c++){
                int8_t out_v = out[c];
                int8_t real_v = real[c];
                if(out_v != real_v){
                    // LOG_ERROR("offset:%d", row*dims + col);
                    LOG_ERROR("row: %d, col: %d, c:%d, o_v:%d, f_v:%d", row, col, c, out_v, real_v);
                    // return;
                }
            }
        }
    }
    
}

void validate_corr(ADT32 *fm, int addr, const char* filename, int dims, int channels)
{
    ADT32 *out_fm = &fm[addr];
    ADT32 *file_fm = new ADT32[3*384*384];
    int layerNum = 0;
    int c_offset = 0;
    load_fm(file_fm, filename);
    int c_iter = (channels + 32 - 1) / 32;
    for (int c_p = 0; c_p < c_iter; c_p++){
        for (int row = 0; row < dims; row++){
            for (int col = 0; col < dims; col++){
                ADT32 out = out_fm[c_p*dims*dims + row*dims + col];
                ADT32 real = file_fm[c_p*dims*dims + row*dims + col];
                for (int c = 0; c < 32; c++){
                    int8_t out_v = out[c];
                    int8_t real_v = real[c];
                    if(c_p == 0 || (c_p == 5 && c < 16) || (c_p == 7 && c == 0)){ layerNum = 0;}
                    else if(c_p == 1 || (c_p == 5 && c >= 16) || (c_p == 7 && c == 1)){ layerNum = 1;}
                    else if(c_p == 2 || (c_p == 6 && c < 16) || (c_p == 7 && c == 2)){ layerNum = 2;}
                    else { layerNum = 3;}
                    
                    if(c_p < 4) {c_offset = 0;}
                    else if(c_p < 6 && c < 16) {c_offset = 32;}
                    else if(c_p < 6 && c > 16) {c_offset = 16;}
                    else { c_offset = 48-c;}

                    if(c+c_p*32 < channels && (out_v != real_v)){
                        LOG_ERROR("offset:%d", c_p*dims*dims + row*dims + col);
                        LOG_ERROR("layer:%d, ch:%d, nb_row:%d, nb_col:%d, o_v:%d, f_v:%d", layerNum, row*48+col, (c+c_offset)/7, (c+c_offset)%7, out_v, real_v);
                        return;
                    }
                }
            }
        }
    }
    
}

void validate_out(ADT32 *fm, int addr, const char* filename, int dims, int channels, int file_offset)
{
    ADT32 *out_fm = &fm[addr];
    ADT32 *file_fm = new ADT32[3*384*384];
    load_fm(file_fm, filename);
    int c_iter = (channels + 32 - 1) / 32;
    for (int c_p = 0; c_p < c_iter; c_p++){
        for (int row = 0; row < dims; row++){
            for (int col = 0; col < dims; col++){
                ADT32 out = out_fm[c_p*dims*dims + row*dims + col];
                ADT32 real = file_fm[c_p*dims*dims + row*dims + col + file_offset];
                for (int c = 0; c < 32; c++){
                    int8_t out_v = out[c];
                    int8_t real_v = real[c];
                    if(c+c_p*32 < channels && (out_v != real_v)){
                        LOG_ERROR("offset:%d", c_p*dims*dims + row*dims + col);
                        LOG_ERROR("c_p:%d, row: %d, col: %d, c:%d, o_v:%d, f_v:%d", c_p, row, col, c, out_v, real_v);
                        return;
                    }
                }
            }
        }
    }
    
}

void tb_conv()
{
    ADT4* IMAGE1 = new ADT4[384*384+385];
    ADT4* IMAGE2 = new ADT4[384*384+385];
    ADT32* fm = new ADT32[358945 + 1000];
    ADT32* weight = new ADT32[6400];
    FDT* flow = new FDT[2304];

    int fmap1_addr = 83329;     //147841;
    int fmap2_addr = 92545;     //157057;
    int Corr0_addr = 101761;    //166273;    
    int Corr1_addr = 267649;    //332161;
    int Corr2_addr = 309121;    //373633;
    int Corr3_addr = 319489;    //384001;

    load_image(IMAGE1, "../../binary/image1.bin");
    load_image(IMAGE2, "../../binary/image2.bin");
    load_weight(weight, "../../binary/param.bin");
    load_fm(fm+fmap2_addr, "../../binary/fmap2.bin");

    // load_fm(fm+385, "../../binary/new-corr.bin");
    // load_fm(fm+30337, "../../binary/u-net.bin");
    // load_fm(fm+37249, "../../binary/u-inp.bin");
    // load_flow(flow, "../../binary/u-flow.bin");

    // load_fm(fm+101761, "../../binary/corr0.bin");
    // load_fm(fm+267649, "../../binary/corr1.bin");
    // load_fm(fm+309121, "../../binary/corr2.bin");
    // load_fm(fm+319489, "../../binary/corr3.bin");
    
    LOG_INFO_S("start testing!");
    
    conv_hw(IMAGE1, IMAGE2, fm, weight, (ADT*) (fm+385), flow);

    validate_out(fm, fmap1_addr, "../../binary/fmap1.bin", 48, 64, 0);
    // validate_out(fm, 30337, "../../binary/u-net.bin", 48, 96, 0);
    // validate_out(fm, 37249, "../../binary/u-inp.bin", 48, 64, 0);
    validate_out(fm, Corr0_addr, "../../binary/corr0.bin", 48, 2304, 0);
    validate_out(fm, Corr1_addr, "../../binary/corr1.bin", 24, 2304, 0);
    validate_out(fm, Corr2_addr, "../../binary/corr2.bin", 12, 2304, 0);
    validate_out(fm, Corr3_addr, "../../binary/corr3.bin", 6, 2304, 0);
    // validate_out(fm, 16513, "../../binary/me-cc1.bin", 48, 96, 0);
    // validate_out(fm, 25729, "../../binary/me-cf1.bin", 48, 64, 0);
    validate_corr(fm, 385, "../../binary/new-corr.bin", 48, 196);
    // validate_out(fm, 55681, "../../binary/gru-sig1.bin", 48, 96, 0);
    // validate_out(fm, 48769, "../../binary/gru-constquant.bin", 48, 96, 0);
    // validate_out(fm, 30337, "../../binary/gru-outnet.bin", 48, 96, 0);
    // validate_out(fm, 62593, "../../binary/fh-p1.bin", 48, 96, 0);
    validate_flow(flow, "../../binary/final-flow.bin", 48);
    save_flow(flow, "../../binary/hardware-flow.bin");


    LOG_INFO_S("success!");

    delete[] fm;
    delete[] weight;

}


int main()
{
    tb_conv();


    return 0;
    
}