#include "include/convolution.hpp"
#include "include/log.hpp"

#define LOCAL_LOG_LEVEL LOG_LEVEL_DEBUG


RDT MAC9(
    ADT A_00, ADT A_01, ADT A_02,
    ADT A_10, ADT A_11, ADT A_12,
    ADT A_20, ADT A_21, ADT A_22,
    ADT B_00, ADT B_01, ADT B_02,
    ADT B_10, ADT B_11, ADT B_12,
    ADT B_20, ADT B_21, ADT B_22
){
    #pragma HLS INLINE off
    ap_int<16> prod_00 = A_00 * B_00;
    ap_int<16> prod_01 = A_01 * B_01;
    ap_int<16> prod_02 = A_02 * B_02;
    ap_int<16> prod_10 = A_10 * B_10;
    ap_int<16> prod_11 = A_11 * B_11;
    ap_int<16> prod_12 = A_12 * B_12;
    ap_int<16> prod_20 = A_20 * B_20;
    ap_int<16> prod_21 = A_21 * B_21;
    ap_int<16> prod_22 = A_22 * B_22;
    #pragma HLS bind_op variable=prod_00 op=mul impl=fabric latency=0


    ap_int<17> sum_0 = prod_00 + prod_11;
    ap_int<17> sum_1 = prod_01 + prod_12;
    ap_int<17> sum_2 = prod_02 + prod_20;
    ap_int<17> sum_3 = prod_10 + prod_21;

    ap_int<18> res_0 = sum_0 + sum_1;
    ap_int<18> res_1 = sum_2 + sum_3;
    ap_int<17> res_2 = prod_22;

    RDT res = res_0 + res_1 + res_2;
    return res;
}

void dw_conv_3x3(const hls::vector<ADT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50], const hls::vector<WDT, 32> wBUF3x3[3][3], const BDT b_buf[32], SDT scale)
{

    #pragma HLS ARRAY_PARTITION variable=wBUF3x3 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=b_buf dim=1 complete

    ADT window_buffer[3][3][32];
    #pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=2
    #pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=3

    ADT line_buffer[3][50][32];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=3

    for (int row = 0; row < 50; row++)
    {
        for (int col = 0; col < 50; col++)
        {
            #pragma HLS PIPELINE II=1
            hls::vector<ADT, 32> temp_out;
            
            for (int c = 0; c < 32; c++)
            {
                #pragma HLS UNROLL
                ADT read_in = ifm[row][col][c];
                line_buffer[row % 3][col][c] = read_in;

                window_buffer[2][2][c] = read_in;
                window_buffer[1][2][c] = line_buffer[(row + 2) % 3][col][c];
                window_buffer[0][2][c] = line_buffer[(row + 1) % 3][col][c];

                RDT res =   window_buffer[0][0][c] * wBUF3x3[0][0][c] +
                            window_buffer[0][1][c] * wBUF3x3[0][1][c] +
                            window_buffer[0][2][c] * wBUF3x3[0][2][c] +
                            window_buffer[1][0][c] * wBUF3x3[1][0][c] +
                            window_buffer[1][1][c] * wBUF3x3[1][1][c] +
                            window_buffer[1][2][c] * wBUF3x3[1][2][c] +
                            window_buffer[2][0][c] * wBUF3x3[2][0][c] +
                            window_buffer[2][1][c] * wBUF3x3[2][1][c] +
                            window_buffer[2][2][c] * wBUF3x3[2][2][c];


                ap_int<20> res_b = res + b_buf[c];

                res_b = res_b >> scale;

                temp_out[c] = res_b < amin ? ADT(amin) : res_b > amax ? ADT(amax) : ADT(res_b);

                for (int r = 0; r < 3; r++)
                {
                    #pragma HLS UNROLL
                    window_buffer[r][0][c] = window_buffer[r][1][c];
                    window_buffer[r][1][c] = window_buffer[r][2][c];
                }
            }
            if (row >= 2 && col >= 2)
            {
                ofm[row - 1][col - 1] = temp_out;
            }
        }
    }
}


RDT MAC32(hls::vector<ADT, 32> ifm, hls::vector<WDT, 32> wBUF1x1)
{
#pragma HLS INLINE off
    RDT res = 0;

    for (short c = 0; c < 32; c++){
        #pragma hls pipeline II=1
        res += ifm[c] * wBUF1x1[c];
    }

    return res;
}

void pw_conv_1x1(hls::vector<ADT, 32> ifm[50][50], hls::vector<RDT, 32> ofm[50][50], hls::vector<WDT, 32> wBUF1x1[32])
{
    #pragma HLS ARRAY_PARTITION variable=wBUF1x1 dim=0 complete
    
    for(int row=1; row<49; row++)
    {
        for(int col=1; col<49; col++)
        {
            #pragma HLS PIPELINE II=1
            for(int co=0; co<32; co++)
            {
                #pragma HLS UNROLL
                ofm[row][col][co] += MAC32(wBUF1x1[co], ifm[row][col]);
            }
        }
    }
}

void activation(hls::vector<RDT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50], BDT bbuf[32], SDT scale, bool relu)
{
    #pragma HLS INLINE off
    // #pragma HLS ARRAY_PARTITION variable=bbuf dim=1 complete

    for (int row = 1; row < 49; row++)
    {
        for (int col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            hls::vector<ADT, 32> temp_ofm;
            for (int c = 0; c < 32; c++)
            {
                #pragma HLS UNROLL
                ap_int<20> qy = ifm[row][col][c] + bbuf[c];
                #pragma HLS bind_op variable=qy op=add impl=dsp latency=0

                if (qy < 0 && relu)
                {
                    qy = 0;
                }

                qy = qy >> scale;

                temp_ofm[c] = qy < amin ? ADT(amin) : qy > amax ? ADT(amax) : ADT(qy);

                ifm[row][col][c] = 0;                
            }
            ofm[row][col] = temp_ofm;
        }
    }
}


void res_add_sub(hls::vector<ADT, 32> ifm1[50][50], hls::vector<ADT, 32> ifm2[50][50], hls::vector<ADT, 32> ofm[50][50], bool add_flag, bool relu)
{
    #pragma HLS INLINE off

    for (int row = 1; row < 49; row++)
    {
        for (int col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            hls::vector<ADT, 32> temp_ifm1 = ifm1[row][col];
            hls::vector<ADT, 32> temp_ifm2 = ifm2[row][col];
            hls::vector<ADT, 32> temp_ofm;

            for (int c = 0; c < 32; c++)
            {
                #pragma HLS UNROLL
                ap_int<9> qy = add_flag? (temp_ifm1[c] + temp_ifm2[c]) : (temp_ifm1[c] - temp_ifm2[c]);
                if (qy < 0 && relu)
                {
                    qy = 0;
                }
                temp_ofm[c] = qy < amin ? ADT(amin) : qy > amax ? ADT(amax) : ADT(qy);
            }
            ofm[row][col] = temp_ofm;
        }
    }
}


void res_mul(hls::vector<ADT, 32> ifm1[50][50], hls::vector<ADT, 32> ifm2[50][50], hls::vector<ADT, 32> ofm[50][50], short scale)
{
    #pragma HLS INLINE off

    for (int row = 1; row < 49; row++)
    {
        for (int col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            hls::vector<ADT, 32> temp_ifm1 = ifm1[row][col];
            hls::vector<ADT, 32> temp_ifm2 = ifm2[row][col];
            hls::vector<ADT, 32> temp_ofm;
            for (int c = 0; c < 32; c++)
            {
                ap_int<16> qy = temp_ifm1[c] * temp_ifm2[c];
                #pragma HLS bind_op variable=qy op=mul impl=dsp latency=0
                qy = qy >> scale;
                temp_ofm[c] = qy < amin ? ADT(amin) : qy > amax ? ADT(amax) : ADT(qy);
            }
            ofm[row][col] = temp_ofm;
        }
    }
}

/*
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigm_input_scale = 0.015625
sigm_output_scale = 0.015625
for i in range(256):
    if i < 128:
        print("{},".format(math.floor(sigmoid(i * sigm_input_scale) / sigm_output_scale + 0.5)), end="")
    else:
        print("{},".format(math.floor(sigmoid((i-256) * sigm_input_scale) / sigm_output_scale + 0.5)), end="")
    if (i+1) % 32 == 0:
        print("")
*/
ADT sigm_ROM(ap_uint<8> addr)
{
    #pragma HLS INLINE off    
    const ADT tanh_rom[256]={
        32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,
        40,40,40,41,41,41,41,41,42,42,42,42,43,43,43,43,43,44,44,44,44,45,45,45,45,45,46,46,46,46,46,47,
        47,47,47,47,48,48,48,48,48,48,49,49,49,49,49,50,50,50,50,50,50,51,51,51,51,51,51,52,52,52,52,52,
        52,52,53,53,53,53,53,53,53,54,54,54,54,54,54,54,55,55,55,55,55,55,55,55,55,56,56,56,56,56,56,56,
        8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,11,11,11,11,11,11,11,12,
        12,12,12,12,12,12,13,13,13,13,13,13,14,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16,16,17,17,17,
        17,17,18,18,18,18,18,19,19,19,19,19,20,20,20,20,21,21,21,21,21,22,22,22,22,23,23,23,23,23,24,24,
        24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,
    };
    ADT res = tanh_rom[addr];
    return res;
}

void sigmoid_by_point(hls::vector<ADT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50])
{
    #pragma HLS INLINE off

    for (int row = 1; row < 49; row++)
    {
        for (int col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            hls::vector<ADT, 32> temp_ifm = ifm[row][col];
            hls::vector<ADT, 32> temp_ofm;
            for (int c = 0; c < 32; c++)
            {
                #pragma HLS UNROLL
                ap_uint<8> addr;
                addr = temp_ifm[c](7,0);
                temp_ofm[c] = sigm_ROM(addr);
            }
            ofm[row][col] = temp_ofm;
        }
    }
}

/*
import math
tanh_input_scale = 0.015625
tanh_output_scale = 0.03125
for i in range(256):
    if i < 128:
        print("{},".format(math.floor(math.tanh(i * tanh_input_scale) / tanh_output_scale + 0.5)), end="")
    else:
        print("{},".format(math.floor(math.tanh((i-256) * tanh_input_scale) / tanh_output_scale + 0.5)), end="")
    if (i+1) % 32 == 0:
        print("")
*/
ADT tanh_ROM(ap_uint<8> addr)
{
    #pragma HLS INLINE off    
    const ADT tanh_rom[256]={
        0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11,12,12,13,13,14,14,14,
        15,15,16,16,16,17,17,17,18,18,18,19,19,19,20,20,20,21,21,21,21,22,22,22,23,23,23,23,23,24,24,24,
        24,25,25,25,25,25,26,26,26,26,26,26,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,29,29,29,29,
        29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,31,31,
        -31,-31,-31,-31,-31,-31,-31,-31,-31,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-29,-29,-29,-29,-29,-29,
        -29,-29,-29,-29,-29,-28,-28,-28,-28,-28,-28,-28,-28,-28,-27,-27,-27,-27,-27,-27,-27,-26,-26,-26,-26,-26,-26,-25,-25,-25,-25,-25,
        -24,-24,-24,-24,-23,-23,-23,-23,-23,-22,-22,-22,-21,-21,-21,-21,-20,-20,-20,-19,-19,-19,-18,-18,-18,-17,-17,-17,-16,-16,-16,-15,
        -15,-14,-14,-14,-13,-13,-12,-12,-11,-11,-11,-10,-10,-9,-9,-8,-8,-7,-7,-6,-6,-5,-5,-4,-4,-3,-3,-2,-2,-1,-1,0,
    };
    ADT res = tanh_rom[addr];
    return res;
}

void tanh_by_point(hls::vector<ADT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50])
{
    #pragma HLS INLINE off

    for (int row = 1; row < 49; row++)
    {
        for (int col = 1; col < 49; col++)
        {
            #pragma HLS PIPELINE II=1
            hls::vector<ADT, 32> temp_ifm = ifm[row][col];
            hls::vector<ADT, 32> temp_ofm;
            for (int c = 0; c < 32; c++)
            {
                #pragma HLS UNROLL
                ap_uint<8> addr;
                addr = temp_ifm[c](7,0);
                temp_ofm[c] = tanh_ROM(addr);
            }
            ofm[row][col] = temp_ofm;
        }
    }
}


void pw_conv_group(hls::vector<ADT, 32> ifm[50][50], hls::vector<RDT, 32> rfm[3][50][50], hls::vector<WDT, 32> wBUF1x1[3][32], short chx)
{
    #pragma hls inline off
    for (short outx = 0; outx < chx; outx++) {
        #pragma HLS loop_tripcount min=1 max=3 avg=2
        pw_conv_1x1(ifm, rfm[outx], wBUF1x1[outx]);
    }
}