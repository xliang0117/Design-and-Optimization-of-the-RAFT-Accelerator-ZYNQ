#include "include/convolution.hpp"
#include "include/log.hpp"
#define LOCAL_LOG_LEVEL LOG_LEVEL_DEBUG


void load_pw_fm(HALF8 *ifm_ddr, half ifm[8][18][18], short width)
{
    #pragma HLS ARRAY_PARTITION variable=ifm dim=1 complete

    for (short row = 1; row < 17; row++)
    {
        for (short col = 1; col < 17; col++)
        {
            #pragma HLS PIPELINE II=1
            HALF8 in_ddr_data;
            in_ddr_data = ifm_ddr[(row - 1) * width + col - 1];

            for (short c = 0; c < 8; c++)
            {
                half valid_data;
                valid_data = rawBitsToHalf(in_ddr_data(16*c+15, 16*c));
                
                ifm[c][col][row] = valid_data;
            }
        }
    }
}

void load_res_fm(HALF8 *ifm_ddr, half ifm[4][32][18][18], ap_uint<16> c_id, short width)
{
    assert(c_id <= 12);
    #pragma HLS ARRAY_PARTITION variable=ifm dim=1 complete


    ap_uint<2> cur_part = c_id >> 2;
    ap_uint<5> cur_chan = c_id(1,0) << 3;

    for (short row = 1; row < 17; row++)
    {
        for (short col = 1; col < 17; col++)
        {
            #pragma HLS PIPELINE II=1
            HALF8 in_ddr_data;
            in_ddr_data = ifm_ddr[(row - 1) * width + col - 1];

            for (short c = 0; c < 8; c++)
            {
                half valid_data;
                valid_data = rawBitsToHalf(in_ddr_data(16*c+15, 16*c));
                
                ifm[cur_part][cur_chan+c][col][row] = valid_data;
            }
        }
    }
}

void load_pw_stride2(HALF8 *ifm_ddr, half ifm[8][18][18], short width) // width should be after shrink
{
    #pragma HLS ARRAY_PARTITION variable=ifm dim=1 complete

    for (short row = 1; row < 17; row++)
    {
        for (short col = 1; col < 17; col++)
        {
            #pragma HLS PIPELINE II=1
            HALF8 in_ddr_data;
            in_ddr_data = ifm_ddr[2 * (row - 1) * width + 2 * (col - 1)];

            for (short c = 0; c < 8; c++)
            {
                half valid_data;
                valid_data = rawBitsToHalf(in_ddr_data(16*c+15, 16*c));
                
                ifm[c][col][row] = valid_data;
            }
        }
    }
}

void load_dw_fm(HALF8 *ifm_ddr, half ifm[8][18][18], short row_padding, short col_padding, short width) //all ex addr should before one row and one col
{
    #pragma HLS ARRAY_PARTITION variable=ifm dim=1 complete

    for (short row = 0; row < 18; row++)
    {
        for (short col = 0; col < 18; col++)
        {
            #pragma HLS PIPELINE II=1
            HALF8 in_ddr_data;
            in_ddr_data = ifm_ddr[row * width + col];

            for (short c = 0; c < 8; c++)
            {
                half ddr_data;
                half valid_data;
                half zero_data = 0;
                
                ddr_data = rawBitsToHalf(in_ddr_data(16*c+15, 16*c));
                if ( (row == 0 && row_padding == TOP_ROW) || (row == 17 && row_padding == BOTTOM_ROW) 
                    || (col == 0 && col_padding == TOP_COL) || (col == 17 && col_padding == BOTTOM_COL) )
                {
                    valid_data = zero_data;
                }
                else
                {
                    valid_data = ddr_data;
                }
                ifm[c][col][row] = valid_data;
            }
        }
    }
}

void load_w1x1(HALF16 *weight_ddr, half wbuf1x1[32][4][8][32], short inchannel_limit, short outchannel_limit)
{
    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=4 complete

    assert(inchannel_limit <= 32);
    assert(outchannel_limit <= 4);

    for (short cit = 0; cit < inchannel_limit; cit++)
    {
        for (short cot = 0; cot < outchannel_limit; cot++)
        {
            for (short c_i = 0; c_i < 8; c_i++)
            {
                for (short c_o = 0; c_o < 2; c_o++)
                {
                    #pragma HLS PIPELINE II=1
                    HALF16 weight_ddr_data = weight_ddr[cit*outchannel_limit*16 + cot*16 + c_i*2 + c_o];
                    for (short c_o_p = 0; c_o_p < 16; c_o_p++)
                    {
                        half weight_data = rawBitsToHalf(weight_ddr_data(16*c_i+15, 16*c_i));
                        wbuf1x1[cit][cot][c_i][c_o*16 + c_o_p] = weight_data;
                    }
                }
                
            }
        }
    }
    
       
}

void load_w3x3(HALF16 *weight_ddr, half wbuf3x3[32][8][3][3], short inchannel_limit)
{
    #pragma HLS ARRAY_PARTITION variable=wbuf3x3 dim=2 complete
    
    for (short cit = 0; cit < inchannel_limit; cit++)
    {
        for (short row = 0; row < 3; row++)
        {
            for (short col = 0; col < 3; col++)
            {
                #pragma HLS PIPELINE II=1
                HALF16 weight_ddr_data = weight_ddr[cit*9 + row*3 + col];
                for (short c = 0; c < 8; c++)
                {
                    half weight_data = rawBitsToHalf(weight_ddr_data(16*c+15, 16*c));
                    wbuf3x3[cit][c][col][row] = weight_data;

                }
            }
        }
    }
}

void load_bnbuf(HALF16 *weight_ddr, half bn_buf[96], short part_div32)
{
    #pragma hls inline off
    #pragma HLS ARRAY_PARTITION variable=bn_buf dim=1 cyclic factor=32

    for (short part = 0; part < part_div32; part++)
    {
        for (short group = 0; group < 2; group++)
        {
            #pragma HLS PIPELINE II=1
            HALF16 bn_ddr_data = weight_ddr[part*2 + group];
            for (short c = 0; c < 16; c++)
            {
                half bn_data = rawBitsToHalf(bn_ddr_data(16*c+15, 16*c));
                bn_buf[part*32 + group*16 + c] = bn_data;

            }
        }
    }

}

void load_dwbbuf(HALF16 *weight_ddr, half bn_buf[32][8], short part_div8)
{
    #pragma hls inline off
    #pragma HLS ARRAY_PARTITION variable=bn_buf dim=2 complete

    for (short part = 0; part < part_div8; part++)
    {
            #pragma HLS PIPELINE II=1
            HALF16 bn_ddr_data = weight_ddr[part];
            for (short c = 0; c < 8; c++)
            {
                half bn_data = rawBitsToHalf(bn_ddr_data(16*c+15, 16*c));
                bn_buf[part][c] = bn_data;
            }
    }

}



half MAC9(
    half A_00, half A_01, half A_02,
    half A_10, half A_11, half A_12,
    half A_20, half A_21, half A_22,
    half B_00, half B_01, half B_02,
    half B_10, half B_11, half B_12,
    half B_20, half B_21, half B_22
){
    half prod_00 = A_00 * B_00;
    half prod_01 = A_01 * B_01;
    half prod_02 = A_02 * B_02;
    half prod_10 = A_10 * B_10;
    half prod_11 = A_11 * B_11;
    half prod_12 = A_12 * B_12;
    half prod_20 = A_20 * B_20;
    half prod_21 = A_21 * B_21;
    half prod_22 = A_22 * B_22;

    half sum_0 = prod_00 + prod_11;
    half sum_1 = prod_01 + prod_12;
    half sum_2 = prod_02 + prod_20;
    half sum_3 = prod_10 + prod_21;

    half res_0 = sum_0 + sum_1;
    half res_1 = sum_2 + sum_3;
    half res_2 = prod_22;

    half res = res_0 + res_1 + res_2;
    return res;
}

void dw_conv_3x3(const half ifm[8][18][18], half ofm[8][18][18], const half wbuf3x3[8][3][3], const half dw_bn[8])
{
    #pragma HLS ARRAY_PARTITION variable=ofm dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=ifm dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wbuf3x3 dim=0 complete

    half window_buffer[8][3][3];
    #pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0

    half line_buffer[8][3][18];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2

    for (int row = 0; row < 18; row++)
    {
        for (int col = 0; col < 18; col++)
        {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE II=1
            for (int c = 0; c < 8; c++)
            {
                #pragma HLS UNROLL
                half read_in = ifm[c][col][row];
                line_buffer[c][row % 3][col] = read_in;

                window_buffer[c][2][2] = read_in;
                window_buffer[c][1][2] = line_buffer[c][(row + 2) % 3][col];
                window_buffer[c][0][2] = line_buffer[c][(row + 1) % 3][col];

                if (row >= 2 && col >= 2)
                {
                    half res = MAC9(
                        window_buffer[c][0][0], window_buffer[c][0][1], window_buffer[c][0][2],
                        window_buffer[c][1][0], window_buffer[c][1][1], window_buffer[c][1][2],
                        window_buffer[c][2][0], window_buffer[c][2][1], window_buffer[c][2][2],
                        wbuf3x3[c][0][0], wbuf3x3[c][1][0], wbuf3x3[c][2][0],
                        wbuf3x3[c][0][1], wbuf3x3[c][1][1], wbuf3x3[c][2][1],
                        wbuf3x3[c][0][2], wbuf3x3[c][1][2], wbuf3x3[c][2][2]);

                    ofm[c][col - 1][row - 1] = res + dw_bn[c];
                }

                for (int r = 0; r < 3; r++)
                {
                    #pragma HLS UNROLL
                    window_buffer[c][r][0] = window_buffer[c][r][1];
                    window_buffer[c][r][1] = window_buffer[c][r][2];
                }
            }
        }
    }
}

half MAC8(half w0,  half b0,
        half w1,  half b1,
        half w2,  half b2,
        half w3,  half b3,
        half w4,  half b4,
        half w5,  half b5,
        half w6,  half b6,
        half w7,  half b7)
{
#pragma HLS INLINE off
	half mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	half add0, add1, add2, add3;
	half add4, add5;
    half add6;

    mul0 = w0 * b0;
    mul1 = w1 * b1;
    mul2 = w2 * b2;
    mul3 = w3 * b3;
    mul4 = w4 * b4;
    mul5 = w5 * b5;
    mul6 = w6 * b6;
    mul7 = w7 * b7;

    add0 = mul0 + mul1;
    add1 = mul2 + mul3;
    add2 = mul4 + mul5;
    add3 = mul6 + mul7;

    add4 = add0 + add1;
    add5 = add2 + add3;

    add6 = add4 + add5;

    return add6;
}

void pw_conv_1x1(half ifm[8][18][18], half ofm[32][18][18], half wbuf1x1[8][32])
{
    #pragma HLS ARRAY_PARTITION variable=ofm dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=ifm dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=1 complete

    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=2 complete


    for(short row=1; row<17; row++)
    {
        for(short col=1; col<17; col++)
        {
            #pragma HLS PIPELINE II=1
            for(short co=0; co<32; co++)
            {
                #pragma HLS UNROLL
                half res = ofm[co][col][row];
                res += MAC8(
                                wbuf1x1[0][co],   ifm[0][col][row],
                                wbuf1x1[1][co],   ifm[1][col][row],
                                wbuf1x1[2][co],   ifm[2][col][row],
                                wbuf1x1[3][co],   ifm[3][col][row],
                                wbuf1x1[4][co],   ifm[4][col][row],
                                wbuf1x1[5][co],   ifm[5][col][row],
                                wbuf1x1[6][co],   ifm[6][col][row],
                                wbuf1x1[7][co],   ifm[7][col][row]);
                ofm[co][col][row] = res;
                // LOG_DEBUG("ifm: %f", (float)ifm[0][col][row]);
                // LOG_DEBUG("weight: %f", (float)wbuf1x1[0][co]);
                // LOG_DEBUG("res: %f", (float)res);
            }
        }
    }

}

void batch_norm(half ifm[4][32][18][18], half ofm[4][32][18][18], half bn_buf[96], short part_limit)
{
    #pragma HLS ARRAY_PARTITION variable=bn_buf dim=1 cyclic factor=32
    for (short part = 0; part < part_limit; part++)
    {
        for(short row=1; row<17; row++)
        {
            for(short col=1; col<17; col++)
            {
                #pragma HLS PIPELINE II=1
                for(short co=0; co<32; co++)
                {
                    #pragma HLS UNROLL
                    ofm[part][co][col][row] = ifm[part][co][col][row] + bn_buf[part*32 + co];
                    ifm[part][co][col][row] = 0;
                }
            }
        }
    }
    
}

void export_dw_shrink(HALF8 *ofm_ddr, half ofm[8][18][18], short width)
{

    #pragma HLS ARRAY_PARTITION variable=ofm dim=2 complete

        for (short row = 0; row < 8; row++)
        {
            for (short col = 0; col < 8; col++)
            {
                #pragma HLS PIPELINE II=1
                HALF8 out_data;
                for (short c = 0; c < 8; c++)
                {
                    #pragma HLS unroll
                    ap_uint<16> zero_bits = 0;
                    ap_uint<16> ddr_data = halfToRawBits(ofm[c][2*col+1][2*row+1]);
                    out_data(16*c+15, 16*c) = ddr_data;
                }
                ofm_ddr[(row)*width + col] = out_data;
            }
        }

}

void export_fm(HALF8 *ofm_ddr, half ofm[4][32][18][18], int part_limit, short outchannel_limit, short width, short height, bool relu) //one part is equal to 8 channel
{

    assert(outchannel_limit <= 4);
    #pragma HLS ARRAY_PARTITION variable=ofm dim=2 complete

    for(short cot = 0; cot < outchannel_limit; cot++)
    {
        for (short part = 0; part < part_limit; part++)
        {
            for (short row = 1; row < 17; row++)
            {
                for (short col = 1; col < 17; col++)
                {
                    #pragma HLS PIPELINE II=1
                    HALF8 out_data;
                    for (short c = 0; c < 8; c++)
                    {
                        #pragma HLS unroll
                        ap_uint<16> zero_bits = 0;
                        ap_uint<16> ddr_data = halfToRawBits(ofm[cot][c+part*8][col][row]);

                        out_data(16*c+15, 16*c) = (relu)? ((ddr_data(15, 15) == 1) ? zero_bits : ddr_data) : ddr_data;
                        ofm[cot][c+part*8][col][row] = 0;
                    }
                    ofm_ddr[cot*part_limit*width*height + part*width*height + (row-1)*width + col - 1] = out_data;
                }
            }
        }
    }
}

void pw_conv_group(half dw_fm[8][18][18], half pw_fm[4][32][18][18], half wbuf1x1[4][8][32], short times)
{    
    #pragma HLS ARRAY_PARTITION variable=pw_fm dim=1 complete

    #pragma HLS ALLOCATION function instances=pw_conv_1x1 limit=1
    #pragma HLS inline off

    for (short iter = 0; iter < times; iter++)
    {
        #pragma HLS loop_tripcount min=1 max=3     
        pw_conv_1x1(dw_fm, pw_fm[iter], wbuf1x1[iter]);
    }
    
}


half sigmoid_output(half op)
{
    #pragma HLS INLINE off    
    static half sigm_rom[1024];

    for (int i = 0; i < 1024; i++)
    {
        sigm_rom[i] = 1;
    }

    half res;
    ap_fixed<10,3> fixed_location;
    fixed_location = op;
    res = sigm_rom[fixed_location(9, 0)];
    return res;
}

void mulmat_by_point(half ifm1[4][32][18][18], half ifm2[4][32][18][18], half ofm[4][32][18][18])
{

    #pragma HLS ARRAY_PARTITION variable=ifm1 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ifm2 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ofm dim=2 complete


    for(short row=1; row<17; row++)
    {
        for(short col=1; col<17; col++)
        {
            for(short co_p=0; co_p<3; co_p++)
            {
                #pragma HLS pipeline II=1
                for(short co=0; co<32; co++)
                {
                    #pragma HLS unroll
                    half in1 = ifm1[co_p][co][row][col];
                    half in2 = ifm2[co_p][co][row][col];

                    half sigm_op = sigmoid_output(in1);
                    half temp_mul_res = sigm_op * in2;

                    ifm1[co_p][co][row][col] = 0;
                    ofm[co_p][co][row][col] = temp_mul_res;
                }
            }
        }
    }
}

half tanh_output(half op)
{
    #pragma HLS INLINE off    
    static half tanh_rom[1024];

    for (int i = 0; i < 1024; i++)
    {
        tanh_rom[i] = i;
    }

    half res;
    ap_fixed<10, 3> fixed_location;
    fixed_location = op;
    res = tanh_rom[fixed_location(9, 0)];
    return res;
}

void tanh_by_point(half ifm[4][32][18][18], half ofm[4][32][18][18])
{
    #pragma HLS ARRAY_PARTITION variable=ifm dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ofm dim=2 complete


    for(short row=1; row<17; row++)
    {
        for(short col=1; col<17; col++)
        {
            for(short co_p=0; co_p<3; co_p++)
            {
                #pragma HLS pipeline II=1
                for(short co=0; co<32; co++)
                {
                    #pragma HLS unroll
                    half in1 = ifm[co_p][co][row][col];
                    half tanh_op = tanh_output(in1);
                    ofm[co_p][co][row][col] = tanh_op;
                }
            }
        }
    }
}

void addmat_by_point(half ifm1[4][32][18][18], half ifm2[4][32][18][18],half ofm[4][32][18][18], short part_limit)
{

    #pragma HLS ARRAY_PARTITION variable=ifm1 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ifm2 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ofm dim=2 complete

    for(short row=1; row<17; row++)
    {
        for(short col=1; col<17; col++)
        {
            for(short co_p=0; co_p<part_limit; co_p++)
            {
                #pragma HLS pipeline II=1
                for(short co=0; co<32; co++)
                {
                    #pragma HLS unroll
                    half in1 = ifm1[co_p][co][row][col];
                    half in2 = ifm2[co_p][co][row][col];
                    ofm[co_p][co][row][col] = in1 + in2;
                }
            }
        }
    }
}

void minusmat_by_point(half ifm1[4][32][18][18], half ifm2[4][32][18][18],half ofm[4][32][18][18])
{

    #pragma HLS ARRAY_PARTITION variable=ifm1 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ifm2 dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=ofm dim=2 complete

    for(short row=1; row<17; row++)
    {
        for(short col=1; col<17; col++)
        {
            for(short co_p=0; co_p<3; co_p++)
            {
                #pragma HLS pipeline II=1
                for(short co=0; co<32; co++)
                {
                    #pragma HLS unroll
                    half in1 = ifm1[co_p][co][row][col];
                    half in2 = ifm2[co_p][co][row][col];
                    ofm[co_p][co][row][col] = in1 - in2;
                }
            }
        }
    }
}
