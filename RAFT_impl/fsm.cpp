#include "include/fsm.hpp"
#include "include/corr.hpp"
#include "hls_print.h"


void conv_hw(HALF8 *fm, HALF16 *weight_ddr, half *corr)
{

    #pragma HLS INTERFACE m_axi port=fm depth=3000000 offset=slave bundle=fm
    #pragma HLS INTERFACE m_axi port=weight_ddr depth=25000 offset=slave bundle=weight
    #pragma HLS interface m_axi port=corr depth=24000000 offset=slave bundle=corr_axi

    #pragma HLS ALLOCATION function instances=load_pw_fm limit=1
    #pragma HLS ALLOCATION function instances=load_w1x1 limit=1
    #pragma HLS ALLOCATION function instances=load_bnbuf limit=1
    #pragma HLS ALLOCATION function instances=load_pw_stride2 limit=1
    #pragma HLS ALLOCATION function instances=pw_conv_group limit=1
    #pragma HLS ALLOCATION function instances=load_dw_fm limit=1
    #pragma HLS ALLOCATION function instances=load_w3x3 limit=1
    #pragma HLS ALLOCATION function instances=dw_conv_3x3 limit=1
    #pragma HLS ALLOCATION function instances=export_fm limit=1
    #pragma HLS ALLOCATION function instances=export_dw_shrink limit=1
    #pragma HLS ALLOCATION function instances=tanh_by_point limit=1
    #pragma HLS ALLOCATION function instances=mulmat_by_point limit=1
    #pragma HLS ALLOCATION function instances=addmat_by_point limit=1
    #pragma HLS ALLOCATION function instances=batch_norm limit=1


    static half dw_fm1[8][18][18];      //[channel][col][row]
    static half dw_fm2[8][18][18];
    static half dw_fm3[8][18][18];
    static half dw_fm4[8][18][18];

    static half pw_fm[4][32][18][18];   //[32*i][channel][col][row]
    static half pw_fm_bias[4][32][18][18];
    static half pw_fm_out[4][32][18][18];
    static half pw_fm_net[4][32][18][18];


    static half wbuf3x3[32][8][3][3];
    static half wbuf1x1[32][4][8][32];  //[inpart][outpart][inc][outc]

    static half bn_buf[96];
    static half dw_bn_buf[32][8];

    static half coords[3072][2];

    padding_behaviour row_padding, col_padding;

    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=wbuf1x1 dim=4 complete


EXTRATOR:for (int i = 0; i < 2; i++)
{
    //layer EA: 3->32, stride = 2
    {
        load_w3x3(weight_ddr + config_extw[i][0].weight3x3_addr, wbuf3x3, 1);
        load_dwbbuf(weight_ddr + config_extw[i][0].bn_addr, dw_bn_buf, 1);
        EA3_row:for (short row_id = 0; row_id < 24; row_id++)
        {
            row_padding = row_id == 0? TOP_ROW : row_id == 23? BOTTOM_ROW : NO_PADDING;
            load_dw_fm(fm + config_ext[i][0].im_addr + row_id*8192, dw_fm1, row_padding, TOP_COL, 512);
            EA3_col:for (short col_id = 0; col_id < 24; col_id++)
            {
                col_padding = col_id == 22? BOTTOM_COL : NO_PADDING; // since load before one col
                if (col_id % 2 == 0)
                {
                    load_dw_fm(fm + config_ext[i][0].im_addr + row_id*8192 + (col_id+1)*16, dw_fm3, row_padding, col_padding, 512);
                    dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                    if (col_id != 0) 
                    {
                        export_dw_shrink(fm + config_ext[i][0].ex_addr + row_id*2048 + (col_id-1)*8, dw_fm4, 256);
                    }
                }
                else
                {
                    load_dw_fm(fm + config_ext[i][0].im_addr + row_id*8192 + (col_id+1)*16, dw_fm1, row_padding, col_padding, 512);
                    dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[0], dw_bn_buf[0]);
                    export_dw_shrink(fm + config_ext[i][0].ex_addr + row_id*2048 + (col_id-1)*8, dw_fm2, 256);
                }
                //hls::print("layerEA3: %d\n", row_id);
            }
            export_dw_shrink(fm + config_ext[i][0].ex_addr + row_id*2048 + 248, dw_fm4, 256);
        }
    }

    load_w1x1(weight_ddr + config_extw[i][1].weight1x1_addr, wbuf1x1, 1, OUTCHAN32);
    load_bnbuf(weight_ddr + config_extw[i][1].bn_addr, bn_buf, OUTCHAN32);

    EA_row:for (short row_id = 0; row_id < 12; row_id++)
    {
        load_pw_fm(fm + config_ext[i][1].im_addr + row_id*4096, dw_fm2, 256);
        EA_col:for (short col_id = 0; col_id < 12; col_id++)
        {
            if(col_id % 2 == 0)
            {
                load_pw_fm(fm + config_ext[i][1].im_addr + row_id*4096 + (col_id+1)*16, dw_fm4, 256);
                pw_conv_group(dw_fm2, pw_fm, wbuf1x1[0], OUTCHAN32);
                batch_norm(pw_fm, pw_fm_out, bn_buf, OUTCHAN32);
                if(col_id != 0)
                {
                    export_fm(fm + config_ext[i][1].ex_addr + row_id*4096 + (col_id-1)*16, pw_fm_bias, 4, OUTCHAN32, 256, 192, false);
                }
            }
            else
            {
                load_pw_fm(fm + config_ext[i][1].im_addr + row_id*4096 + (col_id+1)*16, dw_fm2, 256);
                pw_conv_group(dw_fm4, pw_fm, wbuf1x1[0], OUTCHAN32);
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN32);
                export_fm(fm + config_ext[i][1].ex_addr + row_id*4096 + (col_id-1)*16, pw_fm_out, 4, OUTCHAN32, 256, 192, false);
            }
            //hls::print("layerEA: %d\n", row_id);
        }
        export_fm(fm + config_ext[i][1].ex_addr + row_id*4096 + 240, pw_fm_bias, 4, OUTCHAN32, 256, 192, false);
    }

    //layer EP1: 32->64 //res unit 192*256 -> 96*128
    {
        load_w1x1(weight_ddr + config_extw[i][2].weight1x1_addr, wbuf1x1, 4, OUTCHAN64);
        load_bnbuf(weight_ddr + config_extw[i][2].bn_addr, bn_buf, OUTCHAN64);
        EP1_row:for (short row_id = 0; row_id < 6; row_id++)
        {
            EP1_col:for (short col_id = 0; col_id < 6; col_id++)
            {
                load_pw_stride2(fm + config_ext[i][2].im_addr + row_id*8192 + col_id*32, dw_fm2, 128);
                EP1_cin:for (short c_id = 0; c_id < 4; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_stride2(fm + config_ext[i][2].im_addr + row_id*8192 + col_id*32 + (c_id+1)*49152, dw_fm4, 128);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN64);
                    }
                    else
                    {
                        load_pw_stride2(fm + config_ext[i][2].im_addr + row_id*8192 + col_id*32 + (c_id+1)*49152, dw_fm2, 128);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN64);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN64);
                export_fm(fm + config_ext[i][2].ex_addr + row_id*2048 + col_id*16, pw_fm_bias, 4, OUTCHAN64, 128, 96, false);
                //hls::print("layerEP1: %d\n", row_id);
            }
        }
    }

    //layer EE: 32->16, 192*256
    {
        load_w1x1(weight_ddr + config_extw[i][3].weight1x1_addr, wbuf1x1, 4, OUTCHAN32);
        load_bnbuf(weight_ddr + config_extw[i][3].bn_addr, bn_buf, OUTCHAN32);
        EE_row:for (short row_id = 0; row_id < 12; row_id++)
        {
            EE_col:for (short col_id = 0; col_id < 12; col_id++)
            {
                load_pw_fm(fm + config_ext[i][3].im_addr + row_id*4096 + col_id*16, dw_fm2, 256);
                EE_cin:for (short c_id = 0; c_id < 4; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][3].im_addr + row_id*4096 + col_id*16 + (c_id+1)*49152, dw_fm4, 256);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][3].im_addr + row_id*4096 + col_id*16 + (c_id+1)*49152, dw_fm2, 256);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN32);
                export_fm(fm + config_ext[i][3].ex_addr + row_id*4096 + col_id*16, pw_fm_bias, 2, OUTCHAN32, 256, 192, true);
                //hls::print("layerEE: %d\n", row_id);
            }
        }
    }

    //layer EF：16 -> 16, 192*256 -> 96*128
    {
        load_w3x3(weight_ddr + config_extw[i][4].weight3x3_addr, wbuf3x3, 2);
        load_dwbbuf(weight_ddr + config_extw[i][4].bn_addr, dw_bn_buf, 2);
        EF3_row:for (short row_id = 0; row_id < 12; row_id++)
        {
            EF3_col:for (short col_id = 0; col_id < 12; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 11? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 11? BOTTOM_COL : NO_PADDING;
                load_dw_fm(fm + config_ext[i][4].im_addr + row_id*4096 + col_id*16, dw_fm1, row_padding, col_padding, 256);
                EF3_cid:for (short c_id = 0; c_id < 2; c_id++)
                {
                    if (c_id == 0)
                    {
                        load_dw_fm(fm + config_ext[i][4].im_addr + row_id*4096 + col_id*16 + 49152, dw_fm3, row_padding, col_padding, 256);
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[1], dw_bn_buf[1]);
                        export_dw_shrink(fm + config_ext[i][4].ex_addr + row_id*1024 + col_id*8, dw_fm2, 128);
                    }
                }
                export_dw_shrink(fm + config_ext[i][4].ex_addr + row_id*1024 + col_id*8 + 12288, dw_fm4, 128);
                //hls::print("layerEF3: %d\n", row_id);
            }
        }

        load_w1x1(weight_ddr + config_extw[i][5].weight1x1_addr, wbuf1x1, 2, OUTCHAN32);
        load_bnbuf(weight_ddr + config_extw[i][5].bn_addr, bn_buf, OUTCHAN32);
        EF_row:for (short row_id = 0; row_id < 6; row_id++)
        {
            EF_col:for (short col_id = 0; col_id < 6; col_id++)
            {
                load_pw_fm(fm + config_ext[i][5].im_addr + row_id*2048 + col_id*16, dw_fm2, 128);
                EF_cin:for (short c_id = 0; c_id < 2; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][5].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm4, 128);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][5].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm2, 128);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN32);
                export_fm(fm + config_ext[i][5].ex_addr + row_id*2048 + col_id*16, pw_fm_bias, 2, OUTCHAN32, 128, 96, true);
                //hls::print("layerEF: %d\n", row_id);
            }
        }

    }

    //layer EG: 16 -> 64, 96*128
    {
        load_w1x1(weight_ddr + config_extw[i][6].weight1x1_addr, wbuf1x1, 2, OUTCHAN64);
        load_bnbuf(weight_ddr + config_extw[i][6].bn_addr, bn_buf, OUTCHAN64);
        EG_row:for (short row_id = 0; row_id < 6; row_id++)
        {
            EG_col:for (short col_id = 0; col_id < 6; col_id++)
            {
                load_pw_fm(fm + config_ext[i][6].im_addr + row_id*2048 + col_id*16, dw_fm2, 128);
                EG_cin:for (short c_id = 0; c_id < 2; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][6].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm4, 128);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN64);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][6].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm2, 128);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN64);
                    }
                    for (short res = 0; res < 4; res++)
                    {
                        load_res_fm(fm + config_ext[i][2].ex_addr + row_id*2048 + col_id*16 + (4*c_id+res)*12288, pw_fm_out, 4*c_id+res, 128);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN64);
                addmat_by_point(pw_fm_bias, pw_fm_out, pw_fm, OUTCHAN64);
                export_fm(fm + config_ext[i][7].ex_addr + row_id*2048 + col_id*16, pw_fm, 4, OUTCHAN64, 128, 96, true);
                //hls::print("layerEG: %d\n", row_id);
            }
        }

    }

    //layer EP2: 64->96 //res unit 96*128 -> 48*64
    {
        load_w1x1(weight_ddr + config_extw[i][8].weight1x1_addr, wbuf1x1, 8, OUTCHAN96);
        load_bnbuf(weight_ddr + config_extw[i][8].bn_addr, bn_buf, OUTCHAN96);
        EP2_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            EP2_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                load_pw_stride2(fm + config_ext[i][8].im_addr + row_id*4096 + col_id*32, dw_fm2, 64);
                EP2_cin:for (short c_id = 0; c_id < 8; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_stride2(fm + config_ext[i][8].im_addr + row_id*4096 + col_id*32 + (c_id+1)*12288, dw_fm4, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        load_pw_stride2(fm + config_ext[i][8].im_addr + row_id*4096 + col_id*32 + (c_id+1)*12288, dw_fm2, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                export_fm(fm + config_ext[i][8].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN96, 64, 48, false);
                //hls::print("layerEP2: %d\n", row_id);
            }
        }
    }

    //layer EH: 64 -> 24 // 96*128
    {
        load_w1x1(weight_ddr + config_extw[i][9].weight1x1_addr, wbuf1x1, 8, OUTCHAN32);
        load_bnbuf(weight_ddr + config_extw[i][9].bn_addr, bn_buf, OUTCHAN32);
        EH_row:for (short row_id = 0; row_id < 6; row_id++)
        {
            EH_col:for (short col_id = 0; col_id < 6; col_id++)
            {
                load_pw_fm(fm + config_ext[i][9].im_addr + row_id*2048 + col_id*16, dw_fm2, 128);
                EH_cin:for (short c_id = 0; c_id < 8; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][9].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm4, 128);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][9].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm2, 128);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN32);
                export_fm(fm + config_ext[i][9].ex_addr + row_id*2048 + col_id*16, pw_fm_bias, 3, OUTCHAN32, 128, 96, true);
                //hls::print("layerEH: %d\n", row_id);
            }
        }

    }

//layer EI: 24 -> 24, 96*128 -> 48*64
    {
        load_w3x3(weight_ddr + config_extw[i][10].weight3x3_addr, wbuf3x3, 3);
        load_dwbbuf(weight_ddr + config_extw[i][10].bn_addr, dw_bn_buf, 3);
        EI3_row:for (short row_id = 0; row_id < 6; row_id++)
        {
            EI3_col:for (short col_id = 0; col_id < 6; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 5? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 5? BOTTOM_COL : NO_PADDING;
                load_dw_fm(fm + config_ext[i][10].im_addr + row_id*2048 + col_id*16, dw_fm1, row_padding, col_padding, 128);
                EI3_cid:for (short c_id = 0; c_id < 3; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        load_dw_fm(fm + config_ext[i][10].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm3, row_padding, col_padding, 128);
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id], dw_bn_buf[c_id]);
                        if(c_id != 0)
                        {
                            export_dw_shrink(fm + config_ext[i][10].ex_addr + row_id*512 + col_id*8 + 3072, dw_fm4, 64);
                        }
                    }
                    else
                    {
                        load_dw_fm(fm + config_ext[i][10].im_addr + row_id*2048 + col_id*16 + (c_id+1)*12288, dw_fm1, row_padding, col_padding, 128);
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id], dw_bn_buf[c_id]);
                        export_dw_shrink(fm + config_ext[i][10].ex_addr + row_id*512 + col_id*8, dw_fm2, 64);
                    }
                }
                export_dw_shrink(fm + config_ext[i][10].ex_addr + row_id*512 + col_id*8 + 6144, dw_fm2, 64);
                //hls::print("layerEI3: %d\n", row_id);
            }
        }

        load_w1x1(weight_ddr + config_extw[i][11].weight1x1_addr, wbuf1x1, 3, OUTCHAN32);
        load_bnbuf(weight_ddr + config_extw[i][11].bn_addr, bn_buf, OUTCHAN32);
        EI_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            EI_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                load_pw_fm(fm + config_ext[i][11].im_addr + row_id*1024 + col_id*16, dw_fm2, 64);
                EI_cin:for (short c_id = 0; c_id < 3; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][11].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm4, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][11].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm2, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN32);
                export_fm(fm + config_ext[i][11].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 3, OUTCHAN32, 64, 48, true);
                //hls::print("layerEI: %d\n", row_id);
            }
        }
    }

    //layer EJ: 24 -> 96, 48*64
    {
        load_w1x1(weight_ddr + config_extw[i][12].weight1x1_addr, wbuf1x1, 1, OUTCHAN96);
        load_bnbuf(weight_ddr + config_extw[i][12].bn_addr, bn_buf, OUTCHAN96);
        EJ_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            EJ_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                load_pw_fm(fm + config_ext[i][12].im_addr + row_id*1024 + col_id*16, dw_fm2, 64);
                EJ_cin:for (short c_id = 0; c_id < 3; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][12].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm4, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][12].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm2, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    for (short res = 0; res < 4; res++)
                    {
                        load_res_fm(fm + config_ext[i][8].ex_addr + row_id*1024 + col_id*16 + (4*c_id+res)*3072, pw_fm_out, 4*c_id+res, 64);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                addmat_by_point(pw_fm_bias, pw_fm_out, pw_fm, OUTCHAN96);
                export_fm(fm + config_ext[i][13].ex_addr + row_id*1024 + col_id*16, pw_fm, 4, OUTCHAN96, 64, 48, true);
                //hls::print("layerEJ: %d\n", row_id);
            }
        }
    }

    //layer EK: 96 -> 64, 48*64
    {
        load_w1x1(weight_ddr + config_extw[i][14].weight1x1_addr, wbuf1x1, 12, OUTCHAN64);
        load_bnbuf(weight_ddr + config_extw[i][14].bn_addr, bn_buf, OUTCHAN64);
        EK_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            EK_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                load_pw_fm(fm + config_ext[i][14].im_addr + row_id*1024 + col_id*16, dw_fm2, 64);
                EK_cin:for (short c_id = 0; c_id < 12; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[i][14].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm4, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN64);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[i][14].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm2, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN64);
                    }
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN64);
                export_fm(fm + config_ext[i][14].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN64, 64, 48, true);
                //hls::print("layerEK: %d\n", row_id);
            }
        }

    }
}

//layer ECnet: 96 -> 96, 48*64
{
    load_w1x1(weight_ddr + config_extw[1][15].weight1x1_addr, wbuf1x1, 12, OUTCHAN96);
    load_bnbuf(weight_ddr + config_extw[1][15].bn_addr, bn_buf, OUTCHAN96);
    ENet_row:for (short row_id = 0; row_id < 3; row_id++)
    {
        ENet_col:for (short col_id = 0; col_id < 3; col_id++)
        {
            load_pw_fm(fm + config_ext[1][15].im_addr + row_id*1024 + col_id*16, dw_fm2, 64);
            ENet_cin:for (short c_id = 0; c_id < 12; c_id++)
            {
                if(c_id % 2 == 0)
                {
                    load_pw_fm(fm + config_ext[1][15].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm4, 64);
                    pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                }
                else
                {
                    load_pw_fm(fm + config_ext[1][15].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm2, 64);
                    pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                }
            }
            batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
            tanh_by_point(pw_fm_bias, pw_fm);
            export_fm(fm + config_ext[1][15].ex_addr + row_id*1024 + col_id*16, pw_fm, 4, OUTCHAN96, 64, 48, true);
            //hls::print("layerECnet:%d \n", row_id);
        }
    }

}

{
    CORR_MUL:for (short i = 0; i < 32; i++)
    {
        load_corrmul_op(fm + config_corr[0].im_addr + i*64, wbuf1x1);
        CORR_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            CORR_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                load_pw_fm(fm + config_ext[0][14].ex_addr + row_id*1024 + col_id*16, dw_fm2, 64);
                CORR_cin:for (short c_id = 0; c_id < 8; c_id++)
                {
                    if(c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_ext[0][14].ex_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm4, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        load_pw_fm(fm + config_ext[0][14].ex_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm2, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                }
                export_corrmul(fm + config_corr[0].ex_addr + (row_id*4+col_id)*256*3072/8 + i*96/8, pw_fm);
                //hls::print("layerCorrMul: %d\n", row_id);
            }
        }

    }

    CORR_POOL:for (short scale = 0; scale < 3; scale++)
    {
        //hls::print("CORR_POOL: %d\n", scale);
        corr_pool(fm + config_corr[scale].ex_addr, fm + config_corr[scale+1].ex_addr, scale);
    }
    
}

ITERATION:for (int i = 0; i < 3; i++)
{
    load_coords(fm + config_iter[8].ex_addr, coords, true);
    CORR_INDEX:for (int scale = 0; scale < 4; scale++)
    {
        //hls::print("Corr index: %d\n", scale);
        grid_sample(corr + config_corr[scale].ex_addr*8, coords, corr + config_iter[0].im_addr*8 + scale*56*3072, scale);
    }

    // layer A
    {
        load_w1x1(weight_ddr + config_iterw[0].weight1x1_addr, wbuf1x1, 25, OUTCHAN96);
        load_bnbuf(weight_ddr + config_iterw[0].bn_addr, bn_buf, OUTCHAN96);
        A_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            A_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                load_pw_fm(fm + config_iter[0].im_addr + row_id*1024 + col_id*16, dw_fm2, 64);

                A_cin:for (short c_id = 0; c_id < 25; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        load_pw_fm(fm + config_iter[0].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm4, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id],  OUTCHAN96);
                    }
                    else
                    {
                        load_pw_fm(fm + config_iter[0].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm2, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id],  OUTCHAN96);
                    }

                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                export_fm(fm + config_iter[0].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN96, 64, 48, config_iter[0].relu);
                //hls::print("layerA: %d\n", row_id);
            }
            
        }
    }

    //layer B
    {
        load_w3x3(weight_ddr + config_iterw[1].weight3x3_addr, wbuf3x3, 1);
        load_dwbbuf(weight_ddr + config_iterw[1].bn_addr, dw_bn_buf, 1);
        load_w1x1(weight_ddr + config_iterw[1].weight1x1_addr, wbuf1x1, 1, OUTCHAN64);
        load_bnbuf(weight_ddr + config_iterw[1].bn_addr, bn_buf, OUTCHAN64);

        B_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            B_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[1].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);   //row_id*16*64
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                pw_conv_group(dw_fm2, pw_fm, wbuf1x1[0], OUTCHAN64);
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN64);

                export_fm(fm + config_iter[1].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN64, 64, 48, config_iter[1].relu);
                //hls::print("layerB: %d\n", row_id);
            }
            
        }
    }

    //layer C
    {
        load_w3x3(weight_ddr + config_iterw[2].weight3x3_addr, wbuf3x3, 8);
        load_dwbbuf(weight_ddr + config_iterw[2].bn_addr, dw_bn_buf, 8);
        load_w1x1(weight_ddr + config_iterw[2].weight1x1_addr, wbuf1x1, 8, OUTCHAN32);
        load_bnbuf(weight_ddr + config_iterw[2].bn_addr, bn_buf, OUTCHAN32);

        C_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            C_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[2].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm + config_iter[2].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                C_cin:for (short c_id = 0; c_id < 8; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[2].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[2].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }                
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN32);
                export_fm(fm + config_iter[2].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN32, 64, 48, config_iter[2].relu);
                //hls::print("layerC: %d\n", row_id);
            }
        }
    }

    //layer D
    {
        load_w3x3(weight_ddr + config_iterw[3].weight3x3_addr, wbuf3x3, 16);
        load_dwbbuf(weight_ddr + config_iterw[3].bn_addr, dw_bn_buf, 16);
        load_w1x1(weight_ddr + config_iterw[3].weight1x1_addr, wbuf1x1, 16, OUTCHAN96);
        load_bnbuf(weight_ddr + config_iterw[3].bn_addr, bn_buf, OUTCHAN96);

        D_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            D_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[3].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm + config_iter[3].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                D_cin:for (short c_id = 0; c_id < 16; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[3].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[3].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }                
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                export_fm(fm + config_iter[3].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN96, 64, 48, config_iter[3].relu);
                //hls::print("layerD: %d\n", row_id);
            }
        }
    }

    //layer E
    {
        load_w3x3(weight_ddr + config_iterw[4].weight3x3_addr, wbuf3x3, 31);
        load_dwbbuf(weight_ddr + config_iterw[4].bn_addr, dw_bn_buf, 31);
        load_w1x1(weight_ddr + config_iterw[4].weight1x1_addr, wbuf1x1, 31, OUTCHAN96);
        load_bnbuf(weight_ddr + config_iterw[4].bn_addr, bn_buf, OUTCHAN96);

        E_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            E_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[4].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm+ config_iter[4].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                E_cin:for (short c_id = 0; c_id < 31; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[4].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[4].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }                
                    if(c_id < 12) load_res_fm(fm + config_ext[1][15].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, c_id, 64);
                }
                batch_norm(pw_fm, pw_fm_net, bn_buf, OUTCHAN96);
                mulmat_by_point(pw_fm_net, pw_fm_bias, pw_fm_out); // E generate G input
                export_fm(fm + config_iter[4].ex_addr + row_id*1024 + col_id*16, pw_fm_out, 4, OUTCHAN96, 64, 48, config_iter[4].relu);
                //hls::print("layerE: %d\n", row_id);
            }
            
        }
    }

    //layer F
    {
        load_w3x3(weight_ddr + config_iterw[5].weight3x3_addr, wbuf3x3, 31);
        load_dwbbuf(weight_ddr + config_iterw[5].bn_addr, dw_bn_buf, 31);
        load_w1x1(weight_ddr + config_iterw[5].weight1x1_addr, wbuf1x1, 31, OUTCHAN96);
        load_bnbuf(weight_ddr + config_iterw[5].bn_addr, bn_buf, OUTCHAN96);

        F_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            F_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[5].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm + config_iter[5].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                F_cin:for (short c_id = 0; c_id < 31; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[5].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[5].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }                
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                export_fm(fm + config_iter[5].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN96, 64, 48, config_iter[5].relu);    // z
                //hls::print("layerF: %d\n", row_id);
            }
            
        }
    }


    //layer G
    {
        load_w3x3(weight_ddr + config_iterw[6].weight3x3_addr, wbuf3x3, 31);
        load_dwbbuf(weight_ddr + config_iterw[6].bn_addr, dw_bn_buf, 31);
        load_w1x1(weight_ddr + config_iterw[6].weight1x1_addr, wbuf1x1, 31, OUTCHAN96);
        load_bnbuf(weight_ddr + config_iterw[6].bn_addr, bn_buf, OUTCHAN96);
        G_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            G_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[6].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm + config_iter[6].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                G_cin:for (short c_id = 0; c_id < 31; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[6].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[6].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }                
                    if(c_id < 12) 
                    {
                        load_res_fm(fm + config_ext[1][15].ex_addr + row_id*1024 + col_id*16, pw_fm_net, c_id, 64); // load net
                    }
                    else if (c_id < 24)
                    {
                        load_res_fm(fm + config_iter[5].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, c_id - 12, 64); // z
                    }
                }
                batch_norm(pw_fm, pw_fm_out, bn_buf, OUTCHAN96);

                tanh_by_point(pw_fm_out, pw_fm);

                minusmat_by_point(pw_fm, pw_fm_net, pw_fm_out);   //q - net

                mulmat_by_point(pw_fm_out, pw_fm_bias, pw_fm); //z (q - net)

                addmat_by_point(pw_fm, pw_fm_net, pw_fm_out, OUTCHAN96); // + net

                export_fm(fm + config_iter[6].ex_addr + row_id*1024 + col_id*16, pw_fm_out, 4, OUTCHAN96, 64, 48, config_iter[6].relu);
                //hls::print("layerG: %d\n", row_id);
            }
            
        }
    }

    //layer H
    {
        load_w3x3(weight_ddr + config_iterw[7].weight3x3_addr, wbuf3x3, 12);
        load_dwbbuf(weight_ddr + config_iterw[7].bn_addr, dw_bn_buf, 12);
        load_w1x1(weight_ddr + config_iterw[7].weight1x1_addr, wbuf1x1, 12, 3);
        load_bnbuf(weight_ddr + config_iterw[7].bn_addr, bn_buf, OUTCHAN96);

        H_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            H_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[7].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm + config_iter[7].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                H_cin:for (short c_id = 0; c_id < 12; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[7].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN96);

                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[7].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN96);
                    }                
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                export_fm(fm + config_iter[7].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 4, OUTCHAN96, 64, 48, config_iter[7].relu);
                //hls::print("layerH: %d\n", row_id);
            }
        }
    }

    //layer I
    {
        load_w3x3(weight_ddr + config_iterw[8].weight3x3_addr, wbuf3x3, 12);
        load_dwbbuf(weight_ddr + config_iterw[8].bn_addr, dw_bn_buf, 12);
        load_w1x1(weight_ddr + config_iterw[8].weight1x1_addr, wbuf1x1, 12, OUTCHAN32);
        load_bnbuf(weight_ddr + config_iterw[8].bn_addr, bn_buf, OUTCHAN32);

        I_row:for (short row_id = 0; row_id < 3; row_id++)
        {
            I_col:for (short col_id = 0; col_id < 3; col_id++)
            {
                row_padding = row_id == 0? TOP_ROW : row_id == 2? BOTTOM_ROW : NO_PADDING;
                col_padding = col_id == 0? TOP_COL : col_id == 2? BOTTOM_COL : NO_PADDING;

                load_dw_fm(fm + config_iter[8].im_addr + row_id*1024 + col_id*16, dw_fm1, row_padding, col_padding, 64);
                dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[0], dw_bn_buf[0]);
                load_dw_fm(fm + config_iter[8].im_addr + row_id*1024 + col_id*16 + 16, dw_fm3, row_padding, col_padding, 64);

                I_cin:for (short c_id = 0; c_id < 12; c_id++)
                {
                    if (c_id % 2 == 0)
                    {
                        dw_conv_3x3(dw_fm3, dw_fm4, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[8].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm1, row_padding, col_padding, 64);
                        
                        pw_conv_group(dw_fm2, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }
                    else
                    {
                        dw_conv_3x3(dw_fm1, dw_fm2, wbuf3x3[c_id+1], dw_bn_buf[c_id+1]);
                        load_dw_fm(fm + config_iter[8].im_addr + row_id*1024 + col_id*16 + (c_id+1)*3072, dw_fm3, row_padding, col_padding, 64);
                        
                        pw_conv_group(dw_fm4, pw_fm, wbuf1x1[c_id], OUTCHAN32);
                    }                
                }
                batch_norm(pw_fm, pw_fm_bias, bn_buf, OUTCHAN96);
                export_fm(fm + config_iter[8].ex_addr + row_id*1024 + col_id*16, pw_fm_bias, 1, OUTCHAN32, 64, 48, config_iter[8].relu);
                //hls::print("layerI: %d\n", row_id);
            }
        }
    }
}

}
