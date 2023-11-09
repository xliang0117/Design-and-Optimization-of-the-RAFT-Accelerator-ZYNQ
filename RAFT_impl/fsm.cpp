
#include "include/fsm.hpp"
#include "include/corr.hpp"
#include "hls_print.h"

void conv_hw(ADT4* IMAGE, ADT32* IMAGE_OUT, ADT32* fm, WDT32* weight, ADT* corr, bool curFrame)
{
    #pragma hls interface m_axi port=IMAGE depth=148000 offset=slave
    #pragma hls interface m_axi port=IMAGE_OUT depth=260000 offset=slave
    #pragma hls interface m_axi port=fm depth=420000 offset=slave
    #pragma hls interface m_axi port=weight depth=6400 offset=slave
    #pragma hls interface m_axi port=corr depth=527000 offset=slave
	// #pragma hls interface ap_memory port=flow storage_type=ram_1p
    // #pragma hls interface m_axi port=flow depth=2500 offset=slave bundle=fbram
    #pragma hls interface s_axilite register port=curFrame
    #pragma hls interface s_axilite register port=return

    FDT flow[2304];

    int EXOUT_addr = curFrame ? 182401 : 173185; // true:fmap1addr false:fmap2addr
    int CORRGEN_addr = curFrame ? 173185 : 182401;

    static hls::vector<ADT, 32> afm_1[50][50];  // [col][row]
    static hls::vector<ADT, 32> afm_2[50][50];
    static hls::vector<ADT, 32> afm_3[50][50];
    static hls::vector<ADT, 32> afm_4[50][50];
    static hls::vector<RDT, 32> rfm[3][50][50]; // [outpart][col][row]
    #pragma HLS bind_storage variable=rfm type=RAM_2P impl=URAM

    static hls::vector<WDT, 32> wBUF3x3_A[3][3];   // [pingpong][k][k]
    static hls::vector<WDT, 32> wBUF3x3_B[3][3];   // [pingpong][k][k]
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_A dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_A dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_B dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF3x3_B dim=2 complete

    static hls::vector<WDT, 32> wBUF1x1_A[3][32];  // [outpart][output channel]
    static hls::vector<WDT, 32> wBUF1x1_B[3][32];  // [outpart][output channel]
    #pragma HLS ARRAY_PARTITION variable=wBUF1x1_A dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wBUF1x1_B dim=2 complete

    static BDT dwb_buf_A[32];
    static BDT dwb_buf_B[32];
    #pragma HLS ARRAY_PARTITION variable=dwb_buf_A dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=dwb_buf_B dim=1 complete

    static BDT pwb_buf[3][32];
    #pragma HLS ARRAY_PARTITION variable=pwb_buf dim=2 complete

//EA3, 384*384*3 -> 192*192*3
    load_w3x3(weight + 0, wBUF3x3_A, 0);
    load_dwbbuf(weight + 9, dwb_buf_A, 0);
    EA3_R:for (short rowx = 0; rowx < 8; rowx++) {
        load_fm_IMAGE(IMAGE, afm_1, rowx, 0);
        for (short colx = 0; colx < 8; colx++) {
            if(colx % 2 == 0) {
                load_fm_IMAGE(IMAGE, afm_3, rowx, colx+1);
                dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 9);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 385, afm_2, rowx, colx, outx, 192, 192);
                }
            }
            else {
                load_fm_IMAGE(IMAGE, afm_1, rowx, colx+1);
                dw_conv_3x3(afm_3, afm_4, wBUF3x3_A, dwb_buf_A, 9);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 385, afm_4, rowx, colx, outx, 192, 192);
                }
            }
        }
    }

//EA1, 192*192*3 -> 192*192*32
    load_w1x1(weight + 11, wBUF1x1_A, 0, 1);
    load_pwbbuf(weight + 43, pwb_buf, 1);
    EA1_R:for (short rowx = 0; rowx < 4; rowx++) {
        load_fm_tile(fm + 192, afm_1, rowx, 0, 0, 192, 192);
        for (short colx = 0; colx < 4; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 192, afm_3, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 6, true);
                    export_fm_tile(fm + 37249, afm_1, rowx, colx, outx, 192, 192);
                }
            }
            else {
                load_fm_tile(fm + 192, afm_1, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 6, true);
                    export_fm_tile(fm + 37249, afm_3, rowx, colx, outx, 192, 192);
                }
            }
        }
    }

//EE, 192*192*32 -> 192*192*16
    load_w1x1(weight + 45, wBUF1x1_A, 0, 1);
    load_pwbbuf(weight + 77, pwb_buf, 1);
    EE_R:for (short rowx = 0; rowx < 4; rowx++) {
        load_fm_tile(fm + 37056, afm_1, rowx, 0, 0, 192, 192);
        for (short colx = 0; colx < 4; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 37056, afm_3, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 74113, afm_1, rowx, colx, outx, 192, 192);
                }
            }
            else {
                load_fm_tile(fm + 37056, afm_1, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 74113, afm_3, rowx, colx, outx, 192, 192);
                }
            }
        }
    }

//EF3, 192*192*16 -> 96*96*16
    load_w3x3(weight + 79, wBUF3x3_A, 0);
    load_dwbbuf(weight + 88, dwb_buf_A, 0);
    EF3_R:for (short rowx = 0; rowx < 4; rowx++) {
        load_fm_tile(fm + 73920, afm_1, rowx, 0, 0, 192, 192);
        for (short colx = 0; colx < 4; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 73920, afm_3, rowx, colx+1, 0, 192, 192);
                dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 5);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 110977, afm_2, rowx, colx, outx, 96, 96);
                }
            }
            else {
                load_fm_tile(fm + 73920, afm_1, rowx, colx+1, 0, 192, 192);
                dw_conv_3x3(afm_3, afm_4, wBUF3x3_A, dwb_buf_A, 5);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 110977, afm_4, rowx, colx, outx, 96, 96);
                }
            }
        }
    }

//EF1, 96*96*16 -> 96*96*16
    load_w1x1(weight + 90, wBUF1x1_A, 0, 1);
    load_pwbbuf(weight + 122, pwb_buf, 1);
    EF1_R:for (short rowx = 0; rowx < 2; rowx++) {
        load_fm_tile(fm + 110880, afm_1, rowx, 0, 0, 96, 96);
        for (short colx = 0; colx < 2; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 110880, afm_3, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 74113, afm_1, rowx, colx, outx, 96, 96);
                }
            }
            else {
                load_fm_tile(fm + 110880, afm_1, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 74113, afm_3, rowx, colx, outx, 96, 96);
                }
            }
        }
    }

//EG, 96*96*16 -> 96*96*64
    load_w1x1(weight + 124, wBUF1x1_A, 0, 2);
    load_pwbbuf(weight + 188, pwb_buf, 2);
    EG_R:for (short rowx = 0; rowx < 2; rowx++) {
        load_fm_tile(fm + 74016, afm_1, rowx, 0, 0, 96, 96);
        for (short colx = 0; colx < 2; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 74016, afm_3, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
                for (short outx = 0; outx < 2; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 83329, afm_1, rowx, colx, outx, 96, 96);
                }
            }
            else {
                load_fm_tile(fm + 74016, afm_1, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 2);
                for (short outx = 0; outx < 2; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 83329, afm_3, rowx, colx, outx, 96, 96);
                }
            }
        }
    }

//EP1, 192*192*32 -> 96*96*64
    load_pwbbuf(weight + 256, pwb_buf, 2);
    EP1_R:for (short rowx = 0; rowx < 2; rowx++) {
        for (short colx = 0; colx < 2; colx++) {
            for (short inx = 0; inx < 1; inx++) {
                load_fm_s2(fm + 37249, afm_1, rowx, colx, inx, 192, 192);
                load_w1x1(weight + 192, wBUF1x1_A, inx, 2);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
            }
            for (short outx = 0; outx < 2; outx++) {
                activation(rfm[outx], afm_1, pwb_buf[outx], 8, false);
                load_fm_tile(fm + 83232, afm_2, rowx, colx, outx, 96, 96);
                res_add_sub(afm_1, afm_2, afm_3, true, true);
                export_fm_tile(fm + 101761, afm_3, rowx, colx, outx, 96, 96);
            }
        }
    }

//EH, 96*96*64 -> 96*96*24
    load_pwbbuf(weight + 324, pwb_buf, 1);
    EH_R:for (short rowx = 0; rowx < 2; rowx++) {
        for (short colx = 0; colx < 2; colx++) {
            load_fm_tile(fm + 101664, afm_1, rowx, colx, 0, 96, 96);
            load_w1x1(weight + 260, wBUF1x1_A, 0, 1);
            EH_I:for (short inx = 0; inx < 2; inx++) {
                if(inx % 2 == 0) {
                    load_fm_tile(fm + 101664, afm_3, rowx, colx, inx+1, 96, 96);
                    load_w1x1(weight + 260, wBUF1x1_B, inx+1, 1);
                    pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                }
                else {
                    load_fm_tile(fm + 101664, afm_1, rowx, colx, inx+1, 96, 96);
                    load_w1x1(weight + 260, wBUF1x1_A, inx+1, 1);
                    pw_conv_group(afm_3, rfm, wBUF1x1_B, 1);
                }
            }
            EH_O:for (short outx = 0; outx < 1; outx++) {
                activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
                export_fm_tile(fm + 385, afm_1, rowx, colx, outx, 96, 96);
            }
        }
    }

//EI3, 96*96*24 -> 48*48*24
    load_w3x3(weight + 326, wBUF3x3_A, 0);
    load_dwbbuf(weight + 335, dwb_buf_A, 0);
    EI3_R:for (short rowx = 0; rowx < 2; rowx++) {
        load_fm_tile(fm + 288, afm_1, rowx, 0, 0, 96, 96);
        for (short colx = 0; colx < 2; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 288, afm_3, rowx, colx+1, 0, 96, 96);
                dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 8);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 9601, afm_2, rowx, colx, outx, 48, 48);
                }
            }
            else {
                load_fm_tile(fm + 288, afm_1, rowx, colx+1, 0, 96, 96);
                dw_conv_3x3(afm_3, afm_4, wBUF3x3_A, dwb_buf_A, 8);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 9601, afm_4, rowx, colx, outx, 48, 48);
                }
            }
        }
    }

//EI1, 48*48*24 -> 48*48*24
    load_pwbbuf(weight + 369, pwb_buf, 1);
    load_fm_full(fm + 9552, afm_1, 0);
    load_w1x1(weight + 337, wBUF1x1_A, 0, 1);
    pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
    EI1_O:for (short outx = 0; outx < 1; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
        export_fm_full(fm + 385, afm_1, outx);
    }

//EJ, 48*48*24 -> 48*48*96
    load_pwbbuf(weight + 467, pwb_buf, 3);
    load_fm_full(fm + 336, afm_1, 0);
    load_w1x1(weight + 371, wBUF1x1_A, 0, 3);
    pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
    EJ_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 6, true);
        export_fm_full(fm + 2689, afm_1, outx);
    }

//EP2, 96*96*64 -> 48*48*96
    load_pwbbuf(weight + 665, pwb_buf, 3);
    load_fm_s2(fm + 101761, afm_1, 0, 0, 0, 96, 96);
    load_w1x1(weight + 473, wBUF1x1_A, 0, 3);
    EP2_I:for (short inx = 0; inx < 2; inx++) {
        if(inx % 2 == 0) {
            load_fm_s2(fm + 101761, afm_3, 0, 0, inx+1, 96, 96);
            load_w1x1(weight + 473, wBUF1x1_B, inx+1, 3);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_s2(fm + 101761, afm_1, 0, 0, inx+1, 96, 96);
            load_w1x1(weight + 473, wBUF1x1_A, inx+1, 3);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 3);
        }
    }
    EP2_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, false);
        load_fm_full(fm + 2640, afm_2, outx);
        res_add_sub(afm_1, afm_2, afm_3, true, true);
        export_fm_full(fm + 9601, afm_3, outx);
    }

//EK, 48*48*96 -> 48*48*64
    load_pwbbuf(weight + 863, pwb_buf, 2);
    load_fm_full(fm + 9552, afm_1, 0);
    load_w1x1(weight + 671, wBUF1x1_A, 0, 2);
    EK_I:for (short inx = 0; inx < 3; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 9552, afm_3, inx+1);
            load_w1x1(weight + 671, wBUF1x1_B, inx+1, 2);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
        }
        else {
            load_fm_full(fm + 9552, afm_1, inx+1);
            load_w1x1(weight + 671, wBUF1x1_A, inx+1, 2);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 2);
        }
    }
    EK_O:for (short outx = 0; outx < 2; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, false);
        export_fm_full(fm + EXOUT_addr, afm_1, outx);
    }

{
    //2304x64 * 64x2304 = 2034*2304, can be see as input channels 64, output channels 2304
    load_fm_full(fm+EXOUT_addr-48-1, afm_1, 0);  //fnet2 channel part1
    load_fm_full(fm+EXOUT_addr-48-1, afm_2, 1);  //fnet2 channel part2
    CORR_GEN:for (short group = 0; group < 24; group++)
    {
        load_feature_vector(fm+CORRGEN_addr, wBUF1x1_A, group);        //fnet1
        load_feature_vector(fm+CORRGEN_addr+2304, wBUF1x1_B, group);   //fnet1
        pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
        pw_conv_group(afm_2, rfm, wBUF1x1_B, 3);
        for (short outx = 0; outx < 3; outx++) {
            export_corr(fm+191617, rfm[outx], outx, group);
        }
    }
    CORR_PULL:for (short layerNum = 1; layerNum < 4; layerNum++)
    {
        corr_pool(fm+191617, layerNum);
    }
}

for(short iter = 0; iter < 3; iter++) {


    CORR_INDEX:for (short layerNum = 0; layerNum < 4; layerNum++)
    {
        grid_sample(corr, fm+191617, flow, layerNum);
    }

    
//A, 48*48*196 -> 48*48*96
    load_pwbbuf(weight + 2700, pwb_buf, 3);
    load_fm_full(fm + 336, afm_1, 0);
    load_w1x1(weight + 2028, wBUF1x1_A, 0, 3);
    A_I:for (short inx = 0; inx < 7; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 336, afm_3, inx+1);
            load_w1x1(weight + 2028, wBUF1x1_B, inx+1, 3);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 336, afm_1, inx+1);
            load_w1x1(weight + 2028, wBUF1x1_A, inx+1, 3);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 3);
        }
    }
    A_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
        export_fm_full(fm + 16513, afm_1, outx);
    }

//B, 48*48*2 -> 48*48*64
    load_pwbbuf(weight + 2770, pwb_buf, 2);
    load_fm_flow(flow, afm_1);
    load_w1x1(weight + 2706, wBUF1x1_A, 0, 2);
    pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
    B_O:for (short outx = 0; outx < 2; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 5, true);
        export_fm_full(fm + 25729, afm_1, outx);
    }

//C, 48*48*64 -> 48*48*32
    load_pwbbuf(weight + 2838, pwb_buf, 1);
    load_fm_full(fm + 25680, afm_1, 0);
    load_w1x1(weight + 2774, wBUF1x1_A, 0, 1);
    C_I:for (short inx = 0; inx < 2; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 25680, afm_3, inx+1);
            load_w1x1(weight + 2774, wBUF1x1_B, inx+1, 1);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
        }
        else {
            load_fm_full(fm + 25680, afm_1, inx+1);
            load_w1x1(weight + 2774, wBUF1x1_A, inx+1, 1);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 1);
        }
    }
    C_O:for (short outx = 0; outx < 1; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
        export_fm_full(fm + 23425, afm_1, outx);
    }

//D3, 48*48*128 -> 48*48*80
    load_pwbbuf(weight + 3268, pwb_buf, 3);
    load_fm_full(fm + 16464, afm_1, 0);
    load_w3x3(weight + 2840, wBUF3x3_A, 0);
    load_w1x1(weight + 2884, wBUF1x1_A, 0, 3);
    load_dwbbuf(weight + 2876, dwb_buf_A, 0);
    D3_I:for (short inx = 0; inx < 4; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 16464, afm_3, inx+1);
            load_w3x3(weight + 2840, wBUF3x3_B, inx+1);
            load_w1x1(weight + 2884, wBUF1x1_B, inx+1, 3);
            load_dwbbuf(weight + 2876, dwb_buf_B, inx+1);
            dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 7);
            pw_conv_group(afm_2, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 16464, afm_1, inx+1);
            load_w3x3(weight + 2840, wBUF3x3_A, inx+1);
            load_w1x1(weight + 2884, wBUF1x1_A, inx+1, 3);
            load_dwbbuf(weight + 2876, dwb_buf_A, inx+1);
            dw_conv_3x3(afm_3, afm_4, wBUF3x3_B, dwb_buf_B, 7);
            pw_conv_group(afm_4, rfm, wBUF1x1_B, 3);
        }
    }
    D3_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_2, pwb_buf[outx], 7, true);
        out_with_flow(afm_2, flow, outx);
        export_fm_full(fm + 159361, afm_2, outx);
    }

//F3, 48*48*242 -> 48*48*96
    load_pwbbuf(weight + 4130, pwb_buf, 3);
    load_fm_full(fm + 147792, afm_1, 0);
    load_w3x3(weight + 3274, wBUF3x3_A, 0);
    load_w1x1(weight + 3362, wBUF1x1_A, 0, 3);
    load_dwbbuf(weight + 3346, dwb_buf_A, 0);
    F3_I:for (short inx = 0; inx < 8; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 147792, afm_3, inx+1);
            load_w3x3(weight + 3274, wBUF3x3_B, inx+1);
            load_w1x1(weight + 3362, wBUF1x1_B, inx+1, 3);
            load_dwbbuf(weight + 3346, dwb_buf_B, inx+1);
            dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 6);
            pw_conv_group(afm_2, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 147792, afm_1, inx+1);
            load_w3x3(weight + 3274, wBUF3x3_A, inx+1);
            load_w1x1(weight + 3362, wBUF1x1_A, inx+1, 3);
            load_dwbbuf(weight + 3346, dwb_buf_A, inx+1);
            dw_conv_3x3(afm_3, afm_4, wBUF3x3_B, dwb_buf_B, 6);
            pw_conv_group(afm_4, rfm, wBUF1x1_B, 3);
        }
    }
    F3_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_2, pwb_buf[outx], 5, false);
        sigmoid_by_point(afm_2, afm_1);
        export_fm_full(fm + 30337, afm_1, outx);
    }

//E3, 48*48*242 -> 48*48*96
    load_pwbbuf(weight + 4992, pwb_buf, 3);
    load_fm_full(fm + 147792, afm_1, 0);
    load_w3x3(weight + 4136, wBUF3x3_A, 0);
    load_w1x1(weight + 4224, wBUF1x1_A, 0, 3);
    load_dwbbuf(weight + 4208, dwb_buf_A, 0);
    E3_I:for (short inx = 0; inx < 8; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 147792, afm_3, inx+1);
            load_w3x3(weight + 4136, wBUF3x3_B, inx+1);
            load_w1x1(weight + 4224, wBUF1x1_B, inx+1, 3);
            load_dwbbuf(weight + 4208, dwb_buf_B, inx+1);
            dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 6);
            pw_conv_group(afm_2, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 147792, afm_1, inx+1);
            load_w3x3(weight + 4136, wBUF3x3_A, inx+1);
            load_w1x1(weight + 4224, wBUF1x1_A, inx+1, 3);
            load_dwbbuf(weight + 4208, dwb_buf_A, inx+1);
            dw_conv_3x3(afm_3, afm_4, wBUF3x3_B, dwb_buf_B, 6);
            pw_conv_group(afm_4, rfm, wBUF1x1_B, 3);
        }
    }
    E3_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_2, pwb_buf[outx], 5, false);
        sigmoid_by_point(afm_2, afm_1);
        load_fm_full(fm + 147792, afm_3, outx);
        res_mul(afm_1, afm_3, afm_2, 6);
        export_fm_full(fm + 166273, afm_2, outx);
    }

//G3, 48*48*242 -> 48*48*96
    load_pwbbuf(weight + 5854, pwb_buf, 3);
    load_fm_full(fm + 154704, afm_1, 0);
    load_w3x3(weight + 4998, wBUF3x3_A, 0);
    load_w1x1(weight + 5086, wBUF1x1_A, 0, 3);
    load_dwbbuf(weight + 5070, dwb_buf_A, 0);
    G3_I:for (short inx = 0; inx < 8; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 154704, afm_3, inx+1);
            load_w3x3(weight + 4998, wBUF3x3_B, inx+1);
            load_w1x1(weight + 5086, wBUF1x1_B, inx+1, 3);
            load_dwbbuf(weight + 5070, dwb_buf_B, inx+1);
            dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 6);
            pw_conv_group(afm_2, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 154704, afm_1, inx+1);
            load_w3x3(weight + 4998, wBUF3x3_A, inx+1);
            load_w1x1(weight + 5086, wBUF1x1_A, inx+1, 3);
            load_dwbbuf(weight + 5070, dwb_buf_A, inx+1);
            dw_conv_3x3(afm_3, afm_4, wBUF3x3_B, dwb_buf_B, 6);
            pw_conv_group(afm_4, rfm, wBUF1x1_B, 3);
        }
    }
    G3_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_2, pwb_buf[outx], 7, false);
        tanh_by_point(afm_2, afm_1);
        load_fm_full(fm + 147792, afm_3, outx);
        res_add_sub(afm_1, afm_3, afm_2, false, false);
        load_fm_full(fm + 30288, afm_1, outx);
        res_mul(afm_1, afm_2, afm_4, 6);
        res_add_sub(afm_4, afm_3, afm_1, true, false);
        export_fm_full(fm + 147841, afm_1, outx);
    }

//H3, 48*48*96 -> 48*48*96
    load_pwbbuf(weight + 6181, pwb_buf, 3);
    load_fm_full(fm + 147792, afm_1, 0);
    load_w3x3(weight + 5860, wBUF3x3_A, 0);
    load_w1x1(weight + 5893, wBUF1x1_A, 0, 3);
    load_dwbbuf(weight + 5887, dwb_buf_A, 0);
    H3_I:for (short inx = 0; inx < 3; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 147792, afm_3, inx+1);
            load_w3x3(weight + 5860, wBUF3x3_B, inx+1);
            load_w1x1(weight + 5893, wBUF1x1_B, inx+1, 3);
            load_dwbbuf(weight + 5887, dwb_buf_B, inx+1);
            dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 6);
            pw_conv_group(afm_2, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 147792, afm_1, inx+1);
            load_w3x3(weight + 5860, wBUF3x3_A, inx+1);
            load_w1x1(weight + 5893, wBUF1x1_A, inx+1, 3);
            load_dwbbuf(weight + 5887, dwb_buf_A, inx+1);
            dw_conv_3x3(afm_3, afm_4, wBUF3x3_B, dwb_buf_B, 6);
            pw_conv_group(afm_4, rfm, wBUF1x1_B, 3);
        }
    }
    H3_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_2, pwb_buf[outx], 8, true);
        export_fm_full(fm + 37249, afm_2, outx);
    }

//I, 48*48*96 -> 48*48*2
    load_pwbbuf(weight + 6283, pwb_buf, 1);
    load_fm_full(fm + 37200, afm_1, 0);
    load_w1x1(weight + 6187, wBUF1x1_A, 0, 1);
    I_I:for (short inx = 0; inx < 3; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 37200, afm_3, inx+1);
            load_w1x1(weight + 6187, wBUF1x1_B, inx+1, 1);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
        }
        else {
            load_fm_full(fm + 37200, afm_1, inx+1);
            load_w1x1(weight + 6187, wBUF1x1_A, inx+1, 1);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 1);
        }
    }
    I_O:for (short outx = 0; outx < 1; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, false);
        export_flow(afm_1, flow);
    }
}    

    export_interpolate_RGBA_flow(IMAGE_OUT, flow);

//CA3, 384*384*3 -> 192*192*3
    load_w3x3(weight + 867, wBUF3x3_A, 0);
    load_dwbbuf(weight + 876, dwb_buf_A, 0);
    CA3_R:for (short rowx = 0; rowx < 8; rowx++) {
        load_fm_IMAGE(IMAGE, afm_1, rowx, 0);
        for (short colx = 0; colx < 8; colx++) {
            if(colx % 2 == 0) {
                load_fm_IMAGE(IMAGE, afm_3, rowx, colx+1);
                dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 9);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 385, afm_2, rowx, colx, outx, 192, 192);
                }
            }
            else {
                load_fm_IMAGE(IMAGE, afm_1, rowx, colx+1);
                dw_conv_3x3(afm_3, afm_4, wBUF3x3_A, dwb_buf_A, 9);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 385, afm_4, rowx, colx, outx, 192, 192);
                }
            }
        }
    }

//CA1, 192*192*3 -> 192*192*32
    load_w1x1(weight + 878, wBUF1x1_A, 0, 1);
    load_pwbbuf(weight + 910, pwb_buf, 1);
    CA1_R:for (short rowx = 0; rowx < 4; rowx++) {
        load_fm_tile(fm + 192, afm_1, rowx, 0, 0, 192, 192);
        for (short colx = 0; colx < 4; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 192, afm_3, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 6, true);
                    export_fm_tile(fm + 37249, afm_1, rowx, colx, outx, 192, 192);
                }
            }
            else {
                load_fm_tile(fm + 192, afm_1, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 6, true);
                    export_fm_tile(fm + 37249, afm_3, rowx, colx, outx, 192, 192);
                }
            }
        }
    }

//CE, 192*192*32 -> 192*192*16
    load_w1x1(weight + 912, wBUF1x1_A, 0, 1);
    load_pwbbuf(weight + 944, pwb_buf, 1);
    CE_R:for (short rowx = 0; rowx < 4; rowx++) {
        load_fm_tile(fm + 37056, afm_1, rowx, 0, 0, 192, 192);
        for (short colx = 0; colx < 4; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 37056, afm_3, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 8, true);
                    export_fm_tile(fm + 74113, afm_1, rowx, colx, outx, 192, 192);
                }
            }
            else {
                load_fm_tile(fm + 37056, afm_1, rowx, colx+1, 0, 192, 192);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 8, true);
                    export_fm_tile(fm + 74113, afm_3, rowx, colx, outx, 192, 192);
                }
            }
        }
    }

//CF3, 192*192*16 -> 96*96*16
    load_w3x3(weight + 946, wBUF3x3_A, 0);
    load_dwbbuf(weight + 955, dwb_buf_A, 0);
    CF3_R:for (short rowx = 0; rowx < 4; rowx++) {
        load_fm_tile(fm + 73920, afm_1, rowx, 0, 0, 192, 192);
        for (short colx = 0; colx < 4; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 73920, afm_3, rowx, colx+1, 0, 192, 192);
                dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 5);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 110977, afm_2, rowx, colx, outx, 96, 96);
                }
            }
            else {
                load_fm_tile(fm + 73920, afm_1, rowx, colx+1, 0, 192, 192);
                dw_conv_3x3(afm_3, afm_4, wBUF3x3_A, dwb_buf_A, 5);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 110977, afm_4, rowx, colx, outx, 96, 96);
                }
            }
        }
    }

//CF1, 96*96*16 -> 96*96*16
    load_w1x1(weight + 957, wBUF1x1_A, 0, 1);
    load_pwbbuf(weight + 989, pwb_buf, 1);
    CF1_R:for (short rowx = 0; rowx < 2; rowx++) {
        load_fm_tile(fm + 110880, afm_1, rowx, 0, 0, 96, 96);
        for (short colx = 0; colx < 2; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 110880, afm_3, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 5, true);
                    export_fm_tile(fm + 74113, afm_1, rowx, colx, outx, 96, 96);
                }
            }
            else {
                load_fm_tile(fm + 110880, afm_1, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 1);
                for (short outx = 0; outx < 1; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 5, true);
                    export_fm_tile(fm + 74113, afm_3, rowx, colx, outx, 96, 96);
                }
            }
        }
    }

//CG, 96*96*16 -> 96*96*64
    load_w1x1(weight + 991, wBUF1x1_A, 0, 2);
    load_pwbbuf(weight + 1055, pwb_buf, 2);
    CG_R:for (short rowx = 0; rowx < 2; rowx++) {
        load_fm_tile(fm + 74016, afm_1, rowx, 0, 0, 96, 96);
        for (short colx = 0; colx < 2; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 74016, afm_3, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
                for (short outx = 0; outx < 2; outx++) {
                    activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 83329, afm_1, rowx, colx, outx, 96, 96);
                }
            }
            else {
                load_fm_tile(fm + 74016, afm_1, rowx, colx+1, 0, 96, 96);
                pw_conv_group(afm_3, rfm, wBUF1x1_A, 2);
                for (short outx = 0; outx < 2; outx++) {
                    activation(rfm[outx], afm_3, pwb_buf[outx], 7, true);
                    export_fm_tile(fm + 83329, afm_3, rowx, colx, outx, 96, 96);
                }
            }
        }
    }

//CP1, 192*192*32 -> 96*96*64
    load_pwbbuf(weight + 1123, pwb_buf, 2);
    CP1_R:for (short rowx = 0; rowx < 2; rowx++) {
        for (short colx = 0; colx < 2; colx++) {
            for (short inx = 0; inx < 1; inx++) {
                load_fm_s2(fm + 37249, afm_1, rowx, colx, inx, 192, 192);
                load_w1x1(weight + 1059, wBUF1x1_A, inx, 2);
                pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
            }
            for (short outx = 0; outx < 2; outx++) {
                activation(rfm[outx], afm_1, pwb_buf[outx], 7, false);
                load_fm_tile(fm + 83232, afm_2, rowx, colx, outx, 96, 96);
                res_add_sub(afm_1, afm_2, afm_3, true, true);
                export_fm_tile(fm + 101761, afm_3, rowx, colx, outx, 96, 96);
            }
        }
    }

//CH, 96*96*64 -> 96*96*24
    load_pwbbuf(weight + 1191, pwb_buf, 1);
    CH_R:for (short rowx = 0; rowx < 2; rowx++) {
        for (short colx = 0; colx < 2; colx++) {
            load_fm_tile(fm + 101664, afm_1, rowx, colx, 0, 96, 96);
            load_w1x1(weight + 1127, wBUF1x1_A, 0, 1);
            CH_I:for (short inx = 0; inx < 2; inx++) {
                if(inx % 2 == 0) {
                    load_fm_tile(fm + 101664, afm_3, rowx, colx, inx+1, 96, 96);
                    load_w1x1(weight + 1127, wBUF1x1_B, inx+1, 1);
                    pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
                }
                else {
                    load_fm_tile(fm + 101664, afm_1, rowx, colx, inx+1, 96, 96);
                    load_w1x1(weight + 1127, wBUF1x1_A, inx+1, 1);
                    pw_conv_group(afm_3, rfm, wBUF1x1_B, 1);
                }
            }
            CH_O:for (short outx = 0; outx < 1; outx++) {
                activation(rfm[outx], afm_1, pwb_buf[outx], 6, true);
                export_fm_tile(fm + 385, afm_1, rowx, colx, outx, 96, 96);
            }
        }
    }

//CI3, 96*96*24 -> 48*48*24
    load_w3x3(weight + 1193, wBUF3x3_A, 0);
    load_dwbbuf(weight + 1202, dwb_buf_A, 0);
    CI3_R:for (short rowx = 0; rowx < 2; rowx++) {
        load_fm_tile(fm + 288, afm_1, rowx, 0, 0, 96, 96);
        for (short colx = 0; colx < 2; colx++) {
            if(colx % 2 == 0) {
                load_fm_tile(fm + 288, afm_3, rowx, colx+1, 0, 96, 96);
                dw_conv_3x3(afm_1, afm_2, wBUF3x3_A, dwb_buf_A, 7);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 9601, afm_2, rowx, colx, outx, 48, 48);
                }
            }
            else {
                load_fm_tile(fm + 288, afm_1, rowx, colx+1, 0, 96, 96);
                dw_conv_3x3(afm_3, afm_4, wBUF3x3_A, dwb_buf_A, 7);
                for (short outx = 0; outx < 1; outx++) {
                    export_fm_s2(fm + 9601, afm_4, rowx, colx, outx, 48, 48);
                }
            }
        }
    }

//CI1, 48*48*24 -> 48*48*24
    load_pwbbuf(weight + 1236, pwb_buf, 1);
    load_fm_full(fm + 9552, afm_1, 0);
    load_w1x1(weight + 1204, wBUF1x1_A, 0, 1);
    pw_conv_group(afm_1, rfm, wBUF1x1_A, 1);
    CI1_O:for (short outx = 0; outx < 1; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 6, true);
        export_fm_full(fm + 385, afm_1, outx);
    }

//CJ, 48*48*24 -> 48*48*96
    load_pwbbuf(weight + 1334, pwb_buf, 3);
    load_fm_full(fm + 336, afm_1, 0);
    load_w1x1(weight + 1238, wBUF1x1_A, 0, 3);
    pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
    CJ_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, true);
        export_fm_full(fm + 2689, afm_1, outx);
    }

//CP2, 96*96*64 -> 48*48*96
    load_pwbbuf(weight + 1532, pwb_buf, 3);
    load_fm_s2(fm + 101761, afm_1, 0, 0, 0, 96, 96);
    load_w1x1(weight + 1340, wBUF1x1_A, 0, 3);
    CP2_I:for (short inx = 0; inx < 2; inx++) {
        if(inx % 2 == 0) {
            load_fm_s2(fm + 101761, afm_3, 0, 0, inx+1, 96, 96);
            load_w1x1(weight + 1340, wBUF1x1_B, inx+1, 3);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_s2(fm + 101761, afm_1, 0, 0, inx+1, 96, 96);
            load_w1x1(weight + 1340, wBUF1x1_A, inx+1, 3);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 3);
        }
    }
    CP2_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 7, false);
        load_fm_full(fm + 2640, afm_2, outx);
        res_add_sub(afm_1, afm_2, afm_3, true, true);
        export_fm_full(fm + 9601, afm_3, outx);
    }

//CK1, 48*48*96 -> 48*48*96
    load_pwbbuf(weight + 1826, pwb_buf, 3);
    load_fm_full(fm + 9552, afm_1, 0);
    load_w1x1(weight + 1538, wBUF1x1_A, 0, 3);
    CK1_I:for (short inx = 0; inx < 3; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 9552, afm_3, inx+1);
            load_w1x1(weight + 1538, wBUF1x1_B, inx+1, 3);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
        }
        else {
            load_fm_full(fm + 9552, afm_1, inx+1);
            load_w1x1(weight + 1538, wBUF1x1_A, inx+1, 3);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 3);
        }
    }
    CK1_O:for (short outx = 0; outx < 3; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 9, false);
        tanh_by_point(afm_1, afm_2);
        export_fm_full(fm + 147841, afm_2, outx);
    }

//CK2, 48*48*96 -> 48*48*64
    load_pwbbuf(weight + 2024, pwb_buf, 2);
    load_fm_full(fm + 9552, afm_1, 0);
    load_w1x1(weight + 1832, wBUF1x1_A, 0, 2);
    CK2_I:for (short inx = 0; inx < 3; inx++) {
        if(inx % 2 == 0) {
            load_fm_full(fm + 9552, afm_3, inx+1);
            load_w1x1(weight + 1832, wBUF1x1_B, inx+1, 2);
            pw_conv_group(afm_1, rfm, wBUF1x1_A, 2);
        }
        else {
            load_fm_full(fm + 9552, afm_1, inx+1);
            load_w1x1(weight + 1832, wBUF1x1_A, inx+1, 2);
            pw_conv_group(afm_3, rfm, wBUF1x1_B, 2);
        }
    }
    CK2_O:for (short outx = 0; outx < 2; outx++) {
        activation(rfm[outx], afm_1, pwb_buf[outx], 9, true);
        export_fm_full(fm + 154753, afm_1, outx);
    }


}
