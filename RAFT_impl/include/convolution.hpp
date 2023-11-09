#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "common_datatype.hpp"

void dw_conv_3x3(const hls::vector<ADT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50], const hls::vector<WDT, 32> wBUF3x3[3][3], const BDT b_buf[32], SDT scale);
void pw_conv_1x1(hls::vector<ADT, 32> ifm[50][50], hls::vector<RDT, 32> ofm[50][50], hls::vector<WDT, 32> wBUF1x1[32]);
void activation(hls::vector<RDT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50], BDT bbuf[32], SDT scale, bool relu);
void res_add_sub(hls::vector<ADT, 32> ifm1[50][50], hls::vector<ADT, 32> ifm2[50][50], hls::vector<ADT, 32> ofm[50][50], bool add_flag, bool relu);
void res_mul(hls::vector<ADT, 32> ifm1[50][50], hls::vector<ADT, 32> ifm2[50][50], hls::vector<ADT, 32> ofm[50][50], short scale);
void sigmoid_by_point(hls::vector<ADT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50]);
void tanh_by_point(hls::vector<ADT, 32> ifm[50][50], hls::vector<ADT, 32> ofm[50][50]);

void dw_conv_group(ADT32* fm, WDT32* weight, int fm_addr, int w3_addr, int w1_addr, int dwb_addr, short rowx, short colx, short chx, short dims,
                ADT ifm[32][50][50], ADT ofm[32][50][50], WDT wbuf3x3[32][3][3], WDT wbuf1x1[3][32][32], BDT dwb_buf[32],
                bool full, bool tile, bool loaddw, bool loadpw);
void pw_conv_group(hls::vector<ADT, 32> ifm[50][50], hls::vector<RDT, 32> rfm[3][50][50], hls::vector<WDT, 32> wBUF1x1[3][32], short chx);



void load_fm_IMAGE(ADT4 *ifm_ddr, hls::vector<ADT, 32> ifm[50][50], short rowx, short colx);
void load_fm_tile(ADT32 *ifm_ddr, int32_t addr, hls::vector<ADT, 32> ifm[50][50], short rowx, short colx, short chx, short width, short height);
void load_fm_s2(ADT32 *ifm_ddr, int32_t addr, hls::vector<ADT, 32> ifm[50][50], short rowx, short colx, short chx, short width, short height);
void load_fm_full(ADT32 *ifm_ddr, int32_t addr, hls::vector<ADT, 32> ifm[50][50], short chx);
void load_fm_flow(FDT flow[2304], hls::vector<ADT, 32> ifm[50][50]);
void load_w3x3(WDT32 *weight_ddr, int32_t addr, hls::vector<WDT, 32> wBUF3x3[3][3], short inx);
void load_w1x1(WDT32 *weight_ddr, int32_t addr, hls::vector<WDT, 32> wBUF1x1[3][32], short inx, short outPart);
void load_dwbbuf(WDT32 *weight_ddr, int32_t addr, BDT bbuf[32], short inx);
void load_pwbbuf(WDT32 *weight_ddr, int32_t addr, BDT bbuf[3][32], short outPart);


void out_with_flow(hls::vector<ADT, 32> ifm[50][50], FDT flow[2304], short chx);
void export_fm_tile(ADT32 *ofm_ddr, int32_t addr, hls::vector<ADT, 32> ofm[50][50], short rowx, short colx, short chx, short width, short height);
void export_fm_s2(ADT32 *ofm_ddr, int32_t addr, hls::vector<ADT, 32> ofm[50][50], short rowx, short colx, short chx, short width, short height);
void export_fm_full(ADT32 *ofm_ddr, int32_t addr, hls::vector<ADT, 32> ofm[50][50], short chx);
void export_fm_full_withact(ADT32 *ofm_ddr, int32_t addr, hls::vector<RDT, 32> ofm[3][50][50], short chx, BDT bbuf[3][32], SDT scale, bool relu);
void export_flow(hls::vector<ADT, 32> ofm[50][50], FDT flow[2304], bool update[2304]);


#endif