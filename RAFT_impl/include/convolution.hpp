#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "common_datatype.hpp"

void load_pw_fm(HALF8 *ifm_ddr, half ifm[8][18][18], short width);
void load_res_fm(HALF8 *ifm_ddr, half ifm[4][32][18][18], ap_uint<16> c_id, short width);
void load_pw_stride2(HALF8 *ifm_ddr, half ifm[8][18][18], short width);
void load_dw_fm(HALF8 *ifm_ddr, half ifm[8][18][18], short row_padding, short col_padding, short width);
void load_w1x1(HALF16 *weight_ddr, half wbuf1x1[32][4][8][32], short inchannel_limit, short outchannel_limit);
void load_w3x3(HALF16 *weight_ddr, half wbuf3x3[32][8][3][3], short inchannel_limit);
void load_bnbuf(HALF16 *weight_ddr, half bnbuf[96], short part_div32);
void load_dwbbuf(HALF16 *weight_ddr, half bn_buf[32][8], short part_div8);

void dw_conv_3x3(const half ifm[8][18][18], half ofm[8][18][18], const half wbuf3x3[8][3][3], const half dw_bn[8]);
void pw_conv_1x1(half ifm[8][18][18], half ofm[32][18][18], half wbuf1x1[8][32]);
void batch_norm(half ifm[4][32][18][18], half ofm[4][32][18][18], half bn_buf[96], short part_limit);
void export_fm(HALF8 *ofm_ddr, half ofm[4][32][18][18], int part_limit, short outchannel_limit, short width, short height, bool relu);
void export_dw_shrink(HALF8 *ofm_ddr, half ofm[8][18][18], short width);
void pw_conv_group(half dw_fm[8][18][18], half pw_fm[4][32][18][18], half wbuf1x1[4][8][32], short times);

void mulmat_by_point(half ifm1[4][32][18][18], half ifm2[4][32][18][18], half ofm[4][32][18][18]);
void addmat_by_point(half ifm1[4][32][18][18], half ifm2[4][32][18][18],half ofm[4][32][18][18], short part_limit);
void minusmat_by_point(half ifm1[4][32][18][18], half ifm2[4][32][18][18],half ofm[4][32][18][18]);
void tanh_by_point(half ifm[4][32][18][18], half ofm[4][32][18][18]);

#endif