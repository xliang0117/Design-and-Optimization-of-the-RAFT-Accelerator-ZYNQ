#ifndef CORR_H
#define CORR_H

#include "common_datatype.hpp"
#include "string.h"

void load_corrmul_op(HALF8 *ifm_ddr, half wbuf1x1[32][4][8][32]);
void export_corrmul(HALF8 *ofm_ddr, half ofm[4][32][18][18]);
void corr_pool(HALF8 *ifm_ddr, HALF8 *ofm_ddr, short scale);
void grid_sample(half *corr, half coords[3072][2], half *corr_out, int scale);
void load_coords(HALF8 *ifm_ddr, half coords[3072][2], bool coords_init);

#endif
