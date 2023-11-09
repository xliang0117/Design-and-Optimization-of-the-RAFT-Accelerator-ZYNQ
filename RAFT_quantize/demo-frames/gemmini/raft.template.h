#ifndef MOBILENET_PARAMETERS_H
#define MOBILENET_PARAMETERS_H

#include <include/gemmini_params.h>
#include <stdbool.h>

void grid_sample(elem_t *flow, elem_t *corr_whole, elem_t *sample_res, acc_scale_t flow_scale, acc_scale_t sample_res_scale, int in_dim, int neighbourhood, int corr_layer);

//generate code here

#endif
