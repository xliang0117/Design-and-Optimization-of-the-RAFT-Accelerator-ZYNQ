#ifndef CORR_H
#define CORR_H

#include "common_datatype.hpp"
#include "string.h"

void corr_pool(ADT32 *fm, short layerNum);
void load_feature_vector(ADT32 *fm, hls::vector<WDT, 32> feature[3][32], short group);
void export_corr(ADT32 *fm, hls::vector<RDT, 32> rfm[50][50], short outx, short group);
void grid_sample(ADT *corr, ADT32 *fm, FDT flow[2304], int layerNum);

//experimental
void grid_sample_UPDATE(ADT32 *fm, FDT flow[2304], bool update[2304], ADT8* corr_buffer, short layerNum);
void grid_sample_ING(ADT8 *corr, FDT flow[2304], ADT8* corr_buffer, short layerNum);
void grid_predict_UPDATE(ADT32 *fm, FDT flow[2304], bool update[2304], bool des[2304], ADT16* corr_buffer, short layerNum);
void grid_predict_ING(ADT8 *corr, FDT flow[2304], bool des[2304], ADT16* corr_buffer, short layerNum);
void gs_1_stage_predict(ADT4* IMAGE1, ADT4* IMAGE2, ADT32* fm, WDT32* weight, ADT8* corr, FDT flow[2304], ADT8* corr_buffer, 
                        bool update[2304]);
void gs_2_stage_predict(ADT4* IMAGE1, ADT4* IMAGE2, ADT32* fm, WDT32* weight, ADT8* corr, FDT flow[2304], ADT16* corr_buffer, 
                        bool update[2304], bool des[2304]);

#endif
