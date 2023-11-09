#ifndef FSM_H
#define FSM_H

#include "convolution.hpp"

void conv_hw(ADT4* IMAGE1, ADT4* IMAGE2, ADT32* fm, WDT32* weight, ADT8* corr, ADT8* corr_buffer, FDT flow[2304]);

#endif