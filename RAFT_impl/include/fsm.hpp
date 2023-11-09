#ifndef FSM_H
#define FSM_H

#include "convolution.hpp"

void conv_hw(ADT4* IMAGE, ADT32* IMAGE_OUT, ADT32* fm, WDT32* weight, ADT* corr, bool curFrame);

#endif