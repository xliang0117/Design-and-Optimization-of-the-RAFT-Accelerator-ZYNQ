#ifndef COMMON_DATATYPE_H
#define COMMON_DATATYPE_H

#include "ap_int.h"
#include "hls_vector.h"
#include "hls_math.h"
#include "hls_stream.h"

typedef ap_int<8>  ADT;
typedef ap_int<19> RDT;
typedef ap_int<16> BDT;
typedef ap_int<8>  WDT;
typedef ap_uint<8> SDT;
typedef hls::vector<ADT, 4> ADT4;
typedef hls::vector<ADT, 8> ADT8;
typedef hls::vector<WDT, 32> WDT32;
typedef hls::vector<ADT, 32> ADT32;
typedef hls::vector<BDT, 16> BDT16;

typedef ap_int<12> CDT;     //fixed actually, with 5 frac
typedef hls::vector<ADT, 2> FDT;     //flow 



#define amin -128
#define amax 127
#define rmax 262143
#define rmin -262144

enum padding_behaviour {
    NO_PADDING = 0,
    TOP_ROW = 1,
    TOP_COL = 2,
    BOTTOM_ROW = 3,
    BOTTOM_COL = 4
};

#endif
