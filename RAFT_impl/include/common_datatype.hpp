#ifndef COMMON_DATATYPE_H
#define COMMON_DATATYPE_H

#include "ap_int.h"
#include "hls_half.h"
#include "hls_math.h"
#include "hls_stream.h"

typedef ap_uint<512> HALF32;
typedef ap_uint<256> HALF16;
typedef ap_uint<128> HALF8;
typedef ap_uint<64> HALF4;
typedef ap_uint<32> HALF2;

struct fourhalf
{   
    half part[4];
};

enum padding_behaviour {
    NO_PADDING = 0,
    TOP_ROW = 1,
    TOP_COL = 2,
    BOTTOM_ROW = 3,
    BOTTOM_COL = 4
};

struct layer_weight
{
	char name[10];

    int weight3x3_addr;
    int weight1x1_addr;
    int bn_addr;
};

struct layer_fm
{
	char name[10];

    int im_addr;
    int ex_addr;
    bool relu;
};

#define OUTCHAN8 1
#define OUTCHAN16 2
#define OUTCHAN24 3

#define OUTCHAN32 1
#define OUTCHAN64 2
#define OUTCHAN96 3

#endif
