regular_conv = """
    start = read_cycles();

    tiled_conv_auto(
        {name}_params.batch_size, {name}_params.in_dim, {name}_params.in_channels,
        {name}_params.out_channels, {name}_params.out_dim,
        {name}_params.stride, 1, 1, {name}_params.padding, {name}_params.kernel_size,
        false, false, false, false, false,

        (elem_t*){input}, (elem_t*){name}_w, (acc_t*){name}_b, (elem_t*){output},

        {relu}, {name}_params.output_scale,
        {name}_params.pool_size, 0, {name}_params.pool_padding,

        tiled_matmul_type);

    end = read_cycles();
    conv_cycles += end - start;
    {pt}_cycles += end - start;
    printf("{name}: %llu\\n", end-start);

"""

dw_conv = """
    start = read_cycles();

    tiled_conv_dw_auto(
        {name}_params.batch_size, {name}_params.in_dim, {name}_params.in_channels,
        {name}_params.out_dim,
        {name}_params.stride, {name}_params.padding, {name}_params.kernel_size,

        (elem_t*){input}, (elem_t*){name}_w, (acc_t*){name}_b, (elem_t*){output},

        {relu}, {name}_params.output_scale,
        {name}_params.pool_size, 0, {name}_params.pool_padding,

        tiled_matmul_type);

    end = read_cycles();
    conv_dw_cycles += end - start;
    {pt}_cycles += end - start;
    printf("{name}: %llu\\n", end-start);

"""

pw_conv = """
    start = read_cycles();

    tiled_matmul_nn_auto({name}_params.I, {name}_params.J, {name}_params.K,
        {input}, {name}_w, {name}_b, {output},
        {relu}, {name}_params.output_scale, true,
        tiled_matmul_type, check, "{name}");

    end = read_cycles();
    matmul_cycles += end - start;
    {pt}_cycles += end - start;
    printf("{name}: %llu\\n", end-start);

"""

res_add = """
    start = read_cycles();

    tiled_resadd_auto({name}_params.I, {name}_params.J,
        {name}_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        {input},
        {output},
        {output},
        {relu},
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    {pt}_cycles += end - start;
    printf("{name}: %llu\\n", end-start);

"""

matmul_template = """
    start = read_cycles();

    tiled_matmul_auto({I}, {J}, {K},                        // A[I][K] * B[K][J] + D[I][J] = C[I][J], stride means mat line skip
            (elem_t*){aArray}, (elem_t*){bArray}, {dArray}, (elem_t*){cArray},
            {aStride}, {bStride}, {J}, {J},                 
            {aScale}, {bScale}, {biasScale},
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            {aTrans}, {bTrans},
            false, false,
            0,
            tiled_matmul_type);
    end = read_cycles();
    matmul_cycles += end - start;
    {pt}_cycles += end - start;
    printf("{name}: %llu\\n", end-start);

"""

conv_config = "static const struct ConvParams {name}_params = {{.batch_size={batch_size}, .in_dim={in_dim}, .kernel_size={kernel_size}, .in_channels={in_channels}, .out_channels={out_channels}, .stride={stride}, .padding={padding}, .bias={bias}, .depthwise={depthwise}, .out_dim={out_dim}, .n_patches={n_patches}, .patch_size={patch_size}, .pool_size={pool_size}, .pool_stride={pool_stride}, .pool_padding={pool_padding}, .out_dim_pooled={out_dim_pooled}, .output_scale=(1.0 / (1 << {output_scale})), .I={I}, .J={J}, .K={K}, .res_scale=(1.0 / (1 << {res_scale}))}};"

# 0 for input_channels*kernel_size*kernel_size, 1 for output channels, 2 for name
regular_conv_array = """
static const elem_t {name}_w[{iksize}][{oc}] row_align(1);
static const acc_t {name}_b[{oc}] row_align_acc(1);
"""

# 0 for input/output channels, 1 for in b*w*h, 2 for out b*w*h,  3 for name
dw_conv_array = """
static const elem_t {name}_w[{c}][3][3] row_align(1);
static const acc_t {name}_b[{c}] row_align_acc(1);
"""

# 0 for input channels, 1 for output channels, 2 for in b*w*h, 3 for out b*w*h, 4 for name
pw_conv_array = """
static const elem_t {name}_w[{ic}][{oc}] row_align(1);
static const acc_t {name}_b[{oc}] row_align_acc(1);
"""

memory_array = """
static elem_t {memory}[{size}][{channel}] row_align(1);
"""


#usercode
usercode = {
    "code1":"""
    start = read_cycles();

    grid_sample((elem_t*)net+2304*240, (elem_t*)corr_1, (elem_t*)corr, 1.0, 1.0, 48, 3, 0);
    grid_sample((elem_t*)net+2304*240, (elem_t*)corr_2, (elem_t*)corr + 2304*49, 1.0, 1.0, 48, 3, 1);
    grid_sample((elem_t*)net+2304*240, (elem_t*)corr_3, (elem_t*)corr + 2304*98, 1.0, 1.0, 48, 3, 2);
    grid_sample((elem_t*)net+2304*240, (elem_t*)corr_4, (elem_t*)corr + 2304*147, 1.0, 1.0, 48, 3, 3);
    
    end = read_cycles();
    sample_cycles += end - start;
    printf("sample: %llu\\n", end-start);

    """,

}