ucodeDict = {
    "A":'''

{{
    //2304x64 * 64x2304 = 2034*2304, can be see as input channels 64, output channels 2304
    load_fm_full(fm+{E-fmap2}-48-1, afm_1, 0);  //fnet2 channel part1
    load_fm_full(fm+{E-fmap2}-48-1, afm_2, 1);  //fnet2 channel part2
    CORR_GEN:for (short group = 0; group < 24; group++)
    {{
        load_feature_vector(fm+{E-fmap1}, wBUF1x1_A, group);        //fnet1
        load_feature_vector(fm+{E-fmap1}+2304, wBUF1x1_B, group);   //fnet1
        for (short outx = 0; outx < 3; outx++) {{
            pw_conv_1x1(afm_1, rfm[outx], wBUF1x1_A[outx]);
            pw_conv_1x1(afm_2, rfm[outx], wBUF1x1_B[outx]);
            export_corr(fm+{Corr1}, rfm[outx], outx, group);
        }}
    }}
    CORR_PULL:for (short layerNum = 1; layerNum < 4; layerNum++)
    {{
        corr_pool(fm+{Corr1}, layerNum);
    }}
}}
    ''',

    "B":'''

    CORR_INDEX:for (short layerNum = 0; layerNum < 4; layerNum++)
    {{
        grid_sample(corr, fm+{Corr1}, flow, layerNum);
    }}

    ''',
    "C":'''

{{
    //2304x64 * 64x2304 = 2034*2304, can be see as input channels 64, output channels 2304
    load_fm_full(fm+{E-fmap2}-48-1, afm_1, 0);  //fnet2 channel part1
    load_fm_full(fm+{E-fmap2}-48-1, afm_2, 1);  //fnet2 channel part2
    CORR_GEN:for (short group = 0; group < 24; group++)
    {{
        load_feature_vector(fm+{E-fmap1}, wBUF1x1_A, group);        //fnet1
        load_feature_vector(fm+{E-fmap1}+2304, wBUF1x1_B, group);   //fnet1
        pw_conv_group(afm_1, rfm, wBUF1x1_A, 3);
        pw_conv_group(afm_2, rfm, wBUF1x1_B, 3);
        for (short outx = 0; outx < 3; outx++) {{
            export_corr(fm+{Corr1}, rfm[outx], outx, group);
        }}
    }}
    CORR_PULL:for (short layerNum = 1; layerNum < 4; layerNum++)
    {{
        corr_pool(fm+{Corr1}, layerNum);
    }}
}}
    ''',
}