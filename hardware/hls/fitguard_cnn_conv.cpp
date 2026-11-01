#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include "fitguard_cnn_weights.hpp"

typedef hls::axis<ap_uint<32>, 0, 0, 0> axis32_t;
typedef ap_fixed<18,5,AP_RND,AP_SAT> act_t;
typedef ap_fixed<40,16,AP_RND,AP_SAT> acc_t;

union FloatBits { float f; unsigned int u; };

template<int T, int CI, int CO, int K>
void conv_relu(const act_t in[T][CI], act_t out[T][CO],
               const weight_t w[K][CI][CO], const weight_t b[CO]) {
    const int PAD = (K - 1) / 2;
    for (int t = 0; t < T; ++t) {
        for (int co = 0; co < CO; ++co) {
            acc_t sum = b[co];
            for (int k = 0; k < K; ++k) {
                int ti = t + k - PAD;
                if (ti >= 0 && ti < T) {
                    // Four independent reduction lanes remove the single-
                    // accumulator dependency while remaining well inside
                    // the XC7Z020 DSP/LUT budget.
                    acc_t sum0 = 0;
                    acc_t sum1 = 0;
                    acc_t sum2 = 0;
                    acc_t sum3 = 0;
                    for (int ci = 0; ci < CI; ci += 4) {
#pragma HLS PIPELINE II=1
                        sum0 += (acc_t)in[ti][ci] * (acc_t)w[k][ci][co];
                        if (ci + 1 < CI)
                            sum1 += (acc_t)in[ti][ci + 1] * (acc_t)w[k][ci + 1][co];
                        if (ci + 2 < CI)
                            sum2 += (acc_t)in[ti][ci + 2] * (acc_t)w[k][ci + 2][co];
                        if (ci + 3 < CI)
                            sum3 += (acc_t)in[ti][ci + 3] * (acc_t)w[k][ci + 3][co];
                    }
                    sum += (sum0 + sum1) + (sum2 + sum3);
                }
            }
            out[t][co] = sum > 0 ? (act_t)sum : (act_t)0;
        }
    }
}

template<int T, int C>
void pool2(const act_t in[T][C], act_t out[T/2][C]) {
    for (int t = 0; t < T/2; ++t)
        for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
            out[t][c] = in[2*t][c] > in[2*t+1][c] ? in[2*t][c] : in[2*t+1][c];
        }
}

template<int T, int C>
void gap(const act_t in[T][C], act_t out[C]) {
    for (int c = 0; c < C; ++c) {
        acc_t sum = 0;
        for (int t = 0; t < T; ++t) {
#pragma HLS PIPELINE II=1
            sum += in[t][c];
        }
        out[c] = (act_t)(sum / T);
    }
}

void fitguard_cnn_conv(hls::stream<axis32_t>& input, hls::stream<axis32_t>& output) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE ap_ctrl_none port=return
    act_t imu0[128][8], imu1[128][32], imu1p[64][32], imu2[64][64], imu2p[32][64], imu3[32][96], imu_gap[96];
    act_t vis0[64][7], vis1[64][24], vis1p[32][24], vis2[32][48], vis2p[16][48], vis3[16][64], vis_gap[64];

    axis32_t word;
    FloatBits bits;
    for (int t = 0; t < 128; ++t)
        for (int c = 0; c < 8; ++c) {
#pragma HLS PIPELINE II=1
            word = input.read(); bits.u = (unsigned int)word.data; imu0[t][c] = (act_t)bits.f;
        }
    for (int t = 0; t < 64; ++t)
        for (int c = 0; c < 7; ++c) {
#pragma HLS PIPELINE II=1
            word = input.read(); bits.u = (unsigned int)word.data; vis0[t][c] = (act_t)bits.f;
        }

    conv_relu<128,8,32,9>(imu0, imu1, imu_conv1_0, imu_conv1_1); pool2<128,32>(imu1, imu1p);
    conv_relu<64,32,64,5>(imu1p, imu2, imu_conv2_0, imu_conv2_1); pool2<64,64>(imu2, imu2p);
    conv_relu<32,64,96,3>(imu2p, imu3, imu_conv3_0, imu_conv3_1); gap<32,96>(imu3, imu_gap);

    conv_relu<64,7,24,9>(vis0, vis1, vision_conv1_0, vision_conv1_1); pool2<64,24>(vis1, vis1p);
    conv_relu<32,24,48,5>(vis1p, vis2, vision_conv2_0, vision_conv2_1); pool2<32,48>(vis2, vis2p);
    conv_relu<16,48,64,3>(vis2p, vis3, vision_conv3_0, vision_conv3_1); gap<16,64>(vis3, vis_gap);

    for (int i = 0; i < 160; ++i) {
#pragma HLS PIPELINE II=1
        float value = i < 96 ? (float)imu_gap[i] : (float)vis_gap[i-96];
        bits.f = value; axis32_t outword; outword.data = bits.u;
        outword.keep = -1; outword.strb = -1; outword.last = (i == 159);
        output.write(outword);
    }
}
