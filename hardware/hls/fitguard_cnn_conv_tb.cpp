#include <cmath>
#include <cstdio>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include "fitguard_cnn_golden.hpp"

typedef hls::axis<ap_uint<32>, 0, 0, 0> axis32_t;
union FloatBits { float f; unsigned int u; };
void fitguard_cnn_conv(hls::stream<axis32_t>& input,
                       hls::stream<axis32_t>& output);

int main() {
    float max_error = 0.0f;
    for (int n = 0; n < GOLDEN_CASES; ++n) {
        hls::stream<axis32_t> input, output;
        for (int i = 0; i < 1472; ++i) {
            FloatBits b; b.f = golden_input[n][i]; axis32_t w;
            w.data = b.u; w.keep = -1; w.strb = -1; w.last = (i == 1471); input.write(w);
        }
        fitguard_cnn_conv(input, output);
        for (int i = 0; i < 160; ++i) {
            axis32_t w = output.read(); FloatBits b; b.u = (unsigned int)w.data;
            float err = std::fabs(b.f - golden_output[n][i]); if (err > max_error) max_error = err;
            if ((i == 159) != (bool)w.last) { std::printf("TLAST mismatch case=%d i=%d\n", n, i); return 2; }
        }
    }
    std::printf("FITGUARD_CNN_CSIM_MAX_ERROR=%.9g\n", max_error);
    return max_error <= 0.003f ? 0 : 1;
}
