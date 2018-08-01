static void LRN(std::vector<float> &input,  const int input_shape[4], const int depth_radius, const float bias, const float alpha, const float beta, std::vector<float> *output) {

    output->resize(input.size());

    for (size_t n = 0; n < input_shape[0]; n++) {
        for (size_t h = 0; h < input_shape[2]; h++) {
            for (size_t w = 0; w < input_shape[1]; w++) {
                for (size_t c = 0; c < input_shape[3]; c++) {

                    double sqr_sum = 0.0;

                    for (int k = c - depth_radius; k < c + depth_radius + 1; k++) {
                        size_t src_idx = size_t(std::min(std::max(0, k), input_shape[3]));
                        double s = input[src_idx];
                        s = s * s; // sqr
                        sqr_sum += s;
                    }

                    size_t idx = c * input_shape[0] * input_shape[1] * input_shape[2] + w * input_shape[0] * input_shape[1] + h * input_shape[0];
                    (*output)[idx] = input[idx] / std::pow((bias + alpha * sqr_sum), beta);
                }
            }
        }
    }
}

