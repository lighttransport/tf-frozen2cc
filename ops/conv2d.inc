static void Conv2D_NHWC(const std::vector<float> &input,
                        const std::vector<float> &filter,
                        const int input_shape[4], const int filter_shape[4],
                        const int strides[4], std::vector<float> *output) {
  // input: [n, height, width, channels]
  // filter: [filter_height, filter_width, in_channels, out_channels]
  output->resize(input.size());

  size_t n = input_shape[0];
  size_t width = input_shape[1];
  size_t height = input_shape[2];
  size_t channels = input_shape[3];

  size_t filter_width = filter_shape[1];
  size_t filter_height = filter_shape[2];
  size_t filter_channels = filter_shape[3];  // in_channels

  // Assume n == 1
  // Assume padding == SAME

  for (int i = 0; i < channels; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        double sum = 0.0;
        for (int q = 0; q < filter_channels; q++) {
          for (int dj = -filter_height / 2; dj <= filter_height / 2;
               dj++) {  // height
            int fy = std::min(std::max(j + dj, 0), int(height));
            for (int di = -filter_width / 2; di <= filter_width;
                 di++) {  // width

              if (((k + di) < 0) || 
                  ((k + di) >= width) || 
                  ((j + dj) < 0) || 
                  ((j + dj) >= height)) {
                // zero padding.
                continue;
              }

              // clamp
              int fx = std::min(std::max(k + di, 0), int(width));

              size_t src_idx = filter_channels * (fy * width + fx) + q;
              size_t filter_idx =
                  filter_channels * (fy * filter_width + fx) + q;

              sum += input[src_idx] * filter[filter_idx];
            }
          }
        }

        size_t out_idx = n * (channels * width * height + j * width) + k;
        (*output)[out_idx] = sum;
      }
    }
  }
}
