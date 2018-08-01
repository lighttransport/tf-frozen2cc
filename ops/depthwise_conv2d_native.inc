
// https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/depthwise-conv2d-native

// for k in 0..in_channels-1
//   for q in 0..channel_multiplier-1
//     output[b, i, j, k * channel_multiplier + q] =
//       sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
//                         filter[di, dj, k, q]

void DepthwiseConv2DNative_NHWC(const std::vector<float> &input,
                                const int input_shape[4], const int strides[4],
                                const std::vector<float> &filter,
                                const int filter_shape[4],
                                std::vector<float> *output) {
  // input [batch, in_height, in_width, in_channels]
  // filter [filter_height, filter_width, in_channels, channel_multiplier]

  const int in_height = input_shape[1];
  const int in_width = input_shape[2];
  const int in_channels = input_shape[3];

  const int filter_height = filter_shape[0];
  const int filter_width = filter_shape[1];
  const int channel_multiplier = filter_shape[3];

  for (size_t k = 0; k < in_channels; k++) {
    for (size_t q = 0; q < channel_multiplier; q++) {
      for (size_t i = 0; i < in_height; i++) {
        for (size_t j = 0; j < in_width; j++) {
          float sum = 0.0f;

          for (int di = -filter_height / 2, fy = 0; di <= filter_height / 2;
               di++, fy++) {  // height
            for (int dj = -filter_width / 2, fx = 0; dj <= filter_width / 2;
                 dj++, fx++) {  // width

              if (((i + di) < 0) || ((i + di) >= in_width) || ((j + dj) < 0) ||
                  ((j + dj) >= in_height)) {
                // zero padding.
                continue;
              }

              size_t src_idx = in_channels * (fy * in_width + fx) + q;
              size_t filter_idx =
                  in_channels * (fy * filter_width + fx) + q;

              sum += input[src_idx] * filter[filter_idx];
            }
          }
        }
      }
    }
  }
}
