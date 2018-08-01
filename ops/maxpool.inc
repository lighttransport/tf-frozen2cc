// Assume padding = SAME
static void MaxPool_NHWC(const std::vector<float> &input, const int input_shape[4], const int filter_shape[4], const int strides[4], const int ksize[4], std::vector<float> *output)
{
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

  for (int i = 0; i < channels; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        float maxvalue = -std::numeric_limits<float>::infinity();
        for (int q = 0; q < filter_channels; q++) {
          for (int dj = -filter_height / 2; dj <= filter_height / 2;
               dj++) {  // height
            int fy = std::min(std::max(j + dj, 0), int(height));
            for (int di = -filter_width / 2; di <= filter_width;
                 di++) {  // width

              // clamp
              int fx = std::min(std::max(k + di, 0), int(width));

              size_t src_idx = filter_channels * (fy * width + fx) + q;

              maxvalue = std::max(input[src_idx], maxvalue);
            }
          }
        }

        size_t out_idx = n * (channels * width * height + j * width) + k;
        (*output)[out_idx] = maxvalue;
      }
    }
  }

}
