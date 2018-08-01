static void Reshape(const std::vector<float> &input, const int input_shape[4], const int input_dim, const int output_shape[4], const int output_dim, const std::vector<float> *output) {

  if (output_dim == 1) {
    assert(input_dim == 3);
  } else if (output_dim == 2) {
    assert(input_dim == 1);
  } else if (output_dim == 3) {
    assert(input_dim == 1);
  } else if (output_dim == 4) {
  } else {
    assert(0);
  }
}
