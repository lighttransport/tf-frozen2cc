// TODO(LTE): Support transpose.
static void MatMul(std::vector<float> &input_a, std::vector<float> &input_b,
                   const int input_shape_a[4], const int input_dim_a,
                   const int input_shape_b[4], const int input_dim_b,
                   std::vector<float> *output) {
  // Assume dimension is 2 and `input_a` and `input_b` has valid shape.
  size_t dim0 = input_shape_a[0];
  size_t dim1 = input_shape_a[1];

  output->resize(dim1);
  for (size_t j = 0; j < dim1; j++) {
    float sum = 0.0f;
    // dot op
    for (size_t i = 0; i < dim0; i++) {
      // Assume minor_to_major is [0, 1]. i.e. y first then x.
      sum += input_a[i] * input_b[i * dim1 + j];
    }
    (*output)[j] = sum;
  }
}
