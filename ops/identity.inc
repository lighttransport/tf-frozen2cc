// TODO(LTE): Code is super simple so we are better to embed it to C++ code generator.

void Identity(const std::vector<float> &input, const int input_shape[4], const int input_dim, std::vector<float> *const output, int output_shape[4], int *output_dim)
{
  output->resize(input.size());

  size_t len = 1;
  for (size_t i = 0; i < input_dim; i++) {
    len *= input_shape[i]; 
  }

  memcpy(output->data(), input.data(), len * sizeof(float));

  (*output_dim) = input_dim;
  output_shape[0] = input_shape[0];
  output_shape[1] = input_shape[1];
  output_shape[2] = input_shape[2];
  output_shape[3] = input_shape[3];
}
