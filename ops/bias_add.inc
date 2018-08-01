// TODO(LTE): Code is super simple so we are better to embed it to C++ code generator.

void BiasAdd(const std::vector<float> &input, const float bias, std::vector<float> *const output)
{
  output->resize(input.size());

  for (size_t i = 0; i < input.size(); i++) {
    (*output)[i] = input[i] + bias;
  }
}
