// TODO(LTE): Code is super simple so we are better to embed it to C++ code generator.

void Relu6(const std::vector<float> &input, std::vector<float> *const output)
{
  output->resize(input.size());

  for (size_t i = 0; i < input.size(); i++) {
    (*output)[i] = std::min(std::max(0.0f, input[i]), 6.0f);
  }
}
