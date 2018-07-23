#include "acutest.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include <random>

#include "../ops/conv2d.inc"
#include "../ops/lrn.inc"
#include "../ops/maxpool.inc"
#include "../ops/matmul.inc"

#include "../base64-encdec.inc"

// used for JSON serialization
#include "nlohmann/json.hpp"

// generate random number in the range [0.0, 1.0).
static void GenerateRandomUniformFloat(std::default_random_engine &engine, size_t n, std::vector<float> *output)
{
  std::uniform_real_distribution<> dist;

  for (size_t i = 0; i < n; i++) {
    output->push_back(dist(engine));
  }
}

static std::string SerializeTensorToJSON(
  const std::string &name,
  const std::vector<float> &data, int shape[4], int shape_dim)
{
  nlohmann::json j;

  // to base64
  std::string b64_str = base64_encode(reinterpret_cast<unsigned char const*>(data.data()), sizeof(float) * data.size());

  j["name"] = name;
  j["op"] = "Const";

  {
    nlohmann::json attr;

    {
      nlohmann::json value;

      {
        nlohmann::json dim = nlohmann::json::array();

        for (int i = 0; i < shape_dim; i++) {
          nlohmann::json o;
          o["size"] = std::to_string(shape[i]);
          dim.push_back(o);
        }
        nlohmann::json tensorShape;
        tensorShape["dim"] = dim;

      
        nlohmann::json tensor;
        tensor["tensorShape"] = tensorShape;
        tensor["tensorContent"] = b64_str;
        tensor["dtype"] = "DT_FLOAT";

        value["tensor"] = tensor;

      }

      attr["value"] = value;
      nlohmann::json dtype;
      dtype["type"] = "DT_FLOAT";
      attr["dtype"] = dtype;
    }

    j["attr"] = attr;

  }
  
  {
    std::stringstream ss;
    // pretty print JSON 
    ss << std::setw(2) << j << std::endl;
    return ss.str();
  }

}

void test_random(void)
{
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());

  std::vector<float> rnd;
  GenerateRandomUniformFloat(engine, 100, &rnd);

  for (size_t i = 0; i < 100; i++) {
    std::cout << rnd[i] << std::endl;
  }
  
}

void test_conv2d(void)
{
  int a = 2;
  TEST_CHECK(a == 1);
}

void test_serialize(void)
{
  std::vector<float> input = {1.0f, 2.0f, 2.2f, 4.0f};
  int shapes[4] = {4, 0, 0, 0};  
  int shape_dim = 1;

  std::string s = SerializeTensorToJSON("const1", input, shapes, shape_dim);

  std::cout << s << std::endl;
}

TEST_LIST = {
    { "conv2d",     test_conv2d },
    { "serialize",  test_serialize },
    { "random",  test_random },
    { NULL, NULL }
};

#if 0
int main(int argc, char **argv)
{
  return EXIT_SUCCESS;
}
#endif
