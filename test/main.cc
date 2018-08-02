#include "acutest.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "../ops/conv2d.inc"
#include "../ops/lrn.inc"
#include "../ops/matmul.inc"
#include "../ops/maxpool.inc"
#include "../ops/depthwise_conv2d_native.inc"
#include "../ops/relu6.inc"
#include "../ops/bias_add.inc"
#include "../ops/identity.inc"

#include "../base64-encdec.inc"

// used for JSON serialization
#include "nlohmann/json.hpp"

#ifdef USE_GEMMLOWP
#include "gemmlowp.h"
#endif

int CallExternalCommand(const std::string &command) {
  // LOG_DEBUG("Call external command: {}", command);

  // Check if command processor is available
  if (system(NULL)) {
    // Use `system`
    return system(command.c_str());

  } else {
    fflush(NULL);  // Ensure flushing I/O buffering.
#if defined(_WIN32)
    // TODO: Test on windows
    FILE* fp = _popen(command.c_str(), "r");
#else
    FILE *fp = popen(command.c_str(), "r");
#endif
    if (!fp) {
      std::cerr << "Failed to open pipe for the command : " << command
                << std::endl;
      throw "Failed to open pipe";
    }

    // Wait for execution
    char buf[1024];
    while (!feof(fp)) {
      fgets(buf, sizeof(buf), fp);
      // Print python log
      printf("%s", buf);
    }

// Close pipe
#if defined(_WIN32)
    int ret = _pclose(fp);
#else
    int ret = pclose(fp);
#endif

    return ret;
  }
}

// generate random number in the range [0.0, 1.0).
static void GenerateRandomUniformFloat(std::default_random_engine &engine,
                                       size_t n, std::vector<float> *output) {
  std::uniform_real_distribution<> dist;

  for (size_t i = 0; i < n; i++) {
    output->push_back(dist(engine));
  }
}

static bool SerializeTensorToJSON(const std::string &name,
                                         const std::vector<float> &data,
                                         int shape[4], int shape_dim, nlohmann::json *j) {
  // to base64
  std::string b64_str =
      base64_encode(reinterpret_cast<unsigned char const *>(data.data()),
                    sizeof(float) * data.size());

  (*j)["name"] = name;
  (*j)["op"] = "Const";

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

    (*j)["attr"] = attr;
  }

  return true;
}

static std::string SerializeTensorToJSONString(const std::string &name,
                                         const std::vector<float> &data,
                                         int shape[4], int shape_dim) {
  nlohmann::json j;
  SerializeTensorToJSON(name, data, shape, shape_dim, &j);

  {
    std::stringstream ss;
    // pretty print JSON
    ss << std::setw(2) << j << std::endl;
    return ss.str();
  }
}

void test_random(void) {
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());

  std::vector<float> rnd;
  GenerateRandomUniformFloat(engine, 100, &rnd);

  for (size_t i = 0; i < 100; i++) {
    std::cout << rnd[i] << std::endl;
  }
}

void test_conv2d(void) {
  int a = 2;
  TEST_CHECK(a == 1);
}

void test_serialize(void) {
  std::vector<float> input = {1.0f, 2.0f, 2.2f, 4.0f};
  int shapes[4] = {4, 0, 0, 0};
  int shape_dim = 1;

  std::string s = SerializeTensorToJSONString("const1", input, shapes, shape_dim);

  std::cout << s << std::endl;
}

void test_matmul(void) {

  std::vector<float> input_a = {1.0f, 2.0f, 2.2f, 4.0f};
  std::vector<float> input_b = {2.0f, 3.0f, 4.3f, 5.2f};
  std::vector<float> result = {2.0f, 3.0f, 4.3f, 5.2f};
  int shapes[4] = {4, 0, 0, 0};
  int shape_dim = 1;

  nlohmann::json j_a, j_b;
  SerializeTensorToJSON("const_a", input_a, shapes, shape_dim, &j_a);
  SerializeTensorToJSON("const_b", input_b, shapes, shape_dim, &j_b);

  nlohmann::json j_r;
  SerializeTensorToJSON("reference", result, shapes, shape_dim, &j_r);

  nlohmann::json op;

  op["name"] = "matmul";
  op["op"] = "MatMul";
  op["input"] = { "const_a", "const_b" };

  nlohmann::json attr;

  attr["T"] = {{"type", "DT_FLOAT"}};
  attr["transpose_a"] = {{"b", false}};
  attr["transpose_b"] = {{"b", false}};

  op["attr"] = attr;

  nlohmann::json node;

  node = {j_a, j_b, op, j_r};
  
  std::cout << std::setw(2) << node << std::endl;
}

TEST_LIST = {{"conv2d", test_conv2d},
             {"serialize", test_serialize},
             {"random", test_random},
             {"matmul", test_matmul},
             {NULL, NULL}};

#if 0
int main(int argc, char **argv)
{
  return EXIT_SUCCESS;
}
#endif
