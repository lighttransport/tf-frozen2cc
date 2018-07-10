#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//#include "mnist-network.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wconversion"
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#if __has_warning("-Wcomma")
#pragma clang diagnostic ignored "-Wcomma"
#endif
#if __has_warning("-Wcast-qual")
#pragma clang diagnostic ignored "-Wcast-qual"
#endif
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

static bool IsBigEndian(void) {
    union {
        unsigned int i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}

static void swap4(uint32_t *val) {
    if (!IsBigEndian()) {
        uint32_t tmp = *val;
        uint8_t *dst = reinterpret_cast<uint8_t *>(val);
        uint8_t *src = reinterpret_cast<uint8_t *>(&tmp);

        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }
}

static bool LoadCIFAR10Data(const std::string &image_filename,
                            const std::string &label_filename,
                            std::vector<uint8_t> *images,
                            std::vector<std::string> *labels) {
    // image
    {
        std::ifstream ifs(image_filename.c_str());
        if (!ifs) {
            std::cerr << "Failed to load image file : " << image_filename
                      << std::endl;
            return false;
        }

        images->resize(10000 * (32 * 32 * 3));

        for (size_t i = 0; i < 10000; i++) {
          uint8_t label;

          ifs.read(reinterpret_cast<char *>(&label), 1);
          assert((label >= 0) && (label < 10));
          
          ifs.read(reinterpret_cast<char *>(&images->at(i * (32 * 32 * 3))), 32*32*3);
        }
    }

    // label
    {
        std::ifstream ifs(label_filename.c_str());
        if (!ifs) {
            std::cerr << "Failed to load label file : " << label_filename
                      << std::endl;
            return false;
        }

        for (size_t i = 0; i < 10; i++) {
          std::string name;
          ifs >> name;
          std::cout << name << std::endl;
          labels->push_back(name);
        }
    }

    return true;
}

static void ConvertImage(const uint8_t *src, const float scale,
                         const float offset, std::vector<float> *dst) {
    // Assume memory is already allocated for `dst`
    // byte -> float conversion.
    for (size_t i = 0; i < 28 * 28; i++) {
        (*dst)[i] = scale * float(src[i]) + offset;
    }
}

#if 0
static void ShowProbability(std::vector<float> &output)
{
  std::cout << "result = " << output[0] << ", "
                           << output[1] << ", "
                           << output[2] << ", "
                           << output[3] << ", "
                           << output[4] << ", "
                           << output[5] << ", "
                           << output[6] << ", "
                           << output[7] << ", "
                           << output[8] << ", "
                           << output[9] << std::endl;
}
#endif

static void SaveImagePNG(uint8_t *data, int width, int height, const std::string &filename)
{
  // RRR...GGG...BBB -> RGBRGBRGB...
  std::vector<uint8_t> buf(size_t(width * height * 3));
  for (size_t y = 0; y < size_t(height); y++) {
    for (size_t x = 0; x < size_t(width); x++) {
      buf[3 * (y * size_t(width) + x) + 0] = data[size_t(0 * width * height) + y * size_t(width) + x];
      buf[3 * (y * size_t(width) + x) + 1] = data[size_t(1 * width * height) + y * size_t(width) + x];
      buf[3 * (y * size_t(width) + x) + 2] = data[size_t(2 * width * height) + y * size_t(width) + x];
    }
  }

  int ret = stbi_write_png(filename.c_str(), width, height, /* comp*/3, reinterpret_cast<const void *>(buf.data()), /* stride */3*width);
  if (ret == 0) {
    std::cerr << "Faile to write image" << std::endl;
  } 
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    std::vector<uint8_t> images;  // 10000x(32x32x3)
    std::vector<std::string> labels;      // 10
    if (!LoadCIFAR10Data("cifar-10-batches-bin/test_batch.bin",
                         "cifar-10-batches-bin/batches.meta.txt",
                       &images, &labels)) {
        return EXIT_FAILURE;
    }

    // // DBG
    // for (size_t i = 0; i < 100; i++) {
    //   char fname[1024];
    //   sprintf(fname, "image-%04d.png", int(i));
    //   SaveImagePNG(images.data() + i * 32 * 32 * 3, 32, 32, fname);
    // }

#if 0
    lainnie::Buffer buffer;
    lainnie::NetworkInit(&buffer);

    // std::cout << buffer.input.size() << std::endl;

    assert(buffer.input.size() == 28 * 28);


    // // DBG
    // for (size_t i = 0; i < 10; i++) {
    //     std::cout << "b[" << i << "] = " << buffer.constant_b[i] << std::endl;
    // }

    // Run inference
    int correct = 0;
    for (size_t i = 0; i < 10000; i++) {
        ConvertImage(images.data() + i * 28 * 28, 1.0f / 255.0f, 0.0f,
                     &(buffer.input));
        lainnie::NetworkForwardExecute(&buffer);

        float max_value = buffer.output[0];
        int infered_label_index = 0;
        for (size_t k = 1; k < 10; k++) {
            if (max_value < buffer.output[k]) {
                max_value = buffer.output[k];
                infered_label_index = int(k);
            }
        }

        if (labels[i] == infered_label_index) {
            correct++;
        }

        // std::cout << "label = " << labels[i] << std::endl;
        // ShowResult(buffer.output);
    }

    {
        double accuracy = 100.0 * double(correct) / 10000.0;
        std::cout << "Accuracy = " << accuracy << " %" << std::endl;
    }

#endif
    return EXIT_SUCCESS;
}
