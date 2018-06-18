#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "mnist-network.h"

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

static bool LoadMNISTData(const std::string &mnist_label_filename,
                          const std::string &mnist_image_filename,
                          std::vector<uint8_t> *images,
                          std::vector<int> *labels) {
    // NOTE(LTE): Data is stored in big endian(e.g. POWER, non-Intel CPU)

    // label
    {
        std::ifstream ifs(mnist_label_filename.c_str());
        if (!ifs) {
            std::cerr << "Failed to load label file : " << mnist_label_filename
                      << std::endl;
            return false;
        }

        uint32_t magic;

        ifs.read(reinterpret_cast<char *>(&magic), 4);
        swap4(&magic);

        uint8_t data_type = *(reinterpret_cast<uint8_t *>(&magic) + 1);
        uint8_t dim = *(reinterpret_cast<uint8_t *>(&magic) + 0);

        assert(data_type == 8);
        assert(dim == 1);

        uint32_t size = 0;
        ifs.read(reinterpret_cast<char *>(&size), 4);
        swap4(&size);

        assert(size == 10000);

        std::vector<char> buf(size);
        ifs.read(buf.data(), size);

        // byte -> int
        labels->resize(size);
        for (size_t i = 0; i < size_t(size); i++) {
            (*labels)[i] = int(buf[i]);
        }
    }

    // image
    {
        std::ifstream ifs(mnist_image_filename.c_str());
        if (!ifs) {
            std::cerr << "Failed to load image file : " << mnist_image_filename
                      << std::endl;
            return false;
        }

        uint32_t magic;

        ifs.read(reinterpret_cast<char *>(&magic), 4);

        // TODO(LTE): Endian swap
        uint8_t data_type = *(reinterpret_cast<uint8_t *>(&magic) + 2);
        uint8_t dim = *(reinterpret_cast<uint8_t *>(&magic) + 3);

        assert(data_type == 8);
        assert(dim == 3);

        uint32_t size[3];
        ifs.read(reinterpret_cast<char *>(size), 4 * 3);
        swap4(&size[0]);
        swap4(&size[1]);
        swap4(&size[2]);

        assert(size[0] == 10000);
        assert(size[1] == 28);
        assert(size[2] == 28);

        size_t length = size[0] * size[1] * size[2];
        images->resize(length);
        ifs.read(reinterpret_cast<char *>(images->data()),
                 std::streamsize(length));
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

#if 0
static void SaveImagePNG(uint8_t *data, int width, int height, const std::string &filename)
{
  int ret = stbi_write_png(filename.c_str(), width, height, /* comp*/1, reinterpret_cast<const void *>(data), /* stride */width);
  if (ret == 0) {
    std::cerr << "Faile to write image" << std::endl;
  } 
}
#endif

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    std::vector<uint8_t> images;  // 28x28x10000
    std::vector<int> labels;      // 10000
    if (!LoadMNISTData("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
                       &images, &labels)) {
        return EXIT_FAILURE;
    }

    lainnie::Buffer buffer;
    lainnie::NetworkInit(&buffer);

    // std::cout << buffer.input.size() << std::endl;

    assert(buffer.input.size() == 28 * 28);

    // DBG
    // for (size_t i = 0; i < 10000; i++) {
    //   char fname[1024];
    //   sprintf(fname, "image-%04d.png", int(i));
    //   SaveImagePNG(images.data() + i * 28 * 28, 28, 28, fname);
    // }

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

    return EXIT_SUCCESS;
}
