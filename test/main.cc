#include "acutest.h"

#include <cstdio>
#include <cstdlib>

#include <vector>
#include <algorithm>

#include "../ops/conv2d.inc"
#include "../ops/lrn.inc"
#include "../ops/maxpool.inc"

void test_conv2d(void)
{
  int a = 2;
  TEST_CHECK(a == 1);
}

TEST_LIST = {
    { "conv2d",     test_conv2d },
    { NULL, NULL }
};

#if 0
int main(int argc, char **argv)
{
  return EXIT_SUCCESS;
}
#endif
