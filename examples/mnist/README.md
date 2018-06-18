# MNIST example

## Procedure

Generate MNIST C++ code using `convert.sh` script.

```
$ ./convert.sh
```

Please download MNIST `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz` from http://yann.lecun.com/exdb/mnist/ and gunzip it.

Compie C++ code. `Makefile` is provided.

```
$ make
```

Run `mnist`

```
$ ./mnist
```
