EXTRA_CXXFLAGS=-DUSE_GEMMLOWP=1 -I../third_party/gemmlowp/public/

all:
	clang++ -std=c++11 -fsanitize=address -o test -I. $(EXTRA_CXXFLAGS) main.cc
