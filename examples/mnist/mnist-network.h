
#ifndef LAINNIE_H_
#define LAINNIE_H_

#include <vector>

namespace lainnie {


struct Buffer {
    std::vector<float> constant_b;
    std::vector<float> constant_W;
    std::vector<float> input;
    std::vector<float> MatMul;
    std::vector<float> add;
    std::vector<float> output;

};


// Initialize buffer for the network
void NetworkInit(Buffer *buffer);

// Execute network(forward pass).
bool NetworkForwardExecute(Buffer *buffer);

} // namespace lainnie

#endif // LAINNIE_H_
