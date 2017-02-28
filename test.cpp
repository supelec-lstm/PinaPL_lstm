// Copyright PinaPL
//
// test.cpp
// PinaPL
//

#include <Eigen/Dense>
#include "weights.hpp"
#include "cell.hpp"
#include "test.hpp"

void single_cell_test() {
    int input_size = 7;
    int output_size = 8;
    Weights cell_weight = Weights(input_size, output_size);
    Cell cell = Cell(&cell_weight);
    printf("cell created");
}
