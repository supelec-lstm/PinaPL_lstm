// Copyright PinaPL
//
// test.cpp
// PinaPL
//

#include <Eigen/Dense>
#include "weights.hpp"
#include "cell.hpp"
#include "test.hpp"
#include "iostream"

void single_cell_test() {
    int input_size = 3;
    int output_size = 5;

    std::cout << "instantiated: input_size = "
    << input_size << std::endl;
    std::cout << "instantiated: output_size = "
    << output_size << std::endl;

    Weights* cell_weight = new Weights(input_size, output_size);
    std::cout << "instantiated: cell_weight" << std::endl;

    Cell cell = Cell(cell_weight);
    std::cout << "instantiated: cell; arg: cell_weight" << std::endl;

    Eigen::MatrixXd input(input_size, 1);
    input << 0, 4, 1;
    std::cout << "instantiated: input = "
    << input.transpose() << std::endl;

    cell.compute(&input);
    std::cout << "computed: cell.compute(); arg: &input" << std::endl;

    Eigen::MatrixXd output = cell.cell_out.back();
    std::cout << "value: output = "
    << output.transpose() << std::endl;

    Eigen::MatrixXd expected_output(output_size, 1);
    expected_output << 5, 0, 1, 8, 5;
    std::cout << "instantiated: expected_output = "
    << expected_output.transpose() << std::endl;

    Eigen::MatrixXd delta;
    delta = expected_output - output;
    std::cout << "value: delta = "
    << delta.transpose() << std::endl;

    //cell.compute_gate_gradient(delta, 1);
    //std::cout << "computed: compute_gate_gradient; arg: delta, 1" << std::endl;
}
