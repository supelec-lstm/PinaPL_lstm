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

    std::cout << "instantiated: input_size = " << std::endl
    << input_size << std::endl;
    std::cout << "instantiated: output_size = " << std::endl
    << output_size << std::endl;

    Weights* cell_weight = new Weights(input_size, output_size);
    std::cout << "instantiated: cell_weight" << std::endl;

    Cell cell = Cell(cell_weight);
    std::cout << "instantiated: cell; arg: cell_weight" << std::endl;

    std::cout << "value: weight_in_input_gate = " << std::endl
    << cell.weights->weight_in_input_gate << std::endl;

    Eigen::MatrixXd input(input_size, 1);
    input << 0, 4, 1;
    std::cout << "instantiated: input = " << std::endl
    << input.transpose() << std::endl;

    cell.compute(&input);
    std::cout << "computed: cell.compute(&input)" << std::endl;

    Eigen::MatrixXd output = cell.cell_out.back();
    std::cout << "value: output = " << std::endl
    << output.transpose() << std::endl;

    Eigen::MatrixXd expected_output(output_size, 1);
    expected_output << 5, 0, 1, 8, 5;
    std::cout << "instantiated: expected_output = " << std::endl
    << expected_output.transpose() << std::endl;

    Eigen::MatrixXd delta;
    delta = expected_output - output;
    std::cout << "value: delta = " << std::endl
    << delta.transpose() << std::endl;

    cell.compute_gate_gradient(delta, 1);
    std::cout << "computed: compute_gate_gradient(delta, 1)" << std::endl;

    cell.compute_weight_gradient();
    std::cout << "computed: compute_weight_gradient()" << std::endl;

    double lambda = 0.5;
    std::cout << "instantiated: lambda = " << std::endl
    << lambda << std::endl;

    cell.update_weights(lambda);
    std::cout << "computed: update_weights(lambda)" << std::endl;

    std::cout << "value: weight_in_input_gate = " << std::endl
    << cell.weights->weight_in_input_gate << std::endl;
}
