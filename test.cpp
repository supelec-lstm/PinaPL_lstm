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

    std::cout << "sequence_size is 2" << std::endl;

    std::cout << "def: input_size = " << std::endl
    << input_size << std::endl;
    std::cout << "def: output_size = " << std::endl
    << output_size << std::endl;

    Weights* cell_weight = new Weights(input_size, output_size);
    std::cout << "created: cell_weight" << std::endl;

    Cell cell = Cell(cell_weight);
    std::cout << "created: cell; arg: cell_weight" << std::endl;

    std::cout << "value: weight_in_input_gate = " << std::endl
    << cell.weights->weight_in_input_gate << std::endl;

// FIRST INPUT
    std::cout << "First caracter in sequence:" << std::endl
    << "~~~~~" << std::endl;

    Eigen::MatrixXd input1(input_size, 1);
    input1 << 1, 4, 1;
    std::cout << "def: input1 = " << std::endl
    << input1.transpose() << std::endl;

    cell.compute(&input1);
    std::cout << "computed: cell.compute(&input1)" << std::endl;

    std::cout << "value: input_gate_out = " << std::endl
    << cell.input_gate_out.back() << std::endl;

    Eigen::MatrixXd output1 = cell.cell_out.back();
    std::cout << "value: output = " << std::endl
    << output1.transpose() << std::endl;

    Eigen::MatrixXd expected_output1(output_size, 1);
    expected_output1 << 5, 1, 1, 8, 5;
    std::cout << "def: expected_output = " << std::endl
    << expected_output1.transpose() << std::endl;

    Eigen::MatrixXd delta1;
    delta1 = expected_output1 - output1;
    std::cout << "value: delta = " << std::endl
    << delta1.transpose() << std::endl;

    cell.compute_gate_gradient(&delta1, 1);
    std::cout << "computed: compute_gate_gradient(&delta1, 1)" << std::endl;

    std::cout << "value: delta_input_gate_out = " << std::endl
    << cell.delta_input_gate_out.back() << std::endl;

    cell.compute_weight_gradient();
    std::cout << "computed: compute_weight_gradient()" << std::endl;

    std::cout << "value: delta_weight_in_input_gate = " << std::endl
    << cell.weights->delta_weight_in_input_gate << std::endl;

    std::cout << "End of first caracter in sequence" << std::endl
    << "~~~~~" << std::endl;
// END FIRST INPUT
// SECOND INPUT
    std::cout << "second caracter in sequence:" << std::endl
    << "~~~~~" << std::endl;

    Eigen::MatrixXd input2(input_size, 1);
    input1 << 3, 2, 3;
    std::cout << "def: input2 = " << std::endl
    << input2.transpose() << std::endl;

    cell.compute(&input2);
    std::cout << "computed: cell.compute(&input2)" << std::endl;

    std::cout << "value: input_gate_out = " << std::endl
    << cell.input_gate_out.back() << std::endl;

    Eigen::MatrixXd output2 = cell.cell_out.back();
    std::cout << "value: output = " << std::endl
    << output2.transpose() << std::endl;

    Eigen::MatrixXd expected_output2(output_size, 1);
    expected_output2 << 4, 5, 3, 1, 1;
    std::cout << "def: expected_output = " << std::endl
    << expected_output2.transpose() << std::endl;

    Eigen::MatrixXd delta2;
    delta2 = expected_output2 - output2;
    std::cout << "value: delta2 = " << std::endl
    << delta2.transpose() << std::endl;

    cell.compute_gate_gradient(&delta2, 1);
    std::cout << "computed: compute_gate_gradient(&delta, 1)" << std::endl;

    std::cout << "value: delta_input_gate_out = " << std::endl
    << cell.delta_input_gate_out.back() << std::endl;

    cell.compute_weight_gradient();
    std::cout << "computed: compute_weight_gradient()" << std::endl;

    std::cout << "value: delta_weight_in_input_gate = " << std::endl
    << cell.weights->delta_weight_in_input_gate << std::endl;

    std::cout << "End of second caracter in sequence" << std::endl
    << "~~~~~" << std::endl;
// END SECOND INPUT

    double lambda = 0.5;
    std::cout << "def: lambda = " << std::endl
    << lambda << std::endl;

    cell.update_weights(lambda);
    std::cout << "computed: update_weights(lambda)" << std::endl;

    std::cout << "value: weight_in_input_gate = " << std::endl
    << cell.weights->weight_in_input_gate << std::endl;
}
