// Copyright PinaPL
//
// test.cpp
// PinaPL
//

#include <stdlib.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include "test.hpp"

void single_cell_test() {
    int input_size = 26;
    int output_size = 26;

    std::cout << "sequence_size is 4, word is warp" << std::endl;
    std::cout
    << "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    << std::endl
    << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0"
    << std::endl
    << "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    << std::endl
    << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0"
    << std::endl
    << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0"
    << std::endl;

    Weights* cell_weight = new Weights(input_size, output_size);

    std::vector<Eigen::MatrixXd> inputs;

    Eigen::MatrixXd inputW(input_size, 1);
    inputW << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0;

    Eigen::MatrixXd inputA(input_size, 1);
    inputA << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd inputR(input_size, 1);
    inputR << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0;

    Eigen::MatrixXd inputP(input_size, 1);
    inputP << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0;

    inputs.push_back(inputW);
    inputs.push_back(inputA);
    inputs.push_back(inputR);
    inputs.push_back(inputP);

    for (int j=0; j < 10000; j++) {
        // std::cout << "Starting propagation" << std::endl;
        std::vector<Eigen::MatrixXd> deltas;
        std::vector<Eigen::MatrixXd> result;
        Eigen::MatrixXd previous_output =
            Eigen::MatrixXd::Zero(output_size, 1);
        Eigen::MatrixXd previous_memory =
            Eigen::MatrixXd::Zero(output_size, 1);

        std::vector<Cell> network;

        for (int i=0; i < 3; ++i) {
            Cell cell = Cell(cell_weight);

            Eigen::MatrixXd input = inputs.at(i);

            result = cell.compute(previous_output, &previous_memory, input);
            previous_output = result.at(0);
            deltas.push_back((previous_output - inputs.at(i+1))
                .cwiseProduct(previous_output - inputs.at(i+1)));

            previous_memory = result.at(1);
            network.push_back(cell);
        }

    // std::cout << "Porgagation done, starting backpropagation" << std::endl;

        Eigen::MatrixXd previous_delta_cell_in =
            Eigen::MatrixXd::Zero(output_size, 1);
        Eigen::MatrixXd previous_delta_cell_state =
            Eigen::MatrixXd::Zero(output_size, 1);

        for (int i=3-1; i >= 0; --i) {
            result = network.at(i).compute_gradient(&deltas.at(i),
                &previous_delta_cell_in, &previous_delta_cell_state);
        }
        cell_weight->apply_gradient(0.3);
    }

    std::cout << "Learning done" << std::endl;

    std::vector<Eigen::MatrixXd> result;
    Eigen::MatrixXd previous_output =
        Eigen::MatrixXd::Zero(output_size, 1);
    Eigen::MatrixXd previous_memory =
        Eigen::MatrixXd::Zero(output_size, 1);

    std::vector<Cell> network;

    for (int i=0; i < 3; ++i) {
        Cell cell = Cell(cell_weight);

        Eigen::MatrixXd input = inputs.at(i);

        result = cell.compute(previous_output, &previous_memory, input);
        previous_output = result.at(0);
        previous_memory = result.at(1);
        network.push_back(cell);
        std::cout << "=====================================" << std::endl;
        std::cout << previous_output << std::endl;
    }
}


void single_cell_grammar_test() {
    int input_size = 7;
    int output_size = 7;
    int words_to_learn = 50000;

    Weights* cell_weight = new Weights(input_size, output_size);

    std::ifstream file("reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::MatrixXd> deltas;

    while ((std::getline(file, str)) && (0 < words_to_learn)) {
        int lenght_word = str.length();
        std::vector<Eigen::MatrixXd> deltas;
        std::vector<Eigen::MatrixXd> result;

        Eigen::MatrixXd previous_output =
            Eigen::MatrixXd::Zero(output_size, 1);
        Eigen::MatrixXd previous_memory =
            Eigen::MatrixXd::Zero(output_size, 1);

        std::vector<Cell> network;

        for (int i = 0; i < lenght_word-1; ++i) {
            Cell cell = Cell(cell_weight);
            Eigen::MatrixXd in = get_input(str.at(i));
            Eigen::MatrixXd expected = get_input(str.at(i+1));
            result = cell.compute(previous_output, &previous_memory, in);
            previous_output = result.at(0);
            deltas.push_back((previous_output - expected)
                .cwiseProduct(previous_output - expected));
            previous_memory = result.at(1);
            network.push_back(cell);
        }

        Eigen::MatrixXd previous_delta_cell_in =
            Eigen::MatrixXd::Zero(output_size, 1);
        Eigen::MatrixXd previous_delta_cell_state =
            Eigen::MatrixXd::Zero(output_size, 1);

        for (int i = lenght_word-2; i >= 0; --i) {
            result = network.at(i).compute_gradient(&deltas.at(i),
                &previous_delta_cell_in, &previous_delta_cell_state);
        }
        cell_weight->apply_gradient(0.1);
    }
    std::cout << "Learning done" << std::endl;

    Eigen::MatrixXd previous_output =
        Eigen::MatrixXd::Zero(output_size, 1);
    Eigen::MatrixXd previous_memory =
        Eigen::MatrixXd::Zero(output_size, 1);

    Cell cell = Cell(cell_weight);
    std::vector<Eigen::MatrixXd> result;
    Eigen::MatrixXd B = get_input('B');
    Eigen::MatrixXd P = get_input('P');
    Eigen::MatrixXd V = get_input('V');
    Eigen::MatrixXd E = get_input('E');

    std::cout << "========= On donne B ========" << std::endl;
    result = cell.compute(previous_output, &previous_memory, B);
    previous_output = result.at(0);
    previous_memory = result.at(1);
    std::cout << result.at(0) << std::endl;

    std::cout << "========= On donne P ========" << std::endl;
    result = cell.compute(previous_output, &previous_memory, P);
    previous_output = result.at(0);
    previous_memory = result.at(1);
    std::cout << result.at(0) << std::endl;

    std::cout << "========= On donne V ========" << std::endl;
    result = cell.compute(previous_output, &previous_memory, V);
    previous_output = result.at(0);
    previous_memory = result.at(1);
    std::cout << result.at(0) << std::endl;
}
/*
void single_cell_grammar_test() {
    int input_size = 7;
    int output_size = 7;
    int words_to_learn = 50000;

    Weights* cell_weight = new Weights(input_size, output_size);
    Cell cell = Cell(cell_weight);

    std::ifstream file("reber_test_1M.txt");
    std::string str;
    std::vector<Eigen::MatrixXd> deltas;

    while ((std::getline(file, str)) && (0 < words_to_learn)) {
        int lenght_word = str.length();
        for (int i = 0; i < lenght_word; ++i) {
            Eigen::MatrixXd in = get_input(str.at(i));
            cell.compute(&in);
            deltas.push_back((in - cell.cell_out.back())
                .cwiseProduct((in - cell.cell_out.back())));
        }

        for (int i = lenght_word - 1 ; i >= 0; --i) {
            Eigen::MatrixXd delta = deltas.at(i);
            cell.compute_gate_gradient(&delta, i);
        }
        cell.compute_weight_gradient();
        cell.update_weights(0.3);
        cell.reset();
        words_to_learn -= 1;
    }
    Eigen::MatrixXd in(7, 1);
    in << 1, 0, 0, 0, 0, 0, 0;
    cell.compute(&in);
    std::cout << cell.cell_out.back() << std::endl;
}
*/
Eigen::MatrixXd get_input(char letter) {
    Eigen::MatrixXd in(7, 1);
    switch (letter) {
        case 'B':
            in << 1, 0, 0, 0, 0, 0, 0;
            break;
        case 'T':
            in << 0, 1, 0, 0, 0, 0, 0;
            break;
        case 'P':
            in << 0, 0, 1, 0, 0, 0, 0;
            break;
        case 'S':
            in << 0, 0, 0, 1, 0, 0, 0;
            break;
        case 'X':
            in << 0, 0, 0, 0, 1, 0, 0;
            break;
        case 'V':
            in << 0, 0, 0, 0, 0, 1, 0;
            break;
        case 'E':
            in << 0, 0, 0, 0, 0, 0, 1;
            break;
    }
    return in;
}
