// Copyright PinaPL
//
// test.cpp
// PinaPL
//

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include "weights.hpp"
#include "cell.hpp"
#include "test.hpp"
#include "iostream"


void single_cell_test() {
    int input_size = 4;
    int output_size = 4;

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

    Cell cell = Cell(cell_weight);

    std::vector<Eigen::MatrixXd> inputs;

    Eigen::MatrixXd inputW(input_size, 1);
    inputW << 1, 0, 0, 0;

    Eigen::MatrixXd inputA(input_size, 1);
    inputA << 0, 1, 0, 0;

    Eigen::MatrixXd inputR(input_size, 1);
    inputR << 0, 0, 1, 0;

    Eigen::MatrixXd inputP(input_size, 1);
    inputP << 0, 0, 0, 1;

    inputs.push_back(inputW);
    inputs.push_back(inputA);
    inputs.push_back(inputR);
    inputs.push_back(inputP);

    std::vector<Eigen::MatrixXd> deltas;

    for (int j=0; j < 20000; j++) {
        std::cout << "run :" << j << std::endl;

        for (int i=0; i < 4; ++i) {
            Eigen::MatrixXd input = inputs.at(i);

            cell.compute(&input);

            Eigen::MatrixXd output = cell.cell_out.back();

            Eigen::MatrixXd delta(output_size, 1);
            delta = (output-input).cwiseProduct(output-input);

            deltas.push_back(delta);
        }

        for (int i=4-1; i >= 0; --i) {
            cell.compute_gate_gradient(&deltas.at(i), i);
            std::cout << "gate gradient input gate" << std::endl;
            std::cout << cell.delta_input_gate_out.at(4-i) << std::endl;
        }

        double lambda = 0.5;
        cell.compute_weight_gradient();
        std::cout << "====== delta poids input_gate ======" << std::endl;
        std::cout << cell_weight->delta_weight_in_input_gate << std::endl;
        std::cout << cell.weights->delta_weight_st_input_gate << std::endl;

        cell.update_weights(lambda);
        deltas.clear();
        cell.reset();
    }

    cell.compute(&inputW);

    std::cout << "Result of learning :" << std::endl;
    std::cout << cell.cell_out.back() << std::endl;

    cell.compute(&inputA);
    std::cout << cell.cell_out.back() << std::endl;

    cell.compute(&inputR);
    std::cout << cell.cell_out.back() << std::endl;

    cell.compute(&inputP);
    std::cout << cell.cell_out.back() << std::endl;
}



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
            in << 1, 0, 0, 0, 0, 1, 0;
            break;
        case 'E':
            in << 1, 0, 0, 0, 0, 0, 1;
            break;
    }
    return in;
}
