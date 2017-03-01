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

    Cell cell = Cell(cell_weight);

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
      std::vector<Eigen::MatrixXd> deltas;

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
      }

      double lambda = 0.1;
      cell.update_weights(lambda);

      cell.reset();
    }

    cell.compute(&inputW);

    std::cout << "Result of learning :" << std::endl;
    std::cout << cell.cell_out.back() << std::endl;
}

void single_cell_grammar_test() {
    int input_size = 7;
    int output_size = 7;
    int words_to_learn = 50;

    std::cout << "def: input_size = " << std::endl
    << input_size << std::endl;
    std::cout << "def: output_size = " << std::endl
    << output_size << std::endl;

    Weights* cell_weight = new Weights(input_size, output_size);
    std::cout << "created: cell_weight" << std::endl;

    Cell cell = Cell(cell_weight);
    std::cout << "created: cell; arg: cell_weight" << std::endl;

    std::ifstream file("reber_test_1M.txt");
    std::string str;
    while ((std::getline(file, str)) && (0 < words_to_learn)) {
        int lenght_word = str.length();
        for (int i = 0; i < lenght_word; ++i) {
            std::cout << str.at(i);
            Eigen::MatrixXd input1(input_size, 1);
            cell.compute(&input1);
        }
        std::cout << std::endl;
        for (int i = lenght_word + 1; i >= 0; --i) {
            Eigen::MatrixXd deltas;
            cell.compute_gate_gradient(&deltas, i);
        }
        words_to_learn -= 1;
    }
}
