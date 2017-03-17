// Copyright PinaPL
//
// test.hpp
// PinaPL
//
#ifndef TEST_HPP
#define TEST_HPP

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include "weights.hpp"
#include "cell.hpp"
#include "test.hpp"
#include "iostream"

void single_cell_test();
void single_cell_grammar_test();
Eigen::MatrixXd get_input_bias(char letter);
Eigen::MatrixXd get_input_no_bias(char letter);
#endif
