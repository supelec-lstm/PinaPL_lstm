// Copyright PinaPL
//
// cell.hpp
// PinaPL
//
#ifndef CELL_HPP
#define CELL_HPP

#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>

#include "weights.hpp"

class Cell {
    Weights* weights;
    Eigen::MatrixXd full_input;
    Eigen::MatrixXd forget_gate_out;
    Eigen::MatrixXd input_gate_out;
    Eigen::MatrixXd input_bloc_out;
    Eigen::MatrixXd output_bloc_out;
    Eigen::MatrixXd cell_state_out;
    Eigen::MatrixXd cell_state;
    Eigen::MatrixXd cell_out;


 public:
    explicit Cell(Weights* weights);
    void compute(
        Eigen::MatrixXd previous_output,
        Eigen::MatrixXd previous_memory,
        Eigen::MatrixXd input);
    void compute_gradient();
};
#endif
