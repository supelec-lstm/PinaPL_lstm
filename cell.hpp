// Copyright PinaPL
//
// cell.hpp
// PinaPL
//
#ifndef CELL_HPP
#define CELL_HPP

#include <math.h>
#include <Eigen/Dense>
#include <vector>

#include "weights.hpp"

class Cell {
    Weights* weights;
    std::vector<Eigen::MatrixXd> full_input;
    std::vector<Eigen::MatrixXd> forget_gate_out;
    std::vector<Eigen::MatrixXd> input_gate_out;
    std::vector<Eigen::MatrixXd> input_block_out;
    std::vector<Eigen::MatrixXd> output_gate_out;
    std::vector<Eigen::MatrixXd> cell_state;
    std::vector<Eigen::MatrixXd> cell_out;


 public:
    explicit Cell(Weights* weights);
    void compute(Eigen::MatrixXd input);
    Eigen::MatrixXd compute_gradient(Eigen::MatrixXd deltas);
};
#endif
