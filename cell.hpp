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
    Eigen::MatrixXd input;
    Eigen::MatrixXd previous_output;
    Eigen::MatrixXd previous_cell_state;
    Eigen::MatrixXd forget_gate_out;
    Eigen::MatrixXd input_gate_out;
    Eigen::MatrixXd input_block_out;
    Eigen::MatrixXd output_gate_out;
    Eigen::MatrixXd cell_state;
    Eigen::MatrixXd cell_out;


 public:
    explicit Cell(Weights* weights);
    std::vector<Eigen::MatrixXd> compute(
        Eigen::MatrixXd *previous_output,
        Eigen::MatrixXd *previous_memory,
        Eigen::MatrixXd *input);
    std::vector<Eigen::MatrixXd> compute_gradient(Eigen::MatrixXd* deltas,
        Eigen::MatrixXd* previous_delta_cell_in,
        Eigen::MatrixXd* previous_delta_cell_state);
};
#endif
