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
    std::vector<Eigen::MatrixXd> inputs;
    // std::vector<Eigen::MatrixXd> forget_gate_out;
    std::vector<Eigen::MatrixXd> input_gate_out;
    std::vector<Eigen::MatrixXd> input_block_out;
    std::vector<Eigen::MatrixXd> output_gate_out;
    std::vector<Eigen::MatrixXd> cell_state;
    std::vector<Eigen::MatrixXd> cell_out;

    std::vector<Eigen::MatrixXd> delta_cell_out;                // dy
    std::vector<Eigen::MatrixXd> delta_output_gate_out;         // do
    std::vector<Eigen::MatrixXd> delta_cell_state;              // dc
    // std::vector<Eigen::MatrixXd> delta_forget_gate_out;      // df
    std::vector<Eigen::MatrixXd> delta_input_gate_out;          // di
    std::vector<Eigen::MatrixXd> delta_input_block_out;         // dz


 public:
    explicit Cell(Weights* weights);
    void compute(Eigen::MatrixXd input);
    Eigen::MatrixXd compute_gate_gradient(Eigen::MatrixXd deltas, int time);
    void compute_weight_gradient();
    void update_weights(double lambda);
    void reset_gradient();
};
#endif

// TODO(shaka) : initialize delta_* with 0 as first item
