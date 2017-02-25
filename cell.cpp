// Copyright PinaPL
//
// cell.cpp
// PinaPL
//
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include "weights.hpp"
#include "cell.hpp"
#include "functions.hpp"

Cell::Cell(Weights* weights) {
    this->weights = weights;
}

void Cell::compute(
    Eigen::MatrixXd previous_output,
    Eigen::MatrixXd previous_cell_state,
    Eigen::MatrixXd input) {
    this->full_input = Eigen::MatrixXd(
        input.rows()+previous_output.rows(),
        input.cols());
    this->forget_gate_out =
        (this->weights->weight_forget_gate * this->full_input)
        .unaryExpr(&sigmoid);

    this->input_gate_out =
        (this->weights->weight_input_gate * this->full_input)
        .unaryExpr(&sigmoid);

    this->input_block_out =
        (this->weights->weight_input_block * this->full_input)
        .unaryExpr(&tanh);

    this->output_block_out =
        (this->weights->weight_output_block * this->full_input)
        .unaryExpr(&sigmoid);

    this->cell_state =
        (cell_state.cwiseProduct(this->forget_gate_out)
        + this->input_gate_out.cwiseProduct(this->input_block_out));

        this->cell_state.unaryExpr(&tanh);

    this->cell_out =
        this->cell_state.unaryExpr(&tanh).cwiseProduct(this->output_block_out);
}

// Eigen::MatrixXd Cell::compute_gradient(Eigen::MatrixXd deltas) {
// }
