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

    this->forget_gate_out =
        (this->weights->weight_in_forget_gate * input
        + this->weights->weight_st_forget_gate * previous_cell_state)
        .unaryExpr(&sigmoid);

    this->input_gate_out =
        (this->weights->weight_in_input_gate * input
        + this->weights->weight_st_input_gate * previous_cell_state)
        .unaryExpr(&sigmoid);

    this->input_block_out =
        (this->weights->weight_in_input_block * input
        + this->weights->weight_st_input_block * previous_cell_state)
        .unaryExpr(&tan);

    this->output_gate_out =
        (this->weights->weight_in_output_gate * input
        + this->weights->weight_st_output_gate * previous_cell_state)
        .unaryExpr(&sigmoid);

    this->cell_state =
        (previous_cell_state.cwiseProduct(this->forget_gate_out)
        + this->input_gate_out.cwiseProduct(this->input_block_out));

    this->cell_out =
        this->cell_state.unaryExpr(&tanh).cwiseProduct(this->output_gate_out);
}

// Eigen::MatrixXd Cell::compute_gradient(Eigen::MatrixXd deltas) {
// }
