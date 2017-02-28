// Copyright PinaPL
//
// cell.cpp
// PinaPL
//
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include "weights.hpp"
#include "cell.hpp"
#include "functions.hpp"

Cell::Cell(Weights* weights) {
    this->weights = weights;
}

void Cell::compute(Eigen::MatrixXd input) {
/*    this->forget_gate_out =
        (this->weights->weight_in_forget_gate * input
        + this->weights->weight_st_forget_gate * previous_cell_state)
        .unaryExpr(&sigmoid); */

    this->input_gate_out.push_back(
        (this->weights->weight_in_input_gate * input
        + this->weights->weight_st_input_gate * this->cell_out.back())
        .unaryExpr(&sigmoid));

    this->input_block_out.push_back(
        (this->weights->weight_in_input_block * input
        + this->weights->weight_st_input_block * this->cell_out.back())
        .unaryExpr(&tan));

    this->output_gate_out.push_back(
        (this->weights->weight_in_output_gate * input
        + this->weights->weight_st_output_gate * this->cell_out.back())
        .unaryExpr(&sigmoid));

    this->cell_state.push_back(
        (this->cell_state.back()/*.cwiseProduct(this->forget_gate_out)*/
        + this->input_gate_out.back()
        .cwiseProduct(this->input_block_out.back())));

    this->cell_out.push_back(
        this->cell_state.back().unaryExpr(&tanh)
        .cwiseProduct(this->output_gate_out.back()));
}
