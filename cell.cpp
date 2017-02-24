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
    Eigen::MatrixXd previous_memory,
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

    this->input_bloc_out =
        (this->weights->weight_input_bloc * this->full_input)
        .unaryExpr(&tanh);

    this->output_bloc_out =
        (this->weights->weight_output_bloc * this->full_input)
        .unaryExpr(&sigmoid);

    this->memory =
        (previous_memory.cwiseProduct(this->forget_gate_out)
        + this->input_gate_out.cwiseProduct(this->input_bloc_out));

    this->memory_bloc_out = this->memory.unaryExpr(&tanh);

    this->cell_out = this->memory_bloc_out + this->output_bloc_out;
}
