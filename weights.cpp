// Copyright PinaPL
//
// weights.cpp
// PinaPL
//

#include <stdlib.h>
#include <Eigen/Dense>

#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <random>


#include "weights.hpp"

Weights::Weights(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

// We initialize random weights
    this->weight_forget_gate = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

    this->weight_input_gate = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

    this->weight_input_block = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

    this->weight_output_block = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

// We initialize a null gradient
    this->delta_weight_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);

    this->delta_weight_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);

    this->delta_weight_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);

    this->delta_weight_output_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
}

void Weights::apply_gradient(double lambda) {
// We apply the weight variations
    this->weight_forget_gate =
        this->weight_forget_gate - lambda * this->delta_weight_forget_gate;
    this->weight_input_gate =
        this->weight_input_gate - lambda * this->delta_weight_input_gate;
    this->weight_input_block =
        this->weight_input_block - lambda * this->delta_weight_input_block;
    this->weight_output_block =
        this->weight_output_block - lambda * this->delta_weight_output_block;

// We set a null gradient
    this->delta_weight_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
    this->delta_weight_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
    this->delta_weight_input_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
    this->delta_weight_output_block = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
}
