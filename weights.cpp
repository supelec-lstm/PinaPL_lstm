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

// Initialisation

Weights::Weights(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

    this->weight_forget_gate = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

    this->weight_input_gate = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

    this->weight_input_bloc = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);

    this->weight_output_bloc = Eigen::MatrixXd::Random(
        this->output_size,
        this->input_size + this->output_size);


    this->delta_weight_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);

    this->delta_weight_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);

    this->delta_weight_input_bloc = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);

    this->delta_weight_output_bloc = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
}

void Weights::compute_gradient(double lambda) {
    this->weight_forget_gate =
        this->weight_forget_gate - lambda * this->delta_weight_forget_gate;
    this->weight_input_gate =
        this->weight_input_gate - lambda * this->delta_weight_input_gate;
    this->weight_input_bloc =
        this->weight_input_bloc - lambda * this->delta_weight_input_bloc;
    this->weight_output_bloc =
        this->weight_output_bloc - lambda * this->delta_weight_output_bloc;

    this->delta_weight_forget_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
    this->delta_weight_input_gate = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
    this->delta_weight_input_bloc = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
    this->delta_weight_output_bloc = Eigen::MatrixXd::Zero(
        this->output_size,
        this->input_size + this->output_size);
}
