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
    this->reset();
}

void Cell::compute(Eigen::MatrixXd* input) {
/*    this->forget_gate_out =
        (this->weights->weight_in_forget_gate * input
        + this->weights->weight_st_forget_gate * previous_cell_state)
        .unaryExpr(&sigmoid); */
    this->inputs.push_back((*input));

    this->input_gate_out.push_back(
        (this->weights->weight_in_input_gate * (*input)
        + this->weights->weight_st_input_gate * this->cell_out.back())
        .unaryExpr(&sigmoid));

    this->input_block_out.push_back(
        (this->weights->weight_in_input_block * (*input)
        + this->weights->weight_st_input_block * this->cell_out.back())
        .unaryExpr(&tanhyp));

    this->output_gate_out.push_back(
        (this->weights->weight_in_output_gate * (*input)
        + this->weights->weight_st_output_gate * this->cell_out.back())
        .unaryExpr(&sigmoid));

    this->cell_state.push_back(
        (this->cell_state.back()
        + this->input_gate_out.back()
        .cwiseProduct(this->input_block_out.back())));

    this->cell_out.push_back(
        this->cell_state.back().unaryExpr(&tanhyp)
        .cwiseProduct(this->output_gate_out.back()));
}

Eigen::MatrixXd Cell::compute_gate_gradient(Eigen::MatrixXd deltas, int time) {
    int output_size = this->weights->output_size;
    // Computes dy(t)
    delta_cell_out.push_back(
        deltas
        + this->weights->weight_st_input_block * delta_input_block_out.back()
        + this->weights->weight_st_input_gate * delta_input_gate_out.back()
//      + this->weights->weight_st_forget_gate * delta_forget_gate_out.back()
        + this->weights->weight_st_output_gate * delta_output_gate_out.back() );

    // Computes do(t)
    delta_output_gate_out.push_back(delta_cell_out.back()
        .cwiseProduct(cell_state.at(time).unaryExpr(&tanhyp))
        .cwiseProduct(output_gate_out.at(time).cwiseProduct(
            Eigen::MatrixXd::Ones(output_size, 1)-output_gate_out.at(time))));

    // Computes dc(t)
    delta_cell_state.push_back(
        delta_cell_out.back()
        .cwiseProduct(output_gate_out.at(time))
        .cwiseProduct(cell_state.at(time).unaryExpr(&tanh_derivative)));

    // Computes di(t)
    delta_input_gate_out.push_back(
        delta_cell_state.back()
        .cwiseProduct(input_block_out.at(time))
        .cwiseProduct(input_gate_out.at(time).cwiseProduct(
            Eigen::MatrixXd::Ones(output_size, 1)-input_gate_out.at(time))) );

    // Computes dz(t)
    delta_input_block_out.push_back(
        delta_cell_state.back()
        .cwiseProduct(input_gate_out.at(time))
        .cwiseProduct(input_block_out.at(time).cwiseProduct(
            Eigen::MatrixXd::Ones(output_size, 1)-input_block_out.at(time))) );

    // Computes dx(t)
    Eigen::MatrixXd delta_input =
    this->weights->weight_in_input_block.transpose()
      * delta_input_block_out.back()
    + this->weights->weight_in_input_gate.transpose()
      * delta_input_gate_out.back()
//  + this->weights->weight_in_input_block.transpose()
//    * delta_input_block_out.back()
    + this->weights->weight_in_output_gate.transpose()
      * delta_output_gate_out.back();

    return delta_input;
}

void Cell::compute_weight_gradient() {
    int sequence_size = this->inputs.size();

    // Computes dW
    for (int t = 0; t < sequence_size; ++t) {
        // Computes dWz
        this->weights->delta_weight_in_input_block +=
            delta_input_block_out.at(sequence_size - t)
            * inputs.at(t).transpose();

        // Computes dWi
        this->weights->delta_weight_in_input_gate +=
            delta_input_gate_out.at(sequence_size - t)
            * inputs.at(t).transpose();

        // Computes dWf
        /*
        this->weights->delta_weight_in_input_block +=
            delta_input_block_out.at(sequence_size - t)
            * inputs.at(t).transpose(); */

        // Computes dWo
        this->weights->delta_weight_in_output_gate +=
            delta_output_gate_out.at(sequence_size - t)
            * inputs.at(t).transpose();
    }

    // Computes dR
    for (int t = 0; t < sequence_size - 1; ++t) {
        // Computes dRz
        this->weights->delta_weight_st_input_block +=
            delta_input_block_out.at(sequence_size - t - 1)
            * cell_out.at(t).transpose();

        // Computes dRi
        this->weights->delta_weight_st_input_gate +=
            delta_input_gate_out.at(sequence_size - t - 1)
            * cell_out.at(t).transpose();

        // Computes dRo
        this->weights->delta_weight_st_output_gate +=
            delta_output_gate_out.at(sequence_size - t - 1)
            * cell_out.at(t).transpose();
    }
}

void Cell::update_weights(double lambda) {
    this->weights->apply_gradient(lambda);
}

void Cell::reset() {
    int output_size = this->weights->output_size;

    this->inputs.clear();
    this->input_gate_out.clear();
    this->input_block_out.clear();
    this->output_gate_out.clear();
    this->cell_state.clear();
    this->cell_out.clear();

    this->delta_cell_out.clear();
    this->delta_output_gate_out.clear();
    this->delta_cell_state.clear();
    this->delta_input_gate_out.clear();
    this->delta_input_block_out.clear();

    this->input_gate_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->input_block_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->output_gate_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->cell_state.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->cell_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));

    this->delta_cell_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->delta_output_gate_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->delta_cell_state.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->delta_input_gate_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
    this->delta_input_block_out.push_back(
        Eigen::MatrixXd::Zero(output_size, 1));
}
