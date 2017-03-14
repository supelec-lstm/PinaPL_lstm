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

std::vector<Eigen::MatrixXd> Cell::compute(
    Eigen::MatrixXd *previous_output,
    Eigen::MatrixXd *previous_cell_state,
    Eigen::MatrixXd *input) {

/*    this->forget_gate_out =
        (this->weights->weight_in_forget_gate * input
        + this->weights->weight_st_forget_gate * previous_cell_state)
        .unaryExpr(&sigmoid); */

    this->input = *input;
    this->previous_output = *previous_output;

    this->input_gate_out =
        (this->weights->weight_in_input_gate * *input
        + this->weights->weight_st_input_gate * *previous_cell_state)
        .unaryExpr(&sigmoid);

    this->input_block_out =
        (this->weights->weight_in_input_block * *input
        + this->weights->weight_st_input_block * *previous_cell_state)
        .unaryExpr(&tanhyp);

    this->output_gate_out =
        (this->weights->weight_in_output_gate * *input
        + this->weights->weight_st_output_gate * *previous_cell_state)
        .unaryExpr(&sigmoid);

    this->cell_state =
        (*previous_cell_state/*.cwiseProduct(this->forget_gate_out)*/
        + this->input_gate_out.cwiseProduct(this->input_block_out));

    this->cell_out =
        this->cell_state.unaryExpr(&tanh).cwiseProduct(this->output_gate_out);

    std::vector<Eigen::MatrixXd> result;
    result.push_back(cell_out);
    result.push_back(cell_state);
    return result;
}

std::vector<Eigen::MatrixXd> Cell::compute_gradient(Eigen::MatrixXd* deltas,
    Eigen::MatrixXd* previous_delta_cell_in,
    Eigen::MatrixXd* previous_delta_cell_state) {

// Computes dy
    Eigen::MatrixXd delta_cell_out = *previous_delta_cell_in + *deltas;

// Comptutes do
    Eigen::MatrixXd delta_output_gate = delta_cell_out.cwiseProduct(
        cell_state.unaryExpr(&tanh).cwiseProduct(output_gate_out).cwiseProduct(
        Eigen::MatrixXd::Ones(this->weights->output_size, 1)
        - output_gate_out));

    this->weights->delta_weight_in_output_gate +=
        delta_output_gate * this->input.transpose();

    this->weights->delta_weight_st_output_gate +=
        delta_output_gate * this->previous_output.transpose();

// Computes dc
    Eigen::MatrixXd delta_cell_state = *previous_delta_cell_state
        + delta_cell_out.cwiseProduct(output_gate_out.cwiseProduct(
        Eigen::MatrixXd::Ones(this->weights->output_size, 1)
        - cell_state.unaryExpr(&tanh2) ));

// Computes df
//    Eigen::MatrixXd delta_forget_gate;

// Computes di
    Eigen::MatrixXd delta_input_gate = delta_cell_state
        .cwiseProduct(input_block_out).cwiseProduct(input_gate_out)
        .cwiseProduct(
        (Eigen::MatrixXd::Ones(input_gate_out.rows(), input_gate_out.cols()) -
        input_gate_out));

    this->weights->delta_weight_in_input_gate +=
        delta_input_gate * this->input.transpose();

    this->weights->delta_weight_st_input_gate +=
        delta_input_gate * this->previous_output.transpose();

// Computes dz
    Eigen::MatrixXd delta_input_block =
        delta_cell_state.cwiseProduct(input_gate_out)
        .cwiseProduct((Eigen::MatrixXd::Ones(this->weights->output_size, 1) -
        input_block_out.array().pow(2).matrix()));  // Worst line ever :)

    this->weights->delta_weight_in_input_block +=
        delta_input_block * this->input.transpose();

    this->weights->delta_weight_st_input_block +=
        delta_input_block * this->previous_output.transpose();

// Computes dx
    Eigen::MatrixXd delta_input =
        this->weights->weight_in_input_block * delta_input_block +
        this->weights->weight_in_input_gate * delta_input_gate +
//        this->weights->weight_in_forget_gate * delta_forget_gate +
        this->weights->weight_in_output_gate * delta_output_gate;

// Computes dy(t)
//        Eigen::MatrixXd delta_cell_out = deltas +
//        this->weights->weight_st_input_gate * delta_input_gate +
//        this->weights->weight_st_input_block * delta_input_block +
//        /*this->weights->weight_st_forget_gate * delta_forget_gate +*/
//        this->weights->weight_st_output_gate * delta_output_gate;
    std::vector<Eigen::MatrixXd> result;
    result.push_back(delta_cell_state);
    result.push_back(delta_input);
    return result;
}
