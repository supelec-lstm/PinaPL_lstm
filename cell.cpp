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
        .unaryExpr(&tanh));

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

Eigen::MatrixXd Cell::compute_gradient(Eigen::MatrixXd deltas, int time) {
    int output_size = output_gate_out.at(time).rows();
    // Computes dy(t)
    delta_cell_out.push_back(
        deltas
        + this->weights->weight_st_input_block * delta_input_block_out.back()
        + this->weights->weight_st_input_gate * delta_input_gate_out.back()
//      + this->weights->weight_st_forget_gate * delta_forget_gate_out.back()
        + this->weights->weight_st_output_gate * delta_output_gate_out.back() );

    // Computes do(t)
    delta_output_gate_out.push_back(delta_cell_out.back()
        .cwiseProduct(cell_state.at(time).unaryExpr(&tanh))
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
    this->weights->weight_in_input_block * delta_input_block_out.back()
    + this->weights->weight_in_input_gate * delta_input_gate_out.back()
//  + this->weights->weight_in_input_block * delta_input_block_out.back()
    + this->weights->weight_in_output_gate * delta_output_gate_out.back();

    return delta_input;
}
