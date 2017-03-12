// Copyright PinaPL
//
// weights.hpp
// PinaPL
//

#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include <stdlib.h>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <iostream>
#include <map>


class Weights {
 public:
    Weights(int input_size, int output_size);
    ~Weights();
    void apply_gradient(double lambda);

    //   Information :
    // weight_in means the weight matrix applied to the new input
    // weight_st means the weight matrix applied to the previous cell state
    Eigen::MatrixXd weight_in_forget_gate;
    Eigen::MatrixXd weight_in_input_gate;
    Eigen::MatrixXd weight_in_input_block;
    Eigen::MatrixXd weight_in_output_gate;

    Eigen::MatrixXd weight_st_forget_gate;
    Eigen::MatrixXd weight_st_input_gate;
    Eigen::MatrixXd weight_st_input_block;
    Eigen::MatrixXd weight_st_output_gate;

    //   Information :
    // weight_in means the weight matrix applied to the new INPUT
    // weight_st means the weight matrix applied to the previous cell STATE
    Eigen::MatrixXd delta_weight_in_forget_gate;
    Eigen::MatrixXd delta_weight_in_input_gate;
    Eigen::MatrixXd delta_weight_in_input_block;
    Eigen::MatrixXd delta_weight_in_output_gate;

    Eigen::MatrixXd delta_weight_st_forget_gate;
    Eigen::MatrixXd delta_weight_st_input_gate;
    Eigen::MatrixXd delta_weight_st_input_block;
    Eigen::MatrixXd delta_weight_st_output_gate;

    int input_size;
    int output_size;
};
#endif
