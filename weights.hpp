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
    int input_size;
    int output_size;

    Eigen::MatrixXd delta_weight_forget_gate;
    Eigen::MatrixXd delta_weight_input_gate;
    Eigen::MatrixXd delta_weight_input_bloc;
    Eigen::MatrixXd delta_weight_output_bloc;


 public:
    Weights(int input_size, int output_size);
    ~Weights();
    void apply_gradient(double lambda);

    Eigen::MatrixXd weight_forget_gate;
    Eigen::MatrixXd weight_input_gate;
    Eigen::MatrixXd weight_input_bloc;
    Eigen::MatrixXd weight_output_bloc;
};
#endif
