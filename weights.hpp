// Copyright PinaPL
//
// weights.hpp
// PinaPL
//

#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

#include <Eigen/Dense>

class Weights {
 public:
    int input_size;
    int output_size;
    Weights(int input_size, int output_size);
    ~Weights();
    void apply_gradient(double lambda);

    //   Information :
    // weight_in means the weight matrix applied to the new input
    // weight_st means the weight matrix applied to the previous cell OUT
    Eigen::MatrixXd weight_in_forget_gate;                  // Wf
    Eigen::MatrixXd weight_in_input_gate;                   // Wi
    Eigen::MatrixXd weight_in_input_block;                  // Wz
    Eigen::MatrixXd weight_in_output_gate;                  // Wo

    Eigen::MatrixXd weight_st_forget_gate;                  // Rf
    Eigen::MatrixXd weight_st_input_gate;                   // Ri
    Eigen::MatrixXd weight_st_input_block;                  // Rz
    Eigen::MatrixXd weight_st_output_gate;                  // Ro


    //   Information :
    // weight_in means the weight matrix applied to the new INPUT
    // weight_st means the weight matrix applied to the previous cell OUT
    Eigen::MatrixXd delta_weight_in_forget_gate;           // dWf
    Eigen::MatrixXd delta_weight_in_input_gate;            // dWi
    Eigen::MatrixXd delta_weight_in_input_block;           // dWz
    Eigen::MatrixXd delta_weight_in_output_gate;           // dWo

    Eigen::MatrixXd delta_weight_st_forget_gate;           // dRf
    Eigen::MatrixXd delta_weight_st_input_gate;            // dRi
    Eigen::MatrixXd delta_weight_st_input_block;           // dRz
    Eigen::MatrixXd delta_weight_st_output_gate;           // dRo
};
#endif
