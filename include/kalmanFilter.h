#ifndef _KALMANFILTER_H
#define _KALMANFILTER_H
#include <Eigen/Dense>
#include <iostream>

class KalmanFilter
{
private:
    int stateSize; // 状态量尺寸
    int measSize;   // 测量值尺寸
    int uSize;  // 控制量尺寸
    Eigen::VectorXd x;
    Eigen::VectorXd z;
    Eigen::MatrixXd A; // 状态转移矩阵（运动系统参数）
    Eigen::MatrixXd B; // 运动系统参数 
    Eigen::VectorXd u; // 控制量
    Eigen::MatrixXd P; // 预测误差协方差
    Eigen::MatrixXd H; // 缩放因子矩阵(观测矩阵)
    Eigen::MatrixXd R; // 测量噪声（厂家给的测量噪声均值）
    Eigen::MatrixXd Q; // 控制运动过程中引入的过程噪声
public:
    KalmanFilter(int statSize_, int measSize_, int uSize_);
    // ~KalmanFilter();
    void init(Eigen::VectorXd &x_, Eigen::MatrixXd &P_, Eigen::MatrixXd &R_, Eigen::MatrixXd &Q_);
    Eigen::VectorXd predict(Eigen::MatrixXd &A_);
    Eigen::VectorXd predict(Eigen::MatrixXd &A_, Eigen::MatrixXd &B_, Eigen::VectorXd &u_);
    void update(Eigen::MatrixXd &H_, Eigen::VectorXd z_meas);
};

#endif