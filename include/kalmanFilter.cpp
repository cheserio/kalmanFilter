#include "kalmanFilter.h"

KalmanFilter::KalmanFilter(int stateSize_ = 0, int measSize_ = 0, int uSize_ = 0):stateSize(stateSize_),measSize(measSize_),uSize(uSize_)
{
    if(stateSize == 0 || measSize == 0)
    {
        std::cerr << "Error! State size and measurement size must bigger than 0" << std::endl;
    }

    // 状态量的初始化
    x.resize(stateSize);
    x.setZero();

    // 状态转移矩阵的初始化
    A.resize(stateSize, stateSize);
    A.setIdentity();

    // 控制量的初始化
    u.resize(uSize);
    u.transpose(); // 转置成列向量
    u.setZero();

    // 控制量参数矩阵初始化
    B.resize(stateSize, uSize);
    B.setZero();

    // 预测误差协方差矩阵初始化
    P.resize(stateSize, stateSize);
    P.setIdentity();

    // 观测矩阵 作用就是将状态量转换成与低维度的观测值同维度的
    H.resize(measSize, stateSize);
    H.setZero();

    // 观测量初始化
    z.resize(measSize);
    z.setZero();

    // 过程噪声协方差（运动控制中引入的噪声）
    Q.resize(stateSize, stateSize);
    Q.setZero();

    // 测量噪声协方差
    R.resize(measSize, measSize);
    R.setZero();
}

void KalmanFilter::init(Eigen::VectorXd &x_, Eigen::MatrixXd &P_, Eigen::MatrixXd &R_, Eigen::MatrixXd &Q_)
{
    x = x_;
    P = P_;
    R = R_;
    Q = Q_;
}

Eigen::VectorXd KalmanFilter::predict(Eigen::MatrixXd &A_, Eigen::MatrixXd &B_, Eigen::VectorXd &u_)
{
    A = A_;
    B = B_;
    x = A * x + B * u;
    Eigen::MatrixXd A_T = A.transpose();
    P = A * P * A_T + Q;
    return x; // 返回的是当前预测的状态量
}

// 没有控制量的情况
Eigen::VectorXd KalmanFilter::predict(Eigen::MatrixXd &A_)
{
    A = A_;
    x = A * x;
    Eigen::MatrixXd A_T = A.transpose();
    P = A * P * A_T + Q;
    return x;
}

// 需要传入观测矩阵与观测量进行更新
void KalmanFilter::update(Eigen::MatrixXd &H_, Eigen::VectorXd z_meas)
{
    H = H_;
    Eigen::MatrixXd temp1, temp2, Ht;
    Ht = H.transpose();
    temp1 = H * P * Ht + R;
    temp2 = temp1.inverse(); // (H*P*H'+R)^(-1)
    Eigen::MatrixXd K = P * Ht * temp2; // kalman增益
    z = H * x; // 预测出来的测量值
    x = x + K * (z_meas - z); // 使用真实测量值减去预测测量值得到的残差乘卡尔曼增益
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize);
    // 更新预测误差
    P = (I - K * H) * P;
}
