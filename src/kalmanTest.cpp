#include "../include/kalmanFilter.h"
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

#define N 2000
#define T 0.01
using namespace std;

double func(double &x);
double data_x[N], data_y[N];

double func(double &x)
{
    double res = 5 * x * x;
    return res;
}

// 匀加速运动模型
float sample(float x0, float v0, float acc, float t)
{
    return x0 + v0 * t + 1 / 2 * acc * t * t;
}

// 产生噪声
float GetRand()
{
    return 0.5 * rand() / RAND_MAX - 0.25;
}

int main(void)
{
    ofstream fout, fmout;
    fout.open("data.txt");
    float t;
    // 准备测量数据
    for(int i = 0; i < N; i++)
    {
        t = i * T;
        // x方向上的数据 存入的是累加的数据
        data_x[i] = sample(0, -4, 0, t) + GetRand();
        // y方向上的数据
        data_y[i] = sample(0, 6.5, 0, t) + GetRand();
    }
    int stateSize = 6;
    int measSize = 2;
    int controlSize = 0;
    KalmanFilter kf(stateSize, measSize, controlSize);
    
    // 需要用到的变量的初始化
    Eigen::MatrixXd A(stateSize, stateSize);
    // x, y, vx, vy, ax, ay
    A <<  1, 0, T, 0, 1 / 2 * T*T, 0,
        0, 1, 0, T, 0, 1 / 2 * T*T,
        0, 0, 1, 0, T, 0, 
        0, 0, 0, 1, 0, T,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1;
    Eigen::MatrixXd B(0, 0);
    Eigen::MatrixXd H(measSize, stateSize);
    // 将状态量降维到测量值的维度 测量值只能测量 x, y
    H << 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0;
    // 预测误差协方差
    Eigen::MatrixXd P(stateSize, stateSize);
    P.setIdentity();
    // 测量噪声协方差
    Eigen::MatrixXd R(measSize, measSize);
    R.setIdentity()*0.01;
    // 过程噪声协方差
    Eigen::MatrixXd Q(stateSize, stateSize);
    Q.setIdentity()*0.001;
    // 状态量
    Eigen::VectorXd x(stateSize);
    // 控制量
    Eigen::VectorXd u(0);
    // 测量值
    Eigen::VectorXd z(measSize);
    z.setZero();
    // 输出结果
    Eigen::VectorXd res(stateSize);

    for(int i = 0; i < N; i++)
    {
        if(i == 0)
        {
            // 第一次的状态量也只有x，y 没有速度与加速度
            x << data_x[i], data_y[i], 0, 0, 0, 0;
            kf.init(x, P, R, Q);
        }
        // 没有控制量则只需要A 这里predict会输出预测的x 同时求出了预测误差
        res << kf.predict(A);
        z << data_x[i], data_y[i];
        kf.update(H, z); // H是固定的初始化完可以直接传入
        fout << data_x[i] << " " << res[0] << " " << data_y[i] << " " << res[1] << " " << res[2] << " " << res[3] << " " << res[4] << " " << res[5] << endl;
    }
    fout.close();
    cout << "Done, use python script to draw the figure..." << endl;
    system("pause");
    return 0;
}
