from ast import increment_lineno
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, Matrix
from sympy.interactive import printing

# 这个代码是使用python实现的卡尔曼滤波可以直接使用python3运行 
# 注意需要安装上面引入的头文件


# 给需要用到的这些变量初始化
# 行人状态量 x, y, vx, vy
x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
print(x, x.shape)

# 预测误差
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])
print(P, P.shape)

dt = 0.1 # Time Step between Filter Steps

# 这里定义的是状态方程（匀速运动）中的A矩阵
F = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(F, F.shape)

# H是缩放因子矩阵 因为速度是直接可以测量的因此设定为1
H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(H, H.shape)

# ra是传感器厂商提供的测量误差 在引入误差时需要使用协方差矩阵（x,y两个方向）
ra = 0.09
R = np.matrix([[ra, 0.0],
              [0.0, ra]])
print(R, R.shape)

# 运动状态引入的噪声Q 
sv = 0.5
G = np.matrix([[0.5*dt**2],
               [0.5*dt**2],
               [dt],
               [dt]])
Q = G*G.T*sv**2

printing.init_printing()
dts = Symbol('dt') # 声明dts是一个数学符号
Qs = Matrix([[0.5*dts**2],[0.5*dts**2],[dts],[dts]])
# Qs*Qs.T
I = np.eye(4)
print(I, I.shape)
m = 200 # Measurements 测量数量
vx= 20 # in X
vy= 10 # in Y

# 生成一些测量数据
mx = np.array(vx+np.random.randn(m))
my = np.array(vy+np.random.randn(m))
measurements = np.vstack((mx,my))

# 过程值用于绘图
xt = [] 
yt = []
dxt= []
dyt= []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Rdx= []
Rdy= []
Kx = []
Ky = []
Kdx= []
Kdy= []

# 存储过程值
# x是行人位置 包含了 x,y,vx,vy四个值
# Z 是测量值
# P是预测误差
# R是测量误差
# K是卡尔曼增益
def savestates(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Rdx.append(float(R[0,0]))
    Rdy.append(float(R[1,1]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))  

# 卡尔曼滤波的核心过程
for n in range(len(measurements[0])):

    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # 状态方程（匀速运动）
    x = F*x

    # Project the error covariance ahead
    # 更新预测误差 他本质上是我们的估计状态概率分布的协方差矩阵
    # Q是我们的处理噪声的协方差矩阵
    P = F*P*F.T + Q

    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    # 计算卡尔曼增益两步 先计算的括号内的东西S
    # H是缩放因子的矩阵
    S = H*P*H.T + R
    K = (P*H.T) * np.linalg.pinv(S) # pinv求逆

    # Update the estimate via z
    # 通过测量值更新估计
    # 将循环当前的测量值转换成2行一列的形式
    Z = measurements[:,n].reshape(2,1)
    # H 2*4 的矩阵 x 是 4 * 1的矩阵 求出来的y是2*1
    y = Z - (H*x)                            # Innovation or Residual
    x = x + (K*y)

    # Update the error covariance
    # 更新误差协方差
    P = (I - (K*H))*P

    # Save states (for Plotting)
    # 一次更新后存储中间值
    savestates(x, Z, P, R, K)

def plot_mesurement():
    print("measurements matrix shape:")
    print(measurements.shape)
    # 在x方向上的测量平均值
    print('Standard Deviation of Acceleration Measurements=%.2f' % np.std(mx))
    # 测量误差0.9
    print('You assumed %.2f in R.' % R[0,0])

    fig = plt.figure(figsize=(16,5))
    plt.step(range(m),mx, label='$\dot x$') # 步长200 mx方向
    plt.step(range(m),my, label='$\dot y$') # my（mesurement in direction y）方向
    plt.ylabel(r'Velocity $m/s$')
    plt.title('Measurements')
    plt.legend(loc='best',prop={'size':18})

def plot_x():
    # 同一时间只展示一个方向上的数据
    fig = plt.figure(figsize=(16,9))
    # 测量值和vx vy 因为步长时间是1 所以相当于是求了一个单位时间内的距离
    plt.step(range(len(measurements[0])),dxt, label='$estimateVx$')
    plt.step(range(len(measurements[0])),dyt, label='$estimateVy$')

    plt.step(range(len(measurements[0])),measurements[0], label='$measurementVx$')
    plt.step(range(len(measurements[0])),measurements[1], label='$measurementVy$')

    plt.axhline(vx, color='#999999', label='$trueVx$')
    plt.axhline(vy, color='#999999', label='$trueVy$')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best',prop={'size':11})
    plt.ylim([0, 30])
    plt.ylabel('Velocity')

def plot_xy():
    fig = plt.figure(figsize=(16,16))
    plt.scatter(xt,yt, s=20, label='State', c='k')
    plt.scatter(xt[0],yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1],yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')

plot_mesurement()
plot_x() 
plot_xy()
plt.show()