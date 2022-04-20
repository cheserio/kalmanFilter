import matplotlib.pyplot as plt
import numpy as np


def loadData(fileName):
    inFile = open(fileName, 'r')
    mx = []
    my = []
    x = []
    y = []
    vx = []
    vy = []
    ax = []
    ay = []
    for line in inFile:
        lineData = line.split(' ')
        mx.append(float(lineData[0]))
        x.append(float(lineData[1]))
        my.append(float(lineData[2]))
        y.append(float(lineData[3]))
        vx.append(float(lineData[4]))
        vy.append(float(lineData[5]))
        ax.append(float(lineData[6]))
        ay.append(float(lineData[7]))
    return (mx,my, x, y, vx, vy, ax, ay)
    


def plot_xy(mx, my, x, y):
    # fig = plt.figure(figsize=(16,16))
    plt.scatter(x,y, s=20, label='State', c='k')
    plt.scatter(mx,my, s=20, label='Measurement', c='b')
    plt.scatter(x[0],y[0], s=100, label='Start', c='g')
    plt.scatter(x[-1],y[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("KalmanFilter")
    plt.legend(loc='best')
    plt.axis('equal')
    plt.savefig('./kalmanFilter.png')
    plt.show()

def plot_velocity(v, str, str2):
    # fig = plt.figure(figsize=(16,9))
    plt.step(range(len(v)), v, label='$'+ str + '$')

    plt.xlabel('Filter Step')
    plt.legend(loc='best',prop={'size':11})
    # plt.ylim([0, 30])
    plt.ylabel(str2)
    s = './'+ str + '.png'
    plt.savefig(s)
    plt.show()


(mx, my, x, y, vx, vy, ax, ay) = loadData("/home/shiy/kalmanFilter/build/data.txt")
plot_xy(mx, my, x, y)
plot_velocity(vx, "estimate Vx", "Velocity")
plot_velocity(vy, "estimate Vy", "Velocity")
plot_velocity(ax, "estimate ax", "Acceleration")
plot_velocity(ay, "estimate ay", "Acceleration")


