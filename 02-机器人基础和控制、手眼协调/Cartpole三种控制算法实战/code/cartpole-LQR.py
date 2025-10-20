import numpy as np
import random
import gym
import math
import time
from scipy import linalg
import matplotlib.pyplot as plt
from collections import deque
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import *
from copy import deepcopy, copy
from matplotlib import animation
from cartpole_env import *
from matplotlib.pylab import mpl

from matplotlib import font_manager
font_path = "./AiDianFengYaHei（ShangYongMianFei）-2.ttf"  # 改成正确路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

DIRECT_MAG=True
RANDOM_NOISE=False

if DIRECT_MAG:
    env=CartPoleEnv()
else:
    env = gym.make('CartPole-v1')

class CartPoleControl_LQR:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.i=0

    def control_output(self, state_des, state_now):
        S = np.matrix(linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        K = np.matrix(linalg.inv(self.B.T * S * self.B + self.R) * (self.B.T * S * self.A))
        # print("state_now", state_now)
        # print("state_des", state_des)
        action_list  = -10*np.dot(K, state_now - state_des)
        action = action_list[0,0]
        return action


def save_frames_as_gif(frames, path):
    filename = 'LQR_'+ 'CartPole-v1' + '.gif'
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow')

if __name__ == '__main__':
    L = env.length
    M = env.masscart
    m = env.masspole
    del_t = env.tau
    g = env.gravity
    state = env.reset()
    k_steps = 201
    X_k = np.zeros((4, k_steps+1))
    X_k[:,0] = state
    # 开辟所有控制输入u的存储空间
    U_k = np.zeros((1, k_steps))
    x_goal = np.array([0.0,
                       0.0,
                       0.0,
                       0.0])
    Q = np.eye(4)
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    R = 0.5*np.eye(1)

    A = np.eye(4) + del_t * np.array([[0, 1, 0, 0],
                  [0, 0, (-3*m*g)/(m+4*M), 0],
                  [0, 0, 0, 1],
                  [0, 0, (3*(M+m)*g)/(L*(m+4*M)), 0]])
    B = del_t * np.array([[0],
                  [(1/(M+m))+((3*m)/(m+4*M))],
                  [0],
                  [-(3)/(L*(m+4*M))]])

    control=CartPoleControl_LQR(A, B, Q, R)


    rewards=0
    done = False
    i=0
    step_a = 0
    frames = []
    while abs(state[2]<2) & (step_a < k_steps):
        env.render()
        frames.append(env.render(mode='rgb_array'))
        if RANDOM_NOISE and random.random()>0.99:
            i=2

        if i>0:
            if DIRECT_MAG:
                action = 10
            else:
                action=1
            i-=1
        else:
            action = control.control_output(x_goal, state)

        next_state, reward, done, _ = env.step(action)
        state = next_state
        X_k[:,step_a+1] = state
        U_k[:,step_a] = action
        step_a += 1
    env.close()
    save_frames_as_gif(frames, path = './')

    # 绘制结果
    plt.subplot(1, 2, 1)
    plt.plot(X_k[0, :], label=f"x")
    plt.plot(X_k[1, :], label=f"x_dot")
    plt.plot(X_k[2, :], label=f"theta")
    plt.plot(X_k[3, :], label=f"theta_dot")
    plt.legend()
    plt.title("LQR状态变量")
    plt.xlabel("时间步")
    plt.ylabel("状态值")
    # 第二个子图: 控制输入
    plt.subplot(1, 2, 2)
    for i in range(U_k.shape[0]):
        plt.plot(U_k[i, :], label=f"u{i+1}")
    plt.legend()
    plt.title("LQR控制输入")
    plt.xlabel("时间步")
    plt.ylabel("控制输入值")
    
    # 调整布局并显示
    plt.tight_layout()
    plt.savefig('LQR', dpi=1000)
    plt.show()
