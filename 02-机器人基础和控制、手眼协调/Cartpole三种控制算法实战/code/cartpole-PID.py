import random
import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import *
from matplotlib import animation

from cartpole_env import *
from matplotlib.pylab import mpl
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib import font_manager
font_path = "./AiDianFengYaHei（ShangYongMianFei）-2.ttf"  # 改成正确路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

kp_cart = 2
kd_cart = 50
kp_pole = 8
kd_pole = 100
DIRECT_MAG=True
RANDOM_NOISE=False



if DIRECT_MAG:
    env=CartPoleEnv()
else:
    env = gym.make('CartPole-v1')

class CartPoleControl:

    def __init__(self, kp_cart, kd_cart, kp_pole, kd_pole):
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_pole = kp_pole
        self.kd_pole = kd_pole
        self.bias_cart_1 = 0
        self.bias_pole_1 = 0
        self.i=0

    def pid_cart(self, position):
        bias = position  # 
        #bias=self.bias_cart_1*0.8+bias*0.2
        d_bias = bias - self.bias_cart_1
        # print(self.bias_cart_1)
        balance = self.kp_cart * bias + self.kd_cart * d_bias
        self.bias_cart_1 = bias
        return balance

    def pid_pole(self, angle):
        bias = angle  #
        d_bias = bias - self.bias_pole_1
        balance = -self.kp_pole * bias - self.kd_pole * d_bias
        self.bias_pole_1 = bias
        return balance

    def control_output(self, control_cart, control_pole):
        if DIRECT_MAG:
            return -10*(control_pole - control_cart)
        else:
            return 1 if (control_pole - control_cart) < 0 else 0

def save_frames_as_gif(frames, path):
    filename = 'PID_'+ 'CartPole-v1' + '.gif'
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow')

if __name__ == '__main__':

    control=CartPoleControl(kp_cart, kd_cart, kp_pole, kd_pole)

    rewards=0
    state = env.reset()
    k_steps = 401
    X_k = np.zeros((4, k_steps+1))
    X_k[:,0] = state
    # 开辟所有控制输入u的存储空间
    U_k = np.zeros((1, k_steps))
    done = False
    i=0
    step_a = 0
    frames = []
    while abs(state[2]<2) & (step_a < k_steps):
        env.render()
        frames.append(env.render(mode='rgb_array'))
        control_pole = control.pid_pole(state[2])
        control_cart = control.pid_cart(state[0])
        if RANDOM_NOISE and random.random()>0.99:
            i=2

        if i>0:
            if DIRECT_MAG:
                action = 10
            else:
                action=1
            i-=1
        else:
            action = control.control_output(control_cart, control_pole)

        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards+=reward
        X_k[:,step_a+1] = state
        U_k[:,step_a] = action
        step_a += 1
        print(step_a)
    env.close()
    save_frames_as_gif(frames, path = './')


    # 绘制结果
    plt.subplot(2, 1, 1)
    plt.plot(X_k[0, :], label=f"x")
    plt.plot(X_k[1, :], label=f"x_dot")
    plt.plot(X_k[2, :], label=f"theta")
    plt.plot(X_k[3, :], label=f"theta_dot")
    plt.legend()
    plt.title("PID状态变量")
    plt.xlabel("时间步")
    plt.ylabel("状态值")
    
    # 第二个子图: 控制输入
    plt.subplot(2, 1, 2)
    for i in range(U_k.shape[0]):
        plt.plot(U_k[i, :], label=f"u{i+1}")
    plt.legend()
    plt.title("PID控制输入")
    plt.xlabel("时间步")
    plt.ylabel("控制输入值")
    
    # 调整布局并显示
    plt.tight_layout()
    plt.savefig('PID', dpi=1000)
    plt.show()