import numpy as np
import matplotlib.pyplot as plt

def generate_true_states(num_trials, dt, x0, y0, v_x0, v_y0):
    state = np.zeros((5, num_trials), dtype='float64') # x, y, v_x, v_y, theta
    g = -9.8
    state[:,[0]] = np.vstack([x0, y0, v_x0, v_y0, np.arctan2(y0,x0)])
    t = np.arange(0,4,0.1)
    for i in range(1, num_trials):
        state[0,[i]] = x0 + v_x0*t[i]
        state[1,[i]] = v_y0*t[i] + (1/2)*g*(t[i])**2
        state[2,[i]] = state[2,[i-1]]
        state[3,[i]] = state[3,[i-1]] + g*t[i]
        state[4,[i]] = np.arctan2(state[1,i], state[0,i])

    return state

def generate_measurements(num_trials, dt, x0, y0, v_x0, v_y0, std_dev_sensor): # output is theta
    ''' measurements are angle of elevation of the ball in degrees '''
    state = generate_true_states(num_trials, dt, x0, y0, v_x0, v_y0)

    for i in range(0, num_trials):
        state[4,i] = state[4,i] + np.random.normal(0, std_dev_sensor)

    return np.array(state[4,:])

# a = generate_true_states(40, 0.1, 80, 0, -20, 20)
# x = generate_measurements(40, 0.1, 80, 0, -20, 20, 0.001)


# plt.figure(0)
# plt.plot(x)
# # plt.figure(1)
# # plt.plot(a[0], a[1])
# plt.show()

# print(x)