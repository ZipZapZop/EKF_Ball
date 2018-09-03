import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd
import conversion_tools as ct

def kalman(num_trials, dt, x0, y0, v_x0, v_y0, std_dev_x, std_dev_y, std_dev_vx, std_dev_vy, std_dev_sensor):
    noisy = gd.generate_measurements(num_trials, dt, x0, y0, v_x0, v_y0, std_dev_sensor)
    g = -9.8

    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype='float64')

    B = np.array([[0, 0],
                  [0, 0.5*(dt**2)],
                  [0, 0],
                  [0,dt]], dtype='float64')
    u = np.array([[0],
                  [g]])

    howMuchTrust = 0.1
    dt_sq = dt**2
    dt_3rd = (dt_sq*dt)/2
    dt_4th = (dt_sq**2)/4
    Q = np.array([[dt_4th*howMuchTrust, 0, dt_3rd*howMuchTrust, 0],
                    [0, dt_4th*howMuchTrust, 0, dt_3rd*howMuchTrust],
                    [dt_3rd*howMuchTrust, 0, dt_sq*howMuchTrust, 0],
                    [0, dt_3rd*howMuchTrust, 0, dt_sq*howMuchTrust]])

    P = np.diag(np.array([std_dev_x**2, std_dev_y**2, std_dev_vx**2, std_dev_vy**2]))

    R = np.zeros((1,1))
    R[0] = std_dev_sensor**2

    state = np.zeros((4, num_trials), dtype='float64')
    state[:,[0]] = np.vstack([x0, y0, v_x0, v_y0])

    variances = np.zeros((4, num_trials))
    variances[:, [0]] = np.array([[std_dev_x**2, std_dev_y**2, std_dev_vx**2, std_dev_vy**2]]).T

    for i in range(1, num_trials):
        # predict
        state[:, [i]] = np.dot(A, state[:,[i-1]]) + np.dot(B,u)
        # print(state[0,i], state[1,i], state[2, i], state[3, i])
        P = np.dot(np.dot(A,P), A.T) + Q

        # gain and update
        hp = ct.measurement_func(state[:,[i]])
        H = ct.Jacobian(state[:,[i]])
        K_num = np.dot(P, H.T)
        K_denom = np.dot(np.dot(H,P),H.T) + R
        K = np.dot(K_num, np.linalg.inv(K_denom))

        innovation = np.array([noisy[i] - hp])
        state[:,[i]] = state[:,[i]] + np.dot(K,innovation)

        P = np.dot(np.eye(4) - np.dot(K,H), P)
        d = np.array([np.diag(P)]).T
        variances[:,[i]] = d

    return state, variances


t = np.arange(0, 4, 0.1)

# kalman(num_trials, dt, x0, y0, v_x0, v_y0, std_dev_x, std_dev_y, std_dev_vx, std_dev_vy, std_dev_sensor)
state, var = kalman(40, 0.1, 80, 0, -20, 20, 10, 1, 100, 100, 0.001)

# generate_true_states(num_trials, dt, x0, y0, v_x0, v_y0)
true_state = gd.generate_true_states(40,0.1,80,0,-20,20)
# print(true_state[4])
# print(np.shape(state))

plt.figure(0)
# print(true_state[0])
plt.plot(state[0])
plt.plot(true_state[0])
plt.figure(1)
# plt.plot(state[0], state[1])
# plt.plot(state[0])
# plt.plot(true_state[0])

plt.plot(var[0])
plt.plot(var[1])
plt.show()