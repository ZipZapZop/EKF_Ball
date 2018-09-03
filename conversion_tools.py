import numpy as np

def measurement_func(cart_state):  # returns scalar
    # hx = np.array((1,1), dtype = 'float64')
    hx= np.arctan2(cart_state[1], cart_state[0])
    return hx

def Jacobian(cart_state):   # return 1x4
    xsq_ysq = np.sqrt(cart_state[0]**2 + cart_state[1]**2)
    return np.array([[-cart_state[1]/xsq_ysq, cart_state[0]/xsq_ysq, 0, 0]], dtype='float64')

# x = Jacobian(np.array([2,2]))
# print(np.shape(x))