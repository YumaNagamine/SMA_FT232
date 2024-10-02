import matplotlib.pyplot as plt
import numpy as np
import time
from testFuzzy import FUZZY_CONTROL

class TEST_VALUES():
    def __init__(self):
        pass
    
    def exponential(self, first_value, damp, t): #use these functions as error
        y = first_value * np.exp(-damp * t)
        return y
    
    def exp_cos(self, first_value, damp, freq, t):
        y = first_value * np.exp(-damp * t) * np.cos(freq * t)
        return y
    
    def plot_func(self, start, stop, first_value, coef0 = 0, coef1 = 0, func_name = 'exponential'):
        t = np.arange(start, stop, step = 0.01)
        if func_name == 'exponential':
            y = self.exponential(first_value ,coef0 ,t)
        elif func_name == 'exp_cos':
            y = self.exp_cos(first_value, coef0, coef1, t)
        else:
            print('wrong function name!')
            return
        self.values = y
        plt.plot(t, y)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title('error trajectory test')
        plt.show()


        return
    

if __name__ == '__main__':
    process_share_dict = {}
    target = []
    fuzzy = FUZZY_CONTROL(target , process_share_dict, True)
    test = TEST_VALUES()
    test.plot_func(0, 20, 50, 1)
    test.plot_func(0, 20, 50, 0.5, 2, 'exp_cos')
    t = np.arange(0, 20, 0.01)
    angle0 = test.exp_cos(50, 0.5, 2, t)
    angle1 = test.exponential(50, 1, t)
    angle2 = test.exponential(40, 1, t)
    angle_err = np.vstack((angle0, angle1, angle2))
    angle_err = angle_err.T

    fuzzy.set_slope(-180, 180, 0, 1) #membership functions for angle error
    fuzzy.set_slope(-180, 180, 1, 0) #membership functions for angle error
    fuzzy.set_triangle(-180, 180, 0, 1) #membership functions for angle error
    fuzzy.set_invert_triangle(-360, 360, 0, 0) #membership functions for anglular velocity
    fuzzy.set_triangle(-360, 360, 0, 1, 'Angular Velocity') #membership functions for anglular velocity
    output_values = np.array([[0, 0.1, 0, -0.1]
                              [0, 0.1, 0, -0.1]
                              [0, 0, 0.1, -0.1]
                              [0, 0, 0.1, -0.1]
                             ])
    fuzzy.set_delta_func(output_values) #output membership functions
    for i in range(len(t)):
        process_share_dict['angles'] = angle_err[i]
        print(process_share_dict['angles'])
        fuzzy.Fuzzy_process()
        du = fuzzy.deFuzzy_process()
        print(du)
        