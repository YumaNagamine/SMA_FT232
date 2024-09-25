import numpy as np
import time 
import matplotlib.pyplot as plt


class FUZZY_CONTROL():
    def __init__(self, target=[], process_share_dict={}, savedata = False):
        self.target = np.array(target)
        self.prev_angles = np.array(process_share_dict['angles'])

        self.step = 1
        self.prev_time = time.time()
        self.du_coef_list = np.zeros((4,4,2)) #for set_slope_triangle
        self.weight  = np.array([[1,1,1],
                                 [0,1,1],
                                 [1,1,1],
                                 [1,1,1]]) # can be modified
        self.den = np.sum(self.weight, axis=1)
        if savedata == True:
            self.startingtime = time.time()
            st = time.time() - self.startingtime            
            self.output_with_time = np.array([st,0,0,0,0])


    def save_angle(self, process_share_dict = {}): 
        current_angles = np.array(process_share_dict['angles'])
        t = time.time()
        dt = t - self.prev_time 
        self.error = self.target - current_angles
        self.angular_velocity = (current_angles - self.prev_angles)/dt
        self.prev_angles = current_angles
        self.prev_time = t
        return
    
    def LinearFunc_coef(self, start_x, stop_x, start_y, stop_y):
        a = (stop_y - start_y)/(stop_x - start_x)
        b = (stop_x*start_y - start_x*stop_y)/(stop_x - start_x)
        return [a, b]
    def get_LinearFunc_array(self, start_x, stop_x, start_y, stop_y):
        coef_list = self.LinearFunc_coef(start_x, stop_x, start_y, stop_y)
        x = np.arange(start_x, stop_x + self.step, self.step)
        y = coef_list[0]*x + coef_list[1]
        coord = np.vstack((x,y)).T
        return coord
    
    def set_slope(self, start_x, stop_x, start_y, stop_y, name: str): # for IF part
        [a,b] = self.LinearFunc_coef(start_x, stop_x, start_y, stop_y)
        if name == 'Angle Error':
            if start_y < stop_y:  self.AE_uphill_coef = [a, b]
            else : self.AE_downhill_coef = [a,b]
        elif name == 'Angular Velocity':
            if start_y < stop_y:  self.AV_uphill_coef = [a, b]
            else : self.AV_downhill_coef = [a,b]
        return
    
    def set_triangle(self, minimum_x, maximum_x, center_x = 0, center_y = 1, name = 'Angle Error'):
        self.center_x = center_x
        initial_value = 0
        final_value = 0
        if name == 'Angle Error':
            self.AE_triangle_leftroot = minimum_x
            self.AE_triangle_rightroot = maximum_x
            [a,b] = self.LinearFunc_coef(minimum_x, center_x, initial_value, center_y)
            [c,d] = self.LinearFunc_coef(center_x, maximum_x, center_y, final_value)
            self.AE_leftside_triangle = [a,b]
            self.AE_rightside_triangle = [c,d]
        elif name == 'Angular Velocity':
            self.AV_triangle_leftroot = minimum_x
            self.AV_triangle_rightroot = maximum_x
            [a,b] = self.LinearFunc_coef(minimum_x, center_x, initial_value, center_y)
            [c,d] = self.LinearFunc_coef(center_x, maximum_x, center_y, final_value)
            self.AV_leftside_triangle = [a,b]
            self.AV_rightside_triangle = [c,d]

        return

    def set_slope_triangle(self, param:np.ndarray[np.float32], duty_ratio_num: int): # for THEN part, this is no longer necessary
        '''
        duty_ratio_num is 0-3. 0 and 1 for Extension, 2 and 3 for Flexion
        
        param = [[[start_x, stop_x, start_y, stop_y] (for uphill) [start_x, stop_x, start_y, stop_y] (for downhill) [minimum_x, maximum_x, center_x, center_y] (for triangle)] <- for du0
                 [[start_x, stop_x, start_y, stop_y] (for uphill) [start_x, stop_x, start_y, stop_y] (for downhill) [minimum_x, maximum_x, center_x, center_y] (for triangle)] <- for du1
                 [[start_x, stop_x, start_y, stop_y] (for uphill) [start_x, stop_x, start_y, stop_y] (for downhill) [minimum_x, maximum_x, center_x, center_y] (for triangle)] <- for du2
                 [[start_x, stop_x, start_y, stop_y] (for uphill) [start_x, stop_x, start_y, stop_y] (for downhill) [minimum_x, maximum_x, center_x, center_y] (for triangle)] <- for du3
                 ]
                 (4x3x4 array)
        
        self.du_coef_list is like as follow:
        self.du_coef_list = [[[a,b][c,d][e,f][g,h]] coefficients lists for du0 
                             [[a,b][c,d][e,f][g,h]] coefficients lists for du1
                             [[a,b][c,d][e,f][g,h]] coefficients lists for du2
                             [[a,b][c,d][e,f][g,h]] coefficients lists for du3
                            ]
                            (4x4x2 array)
        each coefficients list consists of coef of uphill, downhill, leftside triangle, rightside triangle
        
        this function is designed to be used with for-loop on duty_ratio_num
        
        '''
        for i in range(4):
            self.du_coef_list[duty_ratio_num][i] = self.LinearFunc_coef(param[duty_ratio_num][i][0], param[duty_ratio_num][i][1], param[duty_ratio_num][i][3], param[duty_ratio_num][i][4])

        return

    def set_delta_func(self, output_values:np.ndarray[np.float32]): #for THEN part, this is probably faster than set_slope_triangle
        """ 
        output_values should be:
        [[remain, flex, extend=0, resist] for duty ratio0
         [remain, flex, extend=0,resist] for duty ratio1
         [remain, flex=0, extend, resist] for duty ratio2
         [remain, flex=0, extend, resist] for duty ratio3
         ]
        """
        self.output_values = output_values
        return
        
    def uphill_value(self, x, name = 'Angle Error'): # for IF part #must done after function set_trianglefor 
        if name == 'Angle Error':
            if x > self.center_x:
                y = self.AE_uphill_coef[0]*x + self.AE_uphill_coef[1]
                return y
            else: return 0
        elif name == 'Angular Velocity':
            if x > self.center_x:
                y = self.AV_uphill_coef[0]*x + self.AV_uphill_coef[1]
                return y
            else: return 0
     
    def downhill_value(self, x, name = 'Angle Error'): # for IF part
        if name == 'Angle Error':
            if x < self.center_x:
                y = self.AE_downhill_coef[0]*x + self.AE_downhill_coef[1]
                return y
            else: return 0
        elif name == 'Angular Velocity':
            if x > self.center_x:
                y = self.AV_downhill_coef[0]*x + self.AV_downhill_coef[1]
                return y

    def inverttriangle_value(self, x):# for IF part
        if x <= self.center_x:
            y = self.AV_downhill_coef[0]*x + self.AV_downhill_coef[1]
            return y
        elif x > self.center_x:
            y = self.AV_uphill_coef[0]*x + self.AV_uphill_coef[1]
            return y

    def triangle_value(self, x, name = 'Angle Error'): # for IF part
        if name == 'Angle Error':
            if self.AE_triangle_leftroot <= x <= self.center_x:
                y = self.AE_leftside_triangle[0]*x + self.AE_leftside_triangle[1]
                return y
            elif self.center_x <= x < self.AE_triangle_rightroot:
                y = self.AE_rightside_triangle[0]*x + self.AE_rightside_triangle[1]
                return y
            else : return 0
        elif name == 'Angular Velocity':
            if self.AV_triangle_leftroot <= x <= self.center_x:
                y = self.AV_leftside_triangle[0]*x + self.AV_leftside_triangle[1]
                return y
            elif self.center_x <= x < self.AV_triangle_rightroot:
                y = self.AV_rightside_triangle[0]*x + self.AV_rightside_triangle[1]
                return y
            else : return 0

    def Fuzzy_process(self):
        '''
        IF THEN rules:(E: error, w: absolute value of angle velocity )
        1 IF E > 0 large and w large THEN remain
        2 IF E > 0 large and w small THEN du > 0, FLEX
        3 IF E < 0 large and w large THEN remain
        4 IF E < 0 large and w small THEN du > 0, EXTEND
        5 IF E small and w large THEN du < 0, RESIST
        6 IF E small and w small THEN remain
        '''
        # error0, error1, error2 = self.error[0], self.error[1], self.error[2]
        AE_membership_degree = np.zeros((3,3))
        AE_membership_degree[0] = self.uphill_value(self.error, name ='Angle Error')
        AE_membership_degree[1] = self.downhill_value(self.error, name = 'Angle Error')
        AE_membership_degree[2] = self.triangle_value(self.error, name = 'Angle Error')
        AE_membership_degree = AE_membership_degree.T # AE membership degree[i] is [md(uphill),md(downhill),md(triangle)] for angle i

        AV_membership_degree = np.zeros((2,3))
        AV_membership_degree[0] = self.inverttriangle_value(self.angular_velocity)
        AV_membership_degree[1] = self.triangle_value(self.angular_velocity, name = 'Angular Velocity')
        AV_membership_degree = AV_membership_degree.T # AV membership degree[i] is [md(invert triangle), md(triangle)] for angle i

        #determine membership degree by following IF-THEN rules
        # membership_degree = []
        # for angle_num in range(3):
        #     for AE in range(3):
        #         for AV in range(2):
        #             membership_degree.append(min(AE_membership_degree[angle_num][AE], AV_membership_degree[angle_num][AV]))
        # membership_degree = np.array(membership_degree).reshape(3,6)
        self.membership_degree = np.minimum(AE_membership_degree[:, :, np.newaxis], AV_membership_degree[:, :, np.newaxis]).reshape(3, 6) 
        '''
        membership degree array should be as follows
        [[1,2,3,4,5,6](angle0)
         [1,2,3,4,5,6](angle1)
         [1,2,3,4,5,6](angle2)
        ] 1-6 is membership degree of each IF-THEN rule
        '''
        return
    def weighting_membership_degree(self):
        new_membership_degree = np.zeros((3,4))
        # for i in range(3):
        #     new_membership_degree[i][0] = self.membership_degree[i][0] + self.membership_degree[i][2] + self.membership_degree[i][5] # remain
        #     new_membership_degree[i][1] = self.membership_degree[i][1] # flex
        #     new_membership_degree[i][2] = self.membership_degree[i][3] # extend
        #     new_membership_degree[i][3] = self.membership_degree[i][4] # resist
        new_membership_degree[:, 0] = self.membership_degree[:, 0] + self.membership_degree[:, 2] + self.membership_degree[:, 5] #remain
        new_membership_degree[:, 1] = self.membership_degree[:, 1] #flex
        new_membership_degree[:, 2] = self.membership_degree[:, 3] #extend
        new_membership_degree[:, 3] = self.membership_degree[:, 4] #resist

        #weight = [[1,1,1],[0,1,1],[1,1,1],[1,1,1]] np.ndarray

        self.new_membership_degree = (np.diag(self.weight @ new_membership_degree))/self.den #obtain weighted membership degree:[remain, flex, extend, resist]
        self.new_membership_degree = np.tile(self.new_membership_degree, (4,1))
        return

    def remain_func(self, x):
        pass
    def flex_func(self,):
        pass
    def extend_func(self, ):
        pass
    def centroid_method_delta(self):
        num = np.diag(self.new_membership_degree @ self.output_values.T)
        den = np.sum(self.new_membership_degree, axis=1)
        centroid = num/den
        return centroid

    def deFuzzy_process(self):
        self.weighting_membership_degree()
        output = self.centroid_method_delta()
        return output
    
    def save_output(self, output):
        nptime = np.array( [self.prev_time - self.startingtime] )
        output_with_time = np.hstack((nptime, output))
        self.output_with_time = np.vstack((self.output_with_time, output_with_time))
        return
    
    def get_triangle_array(self, minimum_x, maximum_x, center_x, center_y):
        initial_value = 0
        final_value = 0
        x1 = np.arange(minimum_x, center_x, self.step)
        x2 = np.arange(center_x, maximum_x + self.step, self.step)
        [a,b] = self.LinearFunc_coef(minimum_x, center_x, initial_value, center_y)
        [c,d] = self.LinearFunc_coef(center_x, maximum_x, center_y, final_value)
        
        y1 = a*x1 + b
        y2 = c*x2 + d
        x = np.hstack((x1, x2))
        y = np.hstack((y1, y2))
        triangle = np.vstack((x,y)).T
        return triangle
    
    def set_membership_arrays(self, start_x, stop_x, triangle_root = [], center_x = 0, center_y = 1):
        triangle = self.get_triangle_array(triangle_root[0], triangle_root[1], center_x, center_y)

        if start_x < triangle_root[0]:
            triangle = triangle.T
            x = np.arange(start_x, triangle_root[0], self.step)
            y = np.zeros(len(x))
            triangle = [np.concatenate((row, arr), axis= None) for row, arr in zip([x,y],triangle)]
            triangle = np.array(triangle)
            triangle = triangle.T
        if triangle_root[1] < stop_x:
            triangle = triangle.T
            x = np.arange(triangle_root[1], stop_x + self.step, self.step)
            y = np.zeros(len(x))
            triangle = [np.concatenate((row, arr), axis= None) for row, arr in zip(triangle, [x,y])]
            triangle = np.array(triangle)
            triangle = triangle.T
        
        uphill = self.get_LinearFunc_array(center_x, stop_x, 0, 1)
        downhill = self.get_LinearFunc_array(start_x, center_x, 1, 0)
        uphill = uphill.T
        downhill = downhill.T
        x1 = np.arange(start_x, center_x, self.step)
        x2 = np.arange(center_x + self.step, stop_x + self.step, self.step)
        zeros = np.zeros(len(x1))
        uphill = [np.concatenate((row, arr), axis= None) for row, arr in zip([x1,zeros],uphill)]
        downhill = [np.concatenate((row, arr), axis= None) for row, arr in zip(downhill,[x2,zeros])]
        uphill = np.array(uphill).T
        downhill = np.array(downhill).T

        return triangle, uphill, downhill
    
    
    def Fuzzy_main(self, process_share_dict) -> np.ndarray: # guide how to use the class. parameters must be set previously
        self.save_angle(process_share_dict)
        self.Fuzzy_process()
        du = self.deFuzzy_process()
        return du
    
    def show_membership_func(self):
        
        return
    

    def visualize_output(self):
        S = 1
        if S == 1:
            self.output_with_time = self.output_with_time.T
            plt.plot(self.output_with_time[0],self.output_with_time[1], label = 'du0')
            plt.plot(self.output_with_time[0],self.output_with_time[2], label = 'du1')
            plt.plot(self.output_with_time[0],self.output_with_time[3], label = 'du2')
            plt.plot(self.output_with_time[0],self.output_with_time[4], label = 'du3')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('output')
            plt.show()
        else:
            timedata = self.output_with_time[:, 0]
            data0 = self.output_with_time[:, 1]
            data1 = self.output_with_time[:, 2]
            data2 = self.output_with_time[:, 3]
            data3 = self.output_with_time[:, 4]
            plt.plot(timedata, data0, label = 'du0')
            plt.plot(timedata, data1, label = 'du1')
            plt.plot(timedata, data2, label = 'du2')
            plt.plot(timedata, data3, label = 'du3')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('output')
            plt.show()

        return
    


if __name__ == '__main__':
    target = [150, 100, 200]
    

    

