import numpy as np
import time 

class FUZZY_CONTROL():
    def __init__(self, target=[], process_share_dict={}):
        self.target = np.array(target)
        self.prev_angles = np.array(process_share_dict['angles'])

        self.step = 1
        self.prev_time = time.time()

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
    
    def set_slope(self, start_x, stop_x, start_y, stop_y, name = 'Angle Error'):
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
        
    def uphill_value(self, x, name = 'Angle Error'): #must done after function set_triangle
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
     
    def downhill_value(self, x, name = 'Angle Error'):
        if name == 'Angle Error':
            if x < self.center_x:
                y = self.AE_downhill_coef[0]*x + self.AE_downhill_coef[1]
                return y
            else: return 0
        elif name == 'Angular Velocity':
            if x > self.center_x:
                y = self.AV_downhill_coef[0]*x + self.AV_downhill_coef[1]
                return y

    def inverttriangle_value(self, x):
        if x <= self.center_x:
            y = self.AV_downhill_coef[0]*x + self.AV_downhill_coef[1]
            return y
        elif x > self.center_x:
            y = self.AV_uphill_coef[0]*x + self.AV_uphill_coef[1]
            return y

    def triangle_value(self, x, name = 'Angle Error'):
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
        # error0, error1, error2 = self.error[0], self.error[1], self.error[2]
        AE_membership_degree = np.zeros((3,3))
        AE_membership_degree[0] = self.uphill_value(self.error)
        AE_membership_degree[1] = self.downhill_value(self.error)
        AE_membership_degree[2] = self.triangle_value(self.error)
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
        membership_degree = np.minimum(AE_membership_degree[:, :, np.newaxis], AV_membership_degree[:, :, np.newaxis]).reshape(3, 6) # to be verified


        pass
    def deFuzzy_process(self):
        pass
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
    
        
    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    process_share_dict = {}
    process_share_dict['angles'] = [100,70,120]
    target = [150, 100, 200]
    f = FUZZY_CONTROL(target, process_share_dict)

    mem_tuple = f.set_membership_arrays(-180,180,[-90,90])
    for i in range(3):
        # print(mem_tuple[i])
        plt.plot(mem_tuple[i][:, 0], mem_tuple[i][:, 1])
    plt.show()

    

