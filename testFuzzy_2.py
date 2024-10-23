import numpy as np
import time


class FUZZYCONTROL():
    def __init__(self):
        pass

    def input_target(self, current_angles):
        remain_upper_limit = 5
        remain_lower_limit = -5
        flag = True
        err = [0,0,0]
        #setting target of angle0
        while True:
            print("Satisfy 90 < angle0 < 180")
            angle0 = int(input("angle0: "))
            err[0] = angle0 - current_angles[0]
            if not 90 < angle0 < 180:
                print("This angle is impossible. Try again...")
                print()
                continue
            if remain_lower_limit < err[0] < remain_upper_limit:
                print("Target angle is considered as the same as current angle. ")
                print("Skip setting target angle1... ")
                print()
                flag = False
                DIP_PIP_mode = 'remain'
                break 
            if 90 < angle0 < 180:
                if err[0] > remain_upper_limit:
                    DIP_PIP_mode = 'extend'
                elif err[0] < remain_lower_limit:
                    DIP_PIP_mode = 'flex'
                else:
                    print("Find some bags... Force-quiting..")
                    return
                print("error01 mode: ", DIP_PIP_mode)
                break
        #setting target of angle1
        if flag:
            if DIP_PIP_mode == 'flex':
                while True:
                    print("Satisfy 90 < angle1 < ", current_angles[1]+remain_lower_limit)
                    angle1 = int(input("angle1 :"))
                    if 90  < angle1 < current_angles[1] + remain_lower_limit:
                        err[1] = angle1 - current_angles[1]
                        break
                    else:
                        print("This angle is impossible. Try again...")
                        print()

            elif DIP_PIP_mode == 'extend':
                while True:
                    print(f"Satisfy {current_angles[1]+remain_upper_limit} < angle1 < 180")
                    angle1 = int(input("angle1 :"))
                    if current_angles[1] + remain_upper_limit < angle1 < 180:
                        err[1] = angle1 - current_angles[1]
                        break
                    else: 
                        print("This angle is impossible. Try again...")
                        print()
            
        #setting target of angle2
        while True:
            print("Satisfy 90 < angle2 < 250")
            angle2 = int(input("angle2: "))
            err[2] = angle2 - current_angles[2]
            if remain_lower_limit < err[2] < remain_upper_limit:
                print("Target angle is considered as the same as current angle. ")
                MP_mode = 'remain'                
                break
            if 90 < angle2 < 250:
                if err[2] < remain_lower_limit:
                    MP_mode = 'flex'
                elif err[2] > remain_upper_limit:
                    MP_mode = 'extend'
                break
            else:
                print("This angle is impossible. Try again...")
                print()
        
        self.mode = [DIP_PIP_mode, MP_mode]
        self.err = err
        print("mode=", self.mode)
        print("errors= ", err)

    def control_method(self, err):
        if self.mode[0] == 'flex' and self.mode[1] == 'flex':
            self.controlmethod_FF(err)
            pass
        elif self.mode[0] == 'flex' and self.mode[1] == 'extend':
            self.controlmethod_FE(err)
            pass
        elif self.mode[0] == 'flex' and self.mode[1] == 'remain':
            self.controlmethod_FR(err)
            pass
        elif self.mode[0] == 'extend' and self.mode[1] == 'flex':
            self.controlmethod_EF(err)
            pass
        elif self.mode[0] == 'extend' and self.mode[1] == 'extend':
            self.controlmethod_EE(err)
            pass
        elif self.mode[0] == 'extend' and self.mode[1] == 'remain':
            self.controlmethod_ER(err)
            pass
        elif self.mode[0] == 'remain' and self.mode[1] == 'flex':
            self.controlmethod_RF(err)
            pass
        elif self.mode[0] == 'remain' and self.mode[1] == 'extend':
            self.controlmethod_RE(err)
            pass
        elif self.mode[0] == 'remain' and self.mode[1] == 'remain':
            self.controlmethod_RR(err)
            pass
        
    def controlmethod_FF(self,err):
        # err  = [angle0, angle1, angle2]
        err = np.array(err
                       )
        pass

if __name__ == "__main__":
    f = FUZZYCONTROL()
    target_angles = [135, 135, 200]
    f.input_target(target_angles)
    while True:
        f.control_method()
        pass
