from controller import Robot,Motor,Camera
import numpy as np

TURN_LEFT = np.pi/4     #Wychylenie przednich kół w Lewo
TURN_RIGHT = -np.pi/4   #Wychylenie przednich kół w Prawo
SPEED_MAX = np.pi*4     #Max ustawiana prędkość
SPEED_SLOW = np.pi/6    #Niższa prędkość do ustawiania dla precyzyjnych ruchów

SPEED_REVERSE = -np.pi/2 #Prędkość do nawracania

WHEELRADIUS = 0.15      #Wielkość kół w modelu

MOTOR_NAMES = ["wheel0","wheel1"]
HINGE_NAMES = ["Hinge::Left"]

class MobileRobot():
    hinge_motors:list[Motor]
    wheel_motors:list[Motor]
    def __init__(self,robot:Robot):
        self.hinge_motors = [robot.getDevice(hinge_name) for hinge_name in HINGE_NAMES if robot.getDevice(hinge_name) != None]
        self.wheel_motors = [robot.getDevice(motor_name) for motor_name in MOTOR_NAMES if robot.getDevice(motor_name) != None]
        
        for motor in self.hinge_motors: 
            motor.setPosition(0)
        
        for motor in self.wheel_motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(0)

    def turn_left(self, scale: float=1.0):
        angle = TURN_LEFT*scale
        for motor in self.hinge_motors:
            motor.setPosition(angle)
        return angle

    def turn_right(self, scale: float=1.0):
        angle = TURN_RIGHT*scale
        for motor in self.hinge_motors:
            motor.setPosition(angle)
        return angle

    def go_straight(self):
        for motor in self.hinge_motors:
            motor.setPosition(0)

    def go_forward(self, scale: float=1.0):
        speed = SPEED_MAX*scale
        for motor in self.wheel_motors:
            motor.setVelocity(speed)
        return speed        

    def go_backward(self, scale: float=1.0):
        speed = SPEED_REVERSE*scale
        for motor in self.wheel_motors:
            motor.setVelocity(speed)
        return speed

    def stop(self):
        for motor in self.wheel_motors:
            motor.setVelocity(0)
    
    def go_slow(self):
        for motor in self.wheel_motors:
            motor.setVelocity(SPEED_SLOW)
        return SPEED_SLOW

    def get_current_velocity(self) -> list[float]:
        return [motor.getVelocity() for motor in self.wheel_motors]