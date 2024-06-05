# Python Script
# https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/

import RPi.GPIO as GPIO          
import time

# GPIO config
L_wheels = [23,24,25] #ENB, IN3,IN4
R_wheels = [1,7,8] #ENA, IN1,IN2
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def GPIO_init(EN,IN1,IN2):
    GPIO.setup(EN,GPIO.OUT)
    GPIO.setup(IN1,GPIO.OUT)
    GPIO.setup(IN2,GPIO.OUT)
    #Set 2 motor off
    GPIO.output(IN1,GPIO.LOW)
    GPIO.output(IN2,GPIO.LOW)

# Init motor
GPIO_init(L_wheels[0],L_wheels[1],L_wheels[2])
GPIO_init(R_wheels[0],R_wheels[1],R_wheels[2])

# PMW 
pL=GPIO.PWM(L_wheels[0],1000)
pR=GPIO.PWM(R_wheels[0],1000)
pL.start(20)
pR.start(20)


def motor_status(status,motor_volume,direction,GPIO1,GPIO2,pwm):
    if status == 'start':
        pwm.ChangeDutyCycle(motor_volume)
        if direction == 'toward':
            GPIO.output(GPIO1,GPIO.HIGH)
            GPIO.output(GPIO2,GPIO.LOW)
            
        elif direction == 'backward':
            GPIO.output(GPIO1,GPIO.LOW)
            GPIO.output(GPIO2,GPIO.HIGH)
    elif status == 'stop':
        pwm.start(0)
        GPIO.output(GPIO1,GPIO.LOW)
        GPIO.output(GPIO2,GPIO.LOW)
    

def go_straight(volume):
    motor_status('start',volume,'toward',R_wheels[1],R_wheels[2],pR)
    motor_status('start',volume,'toward',L_wheels[1],L_wheels[2],pL)

def go_back(volume):
    motor_status('start',volume,'backward',R_wheels[1],R_wheels[2],pR)
    motor_status('start',volume,'backward',L_wheels[1],L_wheels[2],pL)

def turn_left():
    motor_status('start',100,'toward',R_wheels[1],R_wheels[2],pR)
    motor_status('stop',0,'toward',L_wheels[1],L_wheels[2],pL)

def turn_right():
    motor_status('stop',0,'toward',R_wheels[1],R_wheels[2],pR)
    motor_status('start',100,'toward',L_wheels[1],L_wheels[2],pL)
def stop():
    motor_status('stop',0,'toward',R_wheels[1],R_wheels[2],pR)
    motor_status('stop',0,'toward',L_wheels[1],L_wheels[2],pL)
def release():
    GPIO.output(L_wheels[0],GPIO.LOW)
    GPIO.output(L_wheels[1],GPIO.LOW)
    GPIO.output(L_wheels[2],GPIO.LOW)
    GPIO.output(R_wheels[0],GPIO.LOW)
    GPIO.output(R_wheels[1],GPIO.LOW)
    GPIO.output(R_wheels[2],GPIO.LOW)
    
go_straight(100)
time.sleep(2)
turn_left()
time.sleep(1)
stop()

    
