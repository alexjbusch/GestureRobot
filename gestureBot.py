import mediapipe
 
import cv2
from enum import IntEnum
import math
import numpy as np

#-*- coding:UTF-8 -*-
import RPi.GPIO as GPIO
import time

#Definition of  motor pin 
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13


LED_R = 22
LED_G = 27
LED_B = 24

#Set the GPIO port to BCM encoding mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_R, GPIO.OUT)
GPIO.setup(LED_G, GPIO.OUT)
GPIO.setup(LED_B, GPIO.OUT)
#Ignore warning information
GPIO.setwarnings(False)




frames_without_detection = 0





#Motor pin initialization operation
def motor_init():
    global pwm_ENA
    global pwm_ENB
    global delaytime
    GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
    #Set the PWM pin and frequency is 2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)

#advance
def forward(delaytime=None):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

#back
def back(delaytime=None):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

#turn left
def left(delaytime=None):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

#turn right
def right(delaytime=None):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

#turn left in place
def spin_left(delaytime=None):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

#turn right in place
def spin_right(delaytime=None):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

#brake
def brake(delaytime=None):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(50)
    pwm_ENB.ChangeDutyCycle(50)
    if delaytime:
        time.sleep(delaytime)

motor_init()

#The try/except statement is used to detect errors in the try block.
#the except statement catches the exception information and processes it.
#The robot car advance 1sï¼Œback 1sï¼Œturn left 2sï¼Œturn right 2sï¼Œturn left  in place 3s
#turn right  in place 3sï¼Œstop 1sã€‚

"""
try:
    motor_init()
    while True:
        run(.01)
        back(.5)
        left(.01)
        right(.01)
        spin_left(.1)
        spin_right(.1)
        brake(1)
except KeyboardInterrupt:
    pass
pwm_ENA.stop()
pwm_ENB.stop()
GPIO.cleanup()
"""






class HandPoint(IntEnum):
    LEFT_HAND = 0
    RIGHT_HAND = 1

    PALM = 0
    PALM_THUMB = 1
    
    THUMB_TIP = 4
    THUMB_MIDDLE = 3
    THUMB_BASE = 2
    

    INDEX_TIP = 8
    INDEX_UNDERTIP = 7
    INDEX_MIDDLE = 6
    INDEX_KNUCKLE = 5

    MIDDLE_TIP = 12
    MIDDLE_UNDERTIP = 11
    MIDDLE_MIDDLE = 10 
    MIDDLE_KNUCKLE = 9

    RING_TIP = 16
    RING_UNDERTIP = 15
    RING_MIDDLE = 14
    RING_KNUCKLE = 13

    PINKY_TIP = 20
    PINKY_UNDERTIP = 19
    PINKY_MIDDLE = 18
    PINKY_KNUCKLE = 17

def get_HandPoint_pos(hand, handPoint):
        if hand <= len(points)-1:
            x1 = points[hand].landmark[handPoint].x
            y1 = points[hand].landmark[handPoint].y
            image_rows, image_cols, _ = frame.shape
            screen_coords = mpDraw._normalized_to_pixel_coordinates(x1, y1,
                                                       image_cols, image_rows)
            return screen_coords
        else:
            raise ValueError
        
    
def euclidian_dist(a,b):
      if a and b:
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
      else:
            return None


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def get_angle(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        return 360 - ang_deg
    else: 
        return ang_deg


def red():
    GPIO.output(LED_R, GPIO.HIGH)
    GPIO.output(LED_G, GPIO.LOW)
    GPIO.output(LED_B, GPIO.LOW)

def blue():
    GPIO.output(LED_R, GPIO.LOW)
    GPIO.output(LED_G, GPIO.LOW)
    GPIO.output(LED_B, GPIO.HIGH)


def green():
    GPIO.output(LED_R, GPIO.LOW)
    GPIO.output(LED_G, GPIO.HIGH)
    GPIO.output(LED_B, GPIO.LOW)



mpDraw = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
      while True:
            ret, frame = cap.read()
            if ret:
                #flipped = cv2.flip(frame, flipCode = -1)
                #frame1 = cv2.resize(frame, (640, 480))
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = results.multi_hand_landmarks



                
                if points:
                    frames_without_detection = 0
                    

                    if len(points) > 1:
                        green()

                        thumb1 = get_HandPoint_pos(0,HandPoint.THUMB_TIP)
                        thumb2 = get_HandPoint_pos(1,HandPoint.THUMB_TIP)
                        
                        index1 = get_HandPoint_pos(0,HandPoint.INDEX_TIP)
                        index2 = get_HandPoint_pos(1,HandPoint.INDEX_TIP)


                        index_mid_1 = get_HandPoint_pos(0,HandPoint.INDEX_MIDDLE)
                        index_mid_2 = get_HandPoint_pos(1,HandPoint.INDEX_MIDDLE)

                        pinky1 = get_HandPoint_pos(0,HandPoint.PINKY_TIP)
                        pinky2 = get_HandPoint_pos(1,HandPoint.PINKY_TIP)


                        middle1 = get_HandPoint_pos(0,HandPoint.MIDDLE_TIP)
                        middle2 = get_HandPoint_pos(1,HandPoint.MIDDLE_TIP)
                        
                        index_distance = euclidian_dist(index1,index2)
                        pinky_distance = euclidian_dist(pinky1,pinky2)
                        middle_distance = euclidian_dist(middle1,middle2)
                        thumb_distance = euclidian_dist(thumb1,thumb2)

                        index_pinky_distance_1 = euclidian_dist(index1,pinky2)
                        index_pinky_distance_2 = euclidian_dist(index2,pinky1)


                        if index_distance and middle_distance and thumb_distance:
                            spade_ratio = index_distance * thumb_distance / middle_distance
                            diamond_ratio = index_distance * middle_distance / thumb_distance
                            if spade_ratio < 3 and diamond_ratio > 3:
                                back()
                            elif spade_ratio > 3 and diamond_ratio < 3:
                                forward()
                            print(f"{round(spade_ratio,3)}          {round(diamond_ratio,3)}")

                        """
                        metal_ratio = None
                        if index_distance and pinky_distance:
                            metal_ratio = index_distance /pinky_distance

                        peace_ratio = None
                        if index_distance and middle_distance:
                            peace_ratio = index_distance /middle_distance

                        

                        Z_ratio = None
                        if index_pinky_distance_1 and index_pinky_distance_2:
                            Z_ratio = index_pinky_distance_1 / index_pinky_distance_2
                            print(Z_ratio)
                            if Z_ratio > 10:
                                back()
                                continue
                                
                            
                            
                            

                        try:
                            if metal_ratio or peace_ratio:
                                if metal_ratio < 0.3 or peace_ratio < 0.3:
                                    run()
                                    continue
                        except:
                            continue
                        """

                        
                    else:
                        blue()

                    kon_ratio = None
                    reverse_kon_ratio = None
                    angle = None
                    

                    thumb = ((points[0].landmark[HandPoint.THUMB_TIP].x,points[0].landmark[HandPoint.THUMB_TIP].y),
                           (points[0].landmark[HandPoint.THUMB_BASE].x,points[0].landmark[HandPoint.THUMB_BASE].y))

                    index = ((points[0].landmark[HandPoint.INDEX_TIP].x,points[0].landmark[HandPoint.INDEX_TIP].y),
                           (points[0].landmark[HandPoint.INDEX_KNUCKLE].x,points[0].landmark[HandPoint.INDEX_KNUCKLE].y))



                    index_pos = get_HandPoint_pos(0,HandPoint.INDEX_TIP)
                    thumb_pos = get_HandPoint_pos(0,HandPoint.THUMB_TIP)
                    pinky_pos = get_HandPoint_pos(0,HandPoint.PINKY_TIP)
                    middle_pos = get_HandPoint_pos(0,HandPoint.MIDDLE_TIP)
                    ring_pos = get_HandPoint_pos(0,HandPoint.RING_TIP)
                    

                    ring_undertip = get_HandPoint_pos(0,HandPoint.RING_UNDERTIP)

                    
                    index_thumb_distance = euclidian_dist(index_pos, thumb_pos)
                    index_pinky_distance = euclidian_dist(index_pos, pinky_pos)
                    
                    thumb_pinky_distance = euclidian_dist(thumb_pos, pinky_pos)
                    index_middle_distance = euclidian_dist(index_pos, middle_pos)

                    middle_ring_distance = euclidian_dist(middle_pos, ring_pos)

                    pinky_ring_distance = euclidian_dist(pinky_pos, ring_pos)
                    thumb_ring_distance = euclidian_dist(ring_pos, thumb_pos)

                    philosopher_ratio = None
                    pointing_ratio = None



                    if middle_ring_distance and index_pinky_distance and index_middle_distance and pinky_ring_distance:
                        pointing_ratio = (middle_ring_distance + index_pinky_distance) / (index_middle_distance + pinky_ring_distance)
                        if pointing_ratio > 9:
                            if points[0].landmark[HandPoint.THUMB_TIP].x > points[0].landmark[HandPoint.THUMB_BASE].x:
                                right()
                            else:
                                left()
                            
                    
                    
                else:
                    red()
                    brake()
                


                        
