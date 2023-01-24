import mediapipe
 
import cv2
from enum import IntEnum
import math
import numpy as np





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
        # As in if statement
        return 360 - ang_deg
    else: 
        return ang_deg

mpDraw = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
      while True:
            ret, frame = cap.read()
            #flipped = cv2.flip(frame, flipCode = -1)
            frame1 = cv2.resize(frame, (640, 480))
            results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            points = results.multi_hand_landmarks
            if points:

                if len(points) > 1:
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


                    metal_ratio = None
                    if index_distance and pinky_distance:
                        metal_ratio = index_distance /pinky_distance

                    peace_ratio = None
                    if index_distance and middle_distance and thumb_distance:
                        spade_ratio = index_distance * thumb_distance / middle_distance
                        diamond_ratio = index_distance * middle_distance / thumb_distance

                        if spade_ratio < 3 and diamond_ratio > 3:
                            print("AAAAAAAAAAAAAAAAAA")
                        elif spade_ratio > 3 and diamond_ratio < 3:
                            print("BBBBBBBBBBBBBBBBBB")
                        print(f"{round(spade_ratio,3)}          {round(diamond_ratio,3)}")



                

                """
                philosopher_ratio = None
                pointing_ratio = None

                            
                if thumb_ring_distance:
                    print(thumb_ring_distance)
                
                if thumb_ring_distance and index_thumb_distance:
                    philosopher_ratio = thumb_ring_distance/index_thumb_distance



                #print(philosopher_ratio)
                if index_thumb_distance and index_pinky_distance:
                    kon_ratio = index_thumb_distance/index_pinky_distance


                """
                
                  
                for hand in points:
                    mpDraw.draw_landmarks(frame1, hand, handsModule.HAND_CONNECTIONS)
                          
            cv2.imshow("Frame", frame1);
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
              break
