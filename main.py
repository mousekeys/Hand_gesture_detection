from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
import cv2
import argparse
import mediapipe as mdp


# GETS THE REQUIRED ARGUMENTS IN CLI 

def get_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--device", type= int,default=0)
    parser.add_argument("--staticimage", action="store_true")
    parser.add_argument("--width",help="Maximun width", type= int,default=960)
    parser.add_argument("--height",help="Maximun height", type= int,default=540)

    parser.add_argument("--minDetection" ,help="Minimum confidence for detection",type=float,default=0.6)
    parser.add_argument("--minTracking" ,help="Minimum confidenxce for tracking",type=float,default=0.8)

    args = parser.parse_args()

    return args


def camera_input(cam_device,cam_width,cam_height):

#POPERLY INITIALIZE CAMERA FOR WORKING
    cam=cv2.VideoCapture(cam_device)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_height)
    print(cam)
    cam.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cam

def load_models(use_static_image_mode,min_detection,min_tracking):

#LOAD MODEL TO DETECT LANDMARKS IN HANDS
    mdp_hands=mdp.solutions.hands
    hands=mdp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking
        )
    return hands

def main():
    args=get_args()
# GET THE CAMERA ARGUMENTS FROM CLI COMMANDS
    cam_device=args.device
    cam_width=args.width
    cam_height=args.height
# GET THE CAMERA ARGUMENTS FROM CLI COMMANDS
    use_static_image_mode=args.staticimage
    min_detection=args.minDetection

    min_tracking=args.minTracking

    cam=camera_input(cam_device,cam_width,cam_height)
    hands=load_models(use_static_image_mode,min_detection,min_tracking)
    
main()




