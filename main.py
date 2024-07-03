from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
import cv2
import argparse
import mediapipe as mdp
import copy
import os
import csv

#GETS THE REQUIRED ARGUMENTS IN CLI 

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

def camera_input(cam_device,cam_width,cam_height):

#POPERLY INITIALIZE CAMERA FOR WORKING
    cam=cv2.VideoCapture(cam_device)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_height)
    print(cam)
    cam.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# NOW TO FIND THE EXACT COORDINATES IN THE SCREEN AS THE IMAGES ARE NORMALIZED RIGHT NPW

# NOW TO FIND THE EXACT COORDINATES IN THE SCREEN AS THE IMAGES ARE NORMALIZED RIGHT NPW

def screen_coordinates(normalized_coodr,image):
    img_height,img_width,_=image.shape

    screen_coordinate=[]

    for _,landmark in enumerate(normalized_coodr.landmark):
        screen_x=int(landmark.x*img_width)
        screen_y=int(landmark.y*img_height)
        #NO NEED FOR LANDMARK Z AND ALSO IMAGE.SHAPE[2] PROVIDES COLOR INDEXES LIKE RGB SO 3
        screen_coordinate.append([screen_x,screen_y])
        cv2.circle(image, (screen_x, screen_y), 5, (0, 255, 0), -1)

    return screen_coordinate

#NORMALIZATION SO THAT WE CAN DO SOMETHING EVEN WHEN THE HAND IS MOVED AND IT STILL WORKS ON THE HAND GEUSTURE
#THE NORMALIZED POINTS MAKES IT SO THA WE NEED LESSER DATA AND SIMPLER MODEL TO TRAIN AND DETECT

def normalized_values(screen_coordinate):

    temp_landmark=copy.deepcopy(screen_coordinate)
    base_x,base_y=0,0

    print(type(temp_landmark))

    for i, pixel_val in enumerate(temp_landmark):

        if i==0:
            base_x,base_y=screen_coordinate[0][0],screen_coordinate[0][1]
        
        temp_landmark[i][0]=pixel_val[0]-base_x
        temp_landmark[i][1]=pixel_val[1]-base_y

    temp_landmark=np.array(temp_landmark).flatten()
    max_val=max(list(map(abs,temp_landmark)))

    def normalize_(val):
        return val/max_val
    
    temp_landmark=list(map(normalize_,temp_landmark))

    return temp_landmark
    

def store_data(number,normalized_one):
    csv_path=r"datas/pre_processed.csv"
    
    with open(csv_path,'a',newline="") as send:
        store=csv.writer(send)
        store.writerow([number,*normalized_one])
    
    return

def image_path(path=''):
    paths=[]
    path_list=[]
    pathx=[]
    if not path:
        path=r"datas\photos_hands"
    all_files=list(map(int,os.listdir(path)))
    for i in range(len(all_files)):
        paths.append(os.path.join(path,str(all_files[i])))
    for path in paths:
        path_list=[]
        for i in os.listdir(path):
            if i.endswith('.png'):
                path_list.append(os.path.join(path,i))
        pathx.append(list(path_list))
        
    return pathx,all_files

def draw_landmarks(image,results,hands,screen_coordinate):
    mdp_hands = mdp.solutions.hands
# Check if any hands are detected.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the image.
            mdp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mdp_hands.HAND_CONNECTIONS,
                mdp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mdp.solutions.drawing_styles.get_default_hand_connections_style()
            )

    # Display the image with landmarks.
    cv2.imshow('Hand Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

# GET THE CAMERA ARGUMENTS FROM CLI COMMANDS
    use_static_image_mode=True
    min_detection=0.6
    min_tracking=0.8

    hands=load_models(use_static_image_mode,min_detection,min_tracking)
    paths,num=image_path()
    idx=0
    print(paths)
    for path in paths:
        for i in path:
            image=cv2.imread(i)
            image = cv2.flip(image, 1)
            # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            final_img=copy.deepcopy(image)
            
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
    
            screen_coordinate=screen_coordinates(results.multi_hand_landmarks[0],final_img)
           
            normalized_coodr=normalized_values(screen_coordinate)

            final_img=draw_landmarks(final_img,results,hands,screen_coordinate)
            store_data(num[idx],normalized_coodr)
          
          
        idx=+1



# def main():
    
# # GET THE CAMERA ARGUMENTS FROM CLI COMMANDS
#     cam_device=0
#     cam_width=960
#     cam_height=540
# # GET THE CAMERA ARGUMENTS FROM CLI COMMANDS
#     use_static_image_mode=True
#     min_detection=0.6
#     min_tracking=0.8

#     cam=camera_input(cam_device,cam_width,cam_height)
#     hands=load_models(use_static_image_mode,min_detection,min_tracking)
#     paths,num=image_path()
#     idx=0
#     print(paths)
#     for path in paths:
#         for i in path:
#             image=cv2.imread(i)
#             image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#             results=hands.process(image)
#             screen_coordinate=screen_coordinates(results.multi_hand_landmarks,image)
#             normalized_coodr=normalized_values(screen_coordinate)
#             store_data(num[idx],normalized_coodr)
#         idx=+1


if __name__=='main':
    main()

main()
