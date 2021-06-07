import cv2
import math
import numpy as np
import argparse
import pandas as pd
import json
import os

def face_orientation(frame, landmarks, center=None):
    size = frame.shape #(height, width, color_channel)

    image_points = landmarks.reshape(6, 2).astype(np.float32)
    
                        
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
    
    if center == None:
        center = (size[1]/2, size[0]/2)

    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], 
        dtype = np.float32)

    dist_coeffs = np.zeros((4,1)) 
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    
    axis = np.array([
        [500,0,0], 
        [0,500,0], 
        [0,0,500]], 
        dtype=np.float32)
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    # pitch = math.degrees(pitch)
    # roll = -math.degrees(roll)
    # yaw = math.degrees(yaw)

    return imgpts, modelpts, (roll, pitch, yaw), image_points[0]

if __name__ == '__main__':
    with open('./config.json') as f:
        config = json.load(f)
    
    # Save
    image_save = config['image_save']

    # Path
    landmark_path = config['landmark_path']
    save_path = config['landmark_path']
    
    # Column (input path)
    image_dir_path = config['image_dir_path']

    # Offset (if landmark values need offset value) 
    
    offset_x, offset_y = config['offset']['x'], config['offset']['y'] 
    offset_w, offset_h = config['offset']['w'], config['offset']['h'] 

    # Column Info
    landmark_order = ["nose_tip", "chin", "left_eye_center", "right_eye_center", "left_mouse_corner", "right_mouse_corner"]
    landmark_order = [config['landmark'][lo] for lo in landmark_order]
    landmark_prefix = config["landmark"]['landmark_prefix']    
    landmark_col = [prefix+lo for lo in landmark_order for prefix in landmark_prefix ]

    # Dataset Load
    landmarks = pd.read_csv(landmark_path)
    print('# of data : ', len(landmarks))
    

    pose_list = []
    # Processing
    for idx in range(len(landmarks)):
        if idx==100:break
        img_info = landmarks.iloc[idx]
        landmark_value = img_info[landmark_col].values
        # if offset_x !=None and offset_y != None:
        #     landmark_value[::2] += img_info[offset_x]
        #     landmark_value[1::2] += img_info[offset_y]
            
        img_path = img_info[image_dir_path]
        frame = cv2.imread(img_path)
        center = None
        if offset_w and offset_h:
            
            center = (img_info[offset_x]+img_info[offset_w]/2, img_info[offset_y]+img_info[offset_h]/2)
            pass


        imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmark_value, center)
        pose_list.append(list(rotate_degree))

        if image_save:
            nose[0] += img_info[offset_x]
            nose[1] += img_info[offset_y]
            nose = nose.astype(int)

            cv2.line(frame, nose, tuple(imgpts[1].flatten().astype(int)), (0,255,0), 3) 
            cv2.line(frame, nose, tuple(imgpts[0].flatten().astype(int)), (255,0,0), 3) 
            cv2.line(frame, nose, tuple(imgpts[2].flatten().astype(int)), (0,0,255), 3) 
            
            for j in range(len(rotate_degree)):
                cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), 
                (10, 30 + (50 * j)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
            
            for index in range(len(landmark_value)//2):
                cv2.circle(frame, (landmark_value[index*2].astype(int), landmark_value[index*2+1].astype(int)), 5, (255, 0, 0), -1)  
            
            cv2.imwrite(os.path.join('./sample', img_path.split('/')[-1]), frame)

    
    pose_df = pd.DataFrame(np.array(pose_list), columns=['roll', 'pitch', 'yaw'])
    pose_df.to_csv('output.csv', index=False)
    