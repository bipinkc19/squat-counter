import argparse
import logging
import time
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    # parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    parser.add_argument('--vidloc', type=str, default='')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    cap = cv2.VideoCapture(args.vidloc)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('input_pose.mp4', fourcc, 30.0, (640, 640))

    parts_frames = {}
    counter = 0
    while True:
        ret,image = cap.read()
        if ret==False:
            break
        print(image.shape)
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        black = np.ones((640, 640, 3))
        image = TfPoseEstimator.draw_humans(black, humans, imgcopy=False)

        # cv2.imshow('tf-pose-estimation result', image)
        parts_points = {}
        list_of_parts = ['nose', 'sternum', 'right_shoulder', 'right_elbow', 'right_palm', 
                        'left_shoulder', 'left_elbow', 'left_palm', 'right_hip', 'right_knee', 
                        'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye', 
                        'left_eye', 'right_ear', 'left_ear']
        for i, part in enumerate(list_of_parts):
            try:
                parts_points[part] = (int(humans[0].body_parts[i].x * 640), int(humans[0].body_parts[i].y * 640))
            except:
                parts_points[part] = (None, None)
        parts_frames[counter] = parts_points
        counter += 1
        # print(parts_points)
        # plt.imshow(image)
        # plt.show()
        image = image * 255
        image = np.uint8(image.astype(int))

        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Saving the objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump(parts_frames, f)
out.release()
cap.release()
