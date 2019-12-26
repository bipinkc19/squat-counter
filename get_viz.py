import pickle
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def angle_between_points(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    try:
        AB = b-a     
        AC = c-a     

        angle_radians = np.arctan2(AB[0]*AC[0] + AB[1]*AC[1], AB[1]*AC[0] - AB[0]*AC[1])

        angle_degrees = angle_radians*180/np.pi
        return angle_degrees

    except TypeError:
        return None


def x_distance_between_points(a, b):
    try:
        return a[0] - b[0]
    except TypeError:
        return None


def extend_lines(a, b, f):
    if f == 'n':
        scale = 2
    if f == 'l':
        scale = 3
    try:
        a, b = np.array(a), np.array(b)
        new_point_1 = (b - a) * scale + b
        new_point_2 = (a - b) * scale + a

        return (int(new_point_1[0]), int(new_point_1[1])), (int(new_point_2[0]), int(new_point_2[1]))
    except:
        return ((None, None), (None, None))


def get_wrist_position(a, b):
    try:
        x = (a[0] + b[0])/2
        y = (a[1] + b[1])/2
        return np.array([x, y])
    except TypeError:
        return np.array([np.nan, np.nan])

def azimuth(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle)if angle>0 else np.degrees(angle) + 360


def give_points_location(vid_location, model_type):
    tensorrt_use = "False"

    w, h = model_wh('432x368')
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model_type), target_size=(w, h), trt_bool=str2bool(tensorrt_use))
    else:
        e = TfPoseEstimator(get_graph_path(model_type), target_size=(432, 368), trt_bool=str2bool(tensorrt_use))
    
    cap = cv2.VideoCapture(vid_location)
    ret, image = cap.read()
    # print(image.shape)
    # exit()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('temp.mp4', fourcc, 30.0, (image.shape[1], image.shape[0]))

    parts_frames = {}
    counter = 0    # print(image.shape)
    # exit()
    cap = cv2.VideoCapture(vid_location)
    while True:
        ret, image = cap.read()
        if ret==False:
            break
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        black = np.ones((image.shape[0], image.shape[1], 3))
        image = TfPoseEstimator.draw_humans(black, humans, imgcopy=False)

        # cv2.imshow('tf-pose-estimation result', image)
        parts_points = {}
        list_of_parts = ['nose', 'sternum', 'right_shoulder', 'right_elbow', 'right_palm', 
                        'left_shoulder', 'left_elbow', 'left_palm', 'right_hip', 'right_knee', 
                        'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye', 
                        'left_eye', 'right_ear', 'left_ear']
        for i, part in enumerate(list_of_parts):
            try:
                parts_points[part] = (int(humans[0].body_parts[i].x * image.shape[1]), int(humans[0].body_parts[i].y * image.shape[0]))
            except:
                parts_points[part] = (None, None)
        parts_frames[counter] = parts_points
        counter += 1
        image = image * 255
        image = np.uint8(image.astype(int))
        
        out.write(image)
    out.release()

    return parts_frames


def process_video(vid_location, model_type, processed_vid_location):

    parts_frames = give_points_location(vid_location, model_type)
    cap = cv2.VideoCapture("input_pose.mp4")
    ret, image = cap.read()
    w, h = (image.shape[1], image.shape[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 0, 0) 
    thickness = 2

    counter = 0
    counters = []
    wrist_positions = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(processed_vid_location, fourcc, 30.0, (w, h))
    cap = cv2.VideoCapture("temp.mp4")
    while True:
        ret,image = cap.read()
        if ret==False:
            break
        counters.append(counter)
        parts = parts_frames[counter]

        wrist_pos = get_wrist_position(parts['left_palm'], parts['right_palm'])
        
        wrist_positions.append(wrist_pos)
    
        try:
            l_shoulder, r_shoulder = extend_lines(parts['left_shoulder'], parts['right_shoulder'], 'n')
            image = cv2.line(image, l_shoulder, r_shoulder, (0, 0, 0), 6)
        except:
            l_shoulder, r_shoulder = None, None

        try:
            l_hip, r_hip = extend_lines(parts['left_hip'], parts['right_hip'], 'l')
            image = cv2.line(image, l_hip, r_hip, (0, 0, 0), 6)
        except:
            l_hip, r_hip = None, None

        try:
            l_knee, r_knee = extend_lines(parts['left_knee'], parts['right_knee'], 'l')
            image = cv2.line(image, l_knee, r_knee, (0, 0, 0), 6)
        except:
            l_knee, r_knee = None, None

        try:
            top, bottom = extend_lines(parts['sternum'], ((parts['right_hip'][0] + parts['left_hip'][0])/2, (parts['right_hip'][1] + parts['left_hip'][1])), 'n')
            image = cv2.line(image, top, bottom, (0, 0, 0), 6)
        except:
            top, bottom = None, None
        counter += 1
        print(parts['left_palm'], parts['right_palm'])
        try:
            if parts['right_hip'][0] < parts['left_hip'][0]:
                hip_angle = angle_between_points(parts['left_hip'], parts['right_hip'], (0, parts['right_hip'][1])) - 90
            else:
                hip_angle = angle_between_points(parts['right_hip'], parts['left_hip'], (0, parts['left_hip'][1])) + 90
        except:
            hip_angle = None
        
        try:
            knee_angle = angle_between_points(parts['left_knee'], parts['right_knee'], (0, parts['right_knee'][1])) - 90
        except:
            knee_angle= None
        
        try:
            # Failed attempt at finding the angular movement
            if (parts['right_shoulder'][0] > parts['left_shoulder'][0]) and (hip_angle < 90):
                # shoulder_angle = angle_between_points(parts['right_shoulder'], parts['left_shoulder'], (0, parts['left_shoulder'][1])) + 90
                if parts['left_palm'][0] is not None and (parts['left_palm'][0]<parts['right_shoulder'][0] and parts['left_palm'][1]<parts['right_shoulder'][1]):
                    print("yes")
                    shoulder_angle = angle_between_points(parts['left_shoulder'], parts['right_shoulder'], (0, parts['right_shoulder'][1])) - 90
                elif parts['right_palm'][0] is not None and (parts['right_palm'][0]<parts['right_shoulder'][0] and parts['right_palm'][1]<parts['right_shoulder'][1]):
                    print("yes_1")
                    shoulder_angle = angle_between_points(parts['left_shoulder'], parts['right_shoulder'], (0, parts['right_shoulder'][1])) - 90
                elif parts['left_palm'][0] is None and parts['right_palm'][0] is None:
                    print("None")
                    shoulder_angle = None
                else:
                    print('latest')
                    shoulder_angle = angle_between_points(parts['right_shoulder'], parts['left_shoulder'], (0, parts['left_shoulder'][1])) + 90
            else:
                print('no')
                shoulder_angle = angle_between_points(parts['right_shoulder'], parts['left_shoulder'], (0, parts['left_shoulder'][1])) + 90
        except:
            print('exp')
            shoulder_angle = None
        print(shoulder_angle)
        
        
        # Using cv2.putText() method 
        try:
            image = cv2.putText(image, str(round(shoulder_angle, 2)), l_shoulder, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
        except:
            pass
        try:
            image = cv2.putText(image, str(round(hip_angle, 2)), l_hip, font,  
                fontScale, color, thickness, cv2.LINE_AA)
        except:
            pass

        try:
            image = cv2.putText(image, str(round(knee_angle, 2)), l_knee, font,  
                fontScale, color, thickness, cv2.LINE_AA)
        except:
            pass
        
        # get x and y vectors
        x = np.array(wrist_positions)[:, 0]
        y = np.array(wrist_positions)[:, 1]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        matplotlib.rcParams['figure.figsize'] = [w/100, h/100]
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.plot(x, y)
        # ax.scatter(parts['right_shoulder'][0], parts['right_shoulder'][1])
        fig.canvas.draw()
        # plt.show()
        image_ = np.array(fig.canvas.renderer.buffer_rgba())

        cv2.imshow('Left Elbow Angle', image_[:, :, 0:3])
        out.write(image_[:, :, 0:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break #wait until any key is pressed
    cv2.destroyAllWindows()
    out.release()

# process_video(vid_location="input.mp4", model_type="cmu", processed_vid_location="test_out.mp4")
process_video(vid_location="input_in.mp4", model_type="cmu", processed_vid_location="test_out.mp4")
# mobilenet_thin
