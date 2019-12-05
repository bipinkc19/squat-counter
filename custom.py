import argparse

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# python custom.py --model=mobilenet_thin --resize=432x368
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def angle_between_points(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
fontColor = (125, 125, 0)
lineType = 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation ')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    parser.add_argument('--leg', type=str, default='left')

    parser.add_argument('--vidlocation', type=str, default='')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    cam = cv2.VideoCapture(args.vidlocation)
    ret_val, image = cam.read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('squat_counter.mp4', fourcc, 30.0, (1280, 720))

    count_of_squats = 0
    squat_pos = 0
    prev_squat_pos = 0

    while True:

        ret_val, image = cam.read()
        if ret_val==False:
            break
        print(image.shape)
        # image = cv2.resize(image, (432, 368))
        # image = cv2.resize(cv2.imread('./squat.jpg'), (432, 368))
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    
        # for i in humans[0].body_parts.keys():
        #     center = (int(humans[0].body_parts[i].x * w + 0.5), int(humans[0].body_parts[i].y * h + 0.5))
        #     print(center)
        #     print(i, humans[0].body_parts[i], humans[0].body_parts[i].x, humans[0].body_parts[i].y, humans[0].body_parts[i].score)
        if len(humans) != 1:
            continue

        if args.leg == 'left':
            try:
                center_11 = (int(humans[0].body_parts[11].x * w), int(humans[0].body_parts[11].y * h)) # left hip
                center_12 = (int(humans[0].body_parts[12].x * w), int(humans[0].body_parts[12].y * h)) # left knee
                center_13 = (int(humans[0].body_parts[13].x * w), int(humans[0].body_parts[13].y * h)) # left ankle
 
                squat_left_angle = angle_between_points(center_11, center_12, center_13)
                squat_pos = 1 if squat_left_angle <= 110 else 0
                if prev_squat_pos - squat_pos == 1:
                    count_of_squats +=1
                prev_squat_pos = squat_pos
                print(squat_right_angle, squat_pos, count_of_squats)
                cv2.putText(image, 'Number of squats: ' + str(count_of_squats),
                    (100, 100),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.putText(image, 'Angle of knee joint: ' + str(round(squat_left_angle, 1)),
                    (100, 200),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Squat position: ' + str('Yes' if squat_pos==1 else 'No'),
                    (100, 300),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Tracking leg: Left',
                    (100, 400),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
            except:
                pass

        if args.leg == 'right':        
            try:
                center_8 = (int(humans[0].body_parts[8].x * w), int(humans[0].body_parts[8].y * h)) # right hip
                center_9 = (int(humans[0].body_parts[9].x * w), int(humans[0].body_parts[9].y * h)) # right knee
                center_10 = (int(humans[0].body_parts[10].x * w), int(humans[0].body_parts[10].y * h)) # right ankle

                squat_right_angle = angle_between_points(center_8, center_9, center_10)
                squat_pos = 1 if squat_right_angle <= 110 else 0
                if prev_squat_pos - squat_pos == 1:
                    count_of_squats +=1
                prev_squat_pos = squat_pos
                print(squat_right_angle, squat_pos, count_of_squats)
                cv2.putText(image, 'Number of squats: ' + str(count_of_squats),
                    (100, 100),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.putText(image, 'Angle of knee joint: ' + str(round(squat_right_angle, 1)),
                    (100, 200),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Squat position: ' + str('Yes' if squat_pos==1 else 'No'),
                    (100, 300),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Tracking leg: Right',
                    (100, 400),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

            except:
                pass                
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        out.write(image)
        cv2.imshow('tf-pose-estimation result', image)
        # image = cv2.resize(image, (1080, 1920))

        if cv2.waitKey(1) == 'q':
            break
    
    out.release()
    # close the already opened camera
    cam.release()
    # close the window and de-allocate any associated memory usage
    cv2.destroyAllWindows()
