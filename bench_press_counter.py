import argparse
import imutils

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
fontScale = 1.5
fontColor = (255, 255, 255)
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
    out = cv2.VideoWriter('press_counter_new_2.mp4', fourcc, 30.0, (1920, 1080))

    count_of_presses = 0
    press_pos = 0
    prev_press_pos = 0

    i = 0
    while True:

        ret_val, image = cam.read()

        #rotation angle in degree
        # image = cv2.imread('test.png')


        if ret_val==False:
            break
        
        image = cv2.flip(image, 1)
        image = imutils.rotate(image, 45)
        image = cv2.resize(image, (1920, 1080))
        # image = cv2.resize(cv2.imread('./squat.jpg'), (432, 368))
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        print(humans)
        for i in humans[0].body_parts.keys():
            center = (int(humans[0].body_parts[i].x * w + 0.5), int(humans[0].body_parts[i].y * h + 0.5))
            print(center)
            print(i, humans[0].body_parts[i], humans[0].body_parts[i].x, humans[0].body_parts[i].y, humans[0].body_parts[i].score)
        if len(humans) != 1:
            continue
        cv2.waitKey(0)
        if args.leg == 'left':
            try:
                radius = 20
                color = (255, 0, 0)
                thickness = 5
                center_4 = (int(humans[0].body_parts[5].x * w), int(humans[0].body_parts[2].y * h)) # left shoulder
                center_5 = (int(humans[0].body_parts[6].x * w), int(humans[0].body_parts[5].y * h)) # left elbow
                center_6 = (int(humans[0].body_parts[7].x * w), int(humans[0].body_parts[6].y * h)) # left palm

                press_left_angle = angle_between_points(center_5, center_6, center_4)
                press_pos = 1 if press_left_angle <= 50 else 0
                if prev_press_pos - press_pos == 1:
                    count_of_presses +=1
                prev_press_pos = press_pos
                # print(press_left_angle, press_pos, count_of_presses)
                cv2.putText(image, 'Number of presses: ' + str(count_of_presses),
                    (10, 100),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.putText(image, 'Angle of elbow joint: ' + str(round(press_left_angle, 1)),
                    (10, 200),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'press position: ' + str('Yes' if press_pos==1 else 'No'),
                    (10, 300),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Tracking leg: Left',
                    (10, 400),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

            except:
                cv2.putText(image, 'Number of presses: ' + str(count_of_presses),
                    (10, 100),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.putText(image, 'Angle of elbow joint: Unknown',
                    (10, 200),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'press position: Unknown',
                    (10, 300),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Tracking arm: Left',
                    (10, 400),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )


        if args.leg == 'right':        
            try:
                center_1 = (int(humans[0].body_parts[1].x * w), int(humans[0].body_parts[1].y * h)) # right shoulder
                center_2 = (int(humans[0].body_parts[2].x * w), int(humans[0].body_parts[2].y * h)) # right elbow
                center_3 = (int(humans[0].body_parts[3].x * w), int(humans[0].body_parts[3].y * h)) # right palm

                press_right_angle = angle_between_points(center_1, center_2, center_3)
                press_pos = 1 if press_right_angle <= 50 else 0
                if prev_press_pos - press_pos == 1:
                    count_of_presses +=1
                prev_press_pos = press_pos
                print(press_right_angle, press_pos, count_of_presses)
                cv2.putText(image, 'Number of presses: ' + str(count_of_presses),
                    (10, 50),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.putText(image, 'Angle of elbow joint: ' + str(round(press_right_angle, 1)),
                    (10, 150),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'press position: ' + str('Yes' if press_pos==1 else 'No'),
                    (10, 250),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Tracking arm: Right',
                    (10, 350),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )


            except:
                cv2.putText(image, 'Number of presses: ' + str(count_of_presses),
                    (10, 100),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
                cv2.putText(image, 'Angle of elbow joint: Unknown',
                    (10, 200),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'press position: Unknown',
                    (10, 300),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )

                cv2.putText(image, 'Tracking leg: Right',
                    (10, 400),
                    font, 
                    fontScale,
                    fontColor,
                    lineType
                )
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        print(image.shape)
        out.write(image)
        # i += 1
        # if i > 10:
        #     break
        cv2.imshow('tf-pose-estimation result', image)
        image = cv2.resize(image, (1920, 1080))

        if cv2.waitKey(1) == 'q':
            break
    
    out.release()
    # close the already opened camera
    cam.release()
    # close the window and de-allocate any associated memory usage
    cv2.destroyAllWindows()
