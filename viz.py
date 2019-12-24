
import pickle
import cv2
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf


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
    a, b = np.array(a), np.array(b)
    new_point_1 = (b - a) * scale + b
    new_point_2 = (a - b) * scale + a

    return (int(new_point_1[0]), int(new_point_1[1])), (int(new_point_2[0]), int(new_point_2[1]))

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

# Getting back the objects:
with open('objs.pkl', 'rb') as f:
    parts_frames = pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation')
    parser.add_argument('--vidloc', type=str, default='')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.vidloc)

    counter = 0
    w, h = (432, 368)
    counters = []

    wrist_positions = []

    while True:
        ret,image = cap.read()
        if ret==False:
            break
        counters.append(counter)
        parts = parts_frames[counter]
    
        l_shoulder, r_shoulder = extend_lines(parts['left_shoulder'], parts['right_shoulder'], 'n')
        image = cv2.line(image, l_shoulder, r_shoulder, (0, 0, 0), 6)


        l_hip, r_hip = extend_lines(parts['left_hip'], parts['right_hip'], 'l')
        image = cv2.line(image, l_hip, r_hip, (0, 0, 0), 6)

        l_knee, r_knee = extend_lines(parts['left_knee'], parts['right_knee'], 'l')
        image = cv2.line(image, l_knee, r_knee, (0, 0, 0), 6)

        top, bottom = extend_lines(parts['sternum'], ((parts['right_hip'][0] + parts['left_hip'][0])/2, (parts['right_hip'][1] + parts['left_hip'][1])), 'n')
        image = cv2.line(image, top, bottom, (0, 0, 0), 6)
        counter += 1

        if parts['right_shoulder'][0] < parts['left_shoulder'][0]:
            shoulder_angle = angle_between_points(parts['left_shoulder'], parts['right_shoulder'], (0, parts['right_shoulder'][1])) - 90
            print(shoulder_angle)
        else:
            print('*')
            shoulder_angle = angle_between_points(parts['right_shoulder'], parts['left_shoulder'], (0, parts['left_shoulder'][1])) + 90
            print(shoulder_angle)
        if parts['right_hip'][0] < parts['left_hip'][0]:
            hip_angle = angle_between_points(parts['left_hip'], parts['right_hip'], (0, parts['right_hip'][1])) - 90
        else:
            hip_angle = angle_between_points(parts['right_hip'], parts['left_hip'], (0, parts['left_hip'][1])) + 90
        knee_angle = angle_between_points(parts['left_knee'], parts['right_knee'], (0, parts['right_knee'][1])) - 90
        # font hip
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (255, 0, 0) 
        thickness = 2
        
        # Using cv2.putText() method 
        image = cv2.putText(image, str(round(shoulder_angle, 2)), l_shoulder, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, str(round(hip_angle, 2)), l_hip, font,  
            fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, str(round(knee_angle, 2)), l_knee, font,  
            fontScale, color, thickness, cv2.LINE_AA)
        wrist_pos = get_wrist_position(parts['left_palm'], parts['right_palm'])
        # cv2.imshow('image', image)
        wrist_positions.append(wrist_pos)
        # plt.imshow(image)
        # plt.scatter(np.array(wrist_positions)[:, 0], np.array(wrist_positions)[:, 1], c='r', s=5)
        # plt.show()
        # make an agg figure
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.scatter(np.array(wrist_positions)[:, 0], np.array(wrist_positions)[:, 1], c='r', s=5)
        fig.canvas.draw()
        image_ = np.array(fig.canvas.renderer.buffer_rgba())
        cv2.imshow('Left Elbow Angle', image_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break #wait until any key is pressed
    cv2.destroyAllWindows()

        # # make an agg figure
        # fig, ax = plt.subplots()
        # ax.plot(angles_l_shldr_elb_palm, counters)
        # ax.set_title('Left Elbow Angle')
        # ax.set_xlim([0, 190])
        # fig.canvas.draw()
        # image_4 = np.array(fig.canvas.renderer.buffer_rgba())
        # cv2.imshow('Left Elbow Angle', image_4)
