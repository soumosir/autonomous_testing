import os
import cv2
import numpy as np

HOMO = None
HOMO_INV = None

N_WINDOWS = 10
INIT_MARGIN = 100
MARGIN = 150
MIN_PIX = 100
MAX_LINE_COUNT = 10
WEIGHT_FACTOR = 0.7

P1X_FACTOR = 0.15
P2X_FACTOR = 0.45
P3X_FACTOR = 0.55
P4X_FACTOR = .85

P1Y_FACTOR = 1.0
P2Y_FACTOR = 0.62
P3Y_FACTOR = 0.62
P4Y_FACTOR = 1.0


class Line:
    def __init__(self):
        self.line_counter = 0
        self.avg_curve_params = np.array([0, 0, 0])
        self.curr_curve_params = np.array([0, 0, 0])

    def update_line(self, curve_params):
        if self.line_counter != 0:
            curve_params = curve_params * WEIGHT_FACTOR + self.avg_curve_params * (1 - WEIGHT_FACTOR)

        self.curr_curve_params = curve_params
        self.line_counter += 1

        if self.line_counter > MAX_LINE_COUNT:
            self.line_counter = 1
            self.avg_curve_params = np.array([0, 0, 0])

        self.avg_curve_params = (self.avg_curve_params * (
                self.line_counter - 1) + self.curr_curve_params) / self.line_counter

        return curve_params


class LaneDetector:

    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()

    def detect_lanes(self, image):

        left_lane, right_lane = self.get_lane_pixels(image)

        if left_lane[0].size == 0 or left_lane[1].size == 0 or right_lane[0].size == 0 or right_lane[1].size == 0:
            self.left_line.line_counter = 0
            self.right_line.line_counter = 0
            left_lane, right_lane = self.get_lane_pixels(image)

        left_curve_params, right_curve_params = self.get_curve_params(left_lane, right_lane)

        return left_curve_params, right_curve_params

    def get_lane_pixels(self, image):

        non_zero = image.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        if self.left_line.line_counter == 0:
            self.left_line = Line()
            self.right_line = Line()
            h, w = image.shape[:2]

            bottom_half = image[3 * h // 5:, :]
            histogram = np.sum(bottom_half, axis=0)

            mid_point = np.int32(histogram.shape[0] // 2)
            left_x_current = np.argmax(histogram[:mid_point])
            right_x_current = np.argmax(histogram[mid_point:]) + mid_point

            window_height = np.int32(h // N_WINDOWS)

            left_lane_pixels = []
            right_lane_pixels = []

            for window_idx in range(N_WINDOWS):

                win_y_bottom = h - (window_idx + 1) * window_height
                win_y_top = h - window_idx * window_height
                win_left_x_bottom = left_x_current - INIT_MARGIN
                win_left_x_top = left_x_current + INIT_MARGIN
                win_right_x_bottom = right_x_current - INIT_MARGIN
                win_right_x_top = right_x_current + INIT_MARGIN

                non_zero_left_pixels = ((non_zero_x >= win_left_x_bottom) & (non_zero_x < win_left_x_top)
                                        & (non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top)).nonzero()[0]
                non_zero_right_pixels = ((non_zero_x >= win_right_x_bottom) & (non_zero_x < win_right_x_top)
                                         & (non_zero_y >= win_y_bottom) & (non_zero_y < win_y_top)).nonzero()[0]

                if len(non_zero_left_pixels) > MIN_PIX:
                    left_x_current = np.int32(np.mean(non_zero_x[non_zero_left_pixels]))
                if len(non_zero_right_pixels) > MIN_PIX:
                    right_x_current = np.int32(np.mean(non_zero_x[non_zero_right_pixels]))

                left_lane_pixels.append(non_zero_left_pixels)
                right_lane_pixels.append(non_zero_right_pixels)

            left_lane_pixels = np.concatenate(left_lane_pixels)
            right_lane_pixels = np.concatenate(right_lane_pixels)

        else:
            left_curve = self.left_line.curr_curve_params
            right_curve = self.right_line.curr_curve_params

            left_lane_pixels = ((non_zero_x > (left_curve[0] * non_zero_y ** 2 + left_curve[1] * non_zero_y +
                                               left_curve[2] - MARGIN)) & (
                                        non_zero_x < (left_curve[0] * non_zero_y ** 2 +
                                                      left_curve[1] * non_zero_y + left_curve[2] + MARGIN)))
            right_lane_pixels = ((non_zero_x > (right_curve[0] * non_zero_y ** 2 + right_curve[1] * non_zero_y +
                                                right_curve[2] - MARGIN)) & (
                                         non_zero_x < (right_curve[0] * non_zero_y ** 2 +
                                                       right_curve[1] * non_zero_y + right_curve[2] + MARGIN)))

        left_x_pixels = non_zero_x[left_lane_pixels]
        left_y_pixels = non_zero_y[left_lane_pixels]
        right_x_pixels = non_zero_x[right_lane_pixels]
        right_y_pixels = non_zero_y[right_lane_pixels]

        return [left_x_pixels, left_y_pixels], [right_x_pixels, right_y_pixels]

    def get_curve_params(self, left_lane, right_lane):

        leftx, lefty = left_lane[0], left_lane[1]
        rightx, righty = right_lane[0], right_lane[1]

        left_curve_params = np.polyfit(lefty, leftx, 2)
        right_curve_params = np.polyfit(righty, rightx, 2)

        left_curve_params = self.left_line.update_line(left_curve_params)
        right_curve_params = self.right_line.update_line(right_curve_params)

        return left_curve_params, right_curve_params


def get_curvature(y_max, left_curve_params, right_curve_params):
    # Conversion parameters after a lot of research on internet
    ym_per_pix = 30 / 700

    left_a, left_b, left_c = left_curve_params[0], left_curve_params[1], left_curve_params[2]
    right_a, right_b, right_c = right_curve_params[0], right_curve_params[1], right_curve_params[2]

    # Calculate the curvature for both lanes
    left_curvature = ((1 + (2 * left_a * y_max * ym_per_pix + left_b) ** 2) ** 1.5) / np.absolute(2 * left_a)
    right_curvature = ((1 + (2 * right_a * y_max * ym_per_pix + right_b)) ** 1.5) / np.absolute(2 * right_a)

    return left_curvature, right_curvature


def draw_lanes(image, left_lane_x_points, right_lane_x_points, y_points):
    h, w = image.shape[:2]

    output_image = np.dstack((image, image, image))
    lane_overlay_image = output_image.copy()
    lane_line_image = output_image.copy()

    left_lane_points = np.asarray([np.vstack([left_lane_x_points, y_points]).T])
    right_lane_points = np.asarray([np.flipud(np.vstack([right_lane_x_points, y_points]).T)])
    lane_points = np.array(np.hstack((left_lane_points, right_lane_points)), dtype=np.int32)
    cv2.fillPoly(lane_overlay_image, lane_points, color=[0, 0, 255])

    # draw blue circles
    prev_y = 0
    for x, y in zip(left_lane_x_points, y_points):
        if 0 <= x < w and 0 <= y < h:
            if abs(np.int32(y) - prev_y) > 60 and output_image[np.int32(y), np.int32(x)][2] == 255:
                cv2.circle(lane_line_image, (np.int32(x), np.int32(y)), 30, (255, 0, 0), 10)
                prev_y = np.int32(y)
    # draw red line
    for x, y in zip(left_lane_x_points, y_points):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(lane_line_image, (np.int32(x), np.int32(y)), 10, (0, 0, 255), -1)

    # draw green circles
    prev_y = 0.0
    for x, y in zip(right_lane_x_points, y_points):
        if 0 <= x < w and 0 <= y < h:
            if abs(np.int32(y) - prev_y) > 60 and output_image[np.int32(y), np.int32(x)][2] == 255:
                cv2.circle(lane_line_image, (np.int32(x), np.int32(y)), 30, (0, 255, 0), 10)
                prev_y = np.int32(y)
    # draw yellow line
    for x, y in zip(right_lane_x_points, y_points):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(lane_line_image, (np.int32(x), np.int32(y)), 10, (0, 255, 255), -1)

    return lane_line_image, lane_overlay_image


def get_lane_points(left_curve, right_curve, shape):
    h, w = shape[:2]

    y_points = np.arange(0, h - 1, 1)
    try:
        left_curve_x_points = left_curve[0] * y_points ** 2 + left_curve[1] * y_points + left_curve[2]
        right_curve_x_points = right_curve[0] * y_points ** 2 + right_curve[1] * y_points + right_curve[2]
    except Exception as e:
        print("EXCEPTION")
        left_curve_x_points = y_points ** 2 + y_points
        right_curve_x_points = y_points ** 2 + y_points

    return left_curve_x_points, right_curve_x_points, y_points


def warp_image(image, homo, shape=None):
    if shape is None:
        shape = image.shape[:2]

    warped_image = cv2.warpPerspective(image, homo, (shape[1], shape[0]), cv2.INTER_LINEAR)

    return warped_image


def preprocess_image(image):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    binary1 = np.zeros_like(s)
    binary1[(s >= 90) & (s <= 100)] = 255

    b, g, r = cv2.split(image)
    binary2 = np.zeros_like(r)
    binary2[(r >= 220)] = 255

    sobel_out = cv2.Sobel(v, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel_out)
    scaled_sobel = (255 * abs_sobel / np.max(abs_sobel)).astype(np.uint8)
    binary3 = np.zeros_like(scaled_sobel)
    binary3[(scaled_sobel >= 40) & (scaled_sobel <= 60)] = 255

    binary_colored = np.dstack((binary1, binary2, binary3))

    return binary_colored


def get_transformation(shape):
    global HOMO
    global HOMO_INV

    if HOMO is None:
        h, w = shape[:2]

        dst_points = np.float32([
            [w // 5, 4 * h - 1],
            [w // 5, 0],
            [4 * w // 5, 0],
            [4 * w // 5, 4 * h - 1]
        ])

        src_points = np.float32([
            [P1X_FACTOR * w - 1, P1Y_FACTOR * h - 1],
            [P2X_FACTOR * w - 1, P2Y_FACTOR * h - 1],
            [P3X_FACTOR * w - 1, P3Y_FACTOR * h - 1],
            [P4X_FACTOR * w - 1, P4Y_FACTOR * h - 1]
        ])

        HOMO = cv2.getPerspectiveTransform(src_points, dst_points)
        HOMO_INV = np.linalg.inv(HOMO)

        return HOMO, HOMO_INV

    else:
        return HOMO, HOMO_INV


def predict_turns(frame, lane_detector,turn_array):
    frame_copy = frame.copy()
    shape = frame.shape[:2]
    opencv_shape1 = (shape[1] // 4, 3 * shape[0] // 4)
    opencv_shape2 = (shape[1] // 4, shape[0] // 4)

    final_frame = np.zeros((shape[0] + 200, shape[1] + 2 * opencv_shape2[0], 3), dtype=np.uint8)
    cv2.rectangle(final_frame, (0, shape[0]), (shape[1] + 2 * opencv_shape2[0] - 1, shape[0] + 200), (240, 220, 210), -1,
                  16)

    resize_frame = cv2.resize(frame, opencv_shape2)
    cv2.putText(resize_frame, "(1)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # get homography and inv homography between src and dst points
    homo, homo_inv = get_transformation(shape)

    # preprocess the input image to remove noise and filter out the objects of interest
    binary_color_image = preprocess_image(frame_copy)

    # warp the processed image according to the desired homography
    warped_image = warp_image(binary_color_image, homo, (shape[0] * 4, shape[1]))

    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, warped_binary_image = cv2.threshold(gray_warped, 50, 255, cv2.THRESH_BINARY)

    unwarped_binary_image = warp_image(warped_binary_image, homo_inv, shape)
    unwarped_binary_image = cv2.cvtColor(unwarped_binary_image, cv2.COLOR_GRAY2BGR)

    resized_unwarped_binary_image = cv2.resize(unwarped_binary_image, opencv_shape2)
    cv2.putText(resized_unwarped_binary_image, "(2)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    warped_binary_color_image = cv2.cvtColor(warped_binary_image, cv2.COLOR_GRAY2BGR)

    resized_warped_binary_color_image = cv2.resize(warped_binary_color_image, opencv_shape1)
    cv2.putText(resized_warped_binary_color_image, "(3)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # get left and right curves
    left_curve_params, right_curve_params = lane_detector.detect_lanes(warped_binary_image)
    left_lane_x_points, right_lane_x_points, y_points = get_lane_points(left_curve_params, right_curve_params, (shape[0] * 4, shape[1]))

    lane_line_image, lane_overlay_image = draw_lanes(warped_binary_image, left_lane_x_points, right_lane_x_points,
                                                     y_points)
    resized_lane_line_image = cv2.resize(lane_line_image, opencv_shape1)
    cv2.putText(resized_lane_line_image, "(4)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    left_curvature, right_curvature = get_curvature(shape[0] - 1, left_curve_params, right_curve_params)

    unwarped_image = warp_image(lane_overlay_image, homo_inv, shape)
    final_image = cv2.addWeighted(frame, 0.9, unwarped_image, 0.3, 0)

    curvature = (left_curvature + right_curvature) / 2

    if left_curvature - right_curvature > 0:
        turn_prediction = "Turn Right"
    elif left_curvature - right_curvature < 0:
        turn_prediction = "Turn Left"
    else:
        turn_prediction = "Go Straight"
    turn_array.append(turn_prediction)


    cv2.putText(final_image, turn_prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    final_frame[0: shape[0], 0: shape[1]] = final_image
    final_frame[0: opencv_shape2[1], shape[1]: shape[1] + opencv_shape2[0]] = resize_frame
    final_frame[0: opencv_shape2[1], shape[1] + opencv_shape2[0]: shape[1] + 2 * opencv_shape2[0]] =\
        resized_unwarped_binary_image
    final_frame[opencv_shape2[1]: 4 * opencv_shape2[1], shape[1]: shape[1] + opencv_shape2[0]] =\
        resized_warped_binary_color_image
    final_frame[opencv_shape2[1]: 4 * opencv_shape2[1], shape[1] + opencv_shape2[0]: shape[1] + 2 * opencv_shape2[0]] =\
        resized_lane_line_image

    text1 = "(1) : Undistored image, (2) : Detected white and yellow lane markings," \
            " (3) : Warped image, (4) : Detected points and curve fitting"
    text2 = "Left Curvature : {0:.2f}, Right Curvature : {1:.2f}".format(left_curvature, right_curvature)
    text3 = "Average Curvature : {0:.2f}".format(curvature)
    cv2.putText(final_frame, text1, (50, shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(final_frame, text2, (50, shape[0] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(final_frame, text3, (50, shape[0] + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return final_frame


def run(video_path, out_dir, display=True, save=False):
    if not os.path.exists(video_path):
        print("[ERROR]: File Does NOT Exists!!! ", video_path)
        exit()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        w_ = w + 2 * w // 4
        h_ = h + 200
        writer = cv2.VideoWriter(os.path.join(out_dir, "turn_prediction_clouds.mp4"), fourcc, fps, (w_, h_))

    if display:
        cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow("frame", 100, 100)

    print("[INFO]: Started Turn Prediction")

    lane_detector = LaneDetector()
    turn_array = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = np.fliplr(frame)  # uncomment this line to flip the frame horizontally
            
            frame = predict_turns(frame, lane_detector,turn_array)

            if display:
                cv2.imshow("frame", frame)
                k = cv2.waitKey(30)
                if k == ord('q') or k == 27:
                    print("[WARN]: QUIT signal received. Exiting Process!!!")
                    exit()
                elif k == ord('p'):
                    print("[WARN]: PAUSE signal received. Halting Process. Press any key to continue")
                    cv2.waitKey(0)

            if save:
                writer.write(frame)
            
        else:
            print("[INFO]: Video Finished!!!")
            break

    if save:
        writer.release()
    with open("output_cloud.txt", "w") as txt_file:
        for line in turn_array:
            txt_file.write(" ".join(line) + "\n") # works with any number of elements in a line    

if __name__ == "__main__":
    video_file = "../Data/challenge.mp4"
    out_dir_path = ""
    run(video_file, out_dir_path, display=True, save=False)
