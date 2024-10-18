import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

import absl.logging
absl.logging.use_absl_handler()

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # Finger tip IDs

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        fingers = []
        # Thumb
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    """
    Main function to run the hand drawing application.
    This function captures video from the webcam, detects hand landmarks, and allows the user to draw on the screen using hand gestures.
    The user can select different colors or an eraser by moving their hand to specific regions on the screen.
    Variables:
    - p_time (float): Previous time for calculating FPS.
    - cap (cv2.VideoCapture): Video capture object.
    - detector (HandDetector): Hand detector object.
    - draw_color (tuple): Current drawing color.
    - brush_thickness (int): Thickness of the brush.
    - eraser_thickness (int): Thickness of the eraser.
    - xp, yp (int): Previous x and y positions of the drawing point.
    - img_canvas (np.ndarray): Canvas for drawing.
    Hand Gestures:
    - Two fingers up: Selection mode to choose color or eraser.
    - Index finger up: Drawing mode to draw on the canvas.
    The function also merges the drawing canvas with the video feed, sets a header image, and displays the FPS on the screen.
    The loop continues until the user presses the 'q' key.
    Note: This function requires the `cv2`, `numpy`, and `time` modules, as well as a `HandDetector` class and a `header` image.
    """
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    draw_color = (255, 0, 255)
    brush_thickness = 15
    eraser_thickness = 50
    xp, yp = 0, 0
    img_canvas = np.zeros((480, 640, 3), np.uint8)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Mirror the image

        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]  # Index finger tip
            x2, y2 = lm_list[12][1:]  # Middle finger tip

            fingers = detector.fingers_up()

            # Selection Mode - Two fingers up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0  # Reset the previous points
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
                if y1 < 65:
                    if 40 < x1 < 140:  # Purple
                        draw_color = (255, 0, 255)
                    elif 160 < x1 < 260:  # Blue
                        draw_color = (255, 0, 0)
                    elif 280 < x1 < 380:  # Green
                        draw_color = (0, 255, 0)
                    elif 400 < x1 < 500:  # Red
                        draw_color = (0, 0, 255)
                    elif 520 < x1 < 620:  # Eraser
                        draw_color = (0, 0, 0)

            # Drawing Mode - Index finger up
            elif fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if draw_color == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

                xp, yp = x1, y1

        # Merge the canvas with the video feed
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        # Setting the header image
        img[0:65, 0:640] = header

        # Calculate and display FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create header image
header = np.zeros((65, 640, 3), np.uint8)
header = cv2.rectangle(header, (40, 1), (140, 65), (255, 0, 255), -1)
header = cv2.rectangle(header, (160, 1), (260, 65), (255, 0, 0), -1)
header = cv2.rectangle(header, (280, 1), (380, 65), (0, 255, 0), -1)
header = cv2.rectangle(header, (400, 1), (500, 65), (0, 0, 255), -1)
header = cv2.rectangle(header, (520, 1), (620, 65), (0, 0, 0), -1)
cv2.putText(header, "ERASER", (525, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

if __name__ == "__main__":
    main()