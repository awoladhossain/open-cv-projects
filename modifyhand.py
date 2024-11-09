import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os
import warnings
from tkinter import filedialog, colorchooser
import tkinter as tk

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
        self.tip_ids = [4, 8, 12, 16, 20]

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
            try:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
            except IndexError:
                pass
        return self.lm_list

    def fingers_up(self):
        fingers = []
        if len(self.lm_list) >= 21:  # Ensure we have all landmarks
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
        return fingers if fingers else [0, 0, 0, 0, 0]

    def find_distance(self, p1, p2, img, draw=True):
        if len(self.lm_list) >= max(p1, p2):
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
        return 0, img, [0, 0, 0, 0, 0, 0]

class DrawingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector()
        self.setup_drawing_parameters()
        self.setup_ui()

    def setup_drawing_parameters(self):
        self.draw_color = (255, 0, 255)
        self.brush_thickness = 15
        self.eraser_thickness = 50
        self.xp, self.yp = 0, 0
        self.img_canvas = np.zeros((480, 640, 3), np.uint8)
        self.undo_stack = []
        self.redo_stack = []
        self.brush_shape = 'round'
        self.opacity = 255
        self.last_tap_time = 0
        self.TAP_THRESHOLD = 0.3
        self.drawing_mode = 'free'  # 'free', 'line', 'rectangle', 'circle'

    def setup_ui(self):
        self.header = np.zeros((65, 640, 3), np.uint8)
        # Color palette
        colors = [
            (255, 0, 255),  # Purple
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (128, 128, 128) # Eraser
        ]

        spacing = 90
        start_x = 40
        for i, color in enumerate(colors):
            cv2.rectangle(self.header,
                         (start_x + i*spacing, 5),
                         (start_x + (i+1)*spacing - 10, 60),
                         color, cv2.FILLED)

    def handle_drawing(self, img, fingers, x1, y1):
        if self.drawing_mode == 'free':
            self.draw_free(img, x1, y1)
        elif self.drawing_mode == 'line':
            self.draw_line(img, x1, y1)
        elif self.drawing_mode == 'rectangle':
            self.draw_rectangle(img, x1, y1)
        elif self.drawing_mode == 'circle':
            self.draw_circle(img, x1, y1)

    def draw_free(self, img, x1, y1):
        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = x1, y1

        if self.draw_color == (0, 0, 0):  # Eraser
            cv2.line(img, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraser_thickness)
            cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraser_thickness)
        else:
            # Draw with opacity
            overlay = np.zeros_like(self.img_canvas)
            if self.brush_shape == 'round':
                cv2.line(overlay, (self.xp, self.yp), (x1, y1), (*self.draw_color, self.opacity), self.brush_thickness)
            else:  # square brush
                cv2.line(overlay, (self.xp, self.yp), (x1, y1), (*self.draw_color, self.opacity), self.brush_thickness)
            cv2.addWeighted(overlay, self.opacity/255, self.img_canvas, 1, 0, self.img_canvas)

        self.xp, self.yp = x1, y1
        self.undo_stack.append(self.img_canvas.copy())

    def save_drawing(self, directory):
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(
                initialdir=directory,
                initialfile=f"drawing_{timestamp}.png",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                cv2.imwrite(file_path, self.img_canvas)
                return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
        return False

    def run(self):
        save_directory = "saved_drawings"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        p_time = 0
        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            img = self.detector.find_hands(img)
            lm_list = self.detector.find_position(img, draw=False)

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]  # Index finger tip
                fingers = self.detector.fingers_up()

                # Selection Mode - Two fingers up
                if fingers[1] and fingers[2]:
                    self.handle_selection(img, x1, y1)

                # Drawing Mode - Index finger up
                elif fingers[1] and not fingers[2]:
                    self.handle_drawing(img, fingers, x1, y1)

                # Handle other gestures (undo, clear, etc.)
                else:
                    self.handle_other_gestures(fingers)

            # Merge canvas with video feed and display
            self.display_output(img, p_time)

            # Handle keyboard input
            if not self.handle_keyboard_input():
                break

            p_time = time.time()

        self.cap.release()
        cv2.destroyAllWindows()

    def handle_selection(self, img, x1, y1):
        self.xp, self.yp = 0, 0
        if y1 < 65:  # In header area
            spacing = 90
            start_x = 40
            for i, color in enumerate([(255, 0, 255), (255, 0, 0), (0, 255, 0),
                                     (0, 0, 255), (255, 255, 0), (0, 0, 0)]):
                if start_x + i*spacing < x1 < start_x + (i+1)*spacing - 10:
                    self.draw_color = color
                    break

    def handle_other_gestures(self, fingers):
        # Undo - Three fingers up
        if fingers[1] and fingers[2] and fingers[3]:
            if self.undo_stack:
                self.redo_stack.append(self.img_canvas.copy())
                self.img_canvas = self.undo_stack.pop()

        # Redo - Four fingers up
        elif fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            if self.redo_stack:
                self.undo_stack.append(self.img_canvas.copy())
                self.img_canvas = self.redo_stack.pop()

        # Clear canvas - Five fingers up
        elif all(fingers):
            self.undo_stack.append(self.img_canvas.copy())
            self.img_canvas = np.zeros((480, 640, 3), np.uint8)

    def display_output(self, img, p_time):
        # Merge canvas with video feed
        img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.img_canvas)

        # Set header
        img[0:65, 0:640] = self.header

        # Display FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time > 0 else 0
        cv2.putText(img, f'FPS: {int(fps)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display current settings
        cv2.putText(img, f'Mode: {self.drawing_mode}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'Brush: {self.brush_shape}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'Opacity: {self.opacity}', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        cv2.imshow("Canvas", self.img_canvas)

    def handle_keyboard_input(self):
        key = cv2.waitKey(1)
        if key == ord('q'):
            return False
        elif key == ord('s'):
            self.save_drawing("saved_drawings")
        elif key == ord('b'):
            self.brush_shape = 'square' if self.brush_shape == 'round' else 'round'
        elif key == ord('m'):
            modes = ['free', 'line', 'rectangle', 'circle']
            self.drawing_mode = modes[(modes.index(self.drawing_mode) + 1) % len(modes)]
        elif key == ord('o'):
            self.opacity = max(self.opacity - 25, 0)
        elif key == ord('p'):
            self.opacity = min(self.opacity + 25, 255)
        elif key == ord('c'):
            color = colorchooser.askcolor()[0]
            if color:
                self.draw_color = tuple(map(int, color))
        return True

if __name__ == "__main__":
    app = DrawingApp()
    app.run()