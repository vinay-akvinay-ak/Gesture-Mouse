import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
import math
import time
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Absolute path to model
model_path = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# =====================================
# SCREEN SIZE
# =====================================
screen_width, screen_height = pyautogui.size()

# =====================================
# WEBCAM
# =====================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# =====================================
# MEDIAPIPE MODEL
# =====================================
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# =====================================
# CURSOR SETTINGS
# =====================================
prev_x = 0
prev_y = 0

smoothening = 6
frame_reduction = 120

# =====================================
# SWIPE SETTINGS
# =====================================
previous_wrist_x = 0

swipe_threshold = 0.18

swipe_delay = 1

last_swipe_time = time.time()

# =====================================
# CLICK SETTINGS
# =====================================
click_threshold = 22

last_click_time = time.time()

# =====================================
# SCROLL SETTINGS
# =====================================
scroll_delay = 0.1

last_scroll_time = time.time()

# =====================================
# FPS
# =====================================
p_time = 0

# =====================================
# MAIN LOOP
# =====================================
while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    detection_result = detector.detect(mp_image)

    h, w, _ = frame.shape

    # =====================================
    # CONTROL AREA
    # =====================================
    cv2.rectangle(
        frame,
        (frame_reduction, frame_reduction),
        (w - frame_reduction, h - frame_reduction),
        (255, 0, 255),
        2
    )

    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            # =====================================
            # DRAW LANDMARKS
            # =====================================
            for landmark in hand_landmarks:

                lx = int(landmark.x * w)
                ly = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (lx, ly),
                    5,
                    (0, 255, 0),
                    -1
                )

            # =====================================
            # IMPORTANT POINTS
            # =====================================
            wrist = hand_landmarks[0]

            thumb_tip = hand_landmarks[4]

            index_tip = hand_landmarks[8]

            middle_tip = hand_landmarks[12]

            # =====================================
            # COORDINATES
            # =====================================
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)

            mx = int(middle_tip.x * w)
            my = int(middle_tip.y * h)

            # =====================================
            # DRAW IMPORTANT POINTS
            # =====================================
            cv2.circle(frame, (ix, iy), 12, (255, 0, 0), -1)

            cv2.circle(frame, (tx, ty), 12, (0, 0, 255), -1)

            cv2.circle(frame, (mx, my), 12, (255, 0, 255), -1)

            # =====================================
            # CURSOR MOVEMENT
            # =====================================
            screen_x = (
                (ix - frame_reduction)
                / (w - 2 * frame_reduction)
            ) * screen_width

            screen_y = (
                (iy - frame_reduction)
                / (h - 2 * frame_reduction)
            ) * screen_height

            curr_x = prev_x + (
                screen_x - prev_x
            ) / smoothening

            curr_y = prev_y + (
                screen_y - prev_y
            ) / smoothening

            # Stability
            if abs(curr_x - prev_x) < 3:
                curr_x = prev_x

            if abs(curr_y - prev_y) < 3:
                curr_y = prev_y

            # Limit mouse inside screen bounds
            curr_x = max(5, min(screen_width - 5, curr_x))
            curr_y = max(5, min(screen_height - 5, curr_y))

            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            # =====================================
            # LEFT CLICK
            # THUMB + INDEX
            # =====================================
            click_distance = math.hypot(
                ix - tx,
                iy - ty
            )

            if (
                click_distance < click_threshold
                and time.time() - last_click_time > 0.5
            ):

                pyautogui.click()

                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                last_click_time = time.time()

            # =====================================
            # RIGHT CLICK
            # THUMB + MIDDLE
            # =====================================
            right_click_distance = math.hypot(
                mx - tx,
                my - ty
            )

            if (
                right_click_distance < 22
                and time.time() - last_click_time > 0.7
            ):

                pyautogui.rightClick()

                cv2.putText(
                    frame,
                    "RIGHT CLICK",
                    (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    3
                )

                last_click_time = time.time()

            # =====================================
            # SCROLL
            # =====================================
            scroll_distance = math.hypot(
                ix - mx,
                iy - my
            )

            if (
                scroll_distance > 45
                and time.time() - last_scroll_time > scroll_delay
            ):

                # Scroll Up
                if iy < my:

                    pyautogui.scroll(80)

                    cv2.putText(
                        frame,
                        "SCROLL UP",
                        (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

                # Scroll Down
                elif iy > my:

                    pyautogui.scroll(-80)

                    cv2.putText(
                        frame,
                        "SCROLL DOWN",
                        (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

                last_scroll_time = time.time()

            # =====================================
            # SWIPE RIGHT / LEFT
            # =====================================
            current_time = time.time()

            current_wrist_x = wrist.x

            movement = (
                current_wrist_x
                - previous_wrist_x
            )

            # Swipe Right
            if (
                movement > swipe_threshold
                and current_time - last_swipe_time > swipe_delay
            ):

                pyautogui.hotkey(
                    'alt',
                    'right'
                )

                cv2.putText(
                    frame,
                    "NEXT PAGE",
                    (40, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

                last_swipe_time = current_time

            # Swipe Left
            elif (
                movement < -swipe_threshold
                and current_time - last_swipe_time > swipe_delay
            ):

                pyautogui.hotkey(
                    'alt',
                    'left'
                )

                cv2.putText(
                    frame,
                    "PREVIOUS PAGE",
                    (40, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

                last_swipe_time = current_time

            previous_wrist_x = current_wrist_x

            # =====================================
            # BRIGHTNESS CONTROL
            # =====================================
            brightness = np.interp(
                iy,
                [50, h - 50],
                [100, 0]
            )

            # =====================================
            # APP LAUNCHER
            # MOVE HAND TO TOP LEFT
            # =====================================
            if iy < 80 and ix < 80:

                os.system('start chrome')

                cv2.putText(
                    frame,
                    "OPENING CHROME",
                    (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

            # =====================================
            # ACTIVE TEXT
            # =====================================
            cv2.putText(
                frame,
                "Gesture AI Active",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

    # =====================================
    # FPS COUNTER
    # =====================================
    c_time = time.time()

    fps = 1 / (c_time - p_time)

    p_time = c_time

    cv2.putText(
        frame,
        f'FPS: {int(fps)}',
        (1050, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =====================================
    # GESTURE GUIDE
    # =====================================
    cv2.putText(
        frame,
        "Thumb + Index = Left Click",
        (20, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Thumb + Middle = Right Click",
        (20, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Two Fingers = Scroll",
        (500, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Swipe Right = Next Page",
        (500, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "Swipe Left = Previous Page",
        (500, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # =====================================
    # SHOW WINDOW
    # =====================================
    cv2.imshow(
        "AI Gesture Mouse",
        frame
    )

    # =====================================
    # EXIT
    # =====================================
    if cv2.waitKey(1) == ord('q'):
        break

# =====================================
# CLEANUP
# =====================================
cap.release()
cv2.destroyAllWindows()
