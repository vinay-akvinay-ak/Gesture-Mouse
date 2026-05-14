import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# SETTINGS
# -------------------------------

pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

smoothening = 5

click_threshold = 28

scroll_threshold = 45

swipe_threshold = 0.12

cooldown = 0.8

prev_x = 0
prev_y = 0

last_click_time = 0
last_swipe_time = 0

previous_wrist_x = 0

running = True

# -------------------------------
# LOAD MODEL
# -------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)

# -------------------------------
# WEBCAM
# -------------------------------

cap = cv2.VideoCapture(0)

# -------------------------------
# MAIN LOOP
# -------------------------------

while running:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    h, w, _ = frame.shape

    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            # -------------------------------
            # DRAW LANDMARKS
            # -------------------------------

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

            # -------------------------------
            # FINGERS
            # -------------------------------

            index_finger = hand_landmarks[8]
            thumb = hand_landmarks[4]
            middle_finger = hand_landmarks[12]
            wrist = hand_landmarks[0]

            # -------------------------------
            # INDEX POSITION
            # -------------------------------

            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            screen_x = screen_width / w * x
            screen_y = screen_height / h * y

            # Smooth movement

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            # Prevent corner jumps

            curr_x = max(5, min(screen_width - 5, curr_x))
            curr_y = max(5, min(screen_height - 5, curr_y))

            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            # -------------------------------
            # DRAW INDEX POINTER
            # -------------------------------

            cv2.circle(
                frame,
                (x, y),
                10,
                (255, 0, 0),
                -1
            )

            # -------------------------------
            # THUMB POSITION
            # -------------------------------

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (0, 0, 255),
                -1
            )

            # -------------------------------
            # CLICK GESTURE
            # -------------------------------

            click_distance = math.hypot(
                x - thumb_x,
                y - thumb_y
            )

            current_time = time.time()

            if (
                click_distance < click_threshold
                and current_time - last_click_time > 0.5
            ):

                pyautogui.click()

                last_click_time = current_time

                cv2.putText(
                    frame,
                    "CLICK",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # -------------------------------
            # SCROLL GESTURE
            # -------------------------------

            middle_x = int(middle_finger.x * w)
            middle_y = int(middle_finger.y * h)

            cv2.circle(
                frame,
                (middle_x, middle_y),
                10,
                (255, 0, 255),
                -1
            )

            finger_distance = math.hypot(
                x - middle_x,
                y - middle_y
            )

            if finger_distance > scroll_threshold:

                # Scroll Up

                if y < middle_y:

                    pyautogui.scroll(60)

                    cv2.putText(
                        frame,
                        "SCROLL UP",
                        (40, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

                # Scroll Down

                elif y > middle_y:

                    pyautogui.scroll(-60)

                    cv2.putText(
                        frame,
                        "SCROLL DOWN",
                        (40, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )

            # -------------------------------
            # SWIPE DETECTION
            # -------------------------------

            wrist_x = wrist.x

            movement = wrist_x - previous_wrist_x

            # NEXT PAGE

            if (
                movement > swipe_threshold
                and current_time - last_swipe_time > cooldown
            ):

                pyautogui.hotkey('alt', 'right')

                last_swipe_time = current_time

                cv2.putText(
                    frame,
                    "NEXT PAGE",
                    (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

            # PREVIOUS PAGE

            elif (
                movement < -swipe_threshold
                and current_time - last_swipe_time > cooldown
            ):

                pyautogui.hotkey('alt', 'left')

                last_swipe_time = current_time

                cv2.putText(
                    frame,
                    "PREVIOUS PAGE",
                    (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

            previous_wrist_x = wrist_x

    # -------------------------------
    # TITLE
    # -------------------------------

    cv2.putText(
        frame,
        "AI Gesture Mouse Active",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -------------------------------
    # SHOW WINDOW
    # -------------------------------

    cv2.imshow("AI Gesture Mouse", frame)

    # -------------------------------
    # EXIT
    # -------------------------------

    key = cv2.waitKey(1)

    if key == ord('q'):

        running = False

        break

# -------------------------------
# CLEANUP
# -------------------------------

cap.release()

cv2.destroyAllWindows()