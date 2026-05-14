# AI Gesture Mouse

Control your computer using hand gestures with AI-powered hand tracking using MediaPipe and OpenCV.

---

# Features

- AI Mouse Control
- Left Click Gesture
- Scroll Up / Down
- Swipe Left → Previous Page
- Swipe Right → Next Page
- Smooth Cursor Movement
- Real-Time Hand Tracking
- Webcam-Based Control
- Desktop GUI Launcher

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- Tkinter

---

# Project Structure

```bash
GestureMouse/
│
├── src/
│   ├── app.py
│   ├── gesture.py
│   ├── hand_landmarker.task
│
├── dist/
│   └── gesture.exe
│
├── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/vinay-akvinay-ak/Gesture-Mouse.git
```

---

## 2. Open Project Folder

```bash
cd Gesture-Mouse
```

---

## 3. Install Requirements

```bash
pip install opencv-python mediapipe pyautogui
```

---

# Download MediaPipe Model

Download:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place file inside:

```bash
src/
```

---

# Run Application

```bash
python src/app.py
```

---

# Controls

| Gesture | Action |
|---|---|
| Index Finger Move | Mouse Movement |
| Thumb + Index Close | Left Click |
| Two Fingers Up | Scroll |
| Fast Swipe Right | Next Page |
| Fast Swipe Left | Previous Page |
| Q Key | Exit |

---

# Build EXE

```bash
pyinstaller --onefile --windowed src/app.py
```

---

# Future Improvements

- Right Click
- Double Click
- Drag & Drop
- Voice Commands
- Brightness Control
- Volume Control
- Virtual Keyboard
- Custom Gesture Training

---

# Author

Vinayak Goud

GitHub:
https://github.com/vinay-akvinay-ak