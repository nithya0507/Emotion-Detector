import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF logs

import cv2
import numpy as np
import time
import base64
import atexit
import multiprocessing

from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque

# 🔧 Fix multiprocessing issues
multiprocessing.set_start_method('fork', force=True)

# 🔧 Disable GPU (prevents Metal crashes on Mac)
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# ✅ Load model ONCE
model = load_model("emotion_model.keras")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_buffer = deque(maxlen=3)
current_emotion = "Detecting..."

# ✅ Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ✅ Initialize camera safely (Mac fix)
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ✅ HERE
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # ✅ HERE

if not camera.isOpened():
    print("❌ Camera not accessible")

emoji_map = {
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😡",
    "Surprise": "😲",
    "Fear": "😨",
    "Disgust": "🤢",
    "Neutral": "😐"
}


# 🔧 Preprocess face
def preprocess_face(gray, x, y, w, h):
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


# 🎥 Video stream generator (optimized)
def generate_frames():
    global current_emotion, camera

    frame_count = 0   # controls prediction frequency

    while True:
        try:
            success, frame = camera.read()

            # Restart camera if it fails
            if not success:
                print("❌ Frame read failed, restarting camera...")
                camera.release()
                time.sleep(1)
                camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = preprocess_face(gray, x, y, w, h)

                # Run prediction only every 5 frames
                frame_count += 1
                if frame_count % 3 == 0:
                    prediction = model.predict(face, verbose=0)
                    pred_index = np.argmax(prediction)
                    emotion_buffer.append(pred_index)

                # Use smoothed result
                if len(emotion_buffer) > 0:
                    smooth_index = max(set(emotion_buffer), key=emotion_buffer.count)
                    emotion = emotions[smooth_index]
                    current_emotion = emotion

                    emoji = emoji_map.get(emotion, "😐")

                    cv2.putText(frame, emoji, (x, y-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            time.sleep(0.01)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("⚠️ Error in stream:", e)
            break

# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/modes')
def modes():
    return render_template('modes.html')


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'GET':
        return render_template('capture.html')

    # POST → capture image
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Camera failed"})

    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "emotion": current_emotion,
        "image": img_base64
    })


@app.route('/meme')
def meme():
    return render_template('meme.html')


@app.route('/advice')
def advice():
    return render_template('advice.html')


@app.route('/avatar')
def avatar():
    return render_template('avatar.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        "Cache-Control": "no-cache",
                        "Pragma": "no-cache"
                    })


@app.route('/current_emotion')
def get_emotion():
    return jsonify({"emotion": current_emotion})


# 😂 MEME
@app.route('/get_meme', methods=['POST'])
def get_meme():
    emotion = request.json.get("emotion", "neutral").lower()

    return jsonify({
        "image": f"/static/meme/{emotion}.jpg"
    })


# 💡 ADVICE
@app.route('/get_advice', methods=['POST'])
def get_advice():
    emotion = request.json.get("emotion", "Neutral")

    advice_map = {
        "Happy": "Keep smiling 😊 Spread positivity!",
        "Sad": "It's okay to feel down 💙",
        "Angry": "Take a deep breath 😌",
        "Fear": "You’re stronger than your fears 💪",
        "Surprise": "Enjoy the moment 😲",
        "Disgust": "Focus on good vibes ✨",
        "Neutral": "Stay steady 👍"
    }

    return jsonify({
        "advice": advice_map.get(emotion, "Stay positive!")
    })


# 🧑‍🎨 AVATAR
@app.route('/get_avatar', methods=['POST'])
def get_avatar():
    emotion = request.json.get("emotion", "Neutral")

    return jsonify({
        "emoji": emoji_map.get(emotion, "😐")
    })


# 🔒 Clean exit
@atexit.register
def release_camera():
    if camera.isOpened():
        camera.release()
    print("✅ Camera released")


# 🚀 Run app (stable mode)
if __name__ == "__main__":
   app.run(debug=False, port=5000, threaded=True) 