import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import os

mp_face = mp.solutions.face_mesh

def extract_video_features(frames_folder):

    face_mesh = mp_face.FaceMesh(static_image_mode=False)

    feature_sequence = []
    prev_landmarks = None

    frame_files = sorted(os.listdir(frames_folder))

    for file in frame_files:

        frame = cv2.imread(os.path.join(frames_folder, file))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

        # Motion
        if prev_landmarks is not None:
            motion = np.mean(np.abs(coords - prev_landmarks))
        else:
            motion = 0

        prev_landmarks = coords

        # Tension
        tension = np.var(coords)

        # Asymmetry
        left = coords[:234]
        right = coords[234:]
        asymmetry = np.mean(np.abs(left - right))

        # Emotion
        try:
            emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
            emotion_vector = list(emotion['emotion'].values())
        except:
            emotion_vector = [0]*7

        feature_vector = np.concatenate([
            [motion, tension, asymmetry],
            emotion_vector
        ])

        feature_sequence.append(feature_vector)

    return np.array(feature_sequence)