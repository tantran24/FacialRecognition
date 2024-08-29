from deepface import DeepFace
import cv2
import numpy as np
import time
import pickle

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'fastmtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

cap = cv2.VideoCapture(0)
# reference_image_path = "photo-2-1491486311951111228.jpg"  # Replace with the actual path
reference_image_path = "db_faces/test1.jpg"
start_time = time.time()

frame_counter = 0
embeddings_file_path="db_face_features/embeddings2.pkl"

with open(embeddings_file_path, "rb") as f:
    loaded_embeddings = pickle.load(f)
    loaded_embeddings=np.array(loaded_embeddings)

while True:
    ret, frame = cap.read()

    try:
        if frame_counter % 40 == 0:
        #     verification_result = DeepFace.verify(
        #         # img1_path=reference_image_path,
        #         # img2_path=frame,
        #         img1_path="db_faces/test.jpg",

        #         img2_path="db_faces/test2.jpg",
        #         model_name = models[7],
        #         detector_backend = backends[2],

        #         # threshold=0.4
        #     )
        #     print(verification_result)
            
            
            embedding_objs = DeepFace.represent(
            img_path = frame,
            model_name = models[7],
            detector_backend = backends[2]
        )
            embedding_objs_frame = embedding_objs[0]['embedding']
            
            cosine_sim = np.dot(embedding_objs_frame, loaded_embeddings.T) / (np.linalg.norm(embedding_objs_frame) * np.linalg.norm(loaded_embeddings))
            
            verification_result = 1 - cosine_sim
            print(verification_result)
            if verification_result < 0.07:
                print("Verified: Same person")
            else:
                print("Not verified: Different persons")

        frame_counter += 1

    except Exception as err:
        print(str(err))


    

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

