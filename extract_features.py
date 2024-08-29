from deepface import DeepFace

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

embedding_objs = DeepFace.represent(
    img_path = "db_faces/test2.jpg",
    model_name = models[7],
    detector_backend = backends[2],
)

import pickle

embeddings_file_path = "db_face_features/embeddings2.pkl"

with open(embeddings_file_path, "wb") as f:
    pickle.dump(embedding_objs[0]['embedding'], f)


print(embedding_objs)