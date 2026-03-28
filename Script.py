import cv2
import numpy as np
import pygame
import mediapipe as mp
import time
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

pygame.mixer.init()
pygame.mixer.music.load('FAHHH (Meme Sound Effect).mp3')
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.load_weights('custom_cnn_mata.h5')
print("Model berhasil dimuat!")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

frame_terbaru = None
skor_kantuk = 0
sistem_berjalan = True

INTERVAL_CEK_AI = 0
BATAS_TIDUR = 15

def potong_mata(frame, landmarks, indices, w, h):
    x_min = min([landmarks[i].x for i in indices])
    x_max = max([landmarks[i].x for i in indices])
    y_min = min([landmarks[i].y for i in indices])
    y_max = max([landmarks[i].y for i in indices])

    padding_x = 5
    padding_y = 5

    x1, x2 = int(x_min * w) - padding_x, int(x_max * w) + padding_x
    y1, y2 = int(y_min * h) - padding_y, int(y_max * h) + padding_y

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    return crop

def tugas_ai_background():
    global skor_kantuk, frame_terbaru, sistem_berjalan
    
    waktu_terakhir_cek = time.time()
    
    while sistem_berjalan:
        waktu_sekarang = time.time()
        
        if waktu_sekarang - waktu_terakhir_cek >= INTERVAL_CEK_AI:
            waktu_terakhir_cek = waktu_sekarang
            
            if frame_terbaru is not None:
                frame_proses = frame_terbaru.copy()
                tinggi, lebar, _ = frame_proses.shape
                mata_tertutup = False
                
                frame_kecil = cv2.resize(frame_proses, (320, 240))
                rgb_frame = cv2.cvtColor(frame_kecil, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        mata_kiri_crop = potong_mata(frame_proses, landmarks, LEFT_EYE_INDICES, lebar, tinggi)
                        mata_kanan_crop = potong_mata(frame_proses, landmarks, RIGHT_EYE_INDICES, lebar, tinggi)
                        
                        kiri_tertutup = False
                        kanan_tertutup = False

                        if mata_kiri_crop.size != 0:
                            img_kiri = cv2.resize(cv2.cvtColor(mata_kiri_crop, cv2.COLOR_BGR2RGB), (64, 64))
                            img_kiri = np.expand_dims(img_kiri, axis=0) / 255.0
                            prob_kiri = model.predict(img_kiri, verbose=0)[0][0]
                            if prob_kiri < 0.5:
                                kiri_tertutup = True

                        if mata_kanan_crop.size != 0:
                            img_kanan = cv2.resize(cv2.cvtColor(mata_kanan_crop, cv2.COLOR_BGR2RGB), (64, 64))
                            img_kanan = np.expand_dims(img_kanan, axis=0) / 255.0
                            prob_kanan = model.predict(img_kanan, verbose=0)[0][0]
                            if prob_kanan < 0.5:
                                kanan_tertutup = True
                        
                        if mata_kiri_crop.size != 0 and mata_kanan_crop.size != 0:
                            print(f"Kiri: {prob_kiri:.3f} | Kanan: {prob_kanan:.3f}")

                        if kiri_tertutup and kanan_tertutup:
                            mata_tertutup = True

                if mata_tertutup:
                    skor_kantuk += 1  
                else:
                    skor_kantuk = 0   

                if skor_kantuk >= BATAS_TIDUR:
                    if skor_kantuk == BATAS_TIDUR:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                    elif skor_kantuk > BATAS_TIDUR:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()
        
        time.sleep(0.01)

thread_ai = threading.Thread(target=tugas_ai_background)
thread_ai.daemon = True
thread_ai.start()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pTime = time.time()
fps_avg = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_terbaru = frame

    cTime = time.time()
    diff = cTime - pTime
    if diff > 0:
        fps_instant = 1 / diff
        fps_avg = (fps_avg * 0.9) + (fps_instant * 0.1)
    pTime = cTime
    
    cv2.putText(frame, f"FPS: {int(fps_avg)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
    cv2.putText(frame, f"Frame Terdeteksi: {skor_kantuk} / {BATAS_TIDUR}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Kantuk Deteksi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        sistem_berjalan = False
        break

thread_ai.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()