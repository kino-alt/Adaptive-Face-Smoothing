import cv2
import mediapipe as mp
import time

# Indices for key landmarks (Face Mesh 478):
# Mouth corners: left=61, right=291
# Inner eyebrows (near glabella): left=70, right=300
KEYPOINTS = {
    "mouth_left": 61,
    "mouth_right": 291,
    "brow_inner_left": 70,
    "brow_inner_right": 300,
}

# Low-load capture settings
FRAME_W, FRAME_H = 640, 480
PRINT_INTERVAL_SEC = 0.2  # throttle console output

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

last_print = 0.0
frame_skip = 1  # set to 1 for full rate; increase to skip frames
frame_count = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # enables 478 landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_skip > 1 and (frame_count % frame_skip) != 0:
            # Show raw frame to keep UI responsive
            cv2.imshow("Face Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = results.multi_face_landmarks[0]

            # Draw only selected points for low render cost
            for name, idx in KEYPOINTS.items():
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)  # yellow

            # Throttle console output
            now = time.time()
            if now - last_print >= PRINT_INTERVAL_SEC:
                last_print = now
                coords = {}
                for name, idx in KEYPOINTS.items():
                    lm = face_landmarks.landmark[idx]
                    coords[name] = (lm.x, lm.y, lm.z)
                print(coords)

        cv2.imshow("Face Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
