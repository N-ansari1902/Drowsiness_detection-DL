import cv2
import dlib
import numpy as np
from imutils import face_utils

# Yawning Detection
def mouth_aspect_ratio(mouth):
    # Compute mouth aspect ratio (MAR) for yawning
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

#Head Pose Detection
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right Mouth corner
], dtype="double")

def get_head_pose(shape, size):
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left Mouth corner
        shape[54]      # Right Mouth corner
    ], dtype="double")

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))  
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = eulerAngles.flatten()
    return pitch, yaw, roll

#MAin logic
mar_thresh = 0.7    
mar_frames = 15     
yawn_counter = 0

pitch_thresh = 30.0 
nod_frames = 15
nod_counter = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #Yawning Detection
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)
        if mar > mar_thresh:
            yawn_counter += 1
        else:
            yawn_counter = 0
        if yawn_counter >= mar_frames:
            cv2.putText(frame, "Yawning Detected!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        # Draw mouth for visualization
        cv2.polylines(frame, [mouth], True, (0,255,0), 1)

        #Head Pose Detection
        try:
            pitch, yaw_angle, roll = get_head_pose(shape, size)
            cv2.circle(frame, tuple(shape[30]), 3, (255,0,255), -1)
            if abs(pitch) > pitch_thresh:
                nod_counter += 1
            else:
                nod_counter = 0
            if nod_counter >= nod_frames:
                cv2.putText(frame, "Drowsy Head Nod!", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,140,255), 3)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,0), 2)
        except:
            pass

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
