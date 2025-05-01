import cv2
import cv2.aruco as aruco
import numpy as np

# === Load calibration data ===
calib = np.load("calibration_data.npz")
cameraMatrix = calib["cameraMatrix"]
distCoeffs = calib["distCoeffs"]

# === Select dictionary and parameters ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
params = aruco.DetectorParameters_create()

# === Open the camera ===
cap = cv2.VideoCapture(1)  # Use 0 or 1 depending on your camera

# === Set resolution to 720p ===
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === Optional: Set target framerate ===
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is not None and len(ids) > 0:
        for corner in corners:
            int_corners = corner.astype(int)
            cv2.polylines(frame, [int_corners], isClosed=True, color=(0, 255, 255), thickness=2)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.04, cameraMatrix, distCoeffs)

        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            top_left = tuple(c[0].astype(int))
            label_pos = (top_left[0], top_left[1] - 10)
            decision_pos = (top_left[0], top_left[1] + 20)
            line2_pos = (top_left[0], top_left[1] + 40)
            line3_pos = (top_left[0], top_left[1] + 60)
            line4_pos = (top_left[0], top_left[1] + 80)

            # === Draw marker ID ===
            cv2.putText(frame, f"ID={marker_id}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # === Show instructions based on marker ID ===
            if marker_id == 297:
                cv2.putText(frame, "Airlock Detected", decision_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif marker_id == 269:
                cv2.putText(frame, "Lava-Tube Detected", decision_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Detected. Go.", decision_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # === Get position and distance ===
            tvec = tvecs[i][0]
            rvec = rvecs[i][0]
            x, y, z = tvec
            distance = np.linalg.norm(tvec)

            # === Display distance and position
            cv2.putText(frame, f"Distance: {distance:.2f} m", line2_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}", line3_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # === Draw 3D axis ===
            aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03)

    cv2.imshow("Aruco Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
