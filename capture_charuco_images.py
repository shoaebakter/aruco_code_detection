import cv2
import cv2.aruco as aruco
import os

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(5, 7, 0.04, 0.02, aruco_dict)

os.makedirs("charuco_images", exist_ok=True)
cap = cv2.VideoCapture(1)
count = 0

print("📸 Press 's' to save frame, 'q' to quit")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Charuco Capture", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"charuco_images/img_{count}.png", frame)
        print(f"✅ Saved img_{count}.png")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
