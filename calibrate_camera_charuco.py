import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Initialize Charuco board and dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(5, 7, 0.04, 0.02, aruco_dict)

all_corners = []
all_ids = []
image_size = None
valid_images = 0

image_folder = "charuco_images"
if not os.path.exists(image_folder):
    print("❌ No 'charuco_images' folder found. Please add your images.")
    exit()

# Accepted image extensions
valid_ext = (".png", ".jpg", ".jpeg")

for fname in sorted(os.listdir(image_folder)):
    if not fname.lower().endswith(valid_ext):
        continue

    path = os.path.join(image_folder, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"{fname} — ⚠️ Failed to load")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    print(f"{fname} — Detected markers: {len(ids) if ids is not None else 0}")

    if ids is not None and len(ids) > 0:
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 15:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            image_size = gray.shape[::-1]
            valid_images += 1
            print(f"{fname} — ✅ Used (Charuco corners: {len(charuco_corners)})")
        else:
            print(f"{fname} — ⚠️ Not enough Charuco corners")
    else:
        print(f"{fname} — ❌ No ArUco markers detected")

if valid_images < 3:
    print("❌ Not enough valid images. You need at least 3 good captures.")
    exit()

print(f"🔧 Starting calibration using {valid_images} images...")

ret, cameraMatrix, distCoeffs, _, _ = aruco.calibrateCameraCharuco(
    all_corners, all_ids, board, image_size, None, None
)

np.savez("calibration_data.npz", cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

print("✅ Calibration complete!")
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Coefficients:\n", distCoeffs)
