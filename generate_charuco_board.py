import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(5, 7, 0.04, 0.02, aruco_dict)
img = board.draw((1000, 1400))
cv2.imwrite("charuco_board.png", img)
print("✅ Charuco board saved as charuco_board.png")
