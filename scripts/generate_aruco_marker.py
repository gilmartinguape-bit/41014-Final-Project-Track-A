from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parents[1]

output_path = (
    project_root
    / "assets"
    / "aruco_0.png"
)


dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)


marker_id = 0

marker_size_pixels = 512


marker_image = cv2.aruco.generateImageMarker(
    dictionary,
    marker_id,
    marker_size_pixels
)


cv2.imwrite(
    str(output_path),
    marker_image
)


print(
    "ArUco marker saved to:",
    output_path
)