from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mujoco


project_root = Path(__file__).resolve().parents[1]

xml_path = (
    project_root
    / "assets"
    / "empty_world.xml"
)


# ---------------------------------
# LOAD MUJOCO MODEL
# ---------------------------------

model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

data = mujoco.MjData(
    model
)


# ---------------------------------
# PLACE ROBOT
# ---------------------------------

robot_x = 1.0
robot_y = 0.0
robot_yaw = 0.0

data.qpos[0] = robot_x
data.qpos[1] = robot_y
data.qpos[2] = robot_yaw

mujoco.mj_forward(
    model,
    data
)


# ---------------------------------
# RENDER CAMERA IMAGE
# ---------------------------------

image_width = 640
image_height = 480

renderer = mujoco.Renderer(
    model,
    height=image_height,
    width=image_width
)

renderer.update_scene(
    data,
    camera="front_camera"
)

image_rgb = renderer.render()

print("Rendered image shape:", image_rgb.shape)


# ---------------------------------
# CONVERT TO GRAYSCALE
# ---------------------------------

image_gray = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2GRAY
)


# ---------------------------------
# ARUCO DETECTOR
# ---------------------------------

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

detector_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    detector_parameters
)

corners, ids, rejected = aruco_detector.detectMarkers(
    image_gray
)

print()
print("Rejected candidate count:")
print(len(rejected))


# ---------------------------------
# CAMERA INTRINSICS
# ---------------------------------
# Based on MuJoCo camera fovy = 70 degrees

fovy_deg = 70.0
fovy_rad = np.deg2rad(fovy_deg)

fy = (image_height / 2.0) / np.tan(fovy_rad / 2.0)
fx = fy

cx = image_width / 2.0
cy = image_height / 2.0

camera_matrix = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.zeros((5, 1))

print()
print("Camera matrix:")
print(camera_matrix)


# ---------------------------------
# MARKER SIZE
# ---------------------------------
# Black ArUco square size = 0.30 m

marker_length = 0.30


# ---------------------------------
# DRAW IMAGE
# ---------------------------------

image_bgr = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR
)


# ---------------------------------
# ESTIMATE POSE
# ---------------------------------

if ids is None:

    print()
    print("No ArUco markers detected.")

else:

    print()
    print("Detected marker IDs:")
    print(ids.flatten())

    print()
    print("Number of detected markers:")
    print(len(ids))

    cv2.aruco.drawDetectedMarkers(
        image_bgr,
        corners,
        ids
    )

# ---------------------------------
# MARKER 3D CORNER COORDINATES
# ---------------------------------

half_marker = marker_length / 2.0

object_points = np.array([
    [-half_marker,  half_marker, 0.0],   # top-left
    [ half_marker,  half_marker, 0.0],   # top-right
    [ half_marker, -half_marker, 0.0],   # bottom-right
    [-half_marker, -half_marker, 0.0]    # bottom-left
], dtype=np.float32)


# ---------------------------------
# ESTIMATE POSE USING SOLVEPNP
# ---------------------------------

for i in range(len(ids)):

    marker_id = int(ids[i])

    image_points = corners[i].reshape(
        4,
        2
    ).astype(np.float32)


    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )


    if not success:

        print()
        print(
            f"Pose estimation failed for marker {marker_id}"
        )

        continue


    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)


    print()
    print(f"Marker ID: {marker_id}")

    print("Rotation vector [rvec]:")
    print(rvec)

    print()

    print("Translation vector [tvec] in metres:")
    print(tvec)


    distance = np.linalg.norm(
        tvec
    )

    expected_tvec = np.array([
        0.0,
        -0.21,
        0.665
    ])

    expected_distance = np.linalg.norm(
        expected_tvec
    )


    translation_error = (
        tvec - expected_tvec
    )

    translation_error_norm = np.linalg.norm(
        translation_error
    )


    distance_error = (
        distance
        - expected_distance
    )


    distance_error_percent = (
        abs(distance_error)
        /
        expected_distance
        *
        100.0
    )


    print()
    print("Expected translation vector [m]:")
    print(expected_tvec)

    print()

    print("Translation error vector [m]:")
    print(translation_error)

    print()

    print("Translation error magnitude [m]:")
    print(translation_error_norm)

    print()

    print("Expected marker distance [m]:")
    print(expected_distance)

    print()

    print("Distance error [m]:")
    print(distance_error)

    print()

    print("Distance error [%]:")
    print(distance_error_percent)


    print()

    print("Marker distance from camera [m]:")
    print(distance)


    cv2.drawFrameAxes(
        image_bgr,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        0.10
    )


# ---------------------------------
# DISPLAY RESULT
# ---------------------------------

image_display = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(image_display)
plt.title("ArUco Pose Estimation")
plt.axis("off")
plt.show()