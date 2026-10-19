# 41014 Sensors and Control in Mechatronic Systems
## Integrated Project — Mobile Robot Visual Docking with EKF, LQR, LiDAR and ArUco

This repository contains the current working simulation for the **41014 Sensors and Control in Mechatronic Systems Integrated Project** at the University of Technology Sydney (UTS).

The project follows **Track A: Mobile Robot — Visual Waypoint Navigation**. A TurtleBot3-like differential-drive mobile robot is simulated in **MuJoCo** with wheel encoders, 2D LiDAR and a forward RGB camera. The robot estimates its pose using an **Extended Kalman Filter (EKF)**, follows the docking path using **LQR/state-feedback control**, avoids an obstacle using a **LiDAR safety layer**, and performs final docking near an **ArUco-marked charging station**.

The controller uses the **estimated state only**. Ground-truth state is retained only for simulation, sensor generation and performance evaluation.

---

## 1. Current Project Status

The complete simulation pipeline is working end-to-end in two final scenarios:

- **Nominal scenario:** unobstructed approach and visual docking.
- **Obstacle scenario:** approach, LiDAR-triggered avoidance, recovery, ArUco reacquisition and final docking.

Both scenarios generate quantitative metrics, trajectory histories, plots and synchronized demonstration videos.

### Current verified final results

| Metric | Nominal | Obstacle |
|---|---:|---:|
| ArUco / visual EKF updates | 80 | 44 |
| Final localisation error | 0.01124 m | 0.00748 m |
| Position RMSE | 0.03069 m | 0.05590 m |
| Lateral RMSE | 0.08806 m | 0.14149 m |
| Heading RMSE | 9.45 deg | 25.66 deg |
| Final X docking error | -0.00962 m | -0.00667 m |
| Final Y docking error | +0.00571 m | -0.00794 m |
| Final planar docking error | **0.01119 m** | **0.01037 m** |
| Final heading error | **-2.34 deg** | **+2.60 deg** |
| Obstacle avoidance | Not applicable | Successful |

The obstacle case produces larger transient tracking and heading errors because the robot deliberately turns away from the direct path, but it recovers and achieves final docking accuracy comparable with the nominal case.

---

## 2. System Architecture

```text
Wheel encoders
      |
      v
EKF prediction
      |
      +----------------------+
      |                      |
      |              RGB camera + ArUco
      |                      |
      |               pose measurement
      |                      |
      +-----------> EKF correction
                         |
                         v
                Estimated robot pose
                         |
            +------------+------------+
            |                         |
            v                         v
      LQR approach              Final pose
       controller               controller
            |                         |
            +------------+------------+
                         |
                         v
                Nominal command
                         |
          +--------------+--------------+
          |                             |
          | nominal scenario            | obstacle scenario
          v                             v
        Robot                 LiDAR safety override
                                        |
                                        v
                                      Robot
```

### Important implementation rule

The controller receives the **EKF estimate**, not the MuJoCo ground-truth pose.

Ground truth is used only to:

- propagate the simulated robot,
- simulate sensor measurements,
- calculate errors and RMSE,
- compare estimated and true trajectories.

---

## 3. Main Components

### 3.1 Differential-drive motion model

The robot is represented by the planar state:

```text
x = [x, y, theta]
```

with control input:

```text
u = [v, omega]
```

The MuJoCo environment provides the 3D scene, cameras and ray casting, while planar robot motion is propagated using the differential-drive/unicycle kinematic model.

### 3.2 Wheel encoders

Wheel angular-speed measurements are simulated with random encoder noise and a small right-wheel encoder bias. The noisy wheel measurements are converted back to measured `v` and `omega` for EKF prediction.

### 3.3 Extended Kalman Filter

The EKF fuses:

1. wheel-encoder odometry for prediction, and
2. ArUco-derived robot pose for visual correction.

Heading innovations are angle-wrapped, and the covariance update uses the Joseph form.

### 3.4 ArUco camera localisation

The forward RGB camera detects an ArUco marker at the docking station. OpenCV is used for marker detection, corner extraction, `solvePnP`, and conversion of marker/camera geometry into robot pose in the world frame.

### 3.5 LQR approach control

The approach controller regulates lateral deviation and heading error.

Current design settings:

```text
q_y       = 8.0
q_heading = 4.0
r_omega   = 1.2
```

The resulting gain is approximately:

```text
K = [2.582, 2.065]
```

### 3.6 Final pose docking controller

Near the docking station, control switches from LQR approach control to a pose controller with target:

```text
[1.20 m, 0.00 m, 0 deg]
```

### 3.7 LiDAR obstacle safety layer

The obstacle scenario uses left, front and right LiDAR rays at approximately +20 deg, 0 deg and -20 deg.

State sequence:

```text
NORMAL -> AVOIDING -> RECOVERY -> NORMAL
```

In the **nominal scenario**, LiDAR is measured and plotted but does not override the controller.

---

## 4. Key Configuration

```text
DT               = 0.05 s
Wheel radius     = 0.033 m
Wheel base       = 0.160 m
Nominal speed    = 0.18 m/s
Simulation time  = 20 s
Camera FOV       = 90 deg
Video frame rate = 20 fps
```

Initial true pose:

```text
[0.00 m, 0.25 m, 8 deg]
```

Initial EKF estimate:

```text
[0.00 m, 0.20 m, 6 deg]
```

Docking target:

```text
[1.20 m, 0.00 m, 0 deg]
```

ArUco marker world position:

```text
[1.755 m, 0.00 m, 0.35 m]
```

---

## 5. Repository Structure

```text
41014_Final_Project/
|
|-- assets/
|   |-- empty_world.xml
|   |-- nominal_world.xml
|   |-- obstacle_world.xml
|   `-- aruco_0.png
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- model.py
|   |-- sensors.py
|   |-- ekf.py
|   |-- controller.py
|   |-- lidar.py
|   |-- vision.py
|   `-- utils.py
|
|-- scripts/
|   |-- simulate_mujoco_nominal.py
|   |-- simulate_mujoco_obstacle_trial.py
|   |-- compare_final_results.py
|   `-- supporting tests and development scripts
|
|-- results/
|   |-- nominal/
|   |-- obstacle/
|   `-- comparison/
|
|-- requirements.txt
`-- README.md
```

---

## 6. Software Environment

```text
Operating system : Windows
Python           : 3.11.9
Virtual env      : .venv
Git              : 2.56.0.windows.2
Editor           : Visual Studio Code
Simulation       : MuJoCo
Computer vision  : OpenCV contrib
Numerical tools  : NumPy / SciPy
Plotting         : Matplotlib
```

Dependencies in `requirements.txt`:

```text
numpy
scipy
matplotlib
mujoco
opencv-contrib-python
```

---

## 7. Setup

```powershell
cd C:\41014_Final_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional checks:

```powershell
python -c "import mujoco; print(mujoco.__version__)"
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
```

---

## 8. Running the Final Simulations

### Nominal scenario

```powershell
python -m py_compile scripts\simulate_mujoco_nominal.py
python -m scripts.simulate_mujoco_nominal
```

Outputs:

```text
results/nominal/
```

### Obstacle scenario

```powershell
python -m py_compile scripts\simulate_mujoco_obstacle_trial.py
python -m scripts.simulate_mujoco_obstacle_trial
```

Outputs:

```text
results/obstacle/
```

### Final comparison

```powershell
python -m py_compile scripts\compare_final_results.py
python -m scripts.compare_final_results
```

Outputs:

```text
results/comparison/
```

---

## 9. Generated Evidence

Each final scenario produces:

```text
metrics.csv
trajectory_history.csv
simulation.mp4
robot_camera.mp4
birdseye_perspective.mp4
top_view.mp4
four_view_grid.mp4
trajectory.png
lateral_error.png
heading_error.png
control_commands.png
lidar_clearance.png
aruco_updates.png
latest_aruco_detection.png
```

The four-view grid combines:

```text
+---------------------------+---------------------------+
| OVERVIEW                  | ROBOT CAMERA              |
+---------------------------+---------------------------+
| BIRD'S-EYE PERSPECTIVE    | TOP VIEW                  |
+---------------------------+---------------------------+
```

Its status bar shows simulation time, active controller, scenario/LiDAR state, ArUco visibility and EKF visual update status.

---

## 10. Comparison Outputs

`compare_final_results.py` generates:

```text
results/comparison/
|-- comparison_metrics.csv
|-- camera_update_comparison.png
|-- docking_error_comparison.png
|-- heading_error_comparison.png
|-- localisation_rmse_comparison.png
|-- trajectory_comparison.png
`-- final_results_summary.txt
```

The trajectory comparison shows the nominal direct approach, the obstacle-avoidance detour, recovery toward the docking line, and convergence of both scenarios near the same final target.

---

## 11. Hardware Lab Induction Demonstration

Recommended presentation sequence:

1. Show this README and repository structure.
2. Explain: **encoders + ArUco -> EKF -> estimated state -> controller -> robot**.
3. Emphasise that the controller does **not** use ground truth.
4. Show the nominal `four_view_grid.mp4`.
5. Show the obstacle `four_view_grid.mp4`.
6. Point out `NORMAL -> AVOIDING -> RECOVERY -> NORMAL`.
7. Show `trajectory_comparison.png`.
8. Show the final nominal-vs-obstacle metrics.
9. Explain why obstacle motion increases transient RMSE but final docking remains accurate.
10. Explain how the simulation interfaces will map to real encoders, LiDAR, camera and motor commands.

### One-sentence project summary

> The simulation integrates wheel odometry, ArUco vision, EKF localisation, LQR path control, final pose control and LiDAR obstacle avoidance to autonomously approach and dock a differential-drive robot using estimated-state feedback.

---

## 12. Hardware Induction Readiness

The current simulation demonstrates:

- differential-drive modelling,
- wheel-encoder sensing,
- camera/ArUco localisation,
- LiDAR ranging,
- EKF sensor fusion,
- estimated-state feedback,
- LQR approach control,
- final pose docking,
- obstacle avoidance and recovery,
- quantitative nominal/obstacle evaluation.

For hardware implementation, the simulation interfaces must be mapped to real:

```text
wheel encoder data
2D LiDAR scans
RGB camera images
motor velocity commands
robot coordinate frames
sensor calibration parameters
```

The simulation should remain the baseline for later simulation-vs-hardware comparison.

---

## 13. Known Notes

### GLFW shutdown warning

After a successful Windows run, the terminal may print:

```text
GLFWError: (65537) b'The GLFW library is not initialized'
```

This has been observed during viewer cleanup after metrics, figures and videos have already been saved. It is not treated as a failed simulation when all expected outputs were generated successfully.

### Kinematic MuJoCo motion

MuJoCo provides the 3D environment, rendering, camera sensing and LiDAR ray casting. Planar robot motion is updated using the differential-drive kinematic model rather than detailed tyre/contact dynamics.

---

## 14. Current Limitations

- static simulation environment,
- single ArUco docking marker,
- simplified LiDAR safety behaviour,
- kinematic rather than wheel-contact dynamics,
- fixed encoder noise/bias model,
- no hardware sensor calibration yet,
- no real communication latency or actuator dynamics.

---

## 15. Next Stage — Hardware

```text
Working MuJoCo simulation
        |
        v
Hardware induction
        |
        v
Real sensor inspection and calibration
        |
        v
Encoder / LiDAR / camera interface mapping
        |
        v
Low-speed motor-command validation
        |
        v
EKF validation on real measurements
        |
        v
Controller validation
        |
        v
Obstacle and docking experiments
        |
        v
Simulation vs hardware comparison
```

Hardware testing should begin with individual sensor and actuator checks before attempting the complete autonomous docking mission.

---

## 16. Academic Integrity and AI Declaration

The final project page should acknowledge external examples, tutorials and code structures used during development.

Generative AI use should also be declared according to subject requirements, including the tool and purpose. For this project, AI-assisted development may be described transparently as support for code explanation, debugging, documentation, simulation planning and viva preparation. The team remains responsible for understanding, verifying and presenting the submitted implementation.

---

## 17. Final Simulation Summary

The project demonstrates an end-to-end sensing, estimation and control pipeline in MuJoCo.

The obstacle scenario causes larger transient localisation, lateral and heading errors during the avoidance manoeuvre. After recovery, however, the robot successfully returns to the docking task and achieves a final planar docking error of approximately **10.4 mm**, comparable with approximately **11.2 mm** in the nominal run.

This simulation provides the baseline for the next phase of hardware validation.
