# 41014 Integrated Project — Mobile Robot Visual Docking

**Subject:** 41014 Sensors and Control for Mechatronic Systems  
**Track:** Track A — Mobile robot visual waypoint navigation / visual docking  

**Robot:** TurtleBot3 Waffle Pi geometry in MuJoCo with planar differential-drive kinematics  
**Core pipeline:** Wheel encoders + camera/ArUco → EKF → LQR / final pose controller  
**Robustness extension:** LiDAR safety override, obstacle recovery, side-obstacle assist and command smoothing

## Project overview

This project implements an end-to-end sensing, estimation and control pipeline for a differential-drive mobile robot in MuJoCo. The robot starts with lateral and heading error, estimates its pose from noisy wheel encoders and intermittent ArUco-camera measurements, avoids an obstacle, recovers toward the docking line, and finishes at a target pose.

The controller uses the **EKF estimated state only**. MuJoCo ground truth is used only to calculate evaluation metrics and to drive the simulated sensor geometry.

The final docking target is

```math
\mathbf{x}_d=
\begin{bmatrix}
1.30 & 0 & 0
\end{bmatrix}^{T}
```

corresponding to \(x=1.30\text{ m}\), \(y=0\text{ m}\), and \(\theta=0^\circ\).

**Platform:** MuJoCo differential-drive mobile robot  
**Core pipeline:** Wheel encoders + camera/ArUco → EKF → LQR / final pose controller  
**Robustness extension:** LiDAR safety override for obstacle avoidance

## Project overview

This project implements an end-to-end sensing, estimation and control pipeline for a differential-drive mobile robot in MuJoCo. The robot starts with lateral and heading error, estimates its pose from noisy wheel encoders and intermittent ArUco-camera measurements, and docks at a target pose.

The controller uses the **EKF estimated state only**. MuJoCo ground truth is used only to calculate evaluation metrics.

Two scenarios are evaluated:

- **Nominal docking:** visual docking without an obstacle.
- **Obstacle docking:** the same estimator and controller, with a LiDAR safety state machine that can temporarily override the nominal command.

The docking target is

\[
\mathbf{x}_d =
\begin{bmatrix}
1.20 & 0 & 0
\end{bmatrix}^{T}
\]

for \(x=1.20\text{ m}\), \(y=0\text{ m}\), and \(\theta=0^\circ\).


---

## 1. System architecture

```mermaid
flowchart LR
    A[Wheel Encoders] --> B[EKF Prediction]
    C[Front Camera + ArUco] --> D[Visual Pose]
    D --> E[EKF Update]
    B --> E
    E --> F[Estimated Pose]
    F --> G[LQR Approach]
    F --> H[Final Pose Docking]
    G --> I[Controller Handover]
    H --> I
    J[LiDAR] --> K[Safety State Machine]
    I --> K
    K --> L[Recovery Handover]
    L --> M[Side-Obstacle Assist]
    M --> N[Linear + Angular Command Smoothing]
    N --> O[v, omega]
    O --> P[Differential-Drive Robot / MuJoCo]
    P --> A
    P --> C
    P --> J
    P -. Ground truth for evaluation .-> Q[Metrics and Plots]
    E --> F[Estimated Pose x_hat, y_hat, theta_hat]
    F --> G[LQR Approach Controller]
    F --> H[Final Pose Controller]
    I[LiDAR] --> J[Safety Override]
    G --> J
    H --> J
    J --> K[v, omega]
    K --> L[Differential-Drive Robot / MuJoCo]
    L --> A
    L --> C
    L --> I
    L -. Ground truth only for evaluation .-> M[Metrics and Plots]
```

### Coordinate frames


- **World frame \(\{W\}\):** fixed MuJoCo map frame.
- **Robot/body frame \(\{B\}\):** attached to the Waffle Pi body.
- **Camera frame \(\{C\}\):** attached to the front camera.
- **ArUco marker frame \(\{M\}\):** fixed at the docking station.

The camera detects the known marker and estimates marker-to-camera pose. This measurement is transformed into robot pose in the world frame before the EKF update.

---

## 2. Robot model

The robot state and control input are

```math
\mathbf{x}=\begin{bmatrix}x & y & \theta\end{bmatrix}^{T},
\qquad
\mathbf{u}=\begin{bmatrix}v & \omega\end{bmatrix}^{T}.
```

The unicycle kinematic model is

```math
\dot{x}=v\cos\theta,\qquad
\dot{y}=v\sin\theta,\qquad
\dot{\theta}=\omega.
```

For wheel radius \(r\), wheelbase \(L\), and wheel angular velocities \(\omega_R,\omega_L\),

```math
v=\frac{r}{2}(\omega_R+\omega_L),
\qquad
\omega=\frac{r}{L}(\omega_R-\omega_L).
```

### Final model parameters

| Parameter | Value |
|---|---:|
| Simulation time step, \(\Delta t\) | 0.05 s |
| Wheel radius, \(r\) | 0.033 m |
| Wheelbase, \(L\) | 0.288 m |
| Maximum forward speed | 0.18 m/s |
| Maximum forward acceleration | 0.12 m/s² |
| Maximum angular acceleration during normal control | 1.5 rad/s² |
| Simulation duration | 25 s |

The MuJoCo model uses TurtleBot3 Waffle Pi meshes and dimensions while preserving the project’s planar \(x,y,\theta\) state representation.

---

## 3. Sensing

### Wheel encoders

Wheel angular velocity is corrupted with bias and Gaussian noise:

```math
\omega_m=\omega_{\text{true}}+b+\eta,
\qquad
\eta\sim\mathcal N(0,\sigma^2).
```

| Encoder parameter | Value |
|---|---:|
| Noise standard deviation | 0.15 rad/s |
| Left-wheel bias | 0.00 rad/s |
| Right-wheel bias | 0.02 rad/s |

### Camera and ArUco marker

The front camera detects ArUco marker ID 0. OpenCV `solvePnP` estimates marker pose from the marker corners and camera model.

| Camera parameter | Value |
|---|---:|
| Image size | 640 × 480 |
| Field of view | 90° |
| Marker side length | 0.30 m |
| Camera update interval | every 5 simulation steps |
| Pose measurement standard deviation in \(x\) | 0.01 m |
| Pose measurement standard deviation in \(y\) | 0.01 m |
| Pose measurement standard deviation in heading | 1° |

### LiDAR and obstacle safety

The obstacle scenario uses left, front and right rays. The front ray drives a `NORMAL → AVOIDING → RECOVERY → NORMAL` safety state machine, while the side rays provide continuous corner clearance assistance around the Waffle Pi footprint.

| LiDAR / avoidance parameter | Value |
|---|---:|
| Maximum range | 3.5 m |
| Slow-obstacle distance | 0.75 m |
| Avoidance-entry distance | 0.40 m |
| Avoidance-exit distance | 0.75 m |
| Avoidance turn rate | 0.8 rad/s |
| Recovery speed | 0.10 m/s |
| Recovery duration | 4.5 s |
| Minimum avoidance duration | 0.75 s |
| Side influence distance | 0.55 m |
| Side hard distance | 0.30 m |
| Side maximum turn rate | 0.45 rad/s |

---

## 4. State estimator — Extended Kalman Filter

The EKF fuses noisy wheel-encoder motion with intermittent camera pose measurements.

### Prediction

```math
\hat{\mathbf{x}}_{k+1}^{-}=f(\hat{\mathbf{x}}_k,\mathbf{u}_k)
```

with

```math
f(\mathbf{x},\mathbf{u})=
\begin{bmatrix}
x+v\cos\theta\,\Delta t\\
y+v\sin\theta\,\Delta t\\
\theta+\omega\Delta t
\end{bmatrix}.
```

The process covariance used in the final simulation is

```math
Q=\mathrm{diag}\left(0.002^2,\;0.002^2,\;(0.5^\circ)^2\right).
```

### Camera update

The camera supplies direct pose measurements, so \(H=I\) when the marker is visible.

```math
K=P^-H^T(HP^-H^T+R)^{-1}
```

```math
\hat{\mathbf{x}}=\hat{\mathbf{x}}^-+K(\mathbf{z}-H\hat{\mathbf{x}}^-)
```

with

```math
R=\mathrm{diag}\left(0.01^2,\;0.01^2,\;(1^\circ)^2\right).
```

When the marker leaves the camera view, the EKF continues encoder-based prediction until another visual correction is available.

---

## 5. Controller design

The controller uses only the EKF estimate

```math
\hat{\mathbf{x}}=[\hat{x},\hat{y},\hat{\theta}]^T.
```

### Phase 1 — LQR approach

LQR regulates lateral and heading error before the final docking region.

```math
\mathbf{e}=\begin{bmatrix}e_y & e_\theta\end{bmatrix}^{T}
```

with weights

```math
Q_c=\begin{bmatrix}8&0\\0&4\end{bmatrix},
\qquad
R_c=[1.2].
```

and steering law

```math
\omega=-K\mathbf{e}.
```

| LQR parameter | Value |
|---|---:|
| \(q_y\) | 8.0 |
| \(q_\theta\) | 4.0 |
| \(r_\omega\) | 1.2 |
| Angular-rate limit | 1.5 rad/s |
| Maximum forward speed | 0.18 m/s |

### Phase 2 — final pose docking

When the EKF estimate reaches approximately

```math
\hat{x}\ge1.05\text{ m},
```

the project switches from LQR approach to the final pose controller.

| Docking parameter | Value |
|---|---:|
| Final docking start \(x\) | 1.05 m |
| Docking target \(x\) | 1.30 m |
| Slow-down distance | 0.50 m |
| Stop tolerance | 0.03 m |
| Final docking maximum angular speed | 0.5 rad/s |
| Final heading alignment gain | 0.8 |

### Command smoothing and handovers

To remove abrupt visible motion while preserving safety, the final simulation includes:

- **Recovery-to-normal handover:** 0.60 s blend from recovery speed back to the nominal controller.
- **LQR-to-final-dock handover:** 0.60 s blend between the last LQR command and the pose-controller command.
- **Linear acceleration limiter:** limits increases in forward speed to 0.12 m/s²; emergency deceleration remains unrestricted.
- **Angular slew-rate limiter:** limits angular acceleration to 1.5 rad/s² during `NORMAL` operation; direct avoidance turns are not rate-limited.

The safety layer is applied after the nominal controller, so obstacle avoidance retains priority over trajectory tracking and docking.

---

## 6. Final obstacle-scenario results

The final frozen obstacle simulation produced:

| Metric | Final result |
|---|---:|
| Camera updates | 21 |
| Final localisation error | 0.02079 m |
| Position RMSE | 0.03733 m |
| Final lateral error | −0.00167 m |
| Final heading error | 1.372° |
| Final X docking error | −0.00164 m |
| Final Y docking error | −0.00167 m |
| Final planar docking error | **0.00233 m** |
| Obstacle avoidance | Successful |

The final planar docking error is approximately **2.3 mm**. The larger whole-run heading RMSE is expected because it includes the deliberate avoidance turn and recovery manoeuvre rather than only the final docking phase.

The final state sequence is:

```text
LQR_APPROACH
→ NORMAL
→ AVOIDING
→ RECOVERY
→ NORMAL
→ FINAL_DOCK
```

### Final result figures

![Obstacle trajectory](results/obstacle/trajectory.png)

![Control commands](results/obstacle/control_commands.png)

![LiDAR clearance](results/obstacle/lidar_clearance.png)

![ArUco update events](results/obstacle/aruco_updates.png)

![Obstacle lateral error tracking](results/obstacle/lateral_error.png)

![Obstacle heading error tracking](results/obstacle/heading_error.png)

The simulation also saves overview, front-camera, bird’s-eye, top-view and four-view-grid videos under `results/obstacle/` when generated locally.

---

## 7. Installation

Tested development environment:

- Python 3.11.9
- Git
- VS Code
- MuJoCo
- NumPy
- SciPy
- Matplotlib
- OpenCV / ArUco

From a fresh clone:

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd 41014_Final_Project
```

The main frames are:

- **World frame \(\{W\}\):** fixed MuJoCo map frame.
- **Robot/body frame \(\{B\}\):** attached to the differential-drive robot.
- **Camera frame \(\{C\}\):** attached to the front camera.
- **ArUco marker frame \(\{M\}\):** fixed at the docking station.

The camera detects the known marker and estimates marker-to-camera pose. This is transformed into a robot pose in the world frame before the EKF measurement update.

---

## 2. Robot model

The robot state and control input are

\[
\mathbf{x}=
\begin{bmatrix}
x & y & \theta
\end{bmatrix}^{T},
\qquad
\mathbf{u}=
\begin{bmatrix}
v & \omega
\end{bmatrix}^{T}.
\]

The unicycle kinematic model is

\[
\dot{x}=v\cos\theta,\qquad
\dot{y}=v\sin\theta,\qquad
\dot{\theta}=\omega.
\]

For a differential-drive robot with wheel radius \(r\), wheelbase \(L\), and wheel angular velocities \(\omega_R,\omega_L\),

\[
v=\frac{r}{2}(\omega_R+\omega_L),
\qquad
\omega=\frac{r}{L}(\omega_R-\omega_L).
\]

### Model parameters

| Parameter | Value |
|---|---:|
| Simulation time step, \(\Delta t\) | 0.05 s |
| Wheel radius, \(r\) | 0.033 m |
| Wheelbase, \(L\) | 0.16 m |
| Maximum forward speed | 0.18 m/s |
| Simulation duration | 20 s |

### Controllability used for LQR approach

The LQR approach controller uses the small-error states

\[
\mathbf{e}=
\begin{bmatrix}
e_y & e_\theta
\end{bmatrix}^{T}.
\]

For approximately constant forward speed \(v\),

\[
\dot{\mathbf{e}}
=
\begin{bmatrix}
0 & v\\
0 & 0
\end{bmatrix}\mathbf{e}
+
\begin{bmatrix}
0\\
1
\end{bmatrix}\omega.
\]

The controllability matrix is

\[
\mathcal{C}=[B\ AB]
=
\begin{bmatrix}
0 & v\\
1 & 0
\end{bmatrix}.
\]

For \(v\neq0\), \(\operatorname{rank}(\mathcal C)=2\), so the lateral-heading error model is controllable.

---

## 3. Sensing

### Wheel encoders

Wheel angular velocity is deliberately corrupted to prevent unrealistically perfect odometry:

\[
\omega_m=\omega_{\text{true}}+b+\eta,
\qquad
\eta\sim\mathcal N(0,\sigma^2).
\]

| Encoder parameter | Value |
|---|---:|
| Noise standard deviation | 0.15 rad/s |
| Left-wheel bias | 0.00 rad/s |
| Right-wheel bias | 0.02 rad/s |

This causes odometry drift and gives the EKF a meaningful estimation problem.

### Camera and ArUco marker

The front camera detects ArUco marker ID 0. OpenCV `solvePnP` estimates the marker pose using the detected corners, marker geometry, camera matrix and zero simulated distortion.

| Camera parameter | Value |
|---|---:|
| Image size | 640 × 480 |
| Field of view | 90° |
| ArUco marker side length | 0.30 m |
| Camera update interval | every 5 simulation steps |
| Pose noise in \(x\) | 0.01 m |
| Pose noise in \(y\) | 0.01 m |
| Pose noise in heading | 1° |

### LiDAR

The obstacle scenario adds left, front and right range rays. LiDAR does not replace the nominal controller; it is used as a supervisory safety layer.

| LiDAR / avoidance parameter | Value |
|---|---:|
| Maximum range | 3.5 m |
| Slow-obstacle distance | 0.65 m |
| Avoidance-entry distance | 0.30 m |
| Avoidance-exit distance | 0.60 m |
| Avoidance turn rate | 0.8 rad/s |
| Recovery speed | 0.10 m/s |

---

## 4. State estimator — Extended Kalman Filter

The EKF fuses noisy encoder motion with intermittent camera pose measurements.

### Prediction

\[
\hat{\mathbf{x}}_{k+1}^{-}
=
f(\hat{\mathbf{x}}_k,\mathbf{u}_k)
\]

with

\[
f(\mathbf{x},\mathbf{u})=
\begin{bmatrix}
x+v\cos\theta\,\Delta t\\
y+v\sin\theta\,\Delta t\\
\theta+\omega\Delta t
\end{bmatrix}.
\]

The state Jacobian is

\[
F_k=
\begin{bmatrix}
1 & 0 & -v\sin\theta\,\Delta t\\
0 & 1 &  v\cos\theta\,\Delta t\\
0 & 0 & 1
\end{bmatrix},
\]

and covariance prediction is

\[
P_{k+1}^{-}=F_kP_kF_k^T+Q.
\]

Current process covariance:

\[
Q=
\operatorname{diag}
\left(
0.002^2,\;
0.002^2,\;
(0.5^\circ)^2
\right).
\]

### Camera update

The camera supplies a pose measurement

\[
\mathbf{z}=
\begin{bmatrix}
x_c & y_c & \theta_c
\end{bmatrix}^{T}
\]

with direct pose measurement model \(H=I\).

\[
K=P^-H^T(HP^-H^T+R)^{-1}
\]

\[
\hat{\mathbf{x}}
=
\hat{\mathbf{x}}^-+
K(\mathbf{z}-H\hat{\mathbf{x}}^-).
\]

The measurement covariance is

\[
R=
\operatorname{diag}
\left(
0.01^2,\;
0.01^2,\;
(1^\circ)^2
\right).
\]

Because \(H=I\) when the marker is visible, all three pose states are directly observed by the visual pose update. When the marker is not visible, the EKF continues encoder-based prediction and uncertainty grows until visual correction becomes available again.

---

## 5. Controller design

The controller uses the EKF estimate

\[
\hat{\mathbf{x}}=
[\hat{x},\hat{y},\hat{\theta}]^T
\]

and never the simulator ground truth.

### Phase 1 — LQR approach

Before the final docking region, LQR corrects lateral and heading error.

\[
J=
\int
\left(
\mathbf{e}^TQ_c\mathbf{e}
+
\omega^TR_c\omega
\right)dt
\]

with

\[
Q_c=
\begin{bmatrix}
8&0\\
0&4
\end{bmatrix},
\qquad
R_c=[1.2].
\]

The steering law is

\[
\omega=-K\mathbf{e}.
\]

| LQR parameter | Value |
|---|---:|
| \(q_y\) | 8.0 |
| \(q_\theta\) | 4.0 |
| \(r_\omega\) | 1.2 |
| Angular-rate limit | 1.5 rad/s |
| Maximum forward speed | 0.18 m/s |

The larger lateral-error weight makes centreline convergence the main objective while \(R_c\) prevents unnecessarily aggressive steering.

### Phase 2 — final pose docking

At approximately

\[
\hat{x}\ge0.95\text{ m},
\]

the controller switches to a final pose controller. It regulates remaining position and heading error and slows the robot near the target.

| Docking parameter | Value |
|---|---:|
| Final docking start \(x\) | 0.95 m |
| Slow-down distance | 0.50 m |
| Stop tolerance | 0.03 m |

The obstacle scenario applies the LiDAR state machine after the nominal LQR / pose-control command, so collision avoidance can temporarily override the nominal command.

---

## 6. Quantitative results

The comparison is generated directly from the submitted code using:

```powershell
python -m scripts.compare_final_results
```

The script reads the nominal and obstacle metrics and writes:

```text
results/comparison/comparison_metrics.csv
results/comparison/final_results_summary.txt
results/comparison/camera_update_comparison.png
results/comparison/docking_error_comparison.png
```

### Latest simulation results

| Metric | Nominal | Obstacle |
|---|---:|---:|
| Camera updates | 80 | 44 |
| Final localisation error | 0.01124 m | 0.00748 m |
| Position RMSE | 0.03069 m | 0.05590 m |
| Final planar docking error | 0.01119 m | 0.01037 m |
| Final heading error | −2.339° | 2.597° |
| Obstacle avoidance | N/A | Successful |

The obstacle trial produced fewer camera updates and a higher position RMSE because obstacle avoidance changes the path and temporarily reduces the quality/frequency of visual correction. However, the final localisation error and final planar docking error remained close to 1 cm, showing that the estimator and final docking controller recovered after avoidance.

The obstacle safety state machine completed the sequence:

```text
NORMAL → AVOIDING → RECOVERY → NORMAL → FINAL_DOCK
```

### Comparison figures

![Docking error comparison](results/comparison/docking_error_comparison.png)

![Camera update comparison](results/comparison/camera_update_comparison.png)

---

## 7. Installation

Tested development environment:

- Python 3.11.9
- Git
- VS Code
- MuJoCo
- NumPy
- SciPy
- Matplotlib
- OpenCV / ArUco

From a fresh clone:

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd 41014_Final_Project
```

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Run instructions

Run commands from the repository root so the `src` package resolves correctly.

### Obstacle docking simulation

```powershell

Run all commands from the repository root so the `src` package is resolved correctly.

### Nominal docking simulation

```powershell
python -m scripts.simulate_mujoco_nominal
```

Outputs are saved under:

```text
results/nominal/
```

### Obstacle docking simulation

```powershell

python -m scripts.simulate_mujoco_obstacle_trial
```

Outputs are saved under:

```text
results/obstacle/
```


Typical outputs include:

```text
trajectory_history.csv
metrics.csv
trajectory.png
lateral_error.png
heading_error.png
control_commands.png
lidar_clearance.png
aruco_updates.png
latest_aruco_detection.png
simulation.mp4
robot_camera.mp4
birdseye_perspective.mp4
top_view.mp4
four_view_grid.mp4
```

### Nominal docking simulation

```powershell
python -m scripts.simulate_mujoco_nominal
```

Outputs are saved under `results/nominal/`.

If final nominal-versus-obstacle comparison numbers are required, rerun both scenarios with the frozen parameter set before regenerating the comparison report so both datasets use the same robot geometry and controller configuration.

---

## 9. Repository structure

```text
41014_Final_Project/
├── assets/                 # MuJoCo worlds, Waffle Pi meshes and simulation assets
├── src/                    # Model, sensors, EKF, controller, LiDAR and vision modules
├── scripts/                # Simulation and analysis scripts
├── results/
│   ├── nominal/
│   ├── obstacle/
│   └── comparison/
├── requirements.txt
└── README.md
```

---

## 10. Contribution statement

| Team member | Contribution |
|---|---|
| Gil Martin A. Guape | Developed and integrated the differential-drive simulation, EKF estimator, LQR and final-pose control workflow, LiDAR obstacle-avoidance logic, TurtleBot3 Waffle Pi MuJoCo environment, command-smoothing refinements, simulation testing, quantitative evaluation, visual outputs and project documentation. |

The complete sensing–estimation–control pipeline is understood as an integrated system for the individual viva voce.

---

## 11. Generative AI declaration

Generative AI was used as a support tool during project development. Final engineering decisions, code retained in the repository, executed results and explanations remain the responsibility of the student.

| Tool | Purpose | Verification |
|---|---|---|
| OpenAI ChatGPT | Coding guidance and debugging support; explanation of EKF/LQR and sensing concepts; assistance refining the MuJoCo simulation; README/documentation drafting; viva preparation | Suggestions were checked against project requirements, source code and executed simulation outputs before being retained. |

AI-generated suggestions were not treated as simulation evidence. Quantitative values reported on this page come from the executed project scripts and saved result files.

---

## 12. Limitations and hardware transition

The final MuJoCo implementation demonstrates the complete simulation pipeline, but hardware testing can introduce wheel slip, motor deadband, camera-calibration error, sensor latency, frame misalignment and different noise characteristics.

Two simulation-specific details require particular attention before hardware deployment:

1. The current MuJoCo safety implementation can identify named geometries such as `obstacle_1` and the docking panel. A physical robot cannot use simulator geometry names, so hardware obstacle classification should instead rely on LiDAR range sectors and geometric thresholds.
2. The simulation uses known planar robot pose to originate the LiDAR rays. On hardware, LiDAR measurements are naturally expressed in the sensor frame and must be transformed using the calibrated sensor-to-body transform.

The intended hardware architecture remains:

```text
Sensors → EKF → Estimated State → Nominal Controller → Safety Layer → Robot
```

### Compare the final results

```powershell
python -m scripts.compare_final_results
```

Outputs are saved under:

```text
results/comparison/
```

The main simulation code records trajectories, localisation errors, lateral/heading errors, docking errors, camera updates and video outputs.

---

## 9. Repository structure

```text
41014_Final_Project/
├── assets/                 # MuJoCo worlds and simulation assets
├── src/                    # Robot model, sensors, EKF, controller and LiDAR modules
├── scripts/                # Nominal, obstacle and comparison scripts
├── results/
│   ├── nominal/
│   ├── obstacle/
│   └── comparison/
├── requirements.txt
└── README.md
```

---

## 10. Contribution statement

> **Complete this table with the agreed team contribution record before final submission.**  
> Contributions should describe what each person actually implemented, tested, analysed or documented.

| Team member | Contribution |
|---|---|
| Gil Martin A. Guape | [Replace with agreed contribution statement — e.g. modelling, EKF/controller integration, simulation testing, quantitative comparison, documentation.] |
| [Member 2] | [Contribution] |
| [Member 3] | [Contribution] |
| [Member 4, if applicable] | [Contribution] |

All team members are expected to understand the complete sensing-estimation-control pipeline for the individual viva voce.

---

## 11. Generative AI declaration

Generative AI was used as a support tool during project development. The submitted design decisions, code, results and explanations remain the responsibility of the project team.

| Tool | Purpose | Verification |
|---|---|---|
| OpenAI ChatGPT | Coding guidance and debugging support; explanation of EKF/LQR concepts; README/documentation drafting; preparation for simulation review and viva questions | Generated suggestions were checked against the lecture material, project code and executed simulation outputs before being retained. |
| [Add any other GenAI tool actually used] | [Purpose] | [How output was checked] |

AI-generated suggestions were not treated as simulation evidence. Quantitative values reported above come from the project scripts and saved result files.

---

## 12. Key limitations and next step

The current results demonstrate the complete pipeline in MuJoCo, but simulation does not reproduce all hardware effects. Real testing may require retuning because of wheel slip, motor deadband, camera calibration error, sensor latency, frame alignment and different measurement noise.

The next stage is staged hardware validation while keeping the same architecture:

```text
Sensors → EKF → Estimated State → Controller → Robot
```

A quantitative simulation-versus-hardware comparison will be used to evaluate how the system changes on the physical platform.

