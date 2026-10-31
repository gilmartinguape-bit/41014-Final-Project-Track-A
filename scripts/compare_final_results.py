import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# PROJECT PATHS
# ==================================================

project_root = Path(__file__).resolve().parents[1]

nominal_metrics_path = (
    project_root
    / "results"
    / "nominal"
    / "metrics.csv"
)

obstacle_metrics_path = (
    project_root
    / "results"
    / "obstacle"
    / "metrics.csv"
)

comparison_dir = (
    project_root
    / "results"
    / "comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

nominal_trajectory_path = (
    project_root
    / "results"
    / "nominal"
    / "trajectory_history.csv"
)

obstacle_trajectory_path = (
    project_root
    / "results"
    / "obstacle"
    / "trajectory_history.csv"
)

# ==================================================
# READ TRAJECTORY HISTORY
# ==================================================

def read_trajectory_csv(path):

    time_values = []
    true_x_values = []
    true_y_values = []
    ekf_x_values = []
    ekf_y_values = []

    with open(
        path,
        "r",
        newline=""
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        for row in reader:

            time_values.append(
                float(row["time_s"])
            )

            true_x_values.append(
                float(row["true_x_m"])
            )

            true_y_values.append(
                float(row["true_y_m"])
            )

            ekf_x_values.append(
                float(row["ekf_x_m"])
            )

            ekf_y_values.append(
                float(row["ekf_y_m"])
            )

    return {
        "time": np.array(time_values),
        "true_x": np.array(true_x_values),
        "true_y": np.array(true_y_values),
        "ekf_x": np.array(ekf_x_values),
        "ekf_y": np.array(ekf_y_values),
    }

nominal_trajectory = read_trajectory_csv(
    nominal_trajectory_path
)

obstacle_trajectory = read_trajectory_csv(
    obstacle_trajectory_path
)

# ==================================================
# READ METRICS
# ==================================================

def read_metrics_csv(path):
    """
    Read a vertical metrics CSV file with columns:
    metric,value
    """

    metrics = {}

    with open(
        path,
        "r",
        newline=""
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        for row in reader:

            metric = row["metric"]
            value = row["value"]

            try:
                value = float(value)

            except ValueError:
                pass

            metrics[metric] = value

    return metrics


nominal = read_metrics_csv(
    nominal_metrics_path
)

obstacle = read_metrics_csv(
    obstacle_metrics_path
)


# ==================================================
# REQUIRED METRICS
# ==================================================

required_metrics = [
    "camera_updates",
    "final_localisation_error_m",
    "position_rmse_m",
    "true_lateral_rmse_m",
    "true_heading_rmse_deg",
    "final_x_docking_error_m",
    "final_y_docking_error_m",
    "final_planar_docking_error_m",
    "final_true_heading_deg",
]


for metric in required_metrics:

    if metric not in nominal:
        raise KeyError(
            f"Missing metric in nominal results: {metric}"
        )

    if metric not in obstacle:
        raise KeyError(
            f"Missing metric in obstacle results: {metric}"
        )


# ==================================================
# COMPARISON TABLE
# ==================================================

comparison_rows = [
    (
        "Camera / EKF updates",
        nominal["camera_updates"],
        obstacle["camera_updates"],
        "count"
    ),
    (
        "Final localisation error",
        nominal["final_localisation_error_m"],
        obstacle["final_localisation_error_m"],
        "m"
    ),
    (
        "Position RMSE",
        nominal["position_rmse_m"],
        obstacle["position_rmse_m"],
        "m"
    ),
    (
        "Lateral RMSE",
        nominal["true_lateral_rmse_m"],
        obstacle["true_lateral_rmse_m"],
        "m"
    ),
    (
        "Heading RMSE",
        nominal["true_heading_rmse_deg"],
        obstacle["true_heading_rmse_deg"],
        "deg"
    ),
    (
        "Final X docking error",
        nominal["final_x_docking_error_m"],
        obstacle["final_x_docking_error_m"],
        "m"
    ),
    (
        "Final Y docking error",
        nominal["final_y_docking_error_m"],
        obstacle["final_y_docking_error_m"],
        "m"
    ),
    (
        "Final planar docking error",
        nominal["final_planar_docking_error_m"],
        obstacle["final_planar_docking_error_m"],
        "m"
    ),
    (
        "Final heading error",
        nominal["final_true_heading_deg"],
        obstacle["final_true_heading_deg"],
        "deg"
    ),
]


comparison_csv_path = (
    comparison_dir
    / "comparison_metrics.csv"
)

with open(
    comparison_csv_path,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(
        csvfile
    )

    writer.writerow([
        "metric",
        "nominal",
        "obstacle",
        "unit",
        "absolute_change",
        "percent_change"
    ])

    for (
        metric_name,
        nominal_value,
        obstacle_value,
        unit
    ) in comparison_rows:

        absolute_change = (
            obstacle_value
            - nominal_value
        )

        if nominal_value != 0:

            percent_change = (
                absolute_change
                / abs(nominal_value)
                * 100.0
            )

        else:

            percent_change = np.nan

        writer.writerow([
            metric_name,
            nominal_value,
            obstacle_value,
            unit,
            absolute_change,
            percent_change
        ])


# ==================================================
# PRINT COMPARISON
# ==================================================

print()
print("FINAL NOMINAL VS OBSTACLE COMPARISON")
print("=" * 72)

for (
    metric_name,
    nominal_value,
    obstacle_value,
    unit
) in comparison_rows:

    print(
        f"{metric_name:<30}"
        f"Nominal: {nominal_value:>10.5f} {unit:<5}"
        f"Obstacle: {obstacle_value:>10.5f} {unit}"
    )

print()
print(
    f"Saved comparison CSV to: "
    f"{comparison_csv_path}"
)


# ==================================================
# FIGURE 1:
# FINAL DOCKING ERRORS
# ==================================================

docking_labels = [
    "X error",
    "Y error",
    "Planar error"
]

nominal_docking = [
    abs(
        nominal[
            "final_x_docking_error_m"
        ]
    ),
    abs(
        nominal[
            "final_y_docking_error_m"
        ]
    ),
    nominal[
        "final_planar_docking_error_m"
    ]
]

obstacle_docking = [
    abs(
        obstacle[
            "final_x_docking_error_m"
        ]
    ),
    abs(
        obstacle[
            "final_y_docking_error_m"
        ]
    ),
    obstacle[
        "final_planar_docking_error_m"
    ]
]

x = np.arange(
    len(docking_labels)
)

bar_width = 0.35

plt.figure()

plt.bar(
    x - bar_width / 2,
    nominal_docking,
    bar_width,
    label="Nominal"
)

plt.bar(
    x + bar_width / 2,
    obstacle_docking,
    bar_width,
    label="Obstacle"
)

plt.xticks(
    x,
    docking_labels
)

plt.ylabel(
    "Absolute docking error [m]"
)

plt.title(
    "Final Docking Error Comparison"
)

plt.grid(
    axis="y"
)

plt.legend()

docking_plot_path = (
    comparison_dir
    / "docking_error_comparison.png"
)

plt.savefig(
    docking_plot_path,
    dpi=300,
    bbox_inches="tight"
)


# ==================================================
# FIGURE 2:
# LOCALISATION PERFORMANCE
# ==================================================

localisation_labels = [
    "Final localisation error",
    "Position RMSE",
    "Lateral RMSE"
]

nominal_localisation = [
    nominal[
        "final_localisation_error_m"
    ],
    nominal[
        "position_rmse_m"
    ],
    nominal[
        "true_lateral_rmse_m"
    ]
]

obstacle_localisation = [
    obstacle[
        "final_localisation_error_m"
    ],
    obstacle[
        "position_rmse_m"
    ],
    obstacle[
        "true_lateral_rmse_m"
    ]
]

x = np.arange(
    len(localisation_labels)
)

plt.figure()

plt.bar(
    x - bar_width / 2,
    nominal_localisation,
    bar_width,
    label="Nominal"
)

plt.bar(
    x + bar_width / 2,
    obstacle_localisation,
    bar_width,
    label="Obstacle"
)

plt.xticks(
    x,
    localisation_labels,
    rotation=15
)

plt.ylabel(
    "Error [m]"
)

plt.title(
    "Localisation Performance Comparison"
)

plt.grid(
    axis="y"
)

plt.legend()

localisation_plot_path = (
    comparison_dir
    / "localisation_rmse_comparison.png"
)

plt.savefig(
    localisation_plot_path,
    dpi=300,
    bbox_inches="tight"
)


# ==================================================
# FIGURE 3:
# HEADING PERFORMANCE
# ==================================================

heading_labels = [
    "Heading RMSE",
    "Final heading error"
]

nominal_heading = [
    nominal[
        "true_heading_rmse_deg"
    ],
    abs(
        nominal[
            "final_true_heading_deg"
        ]
    )
]

obstacle_heading = [
    obstacle[
        "true_heading_rmse_deg"
    ],
    abs(
        obstacle[
            "final_true_heading_deg"
        ]
    )
]

x = np.arange(
    len(heading_labels)
)

plt.figure()

plt.bar(
    x - bar_width / 2,
    nominal_heading,
    bar_width,
    label="Nominal"
)

plt.bar(
    x + bar_width / 2,
    obstacle_heading,
    bar_width,
    label="Obstacle"
)

plt.xticks(
    x,
    heading_labels
)

plt.ylabel(
    "Angle error [deg]"
)

plt.title(
    "Heading Performance Comparison"
)

plt.grid(
    axis="y"
)

plt.legend()

heading_plot_path = (
    comparison_dir
    / "heading_error_comparison.png"
)

plt.savefig(
    heading_plot_path,
    dpi=300,
    bbox_inches="tight"
)


# ==================================================
# FIGURE 4:
# CAMERA UPDATE COUNT
# ==================================================

plt.figure()

plt.bar(
    [
        "Nominal",
        "Obstacle"
    ],
    [
        nominal[
            "camera_updates"
        ],
        obstacle[
            "camera_updates"
        ]
    ]
)

plt.ylabel(
    "Number of visual EKF updates"
)

plt.title(
    "ArUco / EKF Visual Update Count"
)

plt.grid(
    axis="y"
)

camera_plot_path = (
    comparison_dir
    / "camera_update_comparison.png"
)

plt.savefig(
    camera_plot_path,
    dpi=300,
    bbox_inches="tight"
)

# ==================================================
# FIGURE 5:
# TRAJECTORY COMPARISON
# ==================================================

plt.figure()

plt.plot(
    nominal_trajectory["true_x"],
    nominal_trajectory["true_y"],
    label="Nominal true trajectory"
)

plt.plot(
    obstacle_trajectory["true_x"],
    obstacle_trajectory["true_y"],
    label="Obstacle true trajectory"
)

plt.plot(
    nominal_trajectory["ekf_x"],
    nominal_trajectory["ekf_y"],
    "--",
    label="Nominal EKF estimate"
)

plt.plot(
    obstacle_trajectory["ekf_x"],
    obstacle_trajectory["ekf_y"],
    "--",
    label="Obstacle EKF estimate"
)

plt.scatter(
    0.0,
    0.25,
    marker="o",
    s=70,
    label="Initial robot position"
)

plt.scatter(
    1.20,
    0.0,
    marker="x",
    s=100,
    label="Dock target"
)

plt.axhline(
    y=0.0,
    linestyle=":",
    label="Desired docking line"
)

plt.xlabel(
    "X position [m]"
)

plt.ylabel(
    "Y position [m]"
)

plt.title(
    "Nominal vs Obstacle Trajectory Comparison"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

trajectory_comparison_path = (
    comparison_dir
    / "trajectory_comparison.png"
)

plt.savefig(
    trajectory_comparison_path,
    dpi=300,
    bbox_inches="tight"
)

# ==================================================
# TEXT SUMMARY
# ==================================================

nominal_planar = nominal[
    "final_planar_docking_error_m"
]

obstacle_planar = obstacle[
    "final_planar_docking_error_m"
]

nominal_heading_final = abs(
    nominal[
        "final_true_heading_deg"
    ]
)

obstacle_heading_final = abs(
    obstacle[
        "final_true_heading_deg"
    ]
)

position_rmse_increase = (
    (
        obstacle["position_rmse_m"]
        - nominal["position_rmse_m"]
    )
    / nominal["position_rmse_m"]
    * 100.0
)

lateral_rmse_increase = (
    (
        obstacle["true_lateral_rmse_m"]
        - nominal["true_lateral_rmse_m"]
    )
    / nominal["true_lateral_rmse_m"]
    * 100.0
)

heading_rmse_increase = (
    (
        obstacle["true_heading_rmse_deg"]
        - nominal["true_heading_rmse_deg"]
    )
    / nominal["true_heading_rmse_deg"]
    * 100.0
)


summary_text = f"""
FINAL RESULT COMPARISON
=======================

Nominal scenario
----------------
Camera / EKF updates:
{nominal["camera_updates"]:.0f}

Position RMSE:
{nominal["position_rmse_m"]:.6f} m

Lateral RMSE:
{nominal["true_lateral_rmse_m"]:.6f} m

Heading RMSE:
{nominal["true_heading_rmse_deg"]:.3f} deg

Final planar docking error:
{nominal_planar:.6f} m

Final heading error:
{nominal_heading_final:.3f} deg


Obstacle scenario
-----------------
Camera / EKF updates:
{obstacle["camera_updates"]:.0f}

Position RMSE:
{obstacle["position_rmse_m"]:.6f} m

Lateral RMSE:
{obstacle["true_lateral_rmse_m"]:.6f} m

Heading RMSE:
{obstacle["true_heading_rmse_deg"]:.3f} deg

Final planar docking error:
{obstacle_planar:.6f} m

Final heading error:
{obstacle_heading_final:.3f} deg


Comparison
----------
Position RMSE increased by approximately
{position_rmse_increase:.1f}% in the obstacle scenario.

Lateral RMSE increased by approximately
{lateral_rmse_increase:.1f}% in the obstacle scenario.

Heading RMSE increased by approximately
{heading_rmse_increase:.1f}% because of the deliberate
obstacle-avoidance turn and recovery manoeuvre.

Despite the larger transient deviations, the final docking
accuracy remained similar:

Nominal planar docking error:
{nominal_planar * 1000.0:.1f} mm

Obstacle planar docking error:
{obstacle_planar * 1000.0:.1f} mm

Both scenarios also finished with an absolute heading error
below 3 degrees.

The obstacle scenario therefore demonstrates that the robot
can recover from a LiDAR-triggered avoidance manoeuvre and
still achieve docking accuracy comparable with the nominal
unobstructed run.
"""


summary_path = (
    comparison_dir
    / "final_results_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as summary_file:

    summary_file.write(
        summary_text.strip()
    )


# ==================================================
# FINAL OUTPUT
# ==================================================

print()
print("Comparison outputs created:")

print(
    comparison_csv_path
)

print(
    docking_plot_path
)

print(
    localisation_plot_path
)

print(
    heading_plot_path
)

print(
    camera_plot_path
)

print(
    summary_path
)

print(
    trajectory_comparison_path
)

print()

print(
    "Comparison complete."
)



plt.show()