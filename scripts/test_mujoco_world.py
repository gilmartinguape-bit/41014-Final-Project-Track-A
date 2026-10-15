from pathlib import Path

import mujoco
import mujoco.viewer


project_root = Path(__file__).resolve().parents[1]

xml_path = (
    project_root
    / "assets"
    / "empty_world.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

data = mujoco.MjData(
    model
)


print("MuJoCo model loaded successfully.")

print(
    "Physics timestep:",
    model.opt.timestep,
    "s"
)


mujoco.viewer.launch(
    model,
    data
)