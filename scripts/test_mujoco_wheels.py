from pathlib import Path
import time

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


left_motor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "left_wheel_motor"
)

right_motor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "right_wheel_motor"
)


print("Left motor ID:", left_motor_id)
print("Right motor ID:", right_motor_id)


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    start_time = time.time()

    while viewer.is_running():

        # Command both wheels forward
        data.ctrl[left_motor_id] = 1.0
        data.ctrl[right_motor_id] = 1.0

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )

        if time.time() - start_time > 8.0:
            break