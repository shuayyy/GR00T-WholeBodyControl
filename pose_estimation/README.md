# G1 Object Pose Estimation

This directory estimates the 6D pose of one known object from the Unitree G1
head-mounted Intel RealSense D435I camera and can publish the result to ROS 2
TF.

## Data flow

```text
G1 D435I RGB-D stream
        |
        | ZMQ tcp://192.168.123.164:5555
        v
detector.py
  GroundingDINO -> FastSAM -> FoundationPose
        |
        | 4x4 object-to-camera pose over ZMQ tcp://127.0.0.1:5561
        v
publish_object_tf.py
        |
        v
ROS 2 TF: d435_link -> detected_object
```

`detector.py` detects and registers the object on the first valid frame. It
then tracks the object on later frames for lower latency.

## Requirements

- The G1 camera server must provide aligned raw color and depth frames.
- The host needs the `foundation_pose` Conda environment.
- The external FoundationPose repository, models, and weights must be
  available. The current dependency root is configured in `detector.py` as:

  ```text
  /home/athena/Camera/pose_estimation
  ```

- ROS 2, `rclpy`, `tf2_ros`, and the ROS message packages are required to
  publish TF.

The tested camera configuration is:

- Camera: Intel RealSense D435I
- Serial: `344522070363`
- Resolution: `640x480`
- Frame rate: `30 FPS`
- Depth scale: `0.001` metres per raw depth unit

The camera intrinsics and depth scale are specific to this camera and
resolution. Recalibrate or re-query them if the hardware or stream settings
change.

## Start the camera stream

On the robot:

```bash
ssh unitree@192.168.123.164
cd /home/unitree/image_server
/usr/bin/python3 image_server.py
```

The robot-side server must align depth to the color stream before sending the
frames.

## Run object detection and tracking

On the host:

```bash
cd /home/athena/GR00T-WholeBodyControl
conda activate foundation_pose
python pose_estimation/detector.py
```

The default object model is `trash_truck`. To select another model and text
prompt:

```bash
python pose_estimation/detector.py OBJECT_NAME \
  --text-prompt "description of the object"
```

The object model must exist at:

```text
<POSE_ESTIMATION_ROOT>/models/OBJECT_NAME/textured.obj
```

Controls:

- `r`: discard the tracked pose and register the object again
- `q`: quit

## Publish the object pose to ROS 2 TF

In a ROS-enabled terminal:

```bash
python pose_estimation/publish_object_tf.py
```

The default transform is:

```text
d435_link -> detected_object
```

Custom names and endpoints can be supplied when needed:

```bash
python pose_estimation/publish_object_tf.py \
  --pose-endpoint tcp://127.0.0.1:5561 \
  --parent-frame d435_link \
  --object-frame detected_object
```

To connect this camera-relative object transform to the complete robot TF
tree, run the joint-state bridge in a ROS-enabled terminal:

```bash
PYTHONPATH=. python pose_estimation/collect_robot_transform.py
```

It republishes the measured G1 state on `/joint_states` and runs
`robot_state_publisher` with the G1 URDF, so TF can compose

```text
pelvis -> torso_link -> d435_link -> detected_object
```

`d435_link` is a fixed joint off `torso_link` in
`decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf`
(47.6 degree downward pitch). To capture one transform for inspection:

```bash
python pose_estimation/save_robot_transform.py --parent pelvis --child d435_link
```

## Frame validation

The published TF was validated against known real-world object positions
(in front of, left and right of, and above and below the camera). The
FoundationPose optical-frame axes agree with `d435_link`, so
`publish_object_tf.py` broadcasts the 4x4 pose unchanged, with no additional
frame conversion.

Repeat this check if the camera, its mounting, or the URDF `d435_joint`
origin changes.

## Camera utilities

- `camera_scripts/receive_video.py`: view the streamed raw color and depth.
- `camera_scripts/capture_image.py`: capture RGB-D samples from known cameras.
- `camera_scripts/check_all_cameras.py`: inspect all RealSense cameras through
  image capture or an HTTP video page.
- `camera_scripts/kill_video_processes.py`: robot-side troubleshooting utility
  for processes holding `/dev/video*` devices.
- `camera_scripts/notes.md`: explains raw and visualized depth images.

## Repository contents

- `detector.py`: detection, segmentation, registration, tracking, and ZMQ pose
  output.
- `estimater.py`: FoundationPose estimator implementation used by
  `detector.py`.
- `publish_object_tf.py`: converts the streamed 4x4 pose to a ROS 2 transform.
- `collect_robot_transform.py`: bridges the measured G1 state to
  `/joint_states` and runs `robot_state_publisher` for the full TF tree.
- `save_robot_transform.py`: saves one live TF transform to JSON.
- `g1_head/`: generated camera captures and pose results; ignored by Git.

`estimater.py` retains its upstream NVIDIA license notice. Confirm that its
distribution is permitted before publishing this repository or contributing
the file to another project.

## Current limitations

- Only one object is tracked at a time.
- Nothing yet consumes the published `detected_object` transform; it is not
  wired into the planner.
- Camera intrinsics, depth scale, network address, and external dependency
  path are currently machine-specific.
- The robot-side `image_server.py` (including its depth-alignment patch)
  exists only on the robot; there is no copy in this repository.
- Models and weights are not included in this repository.
- Hardware and ROS integration require the configured robot environments and
  are not covered by automated tests.
