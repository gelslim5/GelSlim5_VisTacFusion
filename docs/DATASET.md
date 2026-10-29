# GelSlim 5.0 visuo-tactile dataset

Twelve 3D-printed test objects (82 × 82 mm plates with raised line patterns and a rod). Every
sample pairs a **tactile** image (UV on) and a **visual** image (UV off, through the
membrane) from the same fixed fingertip camera, with ground-truth contact depth and planar
object pose. The real and the simulated parts share one layout, so the same loader
(`vistacfusion/data/dataset.py`) reads both.

```
dataset/
  meshes/<object>.obj                           object mesh, millimetres, +z up
  real/<object>/session_000/
      session.json                              object name, base rotation, press-cell list
      sensor_0000/
          samples/NNNN.png                      tactile image, 224x224 (UV on)
          rgb/NNNN.png                          visual image, 224x224 (UV off)
          raw_data/NNNN_gt.npy                  contact depth, float32 224x224, metres (0 = no contact)
          raw_data/NNNN_pose.json               pose of the object in the sensor frame (see below)
  sim/<object>/session_XXX/                     12 sessions = base rotations 0°, 30°, …, 330°
      session.json
      sensor_0000/
          calibration/NNNN.png                  18 no-contact / reference frames (SITR calibration)
          samples/NNNN.png                      tactile image rendered in Blender
          samples_g/NNNN.png                    tactile image with learned real-sensor appearance (used for training)
          rgb/NNNN.png                          visual render
          norms/NNNN_gt.png                     surface normals, RGB-encoded (n = png/127.5 − 1)
          raw_data/NNNN_gt.npy                  contact depth, float32 224x224, metres
          raw_data/NNNN_pose.json
```

| Part | Objects | Sessions | Frames |
|---|---|---|---|
| real | 12 | 1 per object | 4 669 |
| sim | 12 | 12 per object | 50 400 (200–800 per session) |

Objects: `pattern_01_2_lines_angle_1_2`, `pattern_01_2_lines_angle_2`,
`pattern_01_2_lines_angle_3`, `pattern_04_3_lines_angle_1`, `pattern_04_3_lines_angle_2`,
`pattern_06_5_lines_angle_1`, `pattern_31_rod`, `pattern_32`, `pattern_33`, `pattern_35`,
`pattern_36`, `pattern_37`.

## Geometry

- The tactile camera sees a square **17.5 mm** field of view (`gel_view_m: 0.017502` in the
  data configs). The models work in the fixed **0.816 centre crop** (14.3 mm), applied
  identically to every sample at train and test time.
- Depth is the penetration of the object into the gel, in metres; the loader multiplies by
  1000 and trains in millimetres. Press depths are 0.8–1.2 mm.
- The real visual image is captured with the sensor retracted (UV off); its field of view is
  wider than the tactile one and it is not pixel-aligned with the tactile image. The model
  learns the correspondence through cross-attention; no extrinsic calibration is used.

## Pose label

`raw_data/NNNN_pose.json` stores the object placement used to render the depth GT:

```json
{"sample_x": 0.00076, "sample_y": -0.00172, "sample_z": 0.00102,
 "rotation_euler": [0.0, 0.0, 2.0048], ...}
```

- `rotation_euler[2]` — object rotation θ about the sensor normal (radians).
- `sample_x`, `sample_y` — position of the object centre (mesh bounding-box centre) in the
  sensor frame, metres. Increasing `sample_x` moves the object to the left in the image,
  increasing `sample_y` moves it up.
- `sample_z` — press depth, metres.

The training target is `(cos θ, sin θ, sample_x / 0.041, sample_y / 0.041)`: the translation
is normalised by half the 82 mm object size, so ±1 corresponds to the object edge. Rotation
labels cover a 90° window per object (−45°…45°, 45°…135° or 225°…315°, depending on the
object's symmetry); translations lie within about ±12 mm of the sensor centre.

For the real data this label comes from the robot arm that placed the sensor (end-effector
pose relative to the measured object centre); for the simulated data it is the Blender
placement. `configs/data/real_rotation_windows.json` and `real_translation_bounds.json`
record the per-object real label ranges, and the data configs restrict the simulated
sessions / samples to those ranges (`align_real_rotation`, `translation_bounds`).

## Splits

There is no split file: within every session, frames with `index % val_every == 0` form the
validation split (`val_every: 10` for real, `20` for sim). The paper reports real validation
numbers. The real:sim ratio ladder subsamples the *training* frames of each sim session
(`train_samples_per_session`), leaving the validation frames untouched.

## Download

```bash
pip install huggingface_hub
bash scripts/download_dataset.sh          # all parts
bash scripts/download_dataset.sh real     # real + meshes
bash scripts/download_dataset.sh sim      # sim + meshes
```

The Hugging Face repo stores one tarball per object (`real/<object>.tar`, `sim/<object>.tar`),
and `meshes/meshes.tar`; the script unpacks them into `./dataset`.

## License

CC BY 4.0.
