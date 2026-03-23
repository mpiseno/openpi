#!/usr/bin/env python3
"""Preprocess DROID RLDS TFRecord data using SAM3 to mask out background.

For each episode, reads JPEG frames from the three camera views, runs SAM3
masking using object nouns extracted from the language instruction, and saves
the masked frames as a compressed .npz file (one per episode).

Output layout:
    <output_dir>/
        <recording_folderpath>--<file_path>.npz
            exterior_image_1_left : uint8 (N, H, W, 3)
            exterior_image_2_left : uint8 (N, H, W, 3)
            wrist_image_left      : uint8 (N, H, W, 3)
"""

import dataclasses
import json
import math
import os
import shutil
import tempfile
import traceback
import collections
from pathlib import Path

# Must be set before the first CUDA allocation to avoid memory fragmentation OOM.
# With the default allocator, reserved-but-unallocated blocks can't satisfy large
# contiguous requests even when total free memory is sufficient.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mediapy
import mujoco
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import tensorflow as tf  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402

# Keep TensorFlow off the GPU so it stays free for SAM3 / PyTorch.
tf.config.set_visible_devices([], "GPU")

from sam3.model_builder import build_sam3_video_predictor  # noqa: E402

SCENE_XML = Path("/home/michael/src/g2d-tracking/assets/franka_fr3_robotiq/fr3_robotiq.xml")

# DROID data is recorded at 15 Hz.
DATA_FREQ = 15.0

# Wrist image resolution in the RLDS dataset.
RLDS_H, RLDS_W = 180, 320

_GRIPPER_BODY_NAMES = [
    "base_mount", "2f85_base",
    "right_driver", "right_coupler", "right_spring_link", "right_follower", "right_pad", "right_silicone_pad",
    "left_driver",  "left_coupler",  "left_spring_link",  "left_follower",  "left_pad",  "left_silicone_pad",
]

_GRIPPER_BASE_NAMES = [
    "base_mount", "2f85_base",
    "right_driver", "right_coupler",
    "left_driver",  "left_coupler",
]

CAMERA_KEYS = [
    "wrist_image_left",
]
EXTRA_CAMERA_KEYS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
]

SEGMENT_TYPE = "gripper"
EXTRA_SITES = ["gripper_base_site", "left_follower_site", "right_follower_site"]


# ---------------------------------------------------------------------------
# MuJoCo FK helpers
# ---------------------------------------------------------------------------


class Sim:
    """MuJoCo simulator for FK and wrist-camera segmentation rendering."""

    def __init__(self) -> None:
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.seg_renderer = mujoco.Renderer(self.model, height=RLDS_H, width=RLDS_W)
        self.seg_renderer.enable_segmentation_rendering()
        self.gripper_geom_ids, self.base_geom_ids = self._get_gripper_geom_ids()
        self._wrist_cam_id = self.model.cam("wrist_cam").id

    def _build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(SCENE_XML))
        cam = spec.worldbody.add_camera()
        cam.name = "wrist_cam"
        cam.fovy = 60.0  # placeholder; overwritten per episode
        return spec.compile()

    def _get_gripper_geom_ids(self) -> tuple[np.ndarray, np.ndarray]:
        gripper_body_ids = {self.model.body(name).id for name in _GRIPPER_BODY_NAMES}
        base_body_ids = {self.model.body(name).id for name in _GRIPPER_BASE_NAMES}
        all_geom_ids = np.array([
            i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in gripper_body_ids
        ])
        base_geom_ids = np.array([
            i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in base_body_ids
        ])
        return all_geom_ids, base_geom_ids

    def _set_camera_extrinsic(self, extrinsic: np.ndarray) -> None:
        """Update wrist_cam pose from a [x,y,z,rx,ry,rz] cam-to-base extrinsic (CV convention)."""
        R_cv = Rotation.from_euler("xyz", extrinsic[3:]).as_matrix()
        flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        self.data.cam_xpos[self._wrist_cam_id] = extrinsic[:3]
        self.data.cam_xmat[self._wrist_cam_id] = (R_cv @ flip).flatten()

    def run_episode(
        self,
        steps: list,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run FK + segmentation for one episode.

        Updates fovy from intrinsics, then returns left (N, 3), right (N, 3),
        masks (N, H, W) bool (True = any gripper pixel), and
        base_masks (N, H, W) bool (True = gripper base pixel, excluding fingers).
        """
        fy = intrinsics[1]
        self.model.cam_fovy[self._wrist_cam_id] = math.degrees(2.0 * math.atan(RLDS_H / (2.0 * fy)))

        decimation = round(1.0 / (DATA_FREQ * self.model.opt.timestep))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = steps[0]["observation"]["joint_position"].numpy()
        mujoco.mj_forward(self.model, self.data)

        left_positions, right_positions, masks, base_masks = [], [], [], []
        for i, step in enumerate(steps):
            arm_pos = step["observation"]["joint_position"].numpy()
            # NOTE: Directly set qpos each step for more stable arm tracking, even though it disobeys the physics.
            self.data.qpos[:7] = arm_pos
            self.data.qvel[:7] = 0.0
            self.data.ctrl[:7] = arm_pos
            self.data.ctrl[7] = 0.8 * step["observation"]["gripper_position"].numpy().item()
            for _ in range(decimation):
                mujoco.mj_step(self.model, self.data)
            left_positions.append(self.data.site("left_gripper_site").xpos.copy())
            right_positions.append(self.data.site("right_gripper_site").xpos.copy())

            self._set_camera_extrinsic(extrinsics[i])
            self.seg_renderer.update_scene(self.data, camera="wrist_cam")
            seg = self.seg_renderer.render()  # (H, W, 2): channel 0=geom ID, channel 1=geom type
            geom_ids = seg[:, :, 0]
            masks.append(np.isin(geom_ids, self.gripper_geom_ids))
            base_masks.append(np.isin(geom_ids, self.base_geom_ids))

        return np.stack(left_positions), np.stack(right_positions), np.stack(masks), np.stack(base_masks)


# ---------------------------------------------------------------------------
# TFRecord / RLDS helpers
# ---------------------------------------------------------------------------


def find_dataset_dir(data_dir: Path) -> Path:
    """Recursively find the TFDS dataset directory (the one with features.json)."""
    hits = list(data_dir.rglob("features.json"))
    if not hits:
        raise FileNotFoundError(f"No TFDS dataset (features.json) found under {data_dir}")
    return hits[0].parent


def decode_camera_frames(steps, camera_key: str) -> np.ndarray:
    """Decode all frames for one camera. Returns uint8 (N, H, W, 3).

    Handles both JPEG-encoded (dtype=string) and already-decoded (dtype=uint8) tensors.
    """
    frames = []
    for step in steps:
        img = step["observation"][camera_key]
        if img.dtype == tf.string:
            img = tf.io.decode_jpeg(img)
        frames.append(img.numpy())
    return np.stack(frames)


_BG_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # background mask


def dilate_masks(masks: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Dilate each mask with a square kernel to better cover the gripper silhouette."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = np.array([cv2.dilate(mask.astype(np.uint8), kernel) for mask in masks])
    return dilated.astype(bool)


# TODO: Probably want to make this an online augmentation.
def augment_masks_ellipses(
    masks: np.ndarray,
    n_ellipses: int = 16,
    min_radius: int = 8,
    max_radius: int = 32,
) -> np.ndarray:
    """Add random ellipses to each frame's mask to obscure the gripper silhouette.

    masks: (N, H, W) bool array. Returns augmented (N, H, W) bool array.
    Ellipses are centered on randomly sampled pixels within the existing mask,
    so they always overlap the masked region while breaking its exact shape.
    """
    result = masks.copy().astype(np.uint8)
    for i, mask in enumerate(result):
        ys, xs = np.where(mask)
        if len(xs) > 0:
            for _ in range(n_ellipses):
                idx = np.random.randint(len(xs))
                cx, cy = int(xs[idx]), int(ys[idx])
                a = np.random.randint(min_radius, max_radius + 1)
                b = np.random.randint(min_radius, max_radius + 1)
                angle = np.random.randint(0, 180)
                cv2.ellipse(result[i], (cx, cy), (a, b), angle, 0, 360, 1, -1)
    return result.astype(bool)


# TODO: Probably want to make this online as well.
def _fill_shifted(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels using a randomly shifted copy of the frame."""
    h, w = frame.shape[:2]
    dy = np.random.randint(h // 2, h)
    dx = np.random.randint(w // 2, w)
    shifted = np.roll(np.roll(frame, dy, axis=0), dx, axis=1)
    result = frame.copy()
    result[mask] = shifted[mask]
    return result


def apply_masks(
    frames: np.ndarray,
    masks: np.ndarray,
    fill_mask: bool = False,
) -> np.ndarray:
    """Replace background pixels (outside mask) in each frame. Returns uint8 (N, H, W, 3).

    masks: (N, H, W) bool array — True where the gripper is.
    fill_mask=True fills with a randomly shifted copy; False fills with solid _BG_COLOR.
    """
    result = frames.copy()
    for i, mask in enumerate(masks):
        if fill_mask:
            result[i][mask] = _fill_shifted(frames[i], mask)[mask]
        else:
            result[i][mask] = _BG_COLOR
    return result


def mask_camera(
    frames: np.ndarray, masks: np.ndarray, base_masks: np.ndarray, fill_mask: bool = False
) -> np.ndarray:
    base_masks = dilate_masks(base_masks, kernel_size=16)
    masks = np.logical_or(masks, base_masks) # combine
    # masks = augment_masks_ellipses(masks)
    masks = apply_masks(frames, masks, fill_mask=fill_mask)
    return masks


# ---------------------------------------------------------------------------
# RLDS / TFRecord output helpers
# ---------------------------------------------------------------------------


def _encode_jpeg(frame_rgb: np.ndarray, quality: int = 95) -> bytes:
    """Encode an RGB uint8 (H, W, 3) frame to JPEG bytes."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG encoding failed"
    return buf.tobytes()


def save_episode_tfrecord(
    raw_record: bytes,
    masked_frames_by_cam: dict[str, np.ndarray],
    output_path: Path,
    extra_obs: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a modified episode TFRecord by editing the raw proto in place.

    Parses the original SequenceExample, overwrites the JPEG bytes for the
    requested camera keys, appends extra_obs as new float feature lists under
    steps/observation/<key>, then writes the result. All other fields are
    preserved verbatim.
    """
    seq_ex = tf.train.SequenceExample()
    seq_ex.ParseFromString(raw_record)

    for cam_key, frames in masked_frames_by_cam.items():
        feat = seq_ex.context.feature.get(f"steps/observation/{cam_key}")
        if feat is None:
            continue
        for i, frame in enumerate(frames):
            if i >= len(feat.bytes_list.value):
                break
            feat.bytes_list.value[i] = _encode_jpeg(frame)

    for obs_key, values in (extra_obs or {}).items():
        feat = seq_ex.context.feature[f"steps/observation/{obs_key}"]
        for row in values:
            feat.float_list.value.extend(row.flatten().tolist())

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        writer.write(seq_ex.SerializeToString())


def setup_rlds_output_dir(source_dataset_dir: Path, rlds_output_dir: Path) -> None:
    """Copy TFDS metadata files so the output dir is loadable with tfds.builder_from_directory."""
    rlds_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dataset_dir / "features.json", rlds_output_dir / "features.json")


def finalize_rlds_dataset_info(
    source_dataset_dir: Path,
    rlds_output_dir: Path,
    shard_lengths: list[int],
    dataset_name: str = "droid_masked",
) -> None:
    """Write dataset_info.json so tfds.builder_from_directory can load the output."""
    info_path = source_dataset_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    info["name"] = dataset_name
    info["splits"] = [
        {
            "name": "train",
            "numExamples": str(sum(shard_lengths)),
            "shardLengths": [str(n) for n in shard_lengths],
        }
    ]

    with open(rlds_output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Camera projection helpers
# ---------------------------------------------------------------------------


def _extrinsic_to_cam_matrix(extrinsic: np.ndarray) -> np.ndarray:
    """Convert 6-DOF [x,y,z,rx,ry,rz] cam-to-base extrinsic to 4x4 base-to-cam matrix."""
    t = extrinsic[:3]
    R = Rotation.from_euler("xyz", extrinsic[3:]).as_matrix()
    cam_to_base = np.eye(4)
    cam_to_base[:3, :3] = R
    cam_to_base[:3, 3] = t
    return np.linalg.inv(cam_to_base)


def _project_point(point_base: np.ndarray, base_to_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[int, int]:
    p_cam = base_to_cam @ np.append(point_base, 1.0)
    x, y, z = p_cam[:3]
    return int(round(fx * x / z + cx)), int(round(fy * y / z + cy))


def overlay_affordances(frames: np.ndarray, metadata: np.lib.npyio.NpzFile) -> np.ndarray:
    """Overlay projected gripper fingertip dots on wrist frames. Red=left, blue=right.

    If prompt_points and prompt_labels are provided, also draws SAM3 prompt points on every frame:
    positive points (label=1) in blue, negative points (label=0) in red.
    """
    extrinsics = metadata["extrinsics"]        # (N, 6)
    fx, fy, cx, cy = metadata["intrinsics"]
    left_gripper_pos = metadata["left_gripper_pos"]   # (N, 3)
    right_gripper_pos = metadata["right_gripper_pos"]  # (N, 3)
    result = frames.copy()
    h, w = frames.shape[1:3]
    n = min(len(result), len(extrinsics), len(left_gripper_pos))
    for i in range(n):
        base_to_cam = _extrinsic_to_cam_matrix(extrinsics[i])
        for pos, color in ((left_gripper_pos[i], (255, 255, 255)), (right_gripper_pos[i], (255, 255, 255))):
            px = _project_point(pos, base_to_cam, fx, fy, cx, cy)
            cv2.circle(result[i], px, 5, color, -1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    data_dir: Path
    """Root DROID data directory."""
    output_dir: Path | None = None
    """Base output directory. Defaults to <script_dir>/data/. Videos go in <output_dir>/videos/, RLDS in <output_dir>/rlds/."""
    skip_existing: bool = True
    """Skip episodes whose output file already exists."""
    fill_mask: bool = False
    """Fill masked regions with randomly shifted image content instead of solid green."""


def main(args: Args) -> None:
    args.data_dir = args.data_dir.expanduser()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    print("Loading MuJoCo model...")
    sim = Sim()

    print("Finding TFDS dataset...")
    dataset_dir = find_dataset_dir(args.data_dir)
    metadata_dir = dataset_dir.parent / "metadata"
    base_dir = (args.output_dir or Path(__file__).parent / "data").expanduser()
    video_dir = base_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Found: {dataset_dir}")
    print(f"  Metadata dir: {metadata_dir}")
    print(f"  Video dir: {video_dir}")

    builder = tfds.builder_from_directory(builder_dir=str(dataset_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    # Raw TFRecord stream (same order as ds when shuffle_files=False)
    shard_files = sorted(dataset_dir.glob("*.tfrecord*"))
    raw_ds = tf.data.TFRecordDataset([str(f) for f in shard_files])

    rlds_output_dir = base_dir / "rlds"
    setup_rlds_output_dir(dataset_dir, rlds_output_dir)
    shard_lengths: list[int] = []  # one entry per episode written to rlds_output_dir

    num_eps_processed = 0
    for (ep_idx, episode), raw_record in zip(enumerate(tqdm(ds, desc="Episodes")), raw_ds):
        output_path = video_dir / f"{ep_idx:04d}.mp4"
        if args.skip_existing and output_path.exists():
            print(f"  Skipping episode {ep_idx:04d} (output already exists)")
            continue

        steps = list(episode["steps"])

        # Skip if no metadata for this episode
        metadata_path = metadata_dir / f"{ep_idx:04d}_metadata.npz"
        if not metadata_path.exists():
            tqdm.write(f"  [{ep_idx:04d}] Skipping: no metadata file")
            continue

        try:
            cam_metadata = np.load(metadata_path)
            left_gripper_pos, right_gripper_pos, masks, base_masks = sim.run_episode(
                steps, cam_metadata["extrinsics"], cam_metadata["intrinsics"]
            )
            metadata = {
                **cam_metadata,
                "left_gripper_pos": left_gripper_pos,
                "right_gripper_pos": right_gripper_pos,
            }

            exterior_frames = decode_camera_frames(steps, "exterior_image_2_left")

            # Compute masked frames for all camera keys
            masked_frames_by_cam: dict[str, np.ndarray] = {}
            for cam in CAMERA_KEYS:
                frames = decode_camera_frames(steps, cam)
                masked = mask_camera(frames, masks, base_masks, fill_mask=args.fill_mask)
                masked_frames_by_cam[cam] = masked  # raw masked frames, no affordance overlay

            # Save video (with affordance overlay, for visualisation only) — first 30 episodes
            if num_eps_processed < 30:
                for cam in CAMERA_KEYS:
                    frames = decode_camera_frames(steps, cam)
                    masked_vis = overlay_affordances(masked_frames_by_cam[cam].copy(), metadata)
                    unmasked_vis = overlay_affordances(frames.copy(), metadata)
                    combined = np.concatenate([masked_vis, unmasked_vis, exterior_frames], axis=2)
                    mediapy.write_video(str(output_path), combined, fps=15)
                    print(f"  Wrote {output_path}")

            # Save as RLDS TFRecord (same format as input)
            rlds_path = rlds_output_dir / f"episode_{ep_idx:04d}.tfrecord"
            save_episode_tfrecord(
                raw_record.numpy(),
                masked_frames_by_cam,
                rlds_path,
                extra_obs={
                    "left_gripper_pos": left_gripper_pos,
                    "right_gripper_pos": right_gripper_pos,
                },
            )
            shard_lengths.append(1)  # one episode per shard file
            print(f"  Wrote {rlds_path}")

            num_eps_processed += 1
        except Exception as e:
            tqdm.write(f"  Error on episode {ep_idx}: {type(e).__name__}: {e}")
            tqdm.write(traceback.format_exc())

        if num_eps_processed >= 3:
            break

    finalize_rlds_dataset_info(dataset_dir, rlds_output_dir, shard_lengths)
    print(f"Done. Videos saved to {video_dir}.")
    print(f"RLDS dataset saved to {rlds_output_dir}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
