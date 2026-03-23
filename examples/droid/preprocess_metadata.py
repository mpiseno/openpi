#!/usr/bin/env python3
"""Preprocess DROID RLDS dataset: extract and save per-episode camera metadata.

For each episode, downloads the raw H5 file from GCS (temporarily), extracts the
wrist camera serial number and per-step extrinsics, and fetches intrinsics from the
Stereolabs calibration server.

Output layout:
    <metadata_dir>/
        0000_metadata.npz
            extrinsics  : (N, 6)  wrist cam extrinsics [x,y,z,rx,ry,rz]
            intrinsics  : (4,)    [fx, fy, cx, cy] scaled to RLDS resolution
        0001_metadata.npz
        ...
        episode_index.json   # maps ep_idx -> episode key (recording_folderpath--file_path)
"""

import configparser
import dataclasses
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import tensorflow as tf  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402

tf.config.set_visible_devices([], "GPU")

GCS_RAW_ROOT = "gs://gresearch/robotics/droid_raw/1.0.1"
NFS_RAW_PREFIX = "/nfs/kun2/datasets/r2d2/r2d2-data-full/"

# RLDS images are 320x180; ZED HD native is 1280x720.
CAM_SCALE = 320 / 1280  # 0.25

# Manual correction added on top of the per-episode cam_to_gripper offset.
# Format: [dx, dy, dz, drx, dry, drz] in the EE/gripper frame.
CAM_TO_GRIPPER_CORRECTION = np.array([0.0, 0.0, -0.023, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def find_dataset_dir(data_dir: Path) -> Path:
    hits = list(data_dir.rglob("features.json"))
    if not hits:
        raise FileNotFoundError(f"No TFDS dataset found under {data_dir}")
    return hits[0].parent


def episode_key(episode) -> str:
    rfp = episode["episode_metadata"]["recording_folderpath"].numpy().decode()
    fp = episode["episode_metadata"]["file_path"].numpy().decode()
    return f"{rfp}--{fp}"


def fetch_h5_temp(nfs_path: str) -> str:
    """Download the raw H5 for an episode to a temp file. Caller must delete it."""
    rel = nfs_path.removeprefix(NFS_RAW_PREFIX)
    gcs_url = f"{GCS_RAW_ROOT}/{rel}"
    fd, local = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    subprocess.run(["gsutil", "cp", gcs_url, local], check=True, capture_output=True)
    return local


def detect_wrist_serial(h5_path: str) -> str:
    """Return the serial of the wrist camera — the only one with varying extrinsics."""
    with h5py.File(h5_path, "r") as f:
        for key in f["observation/camera_extrinsics"]:
            if not key.endswith("_left"):
                continue
            data = f[f"observation/camera_extrinsics/{key}"][:]
            if np.std(data, axis=0).max() > 1e-6:
                return key.removesuffix("_left")
    raise RuntimeError(f"Could not find wrist camera serial in {h5_path}")


def fetch_intrinsics(serial: str, cache: dict) -> np.ndarray:
    """Return [fx, fy, cx, cy] scaled to RLDS resolution. Queries Stereolabs if not cached."""
    if serial not in cache:
        url = f"https://calib.stereolabs.com/?SN={serial}"
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode()
        cfg = configparser.ConfigParser()
        cfg.read_string(raw)
        hd = cfg["LEFT_CAM_HD"]
        cache[serial] = np.array([
            float(hd["fx"]) * CAM_SCALE,
            float(hd["fy"]) * CAM_SCALE,
            float(hd["cx"]) * CAM_SCALE,
            float(hd["cy"]) * CAM_SCALE,
        ])
    return cache[serial]


def extract_extrinsics(h5_path: str, serial: str) -> np.ndarray:
    """Return per-step cam-to-base extrinsics computed from EE poses and cam-to-gripper offset."""
    with h5py.File(h5_path, "r") as f:
        cam_to_gripper = f[f"observation/camera_extrinsics/{serial}_left_gripper_offset"][0]
        ee_poses = f["observation/robot_state/cartesian_position"][:]

    cam_to_gripper = cam_to_gripper + CAM_TO_GRIPPER_CORRECTION

    T_cam_to_gripper = np.eye(4)
    T_cam_to_gripper[:3, :3] = Rotation.from_euler("xyz", cam_to_gripper[3:]).as_matrix()
    T_cam_to_gripper[:3, 3] = cam_to_gripper[:3]

    result = []
    for pose in ee_poses:
        T_ee = np.eye(4)
        T_ee[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        T_ee[:3, 3] = pose[:3]
        T_cam = T_ee @ T_cam_to_gripper
        t = T_cam[:3, 3]
        r = Rotation.from_matrix(T_cam[:3, :3]).as_euler("xyz")
        result.append(np.concatenate([t, r]))
    return np.array(result)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    data_dir: Path
    """Root DROID data directory."""
    metadata_dir: Path | None = None
    """Where to save per-episode .npz files. Defaults to <builder_dir>/metadata/ (e.g. droid_100/metadata/)."""
    skip_existing: bool = True
    """Skip episodes whose .npz already exists."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    args.data_dir = args.data_dir.expanduser()

    print("Finding TFDS dataset...")
    dataset_dir = find_dataset_dir(args.data_dir)
    metadata_dir = args.metadata_dir or dataset_dir.parent / "metadata"
    builder = tfds.builder_from_directory(builder_dir=str(dataset_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_cache: dict[str, np.ndarray] = {}
    episode_index: dict[int, str] = {}

    for ep_idx, episode in enumerate(tqdm(ds, desc="Episodes")):
        out_path = metadata_dir / f"{ep_idx:04d}_metadata.npz"
        ep_key = episode_key(episode)
        episode_index[ep_idx] = ep_key

        if args.skip_existing and out_path.exists():
            continue

        nfs_path = episode["episode_metadata"]["file_path"].numpy().decode()
        h5_tmp = None
        try:
            h5_tmp = fetch_h5_temp(nfs_path)
            serial = detect_wrist_serial(h5_tmp)
            intrinsics = fetch_intrinsics(serial, intrinsics_cache)
            extrinsics = extract_extrinsics(h5_tmp, serial)

            np.savez(
                out_path,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
            )
            tqdm.write(f"  [{ep_idx:04d}] serial={serial} -> {out_path.name}")
        except Exception as e:
            tqdm.write(f"  [{ep_idx:04d}] ERROR: {e}")
        finally:
            if h5_tmp and Path(h5_tmp).exists():
                os.unlink(h5_tmp)

    index_path = metadata_dir / "episode_index.json"
    with open(index_path, "w") as f:
        json.dump(episode_index, f, indent=2)
    print(f"Done. Metadata saved to {metadata_dir}, index at {index_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
