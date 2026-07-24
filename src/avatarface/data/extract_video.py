"""Discover MEAD / DISFA / Aff-Wild2 videos and extract MediaPipe morph sequences."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from avatarface.data.head_features import empty_head_features, head_features_from_matrices
from avatarface.morphs.arkit_schema import N_FULL
from avatarface.morphs.channel_groups import LOWER_IDX, N_LOWER, UPPER_IDX
from avatarface.morphs.mediapipe_mapping import categories_to_vector, lower_from_full

TARGET_FPS = 30.0

# Spec §16.2 filters
MIN_FACE_CONFIDENCE = 0.8
MAX_ABS_YAW_DEG = 45.0
MAX_ABS_PITCH_DEG = 35.0
MIN_FACE_WIDTH_PX = 160


@dataclass
class VideoClip:
    path: Path
    dataset: str
    person_id: str
    video_id: str
    camera: str = "front"
    emotion: str | None = None
    intensity: str | None = None


def discover_mead(root: Path, camera: str = "front") -> list[VideoClip]:
    """
    Expected layout:
      {root}/{actor}/video/{camera}/{emotion}/level_{n}/{id}.mp4
    e.g. M003/video/front/angry/level_1/001.mp4
    """
    clips: list[VideoClip] = []
    if not root.exists():
        return clips
    for actor_dir in sorted(root.iterdir()):
        if not actor_dir.is_dir():
            continue
        actor = actor_dir.name
        if not re.match(r"^[MW]\d{3}$", actor, re.I):
            # also accept nested video folder at root/actor
            pass
        video_root = actor_dir / "video" / camera
        if not video_root.is_dir():
            # alternate: actor/camera/...
            video_root = actor_dir / camera
        if not video_root.is_dir():
            continue
        for mp4 in sorted(video_root.rglob("*.mp4")):
            parts = mp4.relative_to(video_root).parts
            emotion = parts[0] if len(parts) >= 2 else None
            intensity = parts[1] if len(parts) >= 3 else None
            clips.append(
                VideoClip(
                    path=mp4,
                    dataset="mead",
                    person_id=actor,
                    video_id=f"{actor}/{camera}/{'/'.join(parts)}".replace(".mp4", ""),
                    camera=camera,
                    emotion=emotion,
                    intensity=intensity,
                )
            )
    return clips


def discover_disfa(root: Path, camera: str = "Left") -> list[VideoClip]:
    """
    Expected layout (common):
      {root}/Videos_LeftCamera/*.avi
      {root}/Videos_RightCamera/*.avi
    """
    clips: list[VideoClip] = []
    if not root.exists():
        return clips
    folder = root / f"Videos_{camera}Camera"
    if not folder.is_dir():
        folder = root / f"Videos_{camera}"
    if not folder.is_dir():
        folder = root
    for vid in sorted(folder.glob("*.*")):
        if vid.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
            continue
        # typical name SN001_left.avi / SN001.avi
        person = re.match(r"(SN\d+)", vid.stem, re.I)
        pid = person.group(1).upper() if person else vid.stem
        clips.append(
            VideoClip(
                path=vid,
                dataset="disfa",
                person_id=pid,
                video_id=f"disfa/{camera}/{vid.stem}",
                camera=camera.lower(),
            )
        )
    return clips


def discover_affwild2(root: Path) -> list[VideoClip]:
    """Placeholder discovery — enable after MEAD+DISFA pipeline is solid."""
    clips: list[VideoClip] = []
    if not root.exists():
        return clips
    for vid in sorted(root.rglob("*.*")):
        if vid.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
            continue
        clips.append(
            VideoClip(
                path=vid,
                dataset="affwild2",
                person_id=vid.parent.name,
                video_id=f"affwild2/{vid.stem}",
                camera="unknown",
            )
        )
    return clips


def _yaw_pitch_from_mat(mat: np.ndarray) -> tuple[float, float]:
    """Approximate yaw/pitch (degrees) from rotation matrix."""
    r = mat[:3, :3]
    # yaw around y, pitch around x (OpenCV-ish)
    pitch = float(np.degrees(np.arcsin(np.clip(-r[1, 2], -1, 1))))
    yaw = float(np.degrees(np.arctan2(r[0, 2], r[2, 2])))
    return yaw, pitch


def create_landmarker(model_path: Path):
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options

    options = vision.FaceLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_clip(
    clip: VideoClip,
    landmarker,
    target_fps: float = TARGET_FPS,
    ts_offset_ms: int = 0,
) -> tuple[dict | None, int]:
    """
    Returns (record_or_None, next_ts_offset_ms).
    ts_offset_ms must keep rising across clips — MediaPipe VIDEO mode forbids reset.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(clip.path))
    if not cap.isOpened():
        return None, ts_offset_ms

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(src_fps / target_fps, 1e-6)

    full_list: list[np.ndarray] = []
    mat_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    ts_list: list[float] = []
    valid_flags: list[bool] = []

    frame_i = 0
    next_sample = 0.0
    ms = int(ts_offset_ms)

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_i >= next_sample:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Absolute monotonic clock across the whole extraction run
            ts_ms = ms + max(1, int(1000.0 / max(src_fps, 1.0)))
            ms = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)

            t_sec = frame_i / src_fps
            conf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            mat = np.eye(4, dtype=np.float32)
            morph = np.zeros(N_FULL, dtype=np.float32)
            valid = False

            if result.face_blendshapes and result.face_landmarks:
                morph = categories_to_vector(result.face_blendshapes[0])
                # presence/confidence proxy from landmark extent
                lms = result.face_landmarks[0]
                xs = [p.x for p in lms]
                ys = [p.y for p in lms]
                face_w = (max(xs) - min(xs)) * bgr.shape[1]
                face_conf = 1.0 if face_w >= MIN_FACE_WIDTH_PX else face_w / MIN_FACE_WIDTH_PX
                lower_vis = 1.0
                frame_valid = 1.0
                conf = np.array([face_conf, lower_vis, frame_valid], dtype=np.float32)

                if result.facial_transformation_matrixes:
                    mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
                    yaw, pitch = _yaw_pitch_from_mat(mat)
                    if abs(yaw) > MAX_ABS_YAW_DEG or abs(pitch) > MAX_ABS_PITCH_DEG:
                        frame_valid = 0.0
                    conf[2] = frame_valid
                valid = bool(face_conf >= MIN_FACE_CONFIDENCE and conf[2] > 0.5)

            full_list.append(morph)
            mat_list.append(mat)
            conf_list.append(conf)
            ts_list.append(t_sec)
            valid_flags.append(valid)
            next_sample += frame_interval
        frame_i += 1

    cap.release()
    if len(full_list) < 8:
        return None, ms + 1000

    full = np.stack(full_list, axis=0)
    mats = np.stack(mat_list, axis=0)
    conf = np.stack(conf_list, axis=0)
    timestamps = np.asarray(ts_list, dtype=np.float64)
    head = head_features_from_matrices(mats, timestamps)

    t = full.shape[0]
    valid_mask = np.zeros((t, N_FULL), dtype=bool)
    observed_mask = np.zeros((t, N_FULL), dtype=bool)
    for i, ok in enumerate(valid_flags):
        if ok:
            valid_mask[i] = True
            # training: all channels observed from full-face video
            observed_mask[i] = True
        else:
            # mark invalid frames
            valid_mask[i] = False

    # For training completion, simulate HMD: lower observed, upper generated
    # Store both full GT and training observed_mask variant in meta; dataset applies occlusion mask.
    record = {
        "video_id": clip.video_id,
        "track_id": "0",
        "person_id": clip.person_id,
        "dataset": clip.dataset,
        "camera": clip.camera,
        "emotion": clip.emotion,
        "intensity": clip.intensity,
        "fps": float(target_fps),
        "full_morphs": full.astype(np.float32),
        "lower_morphs": lower_from_full(full),
        "head_features": head.astype(np.float32),
        "tracking_confidence": conf.astype(np.float32),
        "valid_mask": valid_mask,
        "observed_mask": observed_mask,
        "timestamps": timestamps,
    }
    return record, ms + 1000


def save_sequence(record: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        k: record[k]
        for k in (
            "video_id",
            "track_id",
            "person_id",
            "dataset",
            "camera",
            "emotion",
            "intensity",
            "fps",
        )
    }
    np.savez_compressed(
        out_path,
        full_morphs=record["full_morphs"],
        lower_morphs=record["lower_morphs"],
        head_features=record["head_features"],
        tracking_confidence=record["tracking_confidence"],
        valid_mask=record["valid_mask"],
        observed_mask=record["observed_mask"],
        timestamps=record["timestamps"],
        meta_json=np.array(json.dumps(meta)),
    )


def default_model_path() -> Path:
    return Path("assets/face_landmarker.task")


def main():
    ap = argparse.ArgumentParser(description="Extract ARKit morphs from MEAD/DISFA/Aff-Wild2")
    ap.add_argument("--dataset", choices=["mead", "disfa", "affwild2", "all"], default="mead")
    ap.add_argument("--mead-root", type=Path, default=Path("data/raw/mead"))
    ap.add_argument("--disfa-root", type=Path, default=Path("data/raw/disfa"))
    ap.add_argument("--affwild2-root", type=Path, default=Path("data/raw/affwild2"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/sequences"))
    ap.add_argument("--camera", default="front", help="MEAD camera (use front initially)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Max clips (0=all)")
    ap.add_argument("--skip-affwild2", action="store_true", default=True)
    args = ap.parse_args()

    model_path = args.model or default_model_path()
    if not model_path.exists():
        raise SystemExit(
            f"Missing MediaPipe model at {model_path}.\n"
            "Download:\n"
            "  mkdir -p assets && curl -L -o assets/face_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    clips: list[VideoClip] = []
    if args.dataset in ("mead", "all"):
        clips.extend(discover_mead(args.mead_root, camera=args.camera))
    if args.dataset in ("disfa", "all"):
        clips.extend(discover_disfa(args.disfa_root))
    if args.dataset == "affwild2" or (args.dataset == "all" and not args.skip_affwild2):
        clips.extend(discover_affwild2(args.affwild2_root))

    if not clips:
        raise SystemExit(
            f"No videos found for dataset={args.dataset}.\n"
            "Place MEAD under data/raw/mead/{{actor}}/video/front/...\n"
            "Place DISFA under data/raw/disfa/Videos_LeftCamera/...\n"
            "Confirm dataset usage terms before downloading."
        )

    if args.limit > 0:
        clips = clips[: args.limit]

    landmarker = create_landmarker(model_path)
    ok_n = 0
    fail_n = 0
    ts_offset = 0
    for i, clip in enumerate(clips):
        safe = clip.video_id.replace("/", "__")
        out_path = args.out / clip.dataset / f"{safe}.npz"
        if out_path.exists():
            print(f"[{i+1}/{len(clips)}] skip existing {out_path.name}")
            ok_n += 1
            continue
        print(f"[{i+1}/{len(clips)}] {clip.dataset} {clip.path}")
        try:
            rec, ts_offset = extract_clip(clip, landmarker, ts_offset_ms=ts_offset)
        except Exception as e:
            print(f"  ERROR: {e}")
            fail_n += 1
            # recreate landmarker + bump clock after hard failures
            try:
                landmarker.close()
            except Exception:
                pass
            landmarker = create_landmarker(model_path)
            ts_offset += 10_000
            continue
        if rec is None:
            print("  FAILED / too short")
            fail_n += 1
            continue
        save_sequence(rec, out_path)
        ok_n += 1
        print(f"  saved T={rec['full_morphs'].shape[0]} → {out_path}")

    print(f"done: {ok_n}/{len(clips)} sequences ({fail_n} failed)")


if __name__ == "__main__":
    main()
