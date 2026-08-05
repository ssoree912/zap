"""Parallel Video-MME frame extraction.

decord's CPU decode of 8 sampled frames is fast (~1s); the real cost is
VideoReader's own index-building on open, which for a ~44min "long"-duration
clip took 135s alone. 300/900 Video-MME videos are "long". Serial processing
made that ~11h just for indexing. GPU decode isn't an option -- this decord
build wasn't compiled with CUDA (`decord.gpu(0)` raises "CUDA not enabled" on
first use). Fix: run the indexing+extraction for many videos concurrently
across CPU processes instead (128 cores were sitting ~88% idle).
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

VIDEOS_DIR = "/workspace/hd/data/Video-MME/videos/data"
FRAMES_DIR = os.environ.get("FRAMES_DIR", "/workspace/zap/look_rebuttal/data_video/Video-MME/frames")
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "8"))
WORKERS = int(os.environ.get("WORKERS", "24"))


def sample_frame_indices(total_frames, n):
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(step * i + step / 2) for i in range(n)]


def extract_one(vid):
    import decord  # imported per-worker; decord contexts aren't fork-safe

    vid_frame_dir = os.path.join(FRAMES_DIR, vid)
    expected = [os.path.join(vid_frame_dir, f"{k+1}.jpg") for k in range(NUM_FRAMES)]
    if all(os.path.exists(p) for p in expected):
        return vid, "cached"

    video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
    if not os.path.exists(video_path):
        return vid, "missing"

    try:
        os.makedirs(vid_frame_dir, exist_ok=True)
        vr = decord.VideoReader(video_path, num_threads=1)
        idxs = sample_frame_indices(len(vr), NUM_FRAMES)
        batch = vr.get_batch(idxs).asnumpy()
        for k, frame in enumerate(batch):
            Image.fromarray(frame).save(os.path.join(vid_frame_dir, f"{k+1}.jpg"), quality=90)
        return vid, "ok"
    except Exception as exc:
        return vid, f"error: {exc}"


def main():
    df = pd.read_parquet("/workspace/hd/data/Video-MME/videomme/test-00000-of-00001.parquet")
    unique_video_ids = df["videoID"].unique().tolist()
    print(f"[parallel-extract] {len(unique_video_ids)} unique videos, {WORKERS} workers", flush=True)

    done = 0
    errors = []
    with Pool(WORKERS) as pool:
        for vid, status in pool.imap_unordered(extract_one, unique_video_ids, chunksize=1):
            done += 1
            if status not in ("cached", "ok"):
                errors.append((vid, status))
                print(f"[parallel-extract] WARNING {vid}: {status}", file=sys.stderr, flush=True)
            if done % 25 == 0:
                print(f"[parallel-extract] {done}/{len(unique_video_ids)} (errors so far: {len(errors)})", flush=True)

    print(f"[parallel-extract] DONE {done}/{len(unique_video_ids)}, {len(errors)} errors", flush=True)
    if errors:
        print("[parallel-extract] error list:", errors, flush=True)


if __name__ == "__main__":
    main()
