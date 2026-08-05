"""Build look_rebuttal-style annotation.json + extracted frames for
Video-MME and SEED-Bench-video, for the OneVision LOOK-M worker.

Video-MME: 8 uniformly-sampled frames per video (decord), extracted once per
unique video and shared across its ~3 questions.

SEED-Bench-video: uses the lmms-lab/SEED-Bench parquet mirror instead of the
raw AILab-CVC v1_video.zip, because the raw release's `data_id` (e.g.
"45899.webm") does not map to the pre-extracted "v<N>_8_frame" folder names
in v1_video.zip without an undocumented id->folder index. The lmms-lab
parquet embeds the already-selected 8 frames directly per row (same frames
lmms-eval's own seedbench task reads via doc["image"]), sidestepping the
mapping problem entirely.

Output layout matches what evaluate.py needs verbatim:
  <out_root>/<dataset>/<dataset>.json   -- {"meta_data": {"question_type":
      "multi-choice"}, "data": [{"sample_id", "task_instance": {"context",
      "choice_list", "images_path"}, "response", "image_quantity_level"}]}
  <out_root>/<dataset>/frames/<key>/{1..8}.jpg
"""

import argparse
import gc
import json
import os
import re
import resource
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # SEED-Bench-video ships some huge source frames; not attacker data


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _abort_if_too_big(limit_gb, tag):
    rss = _rss_gb()
    if rss > limit_gb:
        print(f"[{tag}] ABORTING: RSS {rss:.1f}GB exceeded {limit_gb}GB safety limit", file=sys.stderr, flush=True)
        sys.exit(1)

NUM_FRAMES = 8


def sample_frame_indices(total_frames, n):
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(step * i + step / 2) for i in range(n)]


def build_video_mme(data_root, out_root, num_frames=NUM_FRAMES, limit_videos=None):
    import decord
    import pandas as pd

    parquet_path = os.path.join(data_root, "videomme", "test-00000-of-00001.parquet")
    videos_dir = os.path.join(data_root, "videos", "data")
    df = pd.read_parquet(parquet_path)

    out_dir = os.path.join(out_root, "Video-MME")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    unique_video_ids = df["videoID"].unique().tolist()
    if limit_videos:
        unique_video_ids = unique_video_ids[:limit_videos]
    allowed = set(unique_video_ids)

    frame_cache = {}
    for i, vid in enumerate(unique_video_ids):
        video_path = os.path.join(videos_dir, f"{vid}.mp4")
        vid_frame_dir = os.path.join(frames_dir, vid)
        expected = [os.path.join(vid_frame_dir, f"{k+1}.jpg") for k in range(num_frames)]
        if all(os.path.exists(p) for p in expected):
            frame_cache[vid] = expected
            continue
        if not os.path.exists(video_path):
            print(f"[video-mme] WARNING: missing video file {video_path}, skipping", file=sys.stderr)
            continue
        os.makedirs(vid_frame_dir, exist_ok=True)
        vr = decord.VideoReader(video_path, num_threads=1)
        idxs = sample_frame_indices(len(vr), num_frames)
        batch = vr.get_batch(idxs).asnumpy()
        paths = []
        for k, frame in enumerate(batch):
            p = os.path.join(vid_frame_dir, f"{k+1}.jpg")
            Image.fromarray(frame).save(p, quality=90)
            paths.append(p)
        frame_cache[vid] = paths
        if (i + 1) % 50 == 0:
            print(f"[video-mme] extracted frames for {i+1}/{len(unique_video_ids)} videos", flush=True)

    data = []
    sample_id = 0
    option_prefix_re = re.compile(r"^[A-Za-z]\.\s*")
    for _, row in df.iterrows():
        vid = row["videoID"]
        if vid not in allowed or vid not in frame_cache:
            continue
        options = list(row["options"])
        choice_list = [option_prefix_re.sub("", opt).strip() for opt in options]
        answer_letter = str(row["answer"]).strip().upper()
        answer_idx = ord(answer_letter) - ord("A")
        if not (0 <= answer_idx < len(choice_list)):
            print(f"[video-mme] WARNING: bad answer letter for question_id={row['question_id']}", file=sys.stderr)
            continue
        data.append(
            {
                "sample_id": sample_id,
                "task_instance": {
                    "context": row["question"],
                    "choice_list": choice_list,
                    "images_path": frame_cache[vid],
                },
                "response": choice_list[answer_idx],
                "image_quantity_level": "Many",
                "meta": {
                    "question_id": row["question_id"],
                    "video_id": vid,
                    "duration": row["duration"],
                    "task_type": row["task_type"],
                },
            }
        )
        sample_id += 1

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "Video-MME.json"), "w") as f:
        json.dump({"meta_data": {"question_type": "multi-choice"}, "data": data}, f)
    print(f"[video-mme] wrote {len(data)} samples over {len(frame_cache)} videos -> {out_dir}")


def build_seedbench_video(parquet_root, out_root, limit_rows=None):
    from datasets import load_dataset

    out_dir = os.path.join(out_root, "SEEDBench_video")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    ds = load_dataset("parquet", data_files=os.path.join(parquet_root, "data", "*.parquet"), split="train")
    print(f"[seedbench-video] loaded {len(ds)} total rows (image+video)")

    data = []
    sample_id = 0
    n_video_seen = 0
    for row in ds:
        if row.get("data_type") != "video":
            continue
        n_video_seen += 1
        if limit_rows and sample_id >= limit_rows:
            break
        qid = row["question_id"]
        frame_dir = os.path.join(frames_dir, str(qid))
        images = row["image"]
        frame_paths = []
        os.makedirs(frame_dir, exist_ok=True)
        for k, img in enumerate(images):
            p = os.path.join(frame_dir, f"{k+1}.jpg")
            if not os.path.exists(p):
                img.convert("RGB").save(p, quality=90)
            frame_paths.append(p)

        choice_list = [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]]
        answer_letter = str(row["answer"]).strip().upper()
        answer_idx = ord(answer_letter) - ord("A")
        if not (0 <= answer_idx < len(choice_list)):
            print(f"[seedbench-video] WARNING: bad answer letter for question_id={qid}", file=sys.stderr)
            continue

        data.append(
            {
                "sample_id": sample_id,
                "task_instance": {
                    "context": row["question"],
                    "choice_list": choice_list,
                    "images_path": frame_paths,
                },
                "response": choice_list[answer_idx],
                "image_quantity_level": "Many",
                "meta": {
                    "question_id": qid,
                    "data_id": row["data_id"],
                    "question_type_id": row["question_type_id"],
                },
            }
        )
        sample_id += 1
        if sample_id % 200 == 0:
            print(f"[seedbench-video] processed {sample_id} video samples", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "SEEDBench_video.json"), "w") as f:
        json.dump({"meta_data": {"question_type": "multi-choice"}, "data": data}, f)
    print(f"[seedbench-video] wrote {len(data)} samples (saw {n_video_seen} video rows total) -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["video_mme", "seedbench_video", "both"], default="both")
    parser.add_argument("--video-mme-root", default="/workspace/hd/data/Video-MME")
    parser.add_argument("--seedbench-parquet-root", default="/workspace/hd/data/SEED-Bench-lmms")
    parser.add_argument("--out-root", default="/workspace/zap/look_rebuttal/data_video")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="Video-MME: frames per video (must match what's on disk)")
    parser.add_argument("--limit-videos", type=int, default=None, help="Video-MME: cap number of unique videos (smoke test)")
    parser.add_argument("--limit-rows", type=int, default=None, help="SEED-Bench-video: cap number of samples (smoke test)")
    args = parser.parse_args()

    if args.dataset in ("video_mme", "both"):
        build_video_mme(args.video_mme_root, args.out_root, num_frames=args.num_frames, limit_videos=args.limit_videos)
    if args.dataset in ("seedbench_video", "both"):
        build_seedbench_video(args.seedbench_parquet_root, args.out_root, limit_rows=args.limit_rows)
