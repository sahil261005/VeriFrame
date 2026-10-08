"""
builds the 50-video evaluation set used with evaluate_pipeline.py, from free public Hugging Face datasets:

  AI-generated (25)  34data/gen-videos-{sora2, veo3, kling, wan2}   clips from Sora 2, Veo 3, Kling, Wan 2
  real HD (12)       nkp37/OpenVid-1M (OpenVidHD, CC-BY-4.0)        real-world web footage
  real low-res (13)  sayakpaul/ucf101-subset (UCF101)               real action clips, 320x240

single clips are pulled out of the large remote zips with HTTP range requests, so only ~150MB is downloaded.
selection is seeded, so re-running gives the same 50 videos.

    pip install huggingface_hub remotezip
    python download_benchmark.py                # writes benchmark/ and benchmark/labels.json
    python evaluate_pipeline.py --dir benchmark --labels benchmark/labels.json
"""
import json
import os
import random
import tarfile

from huggingface_hub import hf_hub_download
from remotezip import RemoteZip

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
SEED = 0
VIDEO_EXT = (".mp4", ".mov", ".webm", ".avi")

AI_SOURCES = {  # category -> (zip url, how many clips)
    "ai_sora2": ("https://huggingface.co/datasets/34data/gen-videos-sora2/resolve/main/data_001.zip", 7),
    "ai_veo3": ("https://huggingface.co/datasets/34data/gen-videos-veo3/resolve/main/data_001.zip", 6),
    "ai_kling": ("https://huggingface.co/datasets/34data/gen-videos-kling/resolve/main/data_001.zip", 6),
    "ai_wan2": ("https://huggingface.co/datasets/34data/gen-videos-wan2/resolve/main/data_001.zip", 6),
}
REAL_HD_ZIP = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_1.zip"
REAL_HD_COUNT = 12
REAL_LOWRES_COUNT = 13


def pick(names, count, rng):
    names = sorted(names)
    return rng.sample(names, min(count, len(names)))


def pull_from_zip(url, count, category, rng, labels, label):
    with RemoteZip(url) as z:
        members = [i.filename for i in z.infolist() if i.filename.lower().endswith(VIDEO_EXT)]
        for name in pick(members, count, rng):
            out_name = f"{category}_{os.path.basename(name)}"
            with open(os.path.join(OUT_DIR, out_name), "wb") as f:
                f.write(z.read(name))
            labels[out_name] = {"label": label, "category": category}
            print(f"  {out_name}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    labels = {}

    for category, (url, count) in AI_SOURCES.items():
        print(f"{category}: {count} clips")
        pull_from_zip(url, count, category, rng, labels, "MANIPULATED")

    print(f"real_hd: {REAL_HD_COUNT} clips")
    pull_from_zip(REAL_HD_ZIP, REAL_HD_COUNT, "real_hd", rng, labels, "AUTHENTIC")

    print(f"real_lowres: {REAL_LOWRES_COUNT} clips")
    tar_path = hf_hub_download("sayakpaul/ucf101-subset", "UCF101_subset.tar.gz", repo_type="dataset")
    with tarfile.open(tar_path) as tar:
        members = {m.name: m for m in tar.getmembers() if m.name.lower().endswith(VIDEO_EXT)}
        for name in pick(members, REAL_LOWRES_COUNT, rng):
            out_name = f"real_lowres_{os.path.basename(name)}"
            with tar.extractfile(members[name]) as src, open(os.path.join(OUT_DIR, out_name), "wb") as dst:
                dst.write(src.read())
            labels[out_name] = {"label": "AUTHENTIC", "category": "real_lowres"}
            print(f"  {out_name}")

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\n{len(labels)} videos written to {OUT_DIR}")


if __name__ == "__main__":
    main()
