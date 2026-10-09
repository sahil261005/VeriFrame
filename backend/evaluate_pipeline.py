import os
import csv
import json
import logging
import time
import statistics
from dotenv import load_dotenv

# same setup as main.py, api keys from .env and langsmith tracing off
load_dotenv()
for _var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
    os.environ[_var] = "false"

from agents.orchestrator import run_pipeline
import preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate_pipeline")

def run_evaluation(test_dir, labels_json_path, scores_csv_path=None):
    """
    runs the pipeline on a folder of test videos and compares it to a labels json that maps
    filenames to "AUTHENTIC" or "MANIPULATED". a label can also have a category for per group results like
        "clip.mp4": {"label": "MANIPULATED", "category": "ai_generated"}
    """
    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found at: {test_dir}")
        return
    
    if not os.path.exists(labels_json_path):
        logger.error(f"Labels JSON file not found at: {labels_json_path}")
        return

    with open(labels_json_path, 'r') as f:
        ground_truth = json.load(f)

    logger.info(f"Loaded {len(ground_truth)} video labels for evaluation.")
    
    results = []
    tp, fp, tn, fn = 0, 0, 0, 0
    start_time = time.time()

    for filename, label_entry in ground_truth.items():
        if isinstance(label_entry, dict):
            true_label = label_entry.get("label", "")
            category = label_entry.get("category") or ("real" if true_label == "AUTHENTIC" else "manipulated")
        else:
            true_label = label_entry
            category = "real" if true_label == "AUTHENTIC" else "manipulated"
        video_path = os.path.join(test_dir, filename)
        if not os.path.exists(video_path):
            logger.warning(f"Video file {filename} not found in {test_dir}. Skipping.")
            continue

        logger.info(f"Processing evaluation for: {filename} (Truth: {true_label})")
        
        try:
            # 1. preprocessing, same as main.process_video_task so the numbers match production
            video_start = time.perf_counter()
            metadata = preprocessing.get_video_metadata(video_path)
            frames = preprocessing.extract_frames(video_path, interval=1.0, target_height=480, max_frames=6)

            # 2. run the agent graph, the original file is passed for the audio check like in production
            output = run_pipeline(frames, metadata, video_path=video_path)
            video_seconds = time.perf_counter() - video_start
            perf = output.get("performance", {})

            predicted_verdict = output.get("final_verdict", "UNCERTAIN")
            confidence = output.get("final_confidence", 0.0)

            logger.info(f"Result for {filename}: Predicted {predicted_verdict} (Conf: {confidence*100:.1f}%)")
            
            status = output.get("agent_status", {})
            results.append({
                "filename": filename,
                "category": category,
                "truth": true_label,
                "prediction": predicted_verdict,
                "confidence": confidence,
                # per agent scores so the fusion weights and thresholds can be tuned on labelled data
                "visual_score": output.get("visual_score", 0.0),
                "visual_status": status.get("visual", ""),
                "temporal_score": output.get("temporal_score", 0.0),
                "temporal_status": status.get("temporal", ""),
                "audio_score": output.get("audio_score", 0.0),
                "audio_status": status.get("audio", ""),
                "llm_score": output.get("llm_score", 0.0),
                "llm_status": status.get("llm", ""),
                "generative_score": output.get("generative_score", 0.0),
                "generative_status": status.get("generative", ""),
                "c2pa_ai_generated": metadata.get("provenance", {}).get("c2pa_ai_generated", False),
                "seconds": round(video_seconds, 3),
                "llm_calls": perf.get("llm_calls", 0),
                "llm_input_tokens": perf.get("llm_input_tokens", 0),
                "llm_output_tokens": perf.get("llm_output_tokens", 0),
                "llm_cost_usd": perf.get("llm_cost_usd")
            })

            # count up the results. UNCERTAIN just counts as not manipulated here
            if true_label == "MANIPULATED":
                if predicted_verdict == "MANIPULATED":
                    tp += 1
                else:
                    fn += 1
            elif true_label == "AUTHENTIC":
                if predicted_verdict == "MANIPULATED":
                    fp += 1
                else:
                    tn += 1

        except Exception as e:
            logger.error(f"Error evaluating video {filename}: {e}", exc_info=True)

    if scores_csv_path and results:
        with open(scores_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Per-agent scores written to {scores_csv_path}")

    # print the summary
    total = tp + fp + tn + fn
    if total == 0:
        logger.warning("No videos successfully evaluated.")
        return

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    elapsed = time.time() - start_time

    print("\n" + "="*50)
    print("📋 VERIFRAME PIPELINE EVALUATION REPORT")
    print("="*50)
    print(f"Total Videos Evaluated: {total}")
    print(f"Total Time Elapsed:    {elapsed:.2f} seconds")
    print(f"Average Time per Video:{elapsed/total:.2f} seconds")
    print("-"*50)
    print(f"True Positives (TP):   {tp}")
    print(f"True Negatives (TN):   {tn}")
    print(f"False Positives (FP):  {fp}")
    print(f"False Negatives (FN):  {fn}")
    print("-"*50)
    print(f"Accuracy:             {accuracy * 100:.2f}%")
    print(f"Precision:            {precision * 100:.2f}%")
    print(f"Recall:               {recall * 100:.2f}%")
    print(f"False Positive Rate:  {fpr * 100:.2f}%")

    # per group results. detection rate for the fake categories and false positive rate for real footage
    categories = sorted(set(r["category"] for r in results))
    if len(categories) > 1:
        print("-"*50)
        for cat in categories:
            rows = [r for r in results if r["category"] == cat]
            flagged = sum(1 for r in rows if r["prediction"] == "MANIPULATED")
            kind = "false-positive rate" if all(r["truth"] == "AUTHENTIC" for r in rows) else "detection rate"
            print(f"{cat:20s} n={len(rows):3d}  {kind}: {flagged / len(rows) * 100:.1f}%")

    # latency and cost per video, same numbers the report shows in production
    seconds = sorted(r["seconds"] for r in results)
    if seconds:
        p95_index = min(len(seconds) - 1, max(0, int(round(0.95 * len(seconds))) - 1))
        print("-"*50)
        print(f"Latency p50 / p95:    {statistics.median(seconds):.2f}s / {seconds[p95_index]:.2f}s")
        print(f"LLM calls per video:  {statistics.mean(r['llm_calls'] for r in results):.2f}")
        print(f"LLM tokens per video: {statistics.mean(r['llm_input_tokens'] for r in results):.0f} in / "
              f"{statistics.mean(r['llm_output_tokens'] for r in results):.0f} out")
        costs = [r["llm_cost_usd"] for r in results]
        if all(c is not None for c in costs):
            print(f"LLM cost per video:   ${statistics.mean(costs):.5f}")
        else:
            print("LLM cost per video:   set GEMINI_INPUT_USD_PER_M / GEMINI_OUTPUT_USD_PER_M (and GROQ_*) to compute")
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VeriFrame multi-agent pipeline.")
    parser.add_argument("--dir", default="test_data", help="Directory containing test videos")
    parser.add_argument("--labels", default="test_labels.json", help="Path to JSON file containing file-to-label map")
    parser.add_argument("--scores-csv", default="evaluation_scores.csv", help="Where to write per-video, per-agent scores")
    
    args = parser.parse_args()
    
    # if theres no test data yet just print how to set it up
    if not os.path.exists(args.dir) or not os.path.exists(args.labels):
        print("\n" + "!"*60)
        print("💡 HOW TO RUN VERIFRAME PIPELINE EVALUATION:")
        print("1. Create a test directory with video samples (e.g. 'backend/test_data/').")
        print("2. Create a JSON labels file (e.g. 'backend/test_labels.json') containing:")
        print('   {\n     "sample1.mp4": "AUTHENTIC",\n     "sample2.mp4": "MANIPULATED"\n   }')
        print(f"3. Run: python evaluate_pipeline.py --dir {args.dir} --labels {args.labels}")
        print("!"*60 + "\n")
    else:
        run_evaluation(args.dir, args.labels, args.scores_csv)
