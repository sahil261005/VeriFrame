import os
import json
import logging
import time
from agents.orchestrator import run_pipeline
import preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate_pipeline")

def run_evaluation(test_dir, labels_json_path):
    """
    Evaluates the multi-agent consensus pipeline against a directory of test videos
    using a labels JSON file mapping video filenames to truth labels ("AUTHENTIC" or "MANIPULATED").
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

    for filename, true_label in ground_truth.items():
        video_path = os.path.join(test_dir, filename)
        if not os.path.exists(video_path):
            logger.warning(f"Video file {filename} not found in {test_dir}. Skipping.")
            continue

        logger.info(f"Processing evaluation for: {filename} (Truth: {true_label})")
        
        try:
            # 1. Preprocessing (downscaling & metadata extraction)
            # Simulating main.py background task worker processing flow
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_eval")
            os.makedirs(temp_dir, exist_ok=True)
            downscaled_path = os.path.join(temp_dir, f"downscaled_{filename}")
            
            # extract metadata
            metadata = preprocessing.get_video_metadata(video_path)
            
            # downscale
            preprocessing.downscale_video(video_path, downscaled_path)
            
            # extract frames
            frames = preprocessing.extract_frames(downscaled_path)
            
            # cleanup downscaled temp file
            if os.path.exists(downscaled_path):
                os.remove(downscaled_path)

            # 2. Invoke multi-agent graph
            output = run_pipeline(frames, metadata)
            
            predicted_verdict = output.get("final_verdict", "UNCERTAIN")
            confidence = output.get("final_confidence", 0.0)

            logger.info(f"Result for {filename}: Predicted {predicted_verdict} (Conf: {confidence*100:.1f}%)")
            
            results.append({
                "filename": filename,
                "truth": true_label,
                "prediction": predicted_verdict,
                "confidence": confidence
            })

            # Calculate classification metrics (ignoring UNCERTAIN for clean binary check, or marking it as error)
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

    # Output metrics summary
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
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VeriFrame multi-agent pipeline.")
    parser.add_argument("--dir", default="test_data", help="Directory containing test videos")
    parser.add_argument("--labels", default="test_labels.json", help="Path to JSON file containing file-to-label map")
    
    args = parser.parse_args()
    
    # Check for placeholder file creation guidance if run directly with no data
    if not os.path.exists(args.dir) or not os.path.exists(args.labels):
        print("\n" + "!"*60)
        print("💡 HOW TO RUN VERIFRAME PIPELINE EVALUATION:")
        print("1. Create a test directory with video samples (e.g. 'backend/test_data/').")
        print("2. Create a JSON labels file (e.g. 'backend/test_labels.json') containing:")
        print('   {\n     "sample1.mp4": "AUTHENTIC",\n     "sample2.mp4": "MANIPULATED"\n   }')
        print(f"3. Run: python evaluate_pipeline.py --dir {args.dir} --labels {args.labels}")
        print("!"*60 + "\n")
    else:
        run_evaluation(args.dir, args.labels)
