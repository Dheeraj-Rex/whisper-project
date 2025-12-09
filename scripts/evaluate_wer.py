import csv
from collections import defaultdict
from jiwer import wer

def normalize(text: str) -> str:
    # simple normalization
    return text.lower().strip()

def main():
    # Read the correct transcripts (ground truth)
    correct_csv = "results/correct_transcript.csv"
    predictions_csv = "results/predictions_raw.csv"
    
    # Build a dictionary of ground truth by speaker_num
    ground_truth = {}
    with open(correct_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("speaker_num") and row.get("actual_transcript"):
                speaker_num = row["speaker_num"]
                ground_truth[speaker_num] = {
                    "reference_transcript": row["actual_transcript"],
                    "speaker": row["speaker"],
                    "file_name": row["file_name"]
                }
    
    print(f"Loaded {len(ground_truth)} ground truth transcripts")
    
    # Read predictions and match with ground truth
    matched_rows = []
    unmatched_predictions = []
    
    with open(predictions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_num = row.get("speaker_num", "").strip()
            predicted = row.get("predicted_transcript", "").strip()
            
            # Skip empty rows
            if not speaker_num or not predicted:
                continue
            
            # Match with ground truth
            if speaker_num in ground_truth:
                matched_row = {
                    "file_name": row["file_name"],
                    "env": row["env"],
                    "noise_type": row["noise_type"],
                    "speaker": row["speaker"],
                    "speaker_num": speaker_num,
                    "reference_transcript": ground_truth[speaker_num]["reference_transcript"],
                    "predicted_transcript": predicted
                }
                matched_rows.append(matched_row)
            else:
                unmatched_predictions.append(row["file_name"])
    
    if not matched_rows:
        print("No matching rows found between correct_transcript.csv and predictions_raw.csv")
        return
    
    print(f"Matched {len(matched_rows)} predictions with ground truth")
    if unmatched_predictions:
        print(f"Warning: {len(unmatched_predictions)} predictions could not be matched")
    
    # Overall WER
    all_refs = [normalize(r["reference_transcript"]) for r in matched_rows]
    all_hyps = [normalize(r["predicted_transcript"]) for r in matched_rows]
    
    overall_wer = wer(all_refs, all_hyps)
    print(f"\nOverall WER: {overall_wer:.3f}")
    
    # WER by environment (clean vs noisy)
    by_env = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in matched_rows:
        env = r.get("env", "unknown")
        by_env[env]["refs"].append(normalize(r["reference_transcript"]))
        by_env[env]["hyps"].append(normalize(r["predicted_transcript"]))
    
    print("\nWER by environment:")
    for env, data in sorted(by_env.items()):
        env_wer = wer(data["refs"], data["hyps"])
        count = len(data["refs"])
        print(f"  {env:>15}: {env_wer:.3f} (n={count})")
    
    # WER by noise type
    by_noise = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in matched_rows:
        noise = r.get("noise_type", "unknown")
        by_noise[noise]["refs"].append(normalize(r["reference_transcript"]))
        by_noise[noise]["hyps"].append(normalize(r["predicted_transcript"]))
    
    print("\nWER by noise type:")
    for noise, data in sorted(by_noise.items()):
        noise_wer = wer(data["refs"], data["hyps"])
        count = len(data["refs"])
        print(f"  {noise:>15}: {noise_wer:.3f} (n={count})")
    
    # WER by speaker
    by_speaker = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in matched_rows:
        spk = r.get("speaker", "unknown")
        by_speaker[spk]["refs"].append(normalize(r["reference_transcript"]))
        by_speaker[spk]["hyps"].append(normalize(r["predicted_transcript"]))
    
    print("\nWER by speaker (top 10 worst):")
    speaker_wers = []
    for spk, data in by_speaker.items():
        spk_wer = wer(data["refs"], data["hyps"])
        speaker_wers.append((spk_wer, spk, len(data["refs"])))
    
    speaker_wers.sort(reverse=True)
    for spk_wer, spk, count in speaker_wers[:10]:
        print(f"  {spk:>8}: {spk_wer:.3f} (n={count})")
    
    # Find the worst examples
    print("\nTop 10 clips with largest errors:")
    def clip_wer(ref, hyp):
        return wer([normalize(ref)], [normalize(hyp)])
    
    error_list = []
    for r in matched_rows:
        w = clip_wer(r["reference_transcript"], r["predicted_transcript"])
        error_list.append((w, r))
    
    # Sort by WER descending and print top 10
    error_list.sort(key=lambda x: x[0], reverse=True)
    
    for w, r in error_list[:10]:
        print("\n" + "="*70)
        print(f"File: {r['file_name']}")
        print(f"Environment: {r['env']} | Noise: {r['noise_type']} | Speaker: {r['speaker']}")
        print(f"WER: {w:.3f}")
        print(f"REF: {r['reference_transcript']}")
        print(f"HYP: {r['predicted_transcript']}")

if __name__ == "__main__":
    main()
