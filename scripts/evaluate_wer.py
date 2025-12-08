import csv
from collections import defaultdict
from jiwer import wer

def normalize(text: str) -> str:
    # simple normalization
    return text.lower().strip()

def main():
    input_csv = "results/predictions_with_ref.csv"

    rows = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip rows without reference transcript filled in
            if not row.get("reference_transcript"):
                continue
            rows.append(row)

    if not rows:
        print("No rows with reference_transcript found. Did you fill them in?")
        return

    # overall WER
    all_refs = [normalize(r["reference_transcript"]) for r in rows]
    all_hyps = [normalize(r["predicted_transcript"]) for r in rows]

    overall_wer = wer(all_refs, all_hyps)
    print(f"Overall WER: {overall_wer:.3f}")

    # WER by environment (clean vs noisy)
    by_env = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in rows:
        env = r.get("env", "unknown")
        by_env[env]["refs"].append(normalize(r["reference_transcript"]))
        by_env[env]["hyps"].append(normalize(r["predicted_transcript"]))

    print("\nWER by environment:")
    for env, data in by_env.items():
        env_wer = wer(data["refs"], data["hyps"])
        print(f"  {env:>6}: {env_wer:.3f}")

    # WER by speaking rate (normal vs fast)
    by_rate = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in rows:
        rate = r.get("rate", "unknown")
        by_rate[rate]["refs"].append(normalize(r["reference_transcript"]))
        by_rate[rate]["hyps"].append(normalize(r["predicted_transcript"]))

    print("\nWER by speaking rate:")
    for rate, data in by_rate.items():
        rate_wer = wer(data["refs"], data["hyps"])
        print(f"  {rate:>6}: {rate_wer:.3f}")

    # WER by speaker (for accent differences)
    by_speaker = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in rows:
        spk = r.get("speaker", "unknown")
        by_speaker[spk]["refs"].append(normalize(r["reference_transcript"]))
        by_speaker[spk]["hyps"].append(normalize(r["predicted_transcript"]))

    print("\nWER by speaker:")
    for spk, data in by_speaker.items():
        spk_wer = wer(data["refs"], data["hyps"])
        print(f"  {spk:>8}: {spk_wer:.3f}")

    # find some of the worst examples (possible hallucinations / big errors)
    print("\nSome clips with large errors:")
    def clip_wer(ref, hyp):
        return wer([normalize(ref)], [normalize(hyp)])

    error_list = []
    for r in rows:
        w = clip_wer(r["reference_transcript"], r["predicted_transcript"])
        error_list.append((w, r))

    # sort by WER descending and print top 10
    error_list.sort(key=lambda x: x[0], reverse=True)

    for w, r in error_list[:10]:
        print("\n----------")
        print(f"File: {r['file_name']}  (env={r['env']}, rate={r['rate']}, speaker={r['speaker']})")
        print(f"WER: {w:.3f}")
        print(f"REF: {r['reference_transcript']}")
        print(f"HYP: {r['predicted_transcript']}")

if __name__ == "__main__":
    main()
