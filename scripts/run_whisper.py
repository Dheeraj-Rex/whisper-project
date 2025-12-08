import os
import csv
import argparse
import whisper

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

def list_audio_files(audio_root):
    for filename in os.listdir(audio_root):
        lower = filename.lower()
        if lower.endswith(AUDIO_EXTENSIONS):
            yield os.path.join(audio_root, filename)

def parse_filename(filename):
    """
    Expecting: <env>_<rate>_<speaker>_<index>.wav
    Example: clean_normal_me_01.wav
    Returns: (env, rate, speaker, index)
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    # pad with 'unknown' if anything is missing
    parts += ["unknown"] * (4 - len(parts))
    env, rate, speaker, index = parts[:4]
    return env, rate, speaker, index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", type=str, default="audio",
                        help="Folder containing audio clips")
    parser.add_argument("--model_size", type=str, default="small",
                        help="Whisper model size: tiny, base, small, medium, large")
    parser.add_argument("--output_csv", type=str, default="results/predictions_raw.csv",
                        help="Where to save raw predictions CSV")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print(f"Loading Whisper model '{args.model_size}'...")
    model = whisper.load_model(args.model_size)
    print("Model loaded.")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "env",
            "rate",
            "speaker",
            "index",
            "predicted_transcript"
        ])

        for path in list_audio_files(args.audio_root):
            rel_name = os.path.basename(path)
            env, rate, speaker, index = parse_filename(rel_name)
            print(f"Transcribing {rel_name}  (env={env}, rate={rate}, speaker={speaker})")

            # All clips are English, so we set language="en"
            result = model.transcribe(path, language="en")
            predicted = result["text"].strip()

            writer.writerow([
                rel_name,
                env,
                rate,
                speaker,
                index,
                predicted
            ])

    print(f"\nDone. Predictions saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
