import os
import csv
import argparse
import whisper

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

def list_audio_files(audio_root):
    for root, dirs, files in os.walk(audio_root):
        for filename in files:
            lower = filename.lower()
            if lower.endswith(AUDIO_EXTENSIONS):
                yield os.path.join(root, filename)

def parse_filename(filepath):
    """
    Extract metadata from filepath.
    Examples:
      audio/clean/sp01.wav -> (clean, clean, sp01, 01)
      audio/airport_10dB/sp01_airport_sn10.wav -> (airport_10dB, airport, sp01, 01)
    Returns: (env, noise_type, speaker, speaker_num)
    """
    # Get directory name as environment (e.g., "clean", "airport_10dB")
    env = os.path.basename(os.path.dirname(filepath))
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    # Extract speaker (e.g., "sp01" -> "sp01")
    if filename.startswith("sp"):
        speaker = filename.split("_")[0]  # e.g., "sp01"
        speaker_num = speaker[2:]  # e.g., "01"
    else:
        speaker = "unknown"
        speaker_num = "unknown"
    
    # Extract noise type from env (e.g., "airport_10dB" -> "airport")
    noise_type = env.replace("_10dB", "")
    
    return env, noise_type, speaker, speaker_num

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
            "noise_type",
            "speaker",
            "speaker_num",
            "predicted_transcript"
        ])

        for path in list_audio_files(args.audio_root):
            rel_name = os.path.basename(path)
            env, noise_type, speaker, speaker_num = parse_filename(path)
            print(f"Transcribing {rel_name}  (env={env}, noise_type={noise_type}, speaker={speaker})")

            # All clips are English, so we set language="en"
            result = model.transcribe(path, language="en")
            predicted = result["text"].strip()

            writer.writerow([
                rel_name,
                env,
                noise_type,
                speaker,
                speaker_num,
                predicted
            ])

    print(f"\nDone. Predictions saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
