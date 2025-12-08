If you closed Terminal and came back:

1) cd ~/Documents/whisper_sota_project
2) source venv/bin/activate

Run the transcription script:
python scripts/run_whisper.py --audio_root audio --model_size small

It creates: results/predictions_raw.csv

Save as predictions_with_ref.csv

Evaluate WER and inspect errors:
cd ~/Documents/whisper_sota_project
source venv/bin/activate   # if not already
python scripts/evaluate_wer.py

Git commands:

git init
git status

git add .
git commit -m "Initial commit: whisper project scripts"

git remote add origin https://github.com/Dheeraj-Rex/whisper-project.git
git branch -M main
git push -u origin main

