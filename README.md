# Whisper ASR Evaluation Project

This project evaluates the performance of OpenAI's Whisper automatic speech recognition (ASR) model under varying acoustic conditions to understand how background noise affects transcription accuracy.

## Project Overview

The primary goal of this project is to assess Whisper's transcription accuracy across three distinct audio environments. We wanted to understand how real-world noise conditions impact the model's ability to accurately convert speech to text. The evaluation includes:

- **Clean audio**: Recordings with no background noise, serving as our baseline
- **Airport noise (10dB SNR)**: Audio with ambient airport terminal sounds added
- **Car noise (10dB SNR)**: Recordings mixed with vehicle interior noise

The SNR (Signal-to-Noise Ratio) of 10dB represents a moderately noisy environment where the speech signal is approximately 10 times more powerful than the background noise.

## Dataset

Our dataset consists of:
- 30 individual speakers (labeled sp01 through sp30)
- Each speaker recorded reading an identical sentence
- Three versions of each recording: clean, with airport noise, and with car noise
- Total dataset size: 90 audio files (30 speakers × 3 conditions)

This controlled approach allows us to isolate the effect of noise type on transcription accuracy while keeping other variables constant.

## Project Structure

```
whisper-project/
├── audio/                      # Audio files organized by condition
├── results/
│   ├── correct_transcript.csv  # Ground truth transcripts
│   └── predictions_raw.csv     # Whisper model predictions
├── scripts/
│   ├── run_whisper.py         # Script for generating transcriptions
│   └── evaluate_wer.py        # Script for calculating WER metrics
└── requirements.txt           # Python package dependencies
```

## Installation

### Step 1: Navigate to Project Directory

```bash
cd ~/Documents/git/whisper-project
```

### Step 2: Create Virtual Environment

Setting up a virtual environment ensures that project dependencies are isolated from your system Python installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following required packages:
- `openai-whisper`: The Whisper ASR model implementation
- `jiwer`: Library for calculating Word Error Rate metrics
- `ffmpeg`: Audio file processing utility
- `certifi`: SSL certificate verification

## Usage

### Generating Transcriptions

To run Whisper on your audio files and generate predictions:

```bash
python scripts/run_whisper.py --audio_root audio --model_size small
```

**Command-line Arguments:**
- `--audio_root`: Directory containing audio files (default: `audio`)
- `--model_size`: Whisper model variant to use - options include `tiny`, `base`, `small`, `medium`, or `large` (default: `small`)

The script processes all audio files and saves the predictions to `results/predictions_raw.csv`. Note that larger models provide better accuracy but require significantly more processing time.

### Evaluating Performance

To calculate Word Error Rate (WER) metrics:

```bash
python scripts/evaluate_wer.py
```

**Input Requirements:**
- `results/correct_transcript.csv`: Reference transcripts for comparison
- `results/predictions_raw.csv`: Whisper-generated predictions

**Output Metrics:**
- Overall WER across all audio samples
- WER breakdown by environment (clean, airport, car)
- WER analysis by noise type
- Per-speaker WER analysis (highlighting the 10 speakers with highest error rates)
- Top 10 individual clips with the largest transcription errors

## Results

Using the Whisper Small model, we obtained the following performance metrics:

| Environment | WER | Sample Count |
|-------------|-----|--------------|
| Clean | 0.069 | 30 |
| Airport (10dB) | 0.120 | 30 |
| Car (10dB) | 0.197 | 30 |
| **Overall** | **0.129** | **90** |

### Analysis of Findings

The results reveal several important patterns in Whisper's performance:

- **Clean audio performance**: Achieved the lowest error rate at 6.9%, demonstrating that Whisper performs well under ideal acoustic conditions
- **Impact of car noise**: Car noise produced the highest degradation in performance, with WER reaching 19.7%. This suggests that low-frequency, continuous noise patterns pose significant challenges for the model
- **Airport noise effects**: Airport noise resulted in an intermediate error rate of 12.0%, indicating moderate performance degradation
- **Overall accuracy**: Across all conditions, the model achieved an average WER of 12.9%

These findings suggest that noise type matters significantly. Car noise, which tends to be more continuous and concentrated in lower frequencies, interferes more with speech recognition compared to the more varied acoustic patterns of airport environments.

## Reactivating the Virtual Environment

If you close your terminal session and need to resume work on the project:

```bash
cd ~/Documents/git/whisper-project
source .venv/bin/activate
```

## Dependencies

Complete list of project dependencies can be found in `requirements.txt`:
```
ffmpeg
openai-whisper
jiwer
certifi
```

## Project Notes

This project was developed for educational purposes as part of a course assignment. The evaluation methodology and metrics follow standard practices in speech recognition research.

## Contact

For questions regarding this project, please refer to the course documentation or contact the course instructor.
