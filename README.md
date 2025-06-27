# Audio segmenter and transcriber

A tool for automatically segmenting audio files into speech fragments and transcribing them using Whisper.

## 📝 Requirements

- Python 3.10
- CUDA 12.1 (for GPU acceleration)
- NVIDIA drivers installed

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repository.git
cd your-repository
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Make sure CUDA works correctly:

```bash
nvidia-smi
```

##🚀 Usage:

1. Place the input audio file in the root directory

2. Edit the configuration in the script:

```python
INPUT_AUDIO = "3.wav" # Input audio file
OUTPUT_DIR = "tts_dataset" # Folder for results
LANGUAGE = "ru" # Recognition language
VAD_THRESHOLD = 0.9 # VAD threshold (0.1-0.9)
```

3. Run the script:

```bash
python main.py
```

## ⚙️ Configuration
### Main parameters:

```MIN_SPEECH_DURATION - minimum duration of a speech segment (ms)

MIN_SILENCE_DURATION - minimum duration of silence between segments (ms)

WHISPER_MODEL - Whisper model (tiny, base, small, medium, large)
```

## 📂 Output data structure
### After processing, the OUTPUT_DIR folder will contain:

1. Segmented WAV files

2. Text files with transcription

3. File metadata.csv in filename|text format

## ℹ️ Supported formats
Input formats: WAV, MP3, OGG, FLAC (automatically converted to WAV)

## ⚠️ Issues
*If you have issues with CUDA:*

*Make sure PyTorch and CUDA versions are compatible*

*Try reinstalling torch with the correct CUDA version:*

``bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
