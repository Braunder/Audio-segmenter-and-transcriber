import torch
import torchaudio
from pydub import AudioSegment
import os
import numpy as np
import whisper 
from tqdm import tqdm

# ======================
# КОНФИГУРАЦИЯ
# ======================
INPUT_AUDIO = "audio_file.wav"        # Входной аудиофайл
OUTPUT_DIR = "tts_dataset"   # Папка для результатов
LANGUAGE = "ru"              # Язык распознавания
SAMPLE_RATE = 16000          # Частота дискретизации
VAD_THRESHOLD = 0.9         # Порог VAD (0.1-0.9)
MIN_SPEECH_DURATION = 1500    # Мин. длительность речи (мс)
MIN_SILENCE_DURATION = 300   # Мин. тишина между сегментами (мс)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Устройство обработки
WHISPER_MODEL = "base"       # Модель Whisper: tiny, base, small, medium, large

# Создаем папку для результатов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# 1. ПОДГОТОВКА АУДИО
# ======================
def convert_to_wav(input_file):
    """Конвертирует аудио в нужный формат"""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)
    wav_path = os.path.join(OUTPUT_DIR, "converted.wav")
    audio.export(wav_path, format="wav")
    return wav_path

print("🔊 Конвертация аудио...")
if not INPUT_AUDIO.endswith(".wav"):
    audio_path = convert_to_wav(INPUT_AUDIO)
else:
    audio_path = INPUT_AUDIO

# ======================
# 2. ОБНАРУЖЕНИЕ РЕЧИ (VAD)
# ======================
print("🔍 Обнаружение речевых сегментов...")
print(DEVICE)
# Загрузка модели VAD
model_vad, utils_vad = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, _, _) = utils_vad

# Чтение аудио
wav = read_audio(audio_path, sampling_rate=SAMPLE_RATE)

# Получение временных меток речи
speech_timestamps = get_speech_timestamps(
    wav,
    model_vad,
    sampling_rate=SAMPLE_RATE,
    threshold=VAD_THRESHOLD,
    min_speech_duration_ms=MIN_SPEECH_DURATION,
    min_silence_duration_ms=MIN_SILENCE_DURATION
)

# ======================
# 3. НАРЕЗКА АУДИО
# ======================
print("✂️ Нарезка аудио на сегменты...")
audio = AudioSegment.from_wav(audio_path)
segments = []

for i, ts in enumerate(tqdm(speech_timestamps)):
    start_ms = ts["start"] / SAMPLE_RATE * 1000
    end_ms = ts["end"] / SAMPLE_RATE * 1000
    duration_ms = end_ms - start_ms
    
    # Пропускаем слишком короткие сегменты
    if duration_ms < MIN_SPEECH_DURATION:
        continue
        
    segment = audio[start_ms:end_ms]
    segment_path = os.path.join(OUTPUT_DIR, f"segment_{i:04d}.wav")
    segment.export(segment_path, format="wav")
    segments.append(segment_path)

# ======================
# 4. ТРАНСКРИПЦИЯ С ПОМОЩЬЮ WHISPER
# ======================
print("📝 Загрузка модели Whisper...")
model = whisper.load_model(WHISPER_MODEL).to(DEVICE)
print(f"✅ Модель {WHISPER_MODEL} загружена на {DEVICE}")

print("🔤 Распознавание речи...")
transcriptions = []
failed_segments = []

for segment_path in tqdm(segments):
    try:
        # Транскрибация с помощью Whisper
        result = model.transcribe(
            segment_path,
            language=LANGUAGE,
            fp16=(DEVICE == "cuda")
        )
        text = result['text'].strip()
        transcriptions.append(text)
        
        # Сохранение текста
        with open(segment_path.replace(".wav", ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
            
    except Exception as e:
        print(f"Ошибка при обработке {segment_path}: {str(e)}")
        failed_segments.append(segment_path)

# Удаление неудачных сегментов
for path in failed_segments:
    if path in segments:
        segments.remove(path)
        txt_path = path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
        os.remove(path)

# ======================
# 5. СОЗДАНИЕ METADATA
# ======================
print("📁 Создание файла метаданных...")
metadata = []
for segment_path in segments:
    txt_path = segment_path.replace(".wav", ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        rel_path = os.path.basename(segment_path)
        metadata.append(f"{rel_path}|{text}")

# Сохранение metadata.csv
metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata))

print(f"✅ Готово! Создано {len(metadata)} сегментов")
print(f"Файл метаданных: {metadata_path}")