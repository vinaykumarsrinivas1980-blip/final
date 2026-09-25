"""
train_stop_keyword.py - Train a local openWakeWord ONNX model for the 'stop' keyword.
Generates synthetic dataset using Edge-TTS & pyttsx3, extracts openWakeWord feature embeddings,
trains a PyTorch binary classifier, and exports a production-ready stop.onnx model.
"""

import os
import sys
import time
import math
import wave
import random
import asyncio
import numpy as np
import soundfile as sf
import scipy.signal as signal

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "training_data_stop")
POS_DIR = os.path.join(DATA_DIR, "positive")
NEG_DIR = os.path.join(DATA_DIR, "negative")

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000  # 2.0s window produces exactly (16, 96) embedding frames

POS_PHRASES = [
    "stop",
    "stop now",
    "please stop",
    "stop talking",
    "just stop",
    "stop please",
    "ruko",
    "chup",
    "be quiet",
    "halt",
]

# Words that sound similar or represent speech during TTS playback that should NOT trigger
NEG_PHRASES = [
    "spark",
    "start",
    "spot",
    "step",
    "store",
    "stock",
    "stay",
    "state",
    "shop",
    "hello",
    "hey jarvis",
    "alexa",
    "continue",
    "okay",
    "what is the time",
    "how are you",
    "tell me a story",
    "I am Spark born from bolts",
    "Welcome to SJC Institute of Technology",
    "artificial intelligence and robotics",
    "one two three four five",
    "the weather is clear today",
]

VOICES = [
    "en-US-AvaNeural",
    "en-US-BrianNeural",
    "en-US-EmmaNeural",
    "en-US-ChristopherNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "hi-IN-SwaraNeural",
    "hi-IN-MadhurNeural",
]

RATES = ["-15%", "+0%", "+15%"]
PITCHES = ["-10Hz", "+0Hz", "+10Hz"]


async def generate_edge_tts_clip(text, voice, rate, pitch, out_path, sem):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        return True
    async with sem:
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(out_path)
            return True
        except Exception as err:
            return False


async def generate_dataset():
    os.makedirs(POS_DIR, exist_ok=True)
    os.makedirs(NEG_DIR, exist_ok=True)

    sem = asyncio.Semaphore(6)
    tasks = []

    print("🎙️  [1/4] Generating synthetic audio samples via Edge-TTS...")
    idx = 0
    # Positive clips
    for phrase in POS_PHRASES:
        for voice in VOICES:
            for rate in RATES:
                pitch = random.choice(PITCHES)
                out_file = os.path.join(POS_DIR, f"pos_{idx}_{voice}_{rate}.mp3")
                tasks.append(generate_edge_tts_clip(phrase, voice, rate, pitch, out_file, sem))
                idx += 1

    # Negative clips
    n_idx = 0
    for phrase in NEG_PHRASES:
        for voice in random.sample(VOICES, min(6, len(VOICES))):
            for rate in ["-10%", "+0%", "+10%"]:
                pitch = random.choice(PITCHES)
                out_file = os.path.join(NEG_DIR, f"neg_{n_idx}_{voice}_{rate}.mp3")
                tasks.append(generate_edge_tts_clip(phrase, voice, rate, pitch, out_file, sem))
                n_idx += 1

    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = sum(1 for r in results if r is True)
    print(f"✅ Generated/verified {successful} synthetic audio clips.")


def load_and_resample(file_path):
    try:
        data, sr = sf.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            target_len = int(round(len(data) * SAMPLE_RATE / float(sr)))
            data = signal.resample(data, target_len)
        # normalize
        max_val = np.max(np.abs(data))
        if max_val > 1e-4:
            data = data / max_val
        return data.astype(np.float32)
    except Exception:
        return None


def augment_clip_into_window(clip, offset_fraction, gain=1.0, noise_snr=None):
    """Places audio clip into a 2.0s (32000 samples) window at specified offset with gain and noise."""
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    clip_len = len(clip)

    if clip_len >= WINDOW_SAMPLES:
        window[:] = clip[:WINDOW_SAMPLES] * gain
    else:
        max_offset = WINDOW_SAMPLES - clip_len
        offset = int(round(max_offset * offset_fraction))
        window[offset:offset + clip_len] = clip * gain

    if noise_snr is not None and noise_snr > 0:
        signal_power = np.mean(window ** 2) + 1e-9
        noise_power = signal_power / (10 ** (noise_snr / 10.0))
        noise = np.random.normal(0, math.sqrt(noise_power), WINDOW_SAMPLES).astype(np.float32)
        window = window + noise

    # Clip and convert to int16 format expected by openwakeword
    window = np.clip(window * 32767.0, -32768, 32767).astype(np.int16)
    return window


def build_augmented_dataset():
    print("🧪 [2/4] Augmenting audio and extracting openWakeWord embeddings...")
    from openwakeword.model import Model
    oww = Model(inference_framework="onnx")

    pos_files = [os.path.join(POS_DIR, f) for f in os.listdir(POS_DIR) if f.endswith(('.mp3', '.wav'))]
    neg_files = [os.path.join(NEG_DIR, f) for f in os.listdir(NEG_DIR) if f.endswith(('.mp3', '.wav'))]

    audio_samples = []
    labels = []

    # 1. Augment Positives
    print(f"  • Processing {len(pos_files)} positive source clips...")
    for pf in pos_files:
        clip = load_and_resample(pf)
        if clip is None or len(clip) < 1600:  # skip < 0.1s
            continue
        # Multiple time-shift offsets, volume scaling, and SNR levels
        for off in [0.05, 0.25, 0.50, 0.75]:
            for gain in [0.6, 1.0, 1.3]:
                snr = random.choice([None, 20.0, 15.0])
                w = augment_clip_into_window(clip, offset_fraction=off, gain=gain, noise_snr=snr)
                audio_samples.append(w)
                labels.append(1.0)

    # 2. Augment Negatives
    print(f"  • Processing {len(neg_files)} negative source clips...")
    for nf in neg_files:
        clip = load_and_resample(nf)
        if clip is None or len(clip) < 1600:
            continue
        for off in [0.1, 0.4, 0.7]:
            gain = random.choice([0.7, 1.0])
            snr = random.choice([None, 25.0])
            w = augment_clip_into_window(clip, offset_fraction=off, gain=gain, noise_snr=snr)
            audio_samples.append(w)
            labels.append(0.0)

    # 3. Add Pure Silence, White Noise, and Ambient Room Noise negatives
    for _ in range(250):
        # Silence with slight hiss
        noise_level = random.uniform(50.0, 400.0)
        pure_noise = np.random.normal(0, noise_level, WINDOW_SAMPLES).astype(np.int16)
        audio_samples.append(pure_noise)
        labels.append(0.0)

    audio_array = np.array(audio_samples, dtype=np.int16)
    labels_array = np.array(labels, dtype=np.float32)

    print(f"  • Total audio windows: {len(audio_array)} (Positives: {int(np.sum(labels_array))}, Negatives: {int(len(labels_array) - np.sum(labels_array))})")
    print("  • Passing audio windows through openWakeWord feature preprocessor...")

    cache_emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    cache_lbl_path = os.path.join(DATA_DIR, "labels.npy")

    if os.path.exists(cache_emb_path) and os.path.exists(cache_lbl_path):
        print("  • Found cached embeddings on disk. Loading...")
        embeddings = np.load(cache_emb_path)
        labels_array = np.load(cache_lbl_path)
        print(f"✅ Loaded cached embedding tensors: shape {embeddings.shape}")
        return embeddings, labels_array

    # Extract (16, 96) feature embeddings in batches
    embeddings = oww.preprocessor.embed_clips(audio_array, batch_size=128)
    np.save(cache_emb_path, embeddings)
    np.save(cache_lbl_path, labels_array)
    print(f"✅ Extracted and cached embedding tensors: shape {embeddings.shape}")

    return embeddings, labels_array


class StopWakeWordModel(nn.Module):
    def __init__(self, mean=0.0, std=1.0, hidden_dim=64):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean), dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(float(std), dtype=torch.float32), requires_grad=False)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 96, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x input shape: (batch, 16, 96)
        x = (x - self.mean) / (self.std + 1e-6)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def train_model(embeddings, labels):
    print("🧠 [3/4] Training PyTorch classifier network...")
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    split = int(0.85 * n_samples)

    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_val, y_val = embeddings[val_idx], labels[val_idx]

    # Compute dataset mean and std for normalizer
    mean_val = float(np.mean(X_train))
    std_val = float(np.std(X_train))
    print(f"  • Dataset statistics: mean={mean_val:.4f}, std={std_val:.4f}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = StopWakeWordModel(mean=mean_val, std=std_val, hidden_dim=64)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 35
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vpreds = model(vx)
                vloss = criterion(vpreds, vy)
                val_loss += vloss.item() * len(vy)
                predicted_labels = (vpreds >= 0.5).float()
                correct += (predicted_labels == vy).sum().item()
                total += len(vy)
        val_loss /= len(val_dataset)
        val_acc = correct / total if total > 0 else 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  • Epoch {epoch:02d}/{epochs}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc*100:.2f}%")

    if best_weights:
        model.load_state_dict(best_weights)

    return model


def export_and_verify(model):
    print("📦 [4/4] Exporting to ONNX and verifying with openWakeWord...")
    output_onnx_path = os.path.join(BASE_DIR, "stop.onnx")

    model.eval()
    dummy_input = torch.zeros((1, 16, 96), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["onnx::Flatten_0"],
        output_names=["output"],
        dynamic_axes={
            "onnx::Flatten_0": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"✅ Successfully exported ONNX model to: {output_onnx_path}")
    print(f"   Model size: {os.path.getsize(output_onnx_path) / 1024:.1f} KB")

    # Verification using openwakeword.model.Model
    from openwakeword.model import Model
    test_model = Model(wakeword_models=[output_onnx_path], inference_framework="onnx")
    print(f"✅ Successfully loaded '{output_onnx_path}' into openWakeWord engine.")
    print(f"   Registered model name: {list(test_model.models.keys())}")

    # Test rejection on zeros / silence
    zero_chunk = np.zeros(1280, dtype=np.int16)
    res_zero = test_model.predict(zero_chunk)
    print(f"  • Baseline silence score: {res_zero}")

    # Test rejection on test_spark.wav if present
    spark_wav = os.path.join(BASE_DIR, "test_spark.wav")
    if os.path.exists(spark_wav):
        scores = [p.get('stop', 0.0) for p in test_model.predict_clip(spark_wav)]
        max_spark = max(scores) if scores else 0.0
        print(f"  • False-positive check on test_spark.wav: max score={max_spark:.4f} (Threshold 0.55)")
        assert max_spark < 0.40, f"False positive alert on spark clip! Score={max_spark}"

    print("\n🎉 Training, export, and verification completed successfully!")
    print("Model 'stop.onnx' is ready to use for instant voice barge-in.")


async def main():
    start_time = time.time()
    await generate_dataset()
    embeddings, labels = build_augmented_dataset()
    model = train_model(embeddings, labels)
    export_and_verify(model)
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
