# Audio_Recognition

Audio recognition using a CNN that recognizes 4 different words: **yes**, **no**, **stop**, **go** with an accuracy of **94.63%**.

Built from scratch with PyTorch.

## Pipeline

```
Audio (.wav) → Padding/Truncating (1 sec) → Mel Spectrogram → CNN → Prediction
```

## How it works

### 1. Dataset

So this project uses the [Speech Commands Dataset](https://www.kaggle.com/datasets/yashdogra/speech-commands/data). It contains many different terms but I only used these ones: yes, no, stop and go (around 9500 samples in total).

In `dataset.py`, you'll find a class called `AudioDataset` that does all the heavy lifting: loading audio files, transforming them into spectrograms, and returning them as tensors.

### 2. Preprocessing

**Padding & Truncating:** Not all audio files are exactly 1 second long — some are shorter, some are longer. So to make sure every spectrogram has the same shape, I padded the short ones with silence (zeros) and truncated the long ones to exactly 16000 samples (= 1 sec at 16kHz).

**Spectrograms:** After that I transformed each audio file into a mel spectrogram using [librosa.feature.melspectrogram()](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html). The final shape for each spectrogram is `(1, 64, 32)` — 1 channel, 64 frequency bands, 32 time frames. Basically the CNN sees it as a small grayscale image.

### 3. CNN

```
Conv2d(1, 128) → ReLU → MaxPool2d(2,2)
Conv2d(128, 64) → ReLU → MaxPool2d(2,2)
Conv2d(64, 32)  → ReLU → MaxPool2d(2,2)
Flatten
Linear(384, 128) → ReLU
Linear(128, 32)  → ReLU
Linear(32, 4)    → Output
```

Quick explanation:
- **Conv2d**: a filter that slides over the spectrogram and detects different patterns (edges, shapes,...)
- **ReLU**: activation function, basically turns negative values to 0
- **MaxPool2d**: a 2×2 window that takes the max value, so we get a smaller image but we keep the important info
- **Linear layers**: takes the flattened result and classifies it into the 4 words

### 4. Microphone

Once the model is trained you can test it live with your own voice using `test_micro.py`. It records 1 second of audio from your mic, creates a spectrogram, and tells you what word you said.

## Usage

**Train:**
```bash
python train.py
```

**Test with your mic:**
```bash
python test_micro.py
```

## Results

| Epochs | Accuracy |
|--------|----------|
| 3      | 87.11%   |
| 20     | 94.63%   |

## Files

```
├── dataset.py      # AudioDataset class (loading, preprocessing, spectrograms)
├── train.py        # CNN model + training/testing
├── test_micro.py   # Real-time microphone prediction
```
