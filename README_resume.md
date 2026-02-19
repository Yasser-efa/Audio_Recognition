# Audio Recognition

A CNN that recognizes 4 spoken words: **yes**, **no**, **stop**, **go** — with an accuracy of **94.63%**.

Built from scratch with PyTorch.

## How it works

```
Audio (.wav) → Padding/Truncating (1 sec) → Mel Spectrogram → CNN → Prediction
```

The CNN takes a spectrogram of shape `(1, 64, 32)` and classifies it into one of the 4 words.

## Architecture

```
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
Flatten → Linear → ReLU → Linear → ReLU → Output (4 classes)
```

## Usage

**Train the model:**
```bash
python train.py
```

**Test with your microphone:**
```bash
python test_micro.py
```

## Dataset

[Speech Commands Dataset](https://www.kaggle.com/datasets/yashdogra/speech-commands/data) — ~9500 samples across 4 classes.

## Results

| Epochs | Accuracy |
|--------|----------|
| 3      | 87.11%   |
| 20     | 94.63%   |
