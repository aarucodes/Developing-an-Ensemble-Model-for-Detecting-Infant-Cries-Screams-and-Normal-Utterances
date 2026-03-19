# Infant Cry Classification System

A Jupyter notebook implementing **YAMNet + Wav2Vec2 ensemble** for **cry/scream/normal** infant audio classification.


##  Project Overview

**Current Results**: 38% accuracy (baseline ensemble without training)  
**Target**: 85-92% accuracy with proper feature extraction + classifiers

**Approach**:
Audio → YAMNet embeddings (1024-dim) → MLP Classifier (target)
↳ Wav2Vec2 SUPERB logits → Fine-tuning (target)
↳ Weighted ensemble → Production model

##  Dataset
C:/Users/Desktop/crying/data/
├── cry/ (63 samples)
├── scream/ (72 samples)
└── norm/ (68 samples)


**Config**:
- Sample Rate: 16kHz
- Classes: `['cry', 'scream', 'norm']`
- Class Weights: `{cry: 1.07, norm: 0.99, scream: 0.94}`


###  Working Components
 Data loading (librosa/soundfile)
 YAMNet model loading
 Wav2Vec2 SUPERB loading
 Ensemble prediction pipeline
 Class weight computation
 Confusion matrix visualization

 ## Current Results (38% Accuracy)
    precision   recall   f1-score   support
cry      0.35     0.70       0.47        10
scream   0.00     0.00       0.00        17
norm     0.42     0.62       0.50        13

accuracy 0.38                            40


