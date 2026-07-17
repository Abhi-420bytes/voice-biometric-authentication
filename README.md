# 🎙️ Voice Biometric Authentication System

> AI-powered Voice Authentication using Deep Learning for Speaker Verification, Replay Attack Detection, and Deepfake Voice Detection.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Librosa](https://img.shields.io/badge/Librosa-AudioProcessing-green)
![OpenCV](https://img.shields.io/badge/ComputerVision-AI-orange)
![License](https://img.shields.io/badge/License-MIT-success)

---

# 📖 Overview

Passwords and PIN-based authentication systems are vulnerable to theft, replay attacks, and phishing. Voice biometrics provide a natural and convenient authentication mechanism, but modern AI-generated voices and replay attacks have introduced new security challenges.

This project presents an **AI-powered Voice Biometric Authentication System** that combines **speaker verification** with **deepfake and spoof detection**. By integrating advanced deep learning models such as **TDNN**, **ECAPA-TDNN**, and **AntiSpoofNet**, the framework authenticates genuine users while detecting replayed or AI-generated speech.

The system supports both **text-dependent** and **text-independent** voice authentication, making it suitable for secure real-world deployments.

---

# 🎯 Objectives

- Develop a secure voice biometric authentication system
- Perform accurate speaker verification
- Detect replay attacks and spoofed speech
- Identify AI-generated deepfake voices
- Improve authentication reliability using score fusion
- Build a scalable AI-based voice authentication framework

---

# 🚀 Features

- Text-Dependent Speaker Verification
- Text-Independent Speaker Verification
- Replay Attack Detection
- Deepfake Voice Detection
- Audio Preprocessing Pipeline
- MFCC Feature Extraction
- Speaker Embedding Generation
- TDNN & ECAPA-TDNN Models
- AntiSpoofNet (ResNet)
- Score Fusion for Final Authentication
- Real-Time Authentication Pipeline

---

# 🏗️ System Architecture

```
Voice Input
      │
      ▼
Audio Preprocessing
      │
      ▼
Feature Extraction
(MFCC + Speaker Embeddings)
      │
      ▼
────────────────────────────
Speaker Verification
────────────────────────────

TDNN

ECAPA-TDNN

────────────────────────────
        │
        ▼
AntiSpoofNet
(Replay + Deepfake Detection)
        │
        ▼
Score Fusion
        │
        ▼
Authentication Result
(Genuine / Impostor)
```

---

# 📂 Project Structure

```
voice-biometric-authentication
│
├── dataset
│   ├── LibriSpeech
│   ├── ASVspoof2019
│   └── custom_samples
│
├── preprocessing
│   ├── noise_reduction.py
│   ├── vad.py
│   ├── normalization.py
│
├── feature_extraction
│   ├── mfcc.py
│   ├── embeddings.py
│
├── models
│   ├── tdnn.py
│   ├── ecapa_tdnn.py
│   ├── antispoofnet.py
│
├── inference
│   ├── authenticate.py
│
├── results
│
├── requirements.txt
│
└── README.md
```

---

# 🎵 Datasets

## LibriSpeech

Used for:

- Speaker Verification
- Speaker Embedding Training
- Genuine Voice Samples

---

## ASVspoof 2019

Used for:

- Replay Attack Detection
- Synthetic Speech Detection
- Deepfake Voice Detection

---

## Custom Voice Samples

Collected for:

- Enrollment
- Authentication
- Real-Time Evaluation

---

# 🎧 Audio Preprocessing

The audio pipeline performs:

- Sampling Rate Standardization
- Noise Reduction
- Silence Removal
- Voice Activity Detection (VAD)
- Loudness Normalization
- Audio Chunking
- Data Balancing

---

# 🎼 Feature Extraction

The system extracts:

- MFCC Features
- Speaker Embeddings
- Spectral Features
- Temporal Features

These representations are used for speaker verification and spoof detection.

---

# 🧠 Deep Learning Models

## TDNN

Time Delay Neural Network for text-dependent speaker verification.

### Advantages

- Robust temporal modeling
- Fast inference
- Accurate speaker representation

---

## ECAPA-TDNN

Enhanced TDNN architecture for text-independent speaker verification.

### Advantages

- Channel attention
- Improved speaker embeddings
- High recognition accuracy

---

## AntiSpoofNet (ResNet)

ResNet-based anti-spoofing model that detects:

- Replay attacks
- Synthetic speech
- AI-generated deepfake voices

---

# 🔄 Score Fusion

The authentication score combines outputs from:

- TDNN
- ECAPA-TDNN
- AntiSpoofNet

This fusion strategy improves overall authentication accuracy and reduces false acceptances.

---

# 📊 Evaluation Metrics

The system is evaluated using:

- Equal Error Rate (EER)
- Area Under Curve (AUC)
- True Acceptance Rate (TAR)
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Accuracy
- Precision
- Recall
- F1 Score

---

# ⚙️ Technology Stack

| Layer | Technology |
|---------|------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Audio Processing | Librosa |
| Feature Extraction | MFCC |
| Speaker Verification | TDNN, ECAPA-TDNN |
| Anti-Spoofing | AntiSpoofNet (ResNet) |
| Visualization | Matplotlib |
| Dataset | LibriSpeech, ASVspoof 2019 |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/voice-biometric-authentication.git

cd voice-biometric-authentication
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Train Speaker Verification Model

```bash
python train_tdnn.py
```

---

# ▶ Train Anti-Spoof Model

```bash
python train_antispoof.py
```

---

# ▶ Authenticate User

```bash
python authenticate.py --audio sample.wav
```

---

# 🔍 Output

The system returns:

- Speaker Identity
- Authentication Score
- Genuine / Impostor Decision
- Spoof Detection Result
- Deepfake Detection Result
- Confidence Score

---

# 🌍 Applications

- Banking Authentication
- Mobile Device Unlock
- Smart Home Devices
- Secure Login Systems
- Voice-Based Digital Identity
- Call Center Verification
- Healthcare Authentication
- Government e-Services
- Remote Workforce Access Control

---

# 🔮 Future Enhancements

- Transformer-based Speaker Models (WavLM, HuBERT)
- Real-Time Streaming Authentication
- Multi-Factor Biometric Authentication
- Mobile Deployment
- Federated Learning for Privacy
- Cross-Language Speaker Verification
- Edge AI Optimization
- Explainable AI for Voice Biometrics

---

# 🌟 Project Highlights

- Deep Learning-Based Voice Authentication
- Text-Dependent Verification
- Text-Independent Verification
- Replay Attack Detection
- AI Deepfake Detection
- Score Fusion Framework
- Secure Biometric Authentication
- Scalable AI Architecture

---

# 👨‍💻 Authors

**Challa Abhiram**

B.Tech Artificial Intelligence Engineering

Amrita School of Computing

Amrita Vishwa Vidyapeetham

---

**Akhil K.**

Amrita School of Computing

---

**Shilpita V.**

Amrita School of Computing

---

**Guide**

**Dr. Debanjali Bhattacharya**

Department of Computer Science & Engineering

Amrita School of Computing

Amrita Vishwa Vidyapeetham

---

# 🙏 Acknowledgement

We sincerely thank **Amrita Vishwa Vidyapeetham** and **Dr. Debanjali Bhattacharya** for their guidance and support throughout the development of this project. We also acknowledge the creators of the **LibriSpeech** and **ASVspoof 2019** datasets, whose publicly available resources enabled the training and evaluation of our speaker verification and anti-spoofing models.

---

# 📜 License

This project is intended for academic and research purposes.

Feel free to fork, improve, and cite the work.

⭐ If you find this project useful, please consider giving it a star!
