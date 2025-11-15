# fillyBounce

**FillyBounce** is a real-time jump counter that tracks jumps of **Filian (VTuber)** using computer vision and machine learning.  
It was inspired by Filian’s idea to attempt a **world record for pogo-stick jumping** during a stream.

![2025-11-12 18 55 50](https://github.com/user-attachments/assets/d2640a5e-476b-40a4-adea-b22ec7b430e3)

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.13-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-vision-red)
![YOLOv11](https://img.shields.io/badge/model-YOLOv11-yellow)


---

## ✨ Features
- 🎥 **Real-Time Twitch Stream Capture** – Connect directly to Twitch streams and count jumps live  
- 🤖 **Automatic Jump Detection** – Uses a YOLOv11-trained model to detect and track jumping movements  
- 📊 **Jump Statistics** – Displays total jump count, jumps per second, and elapsed time  
- ⚙️ **Configurable Settings** – Change Twitch channels, toggle preview.

---

## 🧠 Technology Stack

| Component | Purpose |
|------------|----------|
| **Python 3.13.8** | Core language |
| **OpenCV-Python** | Video processing and frame manipulation |
| **Ultralytics YOLOv11** | Custom-trained model for detecting Filian’s movements |
| **Streamlink** | Twitch stream capture and management |

---

## 🧩 Prerequisites
- Python 3.13

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kilven24/fillyBounce.git
   cd fillyBounce
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Select a capture mode:**
   - **Twitch Capture**: Stream from a live Twitch channel
   - **Process Recorded Video**: Analyze a video file
   - **OBS Capture**: Capture from OBS virtual camera

3. **Configure options:**
   - Click "Options" to adjust settings
   - Enable/disable frame preview
   - Change Twitch channel
   - Option to select a lightweight model for faster peformance at reduced precision

**Getting comercial breaks during twitch capture?**:
If you have turbo or subscription to channel, you can add your OAuth token into the config file.
You can get your token following instructions here. https://streamlink.github.io/cli/plugins/twitch.html









