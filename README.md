# fillyBounce

fillyBounce is a real-time jump counter application that tracks jumps of Filian (VTuber) using computer vision and machine learning. Originally created after Filian expressed interest in attempting a world record for pogostick jumping during a stream.

## Features

- **Real-time Twitch Stream Capture**: Connect directly to Twitch streams and count jumps live
- **Automatic Jump Detection**: Uses YOLOv11 trained model to detect and track jumping movements
- **Jump Statistics**: Displays total jump count, jumps per second, and elapsed time

## Technology Stack

- **Python 3.13.8**
- **OpenCV-Python**: Video processing and frame manipulation
- **Ultralytics (YOLOv11)**: Custom-trained model for detection and tracking of Filian's face/head
- **Streamlink**: Twitch stream capture

## Prerequisites

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
   - Adjust detection parameters

**Getting twitch ads?**:
If you have turbo or subscription to channel, you can add your OAuth token into the config file.
You can get your token following instructions here. https://streamlink.github.io/cli/plugins/twitch.html
