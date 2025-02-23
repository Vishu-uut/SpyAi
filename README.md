# Autonomous Real-Time Crime Detection System

![Project Demo](demo.gif) <!-- Add a demo GIF if available -->

A comprehensive solution for real-time crime detection using computer vision and deep learning. The system analyzes live video streams, detects suspicious activities using a Vision Transformer (ViT) model trained on the UFC Crime Dataset, and provides real-time alerts through a web interface.

## Features

- 🚨 **Real-Time Crime Detection**: Processes live video streams with <500ms latency
- 📹 **Motion-Based Keyframe Extraction**: Identifies significant motion events for analysis
- 🔍 **Deep Learning Analysis**: Uses Vision Transformer (ViT) model with 94% validation accuracy
- 📱 **Web Interface**: Real-time video feed with alert management system
- ⚡ **WebSocket Integration**: Instant notifications for new detections
- 🛡️ **Action System**: "Alert" for confirmation or "Deny" to remove false positives

## Technology Stack

### Backend
- **Flask**: REST API server
- **Socket.IO**: Real-time communication
- **Watchdog**: File system monitoring
- **OpenCV**: Video processing and analysis

### Frontend
- **React**: UI framework
- **HTML5 Video**: Video playback
- **CSS Grid**: Responsive layout

### Machine Learning
- **PyTorch**: Deep learning framework
- **timm**: Vision Transformer implementation
- **NumPy**: Numerical computations
- **PIL/Pillow**: Image processing

## Dataset
- **UFC Crime Dataset**: Contains 15,000+ labeled crime/normal scenes
- **Train/Test Split**: 80/20 stratified split
- **Class Balance**: 60% normal / 40% criminal activities

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- NVIDIA GPU (Recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/crime-detection-system.git
cd crime-detection-system

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
