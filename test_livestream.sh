#!/bin/bash

# Livestream ASR Test Script
# This script demonstrates how to use the livestream_asr application

echo "=== Whisper Livestream ASR Test ==="
echo

# Check if model file exists
MODEL_PATH="resources/ggml-small.en.bin"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure you have downloaded the Whisper model."
    echo "You can download it using:"
    echo "  wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin -O resources/ggml-small.en.bin"
    exit 1
fi

# Check if arecord is available
if ! command -v arecord &> /dev/null; then
    echo "Error: 'arecord' command not found."
    echo "Please install ALSA utilities:"
    echo "  sudo apt-get install alsa-utils  # Ubuntu/Debian"
    echo "  sudo yum install alsa-utils      # CentOS/RHEL"
    exit 1
fi

# Check if build directory exists
if [ ! -d "builddir" ]; then
    echo "Error: Build directory not found. Please build the project first:"
    echo "  meson setup builddir"
    echo "  cd builddir && ninja"
    exit 1
fi

# Check if the executable exists
if [ ! -f "builddir/livestream_asr" ]; then
    echo "Error: livestream_asr executable not found."
    echo "Please build the project first:"
    echo "  cd builddir && ninja"
    exit 1
fi

echo "✓ Model file found: $MODEL_PATH"
echo "✓ arecord utility available"
echo "✓ livestream_asr executable ready"
echo

echo "Starting livestream ASR..."
echo "Instructions:"
echo "  - Speak clearly into your microphone"
echo "  - The system will process audio in 3-second chunks"
echo "  - Press Ctrl+C to stop"
echo "  - Make sure your microphone is not muted"
echo

# Test microphone first
echo "Testing microphone (recording 2 seconds)..."
timeout 2s arecord -f S16_LE -c1 -r16000 -t raw > /dev/null 2>&1
if [ $? -eq 124 ]; then
    echo "✓ Microphone test completed successfully"
else
    echo "Warning: Microphone test failed. Please check your audio setup."
    echo "You can test your microphone manually with:"
    echo "  arecord -f S16_LE -c1 -r16000 -d 3 test.wav"
fi

echo
echo "Starting livestream ASR in 3 seconds..."
sleep 3

# Run the livestream ASR application
cd builddir
./livestream_asr "../$MODEL_PATH"
