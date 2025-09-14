# Whisper.cpp Static Library Build & Test Guide

This project provides a minimal static whisper.cpp library build with CPU-only backend support and includes a test application for speech recognition.

## Quick Start

### 1. Automated Setup (Recommended)

Run the setup script to automatically download models and copy sample files:

```bash
# Download base.en model (recommended) and setup resources
./setup_resources.sh

# Or specify a different model
./setup_resources.sh tiny.en    # Faster, less accurate
./setup_resources.sh small.en   # More accurate, slower
```

This will:
- Create a `resources/` folder with models and sample audio
- Download the specified whisper model
- Copy sample audio files (`jfk.wav`, `jfk.mp3`)
- Create a `test_whisper.sh` script for easy testing

### 1b. Manual Setup (Alternative)

If you prefer manual setup, download the whisper model:

```bash
cd whisper.cpp-1.7.6
./models/download-ggml-model.sh base.en
```

Available models (from smallest to largest):
- `tiny`, `tiny.en` (~39 MB)
- `base`, `base.en` (~74 MB) - **Recommended for testing**
- `small`, `small.en` (~244 MB)
- `medium`, `medium.en` (~769 MB)
- `large` (~1550 MB)

**Note**: The `.en` versions are English-only and faster/more accurate for English audio.

### 2. Build the Project

```bash
# Build using Meson (builds both library and test)
meson setup builddir
meson compile -C builddir
```

### 3. Run the Test

**Easy way (after running setup script):**
```bash
./test_whisper.sh
```

**Manual way:**
```bash
cd builddir
./test_whisper

# Or specify custom paths:
./test_whisper /path/to/model.bin /path/to/audio.wav

# Using resources folder:
./test_whisper ../resources/ggml-base.en.bin ../resources/jfk.wav
```

The test will automatically:
- Load the specified model (or `ggml-base.en.bin` by default)
- Process the audio file (or `jfk.wav` by default)
- Display transcription results with timestamps
- Show performance statistics

## Test Output Example

```
=== Whisper.cpp Test Application ===
Model path: ../resources/ggml-base.en.bin
Audio path: ../resources/jfk.wav

Initializing Whisper model...
✓ Whisper model loaded successfully!

Loading audio file...
Loaded 176000 audio samples from ../resources/jfk.wav
✓ Audio file loaded successfully!

Processing audio...
✓ Audio processing completed!

=== TRANSCRIPTION RESULTS ===
Number of segments: 1

Transcription:
[   0.000 -->   11.000] And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

=== STATISTICS ===
Sample time:     45.23 ms
Encode time:    234.56 ms
Decode time:    189.34 ms
```

## Project Structure

After running the setup script, your project will look like:

```
asr/
├── setup_resources.sh      # Resource setup script
├── test_whisper.sh        # Easy test runner
├── resources/             # Created by setup script
│   ├── ggml-base.en.bin  # Downloaded model
│   ├── jfk.wav           # Sample audio (WAV)
│   └── jfk.mp3           # Sample audio (MP3)
├── builddir/             # Meson build directory
│   └── test_whisper      # Test executable
├── whisper.cpp-1.7.6/    # Whisper.cpp source
└── meson.build           # Build configuration
```

## Build Output Files

After successful compilation, you will find:

- **Test Executable**: `builddir/test_whisper`
- **Static Library**: `builddir/subprojects/whisper.cpp-1.7.6/src/libwhisper.a`
- **GGML Library**: `builddir/subprojects/whisper.cpp-1.7.6/ggml/src/libggml.a`
- **Header File**: `whisper.cpp-1.7.6/include/whisper.h`

## Using the Library in Your Project

### Include Headers

```cpp
#include "whisper.h"
```

### Meson Integration

```meson
# Use the whisper dependency from the subproject
whisper_dep = dependency('whisper', fallback: ['whisper.cpp-1.7.6', 'whisper_dep'])

# Create your executable
executable('your_app',
  'your_app.cpp',
  dependencies: whisper_dep,
  install: true
)
```

### Manual Linking (for other build systems)

If you need to use the built libraries with other build systems:

```bash
# Static libraries are located at:
# - builddir/subprojects/whisper.cpp-1.7.6/src/libwhisper.a
# - builddir/subprojects/whisper.cpp-1.7.6/ggml/src/libggml.a

# Headers are at:
# - whisper.cpp-1.7.6/include/whisper.h

# Example compilation:
g++ -std=c++17 your_app.cpp \
    -I./whisper.cpp-1.7.6/include \
    -L./builddir/subprojects/whisper.cpp-1.7.6/src \
    -L./builddir/subprojects/whisper.cpp-1.7.6/ggml/src \
    -lwhisper -lggml \
    -o your_app
```

## Model Requirements & Performance

| Model  | Disk Size | Memory Usage | Speed | Accuracy |
|--------|-----------|--------------|-------|----------|
| tiny   | 39 MB     | ~125 MB      | Fast  | Basic    |
| base   | 74 MB     | ~210 MB      | Good  | Good     |
| small  | 244 MB    | ~465 MB      | Slow  | Better   |
| medium | 769 MB    | ~1.2 GB      | Slower| High     |
| large  | 1550 MB   | ~2.6 GB      | Slowest| Highest |

**Recommendation**: Use `base.en` for English audio - it provides the best balance of accuracy and speed for most applications.

## Audio Requirements

The test application expects:
- **Format**: WAV files (16-bit, mono preferred)
- **Sample Rate**: 16kHz (automatically resampled if different)
- **Channels**: Mono (stereo will be converted)

## Troubleshooting

### Setup Script Issues
```bash
# Make sure the script is executable
chmod +x setup_resources.sh

# Check if whisper.cpp directory exists
ls whisper.cpp-1.7.6/
```

### Model Not Found
```bash
# Re-run setup script
./setup_resources.sh base.en

# Or download manually
cd whisper.cpp-1.7.6
./models/download-ggml-model.sh base.en
```

### Build Fails
```bash
# Install required dependencies
sudo apt update
sudo apt install build-essential cmake ninja-build

# For Meson builds
pip install meson
```

### Audio File Issues
- Ensure WAV file is not corrupted
- Try converting: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`

## Advanced Usage

### Custom Model and Audio
```bash
./test_whisper.sh  # Uses resources/ folder
# Or manually:
cd builddir
./test_whisper /path/to/custom-model.bin /path/to/audio.wav
```

### Using Different Models
```bash
# Setup with different model
./setup_resources.sh tiny.en     # Fast, basic accuracy
./setup_resources.sh medium.en   # Slower, high accuracy

# Test specific model
cd builddir
./test_whisper ../resources/ggml-tiny.en.bin ../resources/jfk.wav
```

### Batch Testing
```bash
# Test multiple audio files
cd builddir
for audio in ../resources/*.wav; do
    echo "Testing: $audio"
    ./test_whisper ../resources/ggml-base.en.bin "$audio"
done
```

## Scripts Reference

### `setup_resources.sh`
- Downloads whisper models
- Copies sample audio files to `resources/`
- Creates `test_whisper.sh` runner script

**Usage:**
```bash
./setup_resources.sh [model_name]
```

**Examples:**
```bash
./setup_resources.sh              # Downloads base.en
./setup_resources.sh tiny.en       # Downloads tiny.en
./setup_resources.sh small         # Downloads small (multilingual)
```

### `test_whisper.sh`
- Builds project if needed
- Runs test with resources from `resources/` folder

**Usage:**
```bash
./test_whisper.sh
```

This build configuration provides all core functionality for CPU-based speech recognition with optimized performance and minimal dependencies.