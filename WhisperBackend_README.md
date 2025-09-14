# WhisperBackend API Documentation

## Overview

The `WhisperBackend` class provides a high-level C++ API for live streaming Automatic Speech Recognition (ASR) using OpenAI's Whisper model. It features:

- **Live streaming ASR**: Real-time speech recognition with callback-based results
- **Multi-language support**: Runtime switching between English and Korean models
- **Internal audio queue**: Thread-safe audio buffering for smooth processing
- **Worker thread**: Dedicated thread for audio processing to avoid blocking the main thread
- **Speech tag handling**: Support for speech segmentation (Start/Continue/End)
- **Asynchronous callbacks**: Non-blocking result delivery via function callbacks
- **Dynamic model loading**: Automatic model unloading and reloading for language switching

## API Reference

### Enums

#### `Language`
```cpp
enum class Language {
    English,    // English language model
    Korean      // Korean language model
};
```

#### `SpeechTag`
```cpp
enum class SpeechTag {
    Start,      // Beginning of speech segment
    Continue,   // Continuation of speech
    End         // End of speech segment
};
```

#### `ResultTag`
```cpp
enum class ResultTag {
    Partial,    // Partial/intermediate transcription result
    Final,      // Final transcription result
    Error       // Error occurred during processing
};
```

### Callback Type

```cpp
using AsrEventCallback = std::function<void(ResultTag resultTag, const std::string& text)>;
```

### WhisperBackend Class

#### Constructor
```cpp
WhisperBackend(const std::string& baseModelPath, Language language, AsrEventCallback asrEventCallback);
```
- **Parameters**:
  - `baseModelPath`: Base path to the Whisper model files without language suffix (e.g., "resources/ggml-small")
  - `language`: Initial language to use (Language::English or Language::Korean)
  - `asrEventCallback`: Function to call when transcription results are available
- **Description**: Initializes the Whisper model for the specified language and starts the internal worker thread

#### Destructor
```cpp
~WhisperBackend();
```
- **Description**: Automatically stops the worker thread and cleans up resources

#### Process Audio
```cpp
void processAudio(const std::vector<float>& audio, SpeechTag speechTag);
```
- **Parameters**:
  - `audio`: Audio samples as 32-bit floats (normalized to [-1.0, 1.0])
  - `speechTag`: Speech segmentation hint (Start/Continue/End)
- **Description**: Adds audio to the internal queue for processing
- **Thread Safety**: This method is thread-safe and can be called from any thread

#### Set Language
```cpp
bool setLanguage(Language language);
```
- **Parameters**:
  - `language`: Target language (Language::English or Language::Korean)
- **Returns**: `true` if language switch was successful, `false` otherwise
- **Description**: Dynamically switches the language model by unloading the current model and loading the target language model
- **Thread Safety**: This method is thread-safe but may temporarily pause audio processing during the switch

## Usage Examples

### Basic Usage

```cpp
#include "WhisperBackend.h"
#include <iostream>

using namespace asr;

// Define callback function
void onAsrResult(ResultTag tag, const std::string& text) {
    switch (tag) {
        case ResultTag::Partial:
            std::cout << "Partial: " << text << std::endl;
            break;
        case ResultTag::Final:
            std::cout << "Final: " << text << std::endl;
            break;
        case ResultTag::Error:
            std::cerr << "Error: " << text << std::endl;
            break;
    }
}

int main() {
    // Specify base model path
    std::string baseModelPath = "resources/ggml-small";

    // Create WhisperBackend with base model path, language, and callback
    WhisperBackend backend(baseModelPath, Language::English, onAsrResult);

    // Process audio chunks
    std::vector<float> audioChunk = getAudioFromMicrophone();
    backend.processAudio(audioChunk, SpeechTag::Start);

    // Switch to Korean mid-stream
    backend.setLanguage(Language::Korean);

    // Process more chunks in Korean
    backend.processAudio(nextChunk, SpeechTag::Continue);
    backend.processAudio(finalChunk, SpeechTag::End);

    return 0;  // Destructor automatically cleans up
}
```

### Integration with AudioStreamer

```cpp
#include "WhisperBackend.h"
#include "AudioStreamer.h"
#include <vector>

using namespace asr;

// Convert int16 to float samples
std::vector<float> convertToFloat(const std::vector<short>& samples) {
    std::vector<float> result;
    result.reserve(samples.size());
    for (short sample : samples) {
        result.push_back(static_cast<float>(sample) / 32768.0f);
    }
    return result;
}

void asrCallback(ResultTag tag, const std::string& text) {
    if (tag == ResultTag::Final && !text.empty()) {
        std::cout << "Transcription: " << text << std::endl;
    }
}

int main() {
    // Specify base model path
    std::string baseModelPath = "resources/ggml-small";

    // Initialize components
    WhisperBackend backend(baseModelPath, Language::English, asrCallback);
    AudioStreamer streamer(100, 16000, 1);  // 100ms chunks, 16kHz, mono

    streamer.start();

    bool inSpeech = false;
    while (streamer.isRunning()) {
        std::vector<short> audioChunk;
        if (streamer.popChunk(audioChunk)) {
            auto floatAudio = convertToFloat(audioChunk);

            // Simple voice activity detection
            float energy = calculateEnergy(floatAudio);
            bool hasVoice = energy > threshold;

            SpeechTag tag;
            if (hasVoice && !inSpeech) {
                tag = SpeechTag::Start;
                inSpeech = true;
            } else if (!hasVoice && inSpeech) {
                tag = SpeechTag::End;
                inSpeech = false;
            } else {
                tag = SpeechTag::Continue;
            }

            backend.processAudio(floatAudio, tag);
        }
    }

    return 0;
}
```

## Architecture

### Internal Design

The `WhisperBackend` uses the Pimpl (Pointer to Implementation) pattern for clean separation of interface and implementation:

```
WhisperBackend (Public Interface)
    ↓
WhisperBackend::Impl (Private Implementation)
    ├── Whisper Context & Parameters
    ├── Worker Thread
    ├── Thread-safe Audio Queue
    ├── Audio Accumulation Buffer
    └── Callback Mechanism
```

### Processing Flow

1. **Audio Input**: Client calls `processAudio()` with audio data and speech tag
2. **Queue Management**: Audio is added to internal thread-safe queue
3. **Accumulation**: Audio chunks are accumulated until sufficient duration (3 seconds by default)
4. **Worker Thread**: Dedicated thread processes queued audio using Whisper
5. **Callback**: Results are delivered via the registered callback function

### Threading Model

- **Main Thread**: Handles `processAudio()` calls and queues audio data
- **Worker Thread**: Continuously processes queued audio chunks
- **Thread Safety**: All internal data structures use proper synchronization

## Configuration

### Model Files

The backend requires you to specify a base model path in the constructor. The API automatically appends language-specific suffixes:

**Required Model Files:**
- For English: `[base_path].en.bin` (e.g., `resources/ggml-small.en.bin`)
- For Korean: `[base_path].bin` (e.g., `resources/ggml-small.bin`)

**Example Model Structure:**
```
resources/
├── ggml-small.en.bin    # English model
├── ggml-small.bin       # Korean/multilingual model
├── ggml-base.en.bin     # Larger English model
└── ggml-base.bin        # Larger Korean/multilingual model
```

**Constructor Usage:**
```cpp
// This will look for resources/ggml-small.en.bin and resources/ggml-small.bin
WhisperBackend backend("resources/ggml-small", Language::English, callback);
```

You can download Whisper models from the [official repository](https://github.com/ggerganov/whisper.cpp). Both language-specific model files must exist for language switching to work properly.

### Audio Parameters

- **Sample Rate**: 16 kHz (fixed, as required by Whisper)
- **Channels**: Mono (1 channel)
- **Sample Format**: 32-bit float, normalized to [-1.0, 1.0]
- **Processing Duration**: 3 seconds (configurable in implementation)

### Whisper Parameters

The backend uses these optimized parameters for streaming:
- Language: English (`en`)
- Translation: Disabled
- Context: Disabled (each chunk processed independently)
- Blank suppression: Enabled
- Non-speech token suppression: Enabled

## Performance Considerations

### Latency
- **Processing latency**: ~200-500ms for 3-second audio chunks
- **Queue latency**: Minimal, audio is queued immediately
- **Callback latency**: Results delivered as soon as processing completes

### Memory Usage
- **Audio queue**: Bounded by processing frequency
- **Whisper model**: ~150MB for small.en model
- **Processing buffers**: ~200KB for 3-second audio chunks

### CPU Usage
- **Worker thread**: Uses one dedicated CPU core for Whisper processing
- **Main thread**: Minimal CPU usage for audio queuing
- **OpenMP**: Whisper uses multiple threads for inference acceleration

## Error Handling

The backend handles various error conditions:

- **Model loading failures**: Reported via console and prevents initialization
- **Audio processing errors**: Delivered via callback with `ResultTag::Error`
- **Thread synchronization**: Automatic cleanup on destruction
- **Queue overflow**: Currently unbounded, consider rate limiting in production

## Building and Dependencies

### Dependencies
- **Whisper.cpp**: OpenAI Whisper C++ implementation
- **C++17**: Modern C++ standard
- **Threading**: `std::thread`, `std::mutex`, `std::condition_variable`
- **OpenMP**: Optional, for performance acceleration

### Build System
The backend integrates with Meson build system:
```bash
cd builddir
meson compile
./example_whisper_backend [base_model_path]  # Run the example
# Example: ./example_whisper_backend resources/ggml-small
# This will use resources/ggml-small.en.bin and resources/ggml-small.bin
```

## Future Enhancements

Potential improvements for the API:
- **Additional languages**: Support for more languages beyond English and Korean
- **Real-time factor configuration**: Adjustable processing chunk duration
- **Voice activity detection**: Built-in VAD for automatic speech segmentation
- **Streaming confidence scores**: Confidence metrics for transcription quality
- **Custom callback contexts**: User data in callback functions
- **Model size detection**: Automatic detection of available model variants
