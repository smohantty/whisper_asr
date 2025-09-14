# WhisperBackend API Documentation

## Overview

The `WhisperBackend` class provides a high-level C++ API for live streaming Automatic Speech Recognition (ASR) using OpenAI's Whisper model. It features:

- **Live streaming ASR**: Real-time speech recognition with callback-based results
- **Multi-language support**: Runtime switching between English and Korean models
- **Builder pattern**: Flexible model configuration with method chaining
- **Custom model paths**: Configure different models for each language independently
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

#### Constructors

**Traditional Constructor**
```cpp
WhisperBackend(const std::string& baseModelPath, Language language, AsrEventCallback asrEventCallback);
```
- **Parameters**:
  - `baseModelPath`: Base path to the Whisper model files without language suffix (e.g., "resources/ggml-small")
  - `language`: Initial language to use (Language::English or Language::Korean)
  - `asrEventCallback`: Function to call when transcription results are available
- **Description**: Initializes the Whisper model for the specified language and starts the internal worker thread

**Builder Constructor**
```cpp
WhisperBackend(const WhisperBackendBuilder& builder);
```
- **Parameters**:
  - `builder`: Configured WhisperBackendBuilder instance
- **Description**: Initializes the Whisper model using builder configuration and starts the internal worker thread

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

### WhisperBackendBuilder Class

The builder class provides a flexible way to configure WhisperBackend instances with custom model paths for each language.

#### Constructor
```cpp
WhisperBackendBuilder();
```
- **Description**: Creates a new builder instance with default settings

#### Configuration Methods

**Set Callback**
```cpp
WhisperBackendBuilder& setCallback(AsrEventCallback callback);
```
- **Parameters**: `callback` - Function to call when transcription results are available
- **Returns**: Reference to builder for method chaining
- **Required**: Yes

**Set Initial Language**
```cpp
WhisperBackendBuilder& setInitialLanguage(Language language);
```
- **Parameters**: `language` - Language to use when backend starts
- **Returns**: Reference to builder for method chaining
- **Default**: Language::English

**Set Model for Language**
```cpp
WhisperBackendBuilder& setModelForLanguage(Language language, const std::string& modelPath);
```
- **Parameters**:
  - `language` - Target language
  - `modelPath` - Full path to the model file for this language
- **Returns**: Reference to builder for method chaining
- **Description**: Configure a specific model file for a language

**Set Base Model Path**
```cpp
WhisperBackendBuilder& setBaseModelPath(const std::string& baseModelPath);
```
- **Parameters**: `baseModelPath` - Base path without language suffix
- **Returns**: Reference to builder for method chaining
- **Description**: Convenience method that sets both English (.en.bin) and Korean (.bin) models

**Build**
```cpp
std::unique_ptr<WhisperBackend> build() const;
```
- **Returns**: Configured WhisperBackend instance
- **Throws**: `std::runtime_error` if configuration is invalid
- **Description**: Creates and returns the configured backend instance

## Usage Examples

### Traditional Constructor Usage

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
    // Traditional constructor - uses automatic model path generation
    WhisperBackend backend("resources/ggml-small", Language::English, onAsrResult);

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

### Builder Pattern Usage

```cpp
#include "WhisperBackend.h"
#include <iostream>

using namespace asr;

void onAsrResult(ResultTag tag, const std::string& text) {
    if (tag == ResultTag::Final) {
        std::cout << "Transcription: " << text << std::endl;
    }
}

int main() {
    // Example 1: Using setBaseModelPath (convenience method)
    auto backend1 = WhisperBackendBuilder()
        .setBaseModelPath("resources/ggml-small")      // Auto-configures both languages
        .setInitialLanguage(Language::English)
        .setCallback(onAsrResult)
        .build();

    // Example 2: Using setModelForLanguage (custom configuration)
    auto backend2 = WhisperBackendBuilder()
        .setModelForLanguage(Language::English, "resources/ggml-base.en.bin")   // Larger English model
        .setModelForLanguage(Language::Korean, "resources/ggml-small.bin")      // Smaller Korean model
        .setInitialLanguage(Language::English)
        .setCallback(onAsrResult)
        .build();

    // Example 3: Mixed configuration
    auto backend3 = WhisperBackendBuilder()
        .setBaseModelPath("resources/ggml-small")                               // Sets both languages
        .setModelForLanguage(Language::English, "resources/ggml-large.en.bin") // Override English with larger model
        .setInitialLanguage(Language::Korean)                                   // Start with Korean
        .setCallback(onAsrResult)
        .build();

    // Use any of the backends
    std::vector<float> audioChunk = getAudioFromMicrophone();
    backend1->processAudio(audioChunk, SpeechTag::Start);

    // Language switching works the same way
    backend1->setLanguage(Language::Korean);

    return 0;
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

**Quick Setup:**
Use the provided setup script to download Korean language models:
```bash
# Download both English and Korean models (recommended)
./setup_resources.sh

# Download specific model sizes
./setup_resources.sh --korean medium.en small

# English-only setup
./setup_resources.sh --english-only
```

**Example Model Structure:**
```
resources/
├── ggml-small.en.bin    # English model (~466MB)
├── ggml-small.bin       # Korean/multilingual model (~466MB)
├── ggml-base.en.bin     # Larger English model (~142MB)
└── ggml-base.bin        # Larger Korean/multilingual model (~142MB)
```

**Constructor Usage:**
```cpp
// This will look for resources/ggml-small.en.bin and resources/ggml-small.bin
WhisperBackend backend("resources/ggml-small", Language::English, callback);
```

**Available Model Sizes:**
- **tiny**: ~39MB (fastest, lowest accuracy)
- **base**: ~142MB (balanced)
- **small**: ~466MB (good accuracy, default)
- **medium**: ~1.5GB (high accuracy)
- **large**: ~2.9GB (highest accuracy, multilingual only)

You can download Whisper models manually from the [official repository](https://github.com/ggerganov/whisper.cpp) or use the provided setup script for automated Korean model setup.

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

# Run examples
./example_whisper_backend [base_model_path]    # Live audio streaming with language switching
./language_switching_demo [base_model_path]    # Language switching demo with dummy audio
./builder_pattern_demo                         # Builder pattern configuration demo

# Examples:
./example_whisper_backend resources/ggml-small
./language_switching_demo resources/ggml-small
./builder_pattern_demo
```

## Future Enhancements

Potential improvements for the API:
- **Additional languages**: Support for more languages beyond English and Korean
- **Builder validation**: Enhanced validation of model file existence at build time
- **Model caching**: Intelligent caching of loaded models for faster switching
- **Real-time factor configuration**: Adjustable processing chunk duration
- **Voice activity detection**: Built-in VAD for automatic speech segmentation
- **Streaming confidence scores**: Confidence metrics for transcription quality
- **Custom callback contexts**: User data in callback functions
