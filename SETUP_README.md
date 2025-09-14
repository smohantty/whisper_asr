# Korean Language Model Setup

The `setup_resources.sh` script has been updated to support downloading Korean language models alongside English models for the WhisperBackend API.

## Quick Start

### Default Setup (English + Korean)
```bash
# Downloads both English and Korean models
./setup_resources.sh
```

This will download:
- `ggml-small.en.bin` - English-only model (optimized for English)
- `ggml-small.bin` - Multilingual model (supports Korean and other languages)

### Korean-Only Setup
```bash
# Explicitly download Korean support
./setup_resources.sh --korean
./setup_resources.sh --multilingual  # Same as --korean
```

### English-Only Setup
```bash
# Download only English models
./setup_resources.sh --english-only
```

## Advanced Usage

### Custom Model Sizes
```bash
# Download larger models for better accuracy
./setup_resources.sh --korean base.en base

# Download larger English, smaller Korean
./setup_resources.sh --korean medium.en small

# English-only with larger model
./setup_resources.sh --english-only large.en
```

### Available Models

**English Models** (optimized for English speech):
- `tiny.en` - ~39 MB, fastest, lowest accuracy
- `base.en` - ~142 MB, balanced speed/accuracy
- `small.en` - ~466 MB, good accuracy (default)
- `medium.en` - ~1.5 GB, high accuracy

**Multilingual Models** (supports Korean + other languages):
- `tiny` - ~39 MB, fastest, lowest accuracy
- `base` - ~142 MB, balanced speed/accuracy
- `small` - ~466 MB, good accuracy (default)
- `medium` - ~1.5 GB, high accuracy
- `large` - ~2.9 GB, highest accuracy

## Usage with WhisperBackend

### Builder Pattern (Recommended)
```cpp
// Using downloaded models with builder pattern
auto backend = WhisperBackendBuilder()
    .setBaseModelPath("resources/ggml-small")  // Uses ggml-small.en.bin and ggml-small.bin
    .setInitialLanguage(Language::English)
    .setCallback(myCallback)
    .build();

// Runtime language switching
backend->setLanguage(Language::Korean);  // Switches to ggml-small.bin
backend->setLanguage(Language::English); // Switches to ggml-small.en.bin
```

### Custom Model Configuration
```cpp
// Using different model sizes per language
auto backend = WhisperBackendBuilder()
    .setModelForLanguage(Language::English, "resources/ggml-medium.en.bin")  // Larger English
    .setModelForLanguage(Language::Korean, "resources/ggml-small.bin")       // Smaller Korean
    .setInitialLanguage(Language::English)
    .setCallback(myCallback)
    .build();
```

### Traditional Constructor
```cpp
// Using base path (automatic model selection)
WhisperBackend backend("resources/ggml-small", Language::English, myCallback);
```

## Model Download Details

The script:
1. **Checks existing models** - Skips download if models already exist
2. **Downloads from official source** - Uses whisper.cpp's download script
3. **Copies to resources/** - Makes models available to your application
4. **Validates downloads** - Ensures models downloaded successfully

### Example Output
```bash
$ ./setup_resources.sh --korean

=== Whisper.cpp Resource Setup ===

Configuration:
  English model: small.en
  Multilingual model (Korean): small
  Whisper directory: subprojects/whisper.cpp-1.7.6
  Resources directory: resources

=== Step 1: English Model ===
Checking for English model...
✓ Model ggml-small.en.bin already exists in resources

=== Step 2: Multilingual Model (Korean Support) ===
Checking for Multilingual (Korean) model...
Downloading Multilingual (Korean) model: small...
✓ Successfully downloaded ggml-small.bin
✓ Copied ggml-small.bin to resources

=== Setup Complete ===
Resources available in: resources/

Downloaded models:
-rw-rw-r-- 1 user user 487614201 Sep 14 20:22 ggml-small.en.bin
-rw-rw-r-- 1 user user 466264557 Sep 14 21:30 ggml-small.bin

=== Model Information ===
✓ English model (small.en): ggml-small.en.bin
  - Optimized for English speech recognition
  - Use with WhisperBackendBuilder for Language::English
✓ Multilingual model (small): ggml-small.bin
  - Supports Korean and other languages
  - Use with WhisperBackendBuilder for Language::Korean

=== Usage Examples ===
1. Builder Pattern (recommended for multi-language):
   auto backend = WhisperBackendBuilder()
       .setBaseModelPath("resources/ggml-small")
       .setInitialLanguage(Language::English)
       .setCallback(callback)
       .build();

2. Test applications:
   ./test_whisper.sh                                      # Basic audio file test
   cd builddir && ./example_whisper_backend resources/ggml-small  # Live streaming
   cd builddir && ./language_switching_demo resources/ggml-small   # Language demo
   cd builddir && ./builder_pattern_demo                   # Builder pattern demo

=== All Done! ===
Your WhisperBackend now supports both English and Korean models!
Use Language::English and Language::Korean with the setLanguage() API.
```

## Testing Korean Support

After setup, test Korean language recognition:

```bash
# Build the project
meson setup builddir
meson compile -C builddir

# Test language switching
cd builddir
./language_switching_demo resources/ggml-small

# Test builder pattern
./builder_pattern_demo

# Interactive Korean/English switching
./example_whisper_backend resources/ggml-small
# During runtime: Press 'k' for Korean, 'e' for English
```

## Troubleshooting

### Model Download Issues
```bash
# Check whisper.cpp directory
ls -la subprojects/whisper.cpp-1.7.6/models/

# Manual model download
cd subprojects/whisper.cpp-1.7.6
./models/download-ggml-model.sh small
```

### Storage Requirements
- **Tiny models**: ~40 MB each
- **Small models**: ~470 MB each (default)
- **Medium models**: ~1.5 GB each
- **Large models**: ~3 GB each

### Performance Considerations
- **English-only models**: Faster, more accurate for English
- **Multilingual models**: Slightly slower, supports Korean + other languages
- **Model size**: Larger models = better accuracy, slower processing

## Integration Examples

The updated setup script enables these workflow patterns:

1. **Development**: Use tiny models for fast iteration
2. **Production**: Use medium/large models for accuracy
3. **Mixed optimization**: Large English model + small Korean model
4. **Resource constrained**: Tiny models for embedded applications
