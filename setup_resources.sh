#!/bin/bash

# Whisper.cpp Model Download Script
# Downloads the specified models to resources directory if not already present
# Supports both English-only and multilingual (including Korean) models

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}=== Whisper.cpp Resource Setup ===${NC}"
    echo "Downloads Whisper models for English and Korean language support"
    echo
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0                              # Download both English and Korean models (default)"
    echo "  $0 --korean [en_model] [ml_model]    # Download both models (explicit)"
    echo "  $0 --multilingual [en_model] [ml_model]  # Same as --korean"
    echo "  $0 --english-only [en_model]    # Download only English model"
    echo "  $0 --help                       # Show this help"
    echo
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                              # Downloads small.en + small (default)"
    echo "  $0 --korean base.en base        # Downloads base.en + base models"
    echo "  $0 --english-only medium.en     # Downloads only medium.en model"
    echo
    echo -e "${YELLOW}Available models:${NC}"
    echo "  English: tiny.en, base.en, small.en, medium.en"
    echo "  Multilingual: tiny, base, small, medium, large"
    echo
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=== Whisper.cpp Resource Setup ===${NC}"
echo

# Configuration
WHISPER_DIR="subprojects/whisper.cpp-1.7.6"
RESOURCES_DIR="resources"
DEFAULT_ENGLISH_MODEL="small.en"
DEFAULT_MULTILINGUAL_MODEL="small"

# Parse command line arguments
if [ "$1" = "--korean" ] || [ "$1" = "--multilingual" ]; then
    DOWNLOAD_MULTILINGUAL=true
    ENGLISH_MODEL="${2:-$DEFAULT_ENGLISH_MODEL}"
    MULTILINGUAL_MODEL="${3:-$DEFAULT_MULTILINGUAL_MODEL}"
elif [ "$1" = "--english-only" ]; then
    DOWNLOAD_MULTILINGUAL=false
    ENGLISH_MODEL="${2:-$DEFAULT_ENGLISH_MODEL}"
    MULTILINGUAL_MODEL=""
else
    # Default behavior: download both English and multilingual models
    DOWNLOAD_MULTILINGUAL=true
    ENGLISH_MODEL="${1:-$DEFAULT_ENGLISH_MODEL}"
    MULTILINGUAL_MODEL="${2:-$DEFAULT_MULTILINGUAL_MODEL}"
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  English model: $ENGLISH_MODEL"
if [ "$DOWNLOAD_MULTILINGUAL" = true ]; then
    echo "  Multilingual model (Korean): $MULTILINGUAL_MODEL"
else
    echo "  Multilingual model: [Skipped]"
fi
echo "  Whisper directory: $WHISPER_DIR"
echo "  Resources directory: $RESOURCES_DIR"
echo

# Create resources directory if it doesn't exist
mkdir -p "$RESOURCES_DIR"

# Check if whisper.cpp directory exists
if [ ! -d "$WHISPER_DIR" ]; then
    echo -e "${RED}Error: $WHISPER_DIR directory not found!${NC}"
    echo "Please ensure you have the whisper.cpp source code in this directory."
    exit 1
fi

# Function to download a model
download_model() {
    local model_name="$1"
    local model_description="$2"

    local model_file="ggml-${model_name}.bin"
    local resource_model_path="$RESOURCES_DIR/$model_file"

    echo -e "${BLUE}Checking for $model_description model...${NC}"

    if [ -f "$resource_model_path" ]; then
        echo -e "${GREEN}✓ Model $model_file already exists in $RESOURCES_DIR${NC}"
        return 0
    fi

    echo -e "${YELLOW}Model $model_file not found in $RESOURCES_DIR${NC}"
    echo -e "${YELLOW}Downloading $model_description model: $model_name...${NC}"

    # Check if download script exists
    if [ ! -f "$WHISPER_DIR/models/download-ggml-model.sh" ]; then
        echo -e "${RED}Error: Download script not found at $WHISPER_DIR/models/download-ggml-model.sh${NC}"
        exit 1
    fi

    # Make download script executable
    chmod +x "$WHISPER_DIR/models/download-ggml-model.sh"

    # Download the model to whisper.cpp models directory
    cd "$WHISPER_DIR"
    ./models/download-ggml-model.sh "$model_name"
    cd - > /dev/null

    # Check if download was successful
    local model_path="$WHISPER_DIR/models/$model_file"
    if [ -f "$model_path" ]; then
        echo -e "${GREEN}✓ Successfully downloaded $model_file${NC}"

        # Copy to resources directory
        echo "Copying model to resources directory..."
        cp "$model_path" "$resource_model_path"
        echo -e "${GREEN}✓ Copied $model_file to $RESOURCES_DIR${NC}"
    else
        echo -e "${RED}Error: Failed to download $model_file${NC}"
        echo "Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large"
        exit 1
    fi
}

# Download English model
echo -e "${BLUE}=== Step 1: English Model ===${NC}"
download_model "$ENGLISH_MODEL" "English"

# Download multilingual model if requested
if [ "$DOWNLOAD_MULTILINGUAL" = true ]; then
    echo
    echo -e "${BLUE}=== Step 2: Multilingual Model (Korean Support) ===${NC}"
    download_model "$MULTILINGUAL_MODEL" "Multilingual (Korean)"
else
    echo
    echo -e "${YELLOW}Skipping multilingual model download (English-only mode)${NC}"
fi

# Show current resources
echo
echo -e "${BLUE}=== Setup Complete ===${NC}"
echo -e "${GREEN}Resources available in: $RESOURCES_DIR/${NC}"
echo
echo "Downloaded models:"
ls -la "$RESOURCES_DIR/"*.bin 2>/dev/null || echo "No .bin files found"

echo
echo -e "${BLUE}=== Model Information ===${NC}"
if [ -f "$RESOURCES_DIR/ggml-${ENGLISH_MODEL}.bin" ]; then
    echo -e "${GREEN}✓ English model (${ENGLISH_MODEL}): ggml-${ENGLISH_MODEL}.bin${NC}"
    echo "  - Optimized for English speech recognition"
    echo "  - Use with WhisperBackendBuilder for Language::English"
fi

if [ "$DOWNLOAD_MULTILINGUAL" = true ] && [ -f "$RESOURCES_DIR/ggml-${MULTILINGUAL_MODEL}.bin" ]; then
    echo -e "${GREEN}✓ Multilingual model (${MULTILINGUAL_MODEL}): ggml-${MULTILINGUAL_MODEL}.bin${NC}"
    echo "  - Supports Korean and other languages"
    echo "  - Use with WhisperBackendBuilder for Language::Korean"
fi

echo
echo -e "${BLUE}=== Usage Examples ===${NC}"
echo "1. Builder Pattern (recommended for multi-language):"
echo -e "${YELLOW}   auto backend = WhisperBackendBuilder()${NC}"
echo -e "${YELLOW}       .setBaseModelPath(\"resources/ggml-${MULTILINGUAL_MODEL%.*}\")${NC}"
echo -e "${YELLOW}       .setInitialLanguage(Language::English)${NC}"
echo -e "${YELLOW}       .setCallback(callback)${NC}"
echo -e "${YELLOW}       .build();${NC}"
echo

echo "2. Test applications:"
echo -e "${YELLOW}   ./test_whisper.sh${NC}                                      # Basic audio file test"
if [ "$DOWNLOAD_MULTILINGUAL" = true ]; then
    echo -e "${YELLOW}   cd builddir && ./example_whisper_backend resources/ggml-${MULTILINGUAL_MODEL%.*}${NC}  # Live streaming"
    echo -e "${YELLOW}   cd builddir && ./language_switching_demo resources/ggml-${MULTILINGUAL_MODEL%.*}${NC}   # Language demo"
else
    echo -e "${YELLOW}   cd builddir && ./example_whisper_backend resources/ggml-${ENGLISH_MODEL%.*}${NC}      # Live streaming"
fi
echo -e "${YELLOW}   cd builddir && ./builder_pattern_demo${NC}                   # Builder pattern demo"

echo
echo -e "${GREEN}=== All Done! ===${NC}"
if [ "$DOWNLOAD_MULTILINGUAL" = true ]; then
    echo "Your WhisperBackend now supports both English and Korean models!"
    echo "Use ${YELLOW}Language::English${NC} and ${YELLOW}Language::Korean${NC} with the setLanguage() API."
else
    echo "Your WhisperBackend is configured for English-only recognition!"
    echo "To add Korean support, run: ${YELLOW}./setup_resources.sh --korean${NC}"
fi
