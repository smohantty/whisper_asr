#!/bin/bash

# Simple test script for whisper.cpp
# Usage: ./test_whisper.sh [model_name]
# Examples:
#   ./test_whisper.sh          # Uses default base.en model
#   ./test_whisper.sh tiny     # Uses tiny model
#   ./test_whisper.sh base     # Uses base model

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse command line arguments
MODEL_NAME="${1:-base.en}"  # Default to base.en if no argument provided

# Show help if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo -e "${BLUE}=== Whisper Test Script Help ===${NC}"
    echo "Usage: ./test_whisper.sh [model_name]"
    echo
    echo "Available models:"
    echo "  tiny      - Smallest model (~77MB), fastest but least accurate"
    echo "  tiny.en   - English-only tiny model"
    echo "  base      - Base model (~147MB), good balance of speed/accuracy"
    echo "  base.en   - English-only base model (default)"
    echo "  small     - Small model (~461MB), better accuracy"
    echo "  small.en  - English-only small model"
    echo "  medium    - Medium model (~1.5GB), high accuracy"
    echo "  medium.en - English-only medium model"
    echo "  large     - Largest model (~3GB), highest accuracy"
    echo
    echo "Examples:"
    echo "  ./test_whisper.sh          # Uses default base.en model"
    echo "  ./test_whisper.sh tiny     # Uses tiny model"
    echo "  ./test_whisper.sh base     # Uses base model"
    echo
    exit 0
fi

MODEL_FILE="ggml-${MODEL_NAME}.bin"
MODEL_PATH="../resources/${MODEL_FILE}"

echo -e "${BLUE}=== Running Whisper Test ===${NC}"
echo -e "${YELLOW}Model: ${MODEL_NAME} (${MODEL_FILE})${NC}"

# Check if model file exists
if [ ! -f "resources/${MODEL_FILE}" ]; then
    echo -e "${YELLOW}Model ${MODEL_FILE} not found in resources directory.${NC}"
    echo "Downloading model..."
    ./setup_resources.sh "${MODEL_NAME}"
fi

# Build if builddir doesn't exist
if [ ! -d "builddir" ]; then
    echo "Building project..."
    meson setup builddir
    meson compile -C builddir
fi

# Run the test
cd builddir
echo -e "${GREEN}Running test with ${MODEL_NAME} model...${NC}"
./test_whisper "${MODEL_PATH}" ../resources/jfk.wav
