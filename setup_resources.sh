#!/bin/bash

# Whisper.cpp Model Download Script
# Downloads the specified model to resources directory if not already present

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Whisper.cpp Resource Setup ===${NC}"
echo

# Configuration
WHISPER_DIR="subprojects/whisper.cpp-1.7.6"
RESOURCES_DIR="resources"
DEFAULT_MODEL="base.en"

# Parse command line arguments
MODEL_NAME="${1:-$DEFAULT_MODEL}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL_NAME"
echo "  Whisper directory: $WHISPER_DIR"
echo "  Resources directory: $RESOURCES_DIR"
echo

# Check if whisper.cpp directory exists
if [ ! -d "$WHISPER_DIR" ]; then
    echo -e "${RED}Error: $WHISPER_DIR directory not found!${NC}"
    echo "Please ensure you have the whisper.cpp source code in this directory."
    exit 1
fi

# Check if model exists in resources directory
MODEL_FILE="ggml-${MODEL_NAME}.bin"
RESOURCE_MODEL_PATH="$RESOURCES_DIR/$MODEL_FILE"

echo -e "${BLUE}Step 1: Checking for model in resources...${NC}"
if [ -f "$RESOURCE_MODEL_PATH" ]; then
    echo -e "${GREEN}✓ Model $MODEL_FILE already exists in $RESOURCES_DIR${NC}"
    echo -e "${GREEN}✓ No download needed!${NC}"
else
    echo -e "${YELLOW}Model $MODEL_FILE not found in $RESOURCES_DIR${NC}"
    echo -e "${YELLOW}Downloading model: $MODEL_NAME...${NC}"

    # Check if download script exists
    if [ ! -f "$WHISPER_DIR/models/download-ggml-model.sh" ]; then
        echo -e "${RED}Error: Download script not found at $WHISPER_DIR/models/download-ggml-model.sh${NC}"
        exit 1
    fi

    # Make download script executable
    chmod +x "$WHISPER_DIR/models/download-ggml-model.sh"

    # Download the model to whisper.cpp models directory
    cd "$WHISPER_DIR"
    ./models/download-ggml-model.sh "$MODEL_NAME"
    cd - > /dev/null

    # Check if download was successful
    MODEL_PATH="$WHISPER_DIR/models/$MODEL_FILE"
    if [ -f "$MODEL_PATH" ]; then
        echo -e "${GREEN}✓ Successfully downloaded $MODEL_FILE${NC}"

        # Copy to resources directory
        echo "Copying model to resources directory..."
        cp "$MODEL_PATH" "$RESOURCE_MODEL_PATH"
        echo -e "${GREEN}✓ Copied $MODEL_FILE to $RESOURCES_DIR${NC}"
    else
        echo -e "${RED}Error: Failed to download $MODEL_FILE${NC}"
        echo "Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large"
        exit 1
    fi
fi

# Show current resources
echo
echo -e "${BLUE}=== Setup Complete ===${NC}"
echo -e "${GREEN}Resources available in: $RESOURCES_DIR/${NC}"
echo
echo "Contents:"
ls -la "$RESOURCES_DIR/"

echo
echo -e "${BLUE}=== Next Steps ===${NC}"
echo "1. Use the test script (recommended):"
echo -e "${YELLOW}   ./test_whisper.sh${NC}"
echo
echo "2. Or build and run manually:"
echo -e "${YELLOW}   meson setup builddir${NC}"
echo -e "${YELLOW}   meson compile -C builddir${NC}"
echo -e "${YELLOW}   cd builddir${NC}"
echo -e "${YELLOW}   ./test_whisper ../resources/$MODEL_FILE ../resources/jfk.wav${NC}"

echo
echo -e "${GREEN}=== All Done! ===${NC}"
echo "You can now run: ${YELLOW}./test_whisper.sh${NC} to build and test everything!"
