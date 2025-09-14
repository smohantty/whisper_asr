#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "WhisperBackend.h"

using namespace asr;

// Demo callback function
void builderDemoCallback(ResultTag tag, const std::string& text) {
    switch (tag) {
        case ResultTag::Partial:
            std::cout << "[PARTIAL] " << text << std::endl;
            break;
        case ResultTag::Final:
            std::cout << "[FINAL] " << text << std::endl;
            break;
        case ResultTag::Error:
            std::cerr << "[ERROR] " << text << std::endl;
            break;
    }
}

// Generate some dummy audio data
std::vector<float> generateSilence(size_t samples) {
    return std::vector<float>(samples, 0.0f);
}

int main(int /* argc */, char* /* argv */[]) {
    std::cout << "=== WhisperBackend Builder Pattern Demo ===" << std::endl;
    std::cout << "This demo shows how to use the WhisperBackendBuilder for flexible model configuration." << std::endl;
    std::cout << std::endl;

    try {
        // Demo 1: Using setBaseModelPath (convenience method)
        std::cout << "1. Demo: Using setBaseModelPath() for automatic model configuration" << std::endl;
        std::cout << "   This sets up both English (.en.bin) and Korean (.bin) models automatically" << std::endl;

        auto backend1 = WhisperBackendBuilder()
            .setBaseModelPath("resources/ggml-small")
            .setInitialLanguage(Language::English)
            .setCallback(builderDemoCallback)
            .build();

        std::cout << "   ✓ Created backend with automatic model paths:" << std::endl;
        std::cout << "     - English: resources/ggml-small.en.bin" << std::endl;
        std::cout << "     - Korean: resources/ggml-small.bin" << std::endl;

        // Test language switching
        std::vector<float> audio_chunk = generateSilence(16000);  // 1 second of silence
        backend1->processAudio(audio_chunk, SpeechTag::Start);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << "   Switching to Korean..." << std::endl;
        if (backend1->setLanguage(Language::Korean)) {
            std::cout << "   ✓ Successfully switched to Korean model" << std::endl;
        } else {
            std::cout << "   ✗ Failed to switch to Korean model (model file may not exist)" << std::endl;
        }

        std::cout << std::endl;

        // Demo 2: Using setModelForLanguage for custom model paths
        std::cout << "2. Demo: Using setModelForLanguage() for custom model configuration" << std::endl;
        std::cout << "   This allows you to specify different model sizes/types for each language" << std::endl;

        auto backend2 = WhisperBackendBuilder()
            .setModelForLanguage(Language::English, "resources/ggml-base.en.bin")  // Larger English model
            .setModelForLanguage(Language::Korean, "resources/ggml-small.bin")     // Smaller Korean model
            .setInitialLanguage(Language::English)
            .setCallback(builderDemoCallback)
            .build();

        std::cout << "   ✓ Created backend with custom model configuration:" << std::endl;
        std::cout << "     - English: resources/ggml-base.en.bin (larger, more accurate)" << std::endl;
        std::cout << "     - Korean: resources/ggml-small.bin (smaller, faster)" << std::endl;

        backend2->processAudio(audio_chunk, SpeechTag::Continue);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << "   Switching to Korean (different model size)..." << std::endl;
        if (backend2->setLanguage(Language::Korean)) {
            std::cout << "   ✓ Successfully switched to Korean model" << std::endl;
        } else {
            std::cout << "   ✗ Failed to switch to Korean model (model file may not exist)" << std::endl;
        }

        std::cout << std::endl;

        // Demo 3: Mixed configuration (some custom, some automatic)
        std::cout << "3. Demo: Mixed configuration - start with base path, then override specific languages" << std::endl;

        auto backend3 = WhisperBackendBuilder()
            .setBaseModelPath("resources/ggml-small")                              // Sets up both languages
            .setModelForLanguage(Language::English, "resources/ggml-large.en.bin") // Override English with larger model
            .setInitialLanguage(Language::Korean)                                   // Start with Korean
            .setCallback(builderDemoCallback)
            .build();

        std::cout << "   ✓ Created backend with mixed configuration:" << std::endl;
        std::cout << "     - English: resources/ggml-large.en.bin (overridden to large model)" << std::endl;
        std::cout << "     - Korean: resources/ggml-small.bin (from base path)" << std::endl;
        std::cout << "     - Starting language: Korean" << std::endl;

        backend3->processAudio(audio_chunk, SpeechTag::End);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << std::endl;

        // Demo 4: Error handling
        std::cout << "4. Demo: Builder pattern error handling" << std::endl;

        try {
            auto invalidBackend = WhisperBackendBuilder()
                .setModelForLanguage(Language::English, "resources/ggml-small.en.bin")
                .setInitialLanguage(Language::Korean)  // No Korean model configured!
                // .setCallback(builderDemoCallback)   // No callback set!
                .build();
        } catch (const std::exception& e) {
            std::cout << "   ✓ Caught expected error: " << e.what() << std::endl;
        }

        try {
            auto invalidBackend2 = WhisperBackendBuilder()
                .setCallback(builderDemoCallback)
                .setInitialLanguage(Language::Korean)  // No models configured at all!
                .build();
        } catch (const std::exception& e) {
            std::cout << "   ✓ Caught expected error: " << e.what() << std::endl;
        }

        std::cout << std::endl;
        std::cout << "✓ Builder pattern demo completed successfully!" << std::endl;
        std::cout << std::endl;
        std::cout << "Key Builder Pattern Benefits:" << std::endl;
        std::cout << "  - Flexible model configuration per language" << std::endl;
        std::cout << "  - Method chaining for clean, readable code" << std::endl;
        std::cout << "  - Validation at build time to catch configuration errors" << std::endl;
        std::cout << "  - Support for both automatic and custom model paths" << std::endl;
        std::cout << "  - Backward compatibility with traditional constructor" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
