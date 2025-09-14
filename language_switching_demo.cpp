#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "WhisperBackend.h"

using namespace asr;

// Demo callback function
void demoCallback(ResultTag tag, const std::string& text) {
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

int main(int argc, char* argv[]) {
    std::cout << "=== WhisperBackend Language Switching Demo ===" << std::endl;
    std::cout << "This demo shows how to use the setLanguage API." << std::endl;
    std::cout << std::endl;

    // Default base model path
    std::string base_model_path = "resources/ggml-small";

    if (argc >= 2) {
        base_model_path = argv[1];
    }

    std::cout << "Base model path: " << base_model_path << std::endl;
    std::cout << "Expected files:" << std::endl;
    std::cout << "  - " << base_model_path << ".en.bin (English)" << std::endl;
    std::cout << "  - " << base_model_path << ".bin (Korean/Multilingual)" << std::endl;
    std::cout << std::endl;

    try {
        // Create WhisperBackend starting with English
        std::cout << "1. Initializing WhisperBackend with English..." << std::endl;
        WhisperBackend backend(base_model_path, Language::English, demoCallback);

        // Generate some dummy audio
        std::vector<float> audio_chunk = generateSilence(16000);  // 1 second of silence

        std::cout << "2. Processing audio with English model..." << std::endl;
        backend.processAudio(audio_chunk, SpeechTag::Start);

        // Wait a bit for processing
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Switch to Korean
        std::cout << "\n3. Switching to Korean model..." << std::endl;
        if (backend.setLanguage(Language::Korean)) {
            std::cout << "✓ Successfully switched to Korean model!" << std::endl;
        } else {
            std::cout << "✗ Failed to switch to Korean model!" << std::endl;
            return 1;
        }

        std::cout << "4. Processing audio with Korean model..." << std::endl;
        backend.processAudio(audio_chunk, SpeechTag::Continue);

        // Wait a bit for processing
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Switch back to English
        std::cout << "\n5. Switching back to English model..." << std::endl;
        if (backend.setLanguage(Language::English)) {
            std::cout << "✓ Successfully switched back to English model!" << std::endl;
        } else {
            std::cout << "✗ Failed to switch back to English model!" << std::endl;
            return 1;
        }

        std::cout << "6. Processing final audio with English model..." << std::endl;
        backend.processAudio(audio_chunk, SpeechTag::End);

        // Wait for final processing
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cout << "\n✓ Language switching demo completed successfully!" << std::endl;
        std::cout << "\nKey features demonstrated:" << std::endl;
        std::cout << "  - Dynamic language switching without recreating backend" << std::endl;
        std::cout << "  - Automatic model unloading and reloading" << std::endl;
        std::cout << "  - Seamless audio processing across language changes" << std::endl;
        std::cout << "  - Thread-safe language switching" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
