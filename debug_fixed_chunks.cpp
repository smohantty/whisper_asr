#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "WhisperBackend.h"

// Simple test to check if anything is being processed
void debugCallback(asr::ResultTag resultTag, const std::string& text) {
    std::cout << "CALLBACK TRIGGERED! ";
    switch (resultTag) {
        case asr::ResultTag::Partial:
            std::cout << "[PARTIAL] " << text << std::endl;
            break;
        case asr::ResultTag::Final:
            std::cout << "[FINAL  ] " << text << std::endl;
            break;
        case asr::ResultTag::Error:
            std::cout << "[ERROR  ] " << text << std::endl;
            break;
    }
}

int main() {
    std::cout << "=== Debug Fixed Chunks Processing ===" << std::endl;

    // Create WhisperBackend
    std::unique_ptr<asr::WhisperBackend> backend;
    try {
        backend = std::make_unique<asr::WhisperBackend>(
            "resources/ggml-small",
            asr::Language::English,
            debugCallback
        );
    } catch (const std::exception& e) {
        std::cerr << "Error initializing WhisperBackend: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nSending exactly 300ms of audio (4800 samples)..." << std::endl;

    // Create exactly 300ms worth of audio (4800 samples at 16kHz)
    std::vector<float> exactChunk(4800, 0.1f); // Bigger signal

    std::cout << "1. Start tag with exactly 300ms..." << std::endl;
    backend->processAudio(exactChunk, asr::SpeechTag::Start);

    std::cout << "2. Waiting for processing..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // Wait longer

    std::cout << "3. End tag..." << std::endl;
    backend->processAudio(std::vector<float>(), asr::SpeechTag::End);

    std::cout << "4. Final wait..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    std::cout << "\nIf no callbacks were triggered, there might be an issue with the chunking logic." << std::endl;
    return 0;
}
