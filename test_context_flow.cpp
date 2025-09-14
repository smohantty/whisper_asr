#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "WhisperBackend.h"

// Simple test to verify context management
void testContextCallback(asr::ResultTag resultTag, const std::string& text) {
    std::cout << "[";
    switch (resultTag) {
        case asr::ResultTag::Partial:
            std::cout << "PARTIAL";
            break;
        case asr::ResultTag::Final:
            std::cout << "FINAL  ";
            break;
        case asr::ResultTag::Error:
            std::cout << "ERROR  ";
            break;
    }
    std::cout << "] " << text << std::endl;
}

int main() {
    std::cout << "=== WhisperBackend Context Management Test ===" << std::endl;

    // Create WhisperBackend
    std::unique_ptr<asr::WhisperBackend> backend;
    try {
        backend = std::make_unique<asr::WhisperBackend>(
            "resources/ggml-small",
            asr::Language::English,
            testContextCallback
        );
    } catch (const std::exception& e) {
        std::cerr << "Error initializing WhisperBackend: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nTesting SpeechTag flow: Start -> Continue -> Continue -> End" << std::endl;
    std::cout << "================================================" << std::endl;

    // Create some dummy audio data (silence)
    std::vector<float> audioChunk(16000, 0.0f); // 1 second of silence

    // Test SpeechTag flow
    std::cout << "\n1. Sending Start tag..." << std::endl;
    backend->processAudio(audioChunk, asr::SpeechTag::Start);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\n2. Sending Continue tag..." << std::endl;
    backend->processAudio(audioChunk, asr::SpeechTag::Continue);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\n3. Sending another Continue tag..." << std::endl;
    backend->processAudio(audioChunk, asr::SpeechTag::Continue);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\n4. Sending End tag..." << std::endl;
    backend->processAudio(audioChunk, asr::SpeechTag::End);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\nTesting another speech sequence..." << std::endl;
    std::cout << "=====================================\n" << std::endl;

    std::cout << "5. Sending new Start tag..." << std::endl;
    backend->processAudio(audioChunk, asr::SpeechTag::Start);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\n6. Sending End tag..." << std::endl;
    backend->processAudio(audioChunk, asr::SpeechTag::End);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}
