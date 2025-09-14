#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <atomic>
#include <signal.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include "WhisperBackend.h"
#include "AudioStreamer.h"

using namespace asr;

// Convert int16 audio samples to float and normalize
std::vector<float> convert_to_float(const std::vector<short>& int16_samples) {
    std::vector<float> float_samples;
    float_samples.reserve(int16_samples.size());

    for (short sample : int16_samples) {
        float_samples.push_back(static_cast<float>(sample) / 32768.0f);
    }

    return float_samples;
}

// Callback function for ASR events
void asrEventCallback(ResultTag resultTag, const std::string& text) {
    switch (resultTag) {
        case ResultTag::Partial:
            std::cout << "PARTIAL: " << text << std::endl;
            break;
        case ResultTag::Final:
            std::cout << "FINAL: " << text << std::endl;
            std::cout << "---" << std::endl;
            break;
        case ResultTag::Error:
            std::cerr << "ERROR: " << text << std::endl;
            break;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== WhisperBackend Live Streaming ASR Example ===" << std::endl;
    std::cout << "This example demonstrates the WhisperBackend API for live audio streaming." << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  - Live speech recognition" << std::endl;
    std::cout << "  - Language switching (English/Korean)" << std::endl;
    std::cout << "Speak into your microphone and see real-time transcription results." << std::endl;
    std::cout << "Press 'e' for English, 'k' for Korean, Ctrl+C to stop." << std::endl;
    std::cout << std::endl;

    // Default base model path (without language suffix)
    std::string base_model_path = "resources/ggml-small";

    // Check command line arguments
    if (argc >= 2) {
        base_model_path = argv[1];
    }

    std::cout << "Base model path: " << base_model_path << std::endl;

    // Create WhisperBackend with callback (start with English)
    std::cout << "Initializing WhisperBackend with English model..." << std::endl;
    WhisperBackend backend(base_model_path, Language::English, asrEventCallback);

    // Initialize AudioStreamer
    const size_t chunk_size_ms = 100;  // 100ms chunks for more responsive streaming
    const int sample_rate = 16000;
    const int channels = 1;

    std::cout << "Initializing AudioStreamer..." << std::endl;
    AudioStreamer streamer(chunk_size_ms, sample_rate, channels);

    std::cout << "Starting audio capture..." << std::endl;
    streamer.start();

    if (!streamer.isRunning()) {
        std::cerr << "Error: Failed to start audio streaming. Check if your microphone is available." << std::endl;
        return 1;
    }

    std::cout << "✓ Audio streaming started!" << std::endl;
    std::cout << "Speak now..." << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  Type 'e' + Enter: Switch to English" << std::endl;
    std::cout << "  Type 'k' + Enter: Switch to Korean" << std::endl;
    std::cout << "  Ctrl+C: Exit" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Track speech state
    bool in_speech = false;
    auto last_audio_time = std::chrono::steady_clock::now();
    const std::chrono::milliseconds silence_threshold(1000);  // 1 second of silence

    // Language switching check interval
    auto last_input_check = std::chrono::steady_clock::now();
    const std::chrono::milliseconds input_check_interval(100);

    // Main processing loop
    int chunk_count = 0;
    while (streamer.isRunning()) {
        std::vector<short> audio_chunk;

        // Get audio chunk
        if (streamer.popChunk(audio_chunk)) {
            chunk_count++;

            // Convert to float
            std::vector<float> float_audio = convert_to_float(audio_chunk);

            // Simple voice activity detection (energy-based)
            float energy = 0.0f;
            for (float sample : float_audio) {
                energy += sample * sample;
            }
            energy /= float_audio.size();

            const float energy_threshold = 0.0001f;  // Adjust based on your environment
            bool has_voice = energy > energy_threshold;

            auto current_time = std::chrono::steady_clock::now();

            if (has_voice) {
                last_audio_time = current_time;

                if (!in_speech) {
                    // Speech started
                    in_speech = true;
                    std::cout << "[Speech started]" << std::endl;
                    backend.processAudio(float_audio, SpeechTag::Start);
                } else {
                    // Speech continuing
                    backend.processAudio(float_audio, SpeechTag::Continue);
                }
            } else {
                // Check if we should end speech due to silence
                auto silence_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - last_audio_time);

                if (in_speech && silence_duration > silence_threshold) {
                    // Speech ended
                    in_speech = false;
                    std::cout << "[Speech ended]" << std::endl;
                    backend.processAudio(float_audio, SpeechTag::End);
                } else if (in_speech) {
                    // Still in speech, but silent chunk
                    backend.processAudio(float_audio, SpeechTag::Continue);
                }
            }

            // Progress indicator
            if (chunk_count % 100 == 0) {
                std::cout << "." << std::flush;
            }
        }

        // Check for language switching commands
        auto current_time = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_input_check) > input_check_interval) {
            // Check if there's input available (non-blocking)
            int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
            fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

            char input;
            if (read(STDIN_FILENO, &input, 1) > 0) {
                if (input == 'e' || input == 'E') {
                    std::cout << "\n[Switching to English model...]" << std::endl;
                    if (backend.setLanguage(Language::English)) {
                        std::cout << "[Now using English model]" << std::endl;
                    } else {
                        std::cout << "[Failed to switch to English model]" << std::endl;
                    }
                } else if (input == 'k' || input == 'K') {
                    std::cout << "\n[Switching to Korean model...]" << std::endl;
                    if (backend.setLanguage(Language::Korean)) {
                        std::cout << "[Now using Korean model]" << std::endl;
                    } else {
                        std::cout << "[Failed to switch to Korean model]" << std::endl;
                    }
                }
            }

            // Restore blocking mode
            fcntl(STDIN_FILENO, F_SETFL, flags);
            last_input_check = current_time;
        }

        // Small delay to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << std::endl << "Stopping audio streamer..." << std::endl;
    streamer.stop();

    std::cout << "✓ Example completed!" << std::endl;
    return 0;
}
