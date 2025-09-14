#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <cmath>

// Include WhisperBackend and AudioStreamer headers
#include "WhisperBackend.h"
#include "AudioStreamer.h"

// Global flag for graceful shutdown
std::atomic<bool> g_running(true);

// Global variables for transcription results and statistics
std::mutex g_output_mutex;
int g_total_transcriptions = 0;
auto g_start_time = std::chrono::steady_clock::now();

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    g_running = false;
}

// Convert int16 audio samples to float and normalize
std::vector<float> convert_to_float(const std::vector<short>& int16_samples) {
    std::vector<float> float_samples;
    float_samples.reserve(int16_samples.size());

    for (short sample : int16_samples) {
        float_samples.push_back(static_cast<float>(sample) / 32768.0f);
    }

    return float_samples;
}

// Callback function for handling ASR results
void asrEventCallback(asr::ResultTag resultTag, const std::string& text) {
    std::lock_guard<std::mutex> lock(g_output_mutex);

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - g_start_time).count();

    switch (resultTag) {
        case asr::ResultTag::Partial:
            if (!text.empty()) {
                std::cout << "[" << elapsed << "s] [PARTIAL] " << text << std::endl;
            }
            break;
        case asr::ResultTag::Final:
            g_total_transcriptions++;
            if (!text.empty()) {
                std::cout << "[" << elapsed << "s] [FINAL]   " << text << std::endl;
                std::cout << std::string(50, '-') << std::endl;
            } else {
                std::cout << "[" << elapsed << "s] [No speech detected]" << std::endl;
                std::cout << std::string(50, '-') << std::endl;
            }
            break;
        case asr::ResultTag::Error:
            std::cerr << "[" << elapsed << "s] [ERROR]   " << text << std::endl;
            break;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Whisper Livestream ASR Test Application (WhisperBackend) ===" << std::endl;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Default model path
    std::string model_path = "resources/ggml-small";

    // Check command line arguments
    if (argc >= 2) {
        model_path = argv[1];
    }

    std::cout << "Base model path: " << model_path << std::endl;

    // Check if model file exists (check for .en.bin version)
    std::string test_model_path = model_path + ".en.bin";
    std::ifstream model_check(test_model_path);
    if (!model_check.good()) {
        std::cerr << "Error: Model file not found: " << test_model_path << std::endl;
        std::cerr << "Please ensure the model file exists or provide a valid base path." << std::endl;
        return 1;
    }
    model_check.close();

    std::cout << "\nInitializing WhisperBackend..." << std::endl;

    // Create WhisperBackend using base model path
    std::unique_ptr<asr::WhisperBackend> whisperBackend;
    try {
        whisperBackend = std::make_unique<asr::WhisperBackend>(
            model_path,
            asr::Language::English,
            asrEventCallback
        );
        std::cout << "✓ WhisperBackend initialized successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to initialize WhisperBackend: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nTranscription parameters:" << std::endl;
    std::cout << "  Language: English" << std::endl;
    std::cout << "  Real-time processing: enabled" << std::endl;
    std::cout << "  Chunk-based processing: enabled (managed by WhisperBackend)" << std::endl;

    // Initialize AudioStreamer
    // Parameters: chunk size in ms, sample rate, channels
    // We'll use 1000ms chunks (1 second)
    const size_t chunk_size_ms = 1000;
    const int sample_rate = 16000;
    const int channels = 1;

    AudioStreamer streamer(chunk_size_ms, sample_rate, channels);

    std::cout << "\nAudio streaming parameters:" << std::endl;
    std::cout << "  Chunk size: " << chunk_size_ms << " ms" << std::endl;
    std::cout << "  Sample rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << channels << std::endl;

    std::cout << "\nStarting audio capture..." << std::endl;
    std::cout << "Speak into your microphone. Press Ctrl+C to stop." << std::endl;
    std::cout << "Make sure your microphone is working and 'arecord' is available." << std::endl;
    std::cout << "\n" << std::string(50, '=') << std::endl;

    // Start the audio streamer
    streamer.start();

    if (!streamer.isRunning()) {
        std::cerr << "Error: Failed to start audio streaming. Check if your microphone is available." << std::endl;
        return 1;
    }

    std::cout << "✓ Audio streaming started!" << std::endl;
    std::cout << "Listening for speech..." << std::endl;

    int chunk_count = 0;
    g_start_time = std::chrono::steady_clock::now();
    bool speech_started = false;

    // Main processing loop
    while (g_running && streamer.isRunning()) {
        std::vector<short> audio_chunk;

        // Get audio chunk (this will block until data is available)
        if (streamer.popChunk(audio_chunk)) {
            chunk_count++;

            // Convert to float
            std::vector<float> float_audio = convert_to_float(audio_chunk);

            // Simple voice activity detection (very basic - just check for non-zero samples)
            bool has_audio = false;
            for (float sample : float_audio) {
                if (std::abs(sample) > 0.01f) {  // Threshold for detecting audio
                    has_audio = true;
                    break;
                }
            }

            // Determine speech tag
            asr::SpeechTag speechTag = asr::SpeechTag::Continue;
            if (has_audio && !speech_started) {
                speechTag = asr::SpeechTag::Start;
                speech_started = true;
                std::cout << "\n[Speech detected - starting transcription]" << std::endl;
            } else if (!has_audio && speech_started) {
                speechTag = asr::SpeechTag::End;
                speech_started = false;
            }

            // Process audio through WhisperBackend
            whisperBackend->processAudio(float_audio, speechTag);

            // Show progress indicator every 5 chunks
            if (chunk_count % 5 == 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - g_start_time).count();

                std::lock_guard<std::mutex> lock(g_output_mutex);
                std::cout << "Processing... (" << elapsed << "s, " << chunk_count << " chunks)" << std::endl;
            }
        } else {
            // No more audio data available, streamer might have stopped
            if (!streamer.isRunning()) {
                std::cout << "Audio streaming stopped." << std::endl;
                break;
            }
        }

        // Small delay to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop the streamer
    std::cout << "\nStopping audio streamer..." << std::endl;
    streamer.stop();

    // Signal end of speech if we were in the middle of processing
    if (speech_started) {
        std::vector<float> empty_audio;
        whisperBackend->processAudio(empty_audio, asr::SpeechTag::End);
    }

    // Give WhisperBackend time to process any remaining audio
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Print statistics
    std::cout << "\n=== SESSION STATISTICS ===" << std::endl;
    std::cout << "Total chunks processed: " << chunk_count << std::endl;
    std::cout << "Total transcriptions: " << g_total_transcriptions << std::endl;

    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - g_start_time).count();
    std::cout << "Total duration: " << total_duration << " seconds" << std::endl;

    // Cleanup (WhisperBackend will cleanup automatically when destroyed)
    whisperBackend.reset();

    std::cout << "\n✓ Cleanup completed!" << std::endl;
    std::cout << "\n=== Livestream ASR test completed! ===" << std::endl;

    return 0;
}
