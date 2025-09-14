#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <thread>

// Include whisper and AudioStreamer headers
#include "whisper.h"
#include "AudioStreamer.h"

// Global flag for graceful shutdown
std::atomic<bool> g_running(true);

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

// Accumulate audio chunks until we have enough for processing
class AudioAccumulator {
public:
    AudioAccumulator(size_t target_samples) : target_samples_(target_samples) {
        accumulated_samples_.reserve(target_samples);
    }

    // Add new chunk and return true if we have enough samples for processing
    bool add_chunk(const std::vector<short>& chunk) {
        accumulated_samples_.insert(accumulated_samples_.end(), chunk.begin(), chunk.end());

        if (accumulated_samples_.size() >= target_samples_) {
            return true;
        }
        return false;
    }

    // Get accumulated samples and reset for next batch
    std::vector<float> get_and_reset() {
        std::vector<float> result = convert_to_float(accumulated_samples_);
        accumulated_samples_.clear();
        return result;
    }

    size_t current_size() const {
        return accumulated_samples_.size();
    }

private:
    std::vector<short> accumulated_samples_;
    size_t target_samples_;
};

int main(int argc, char* argv[]) {
    std::cout << "=== Whisper Livestream ASR Test Application ===" << std::endl;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Default model path
    std::string model_path = "resources/ggml-small.en.bin";

    // Check command line arguments
    if (argc >= 2) {
        model_path = argv[1];
    }

    std::cout << "Model path: " << model_path << std::endl;

    // Check if model file exists
    std::ifstream model_check(model_path);
    if (!model_check.good()) {
        std::cerr << "Error: Model file not found: " << model_path << std::endl;
        std::cerr << "Please ensure the model file exists or provide a valid path." << std::endl;
        return 1;
    }
    model_check.close();

    // Initialize whisper context
    std::cout << "\nInitializing Whisper model..." << std::endl;
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Error: Failed to initialize whisper context from " << model_path << std::endl;
        return 1;
    }
    std::cout << "✓ Whisper model loaded successfully!" << std::endl;

    // Set up whisper parameters for streaming
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = "en";
    params.translate = false;
    params.print_realtime = false;
    params.print_progress = false;  // Disable progress for cleaner output
    params.print_timestamps = true;
    params.print_special = false;
    params.no_context = true;       // Process each chunk independently
    params.single_segment = false;
    params.suppress_blank = true;   // Suppress blank outputs
    params.suppress_nst = true;

    std::cout << "\nTranscription parameters:" << std::endl;
    std::cout << "  Language: " << (params.language ? params.language : "auto") << std::endl;
    std::cout << "  Real-time processing: enabled" << std::endl;
    std::cout << "  Chunk-based processing: enabled" << std::endl;

    // Initialize AudioStreamer
    // Parameters: chunk size in ms, sample rate, channels
    // We'll use 1000ms chunks (1 second) for better transcription accuracy
    const size_t chunk_size_ms = 1000;
    const int sample_rate = 16000;
    const int channels = 1;

    AudioStreamer streamer(chunk_size_ms, sample_rate, channels);

    // Calculate target samples for processing
    // We'll process audio in 3-second chunks for better accuracy
    const size_t processing_duration_sec = 3;
    const size_t target_samples = sample_rate * processing_duration_sec;

    AudioAccumulator accumulator(target_samples);

    std::cout << "\nAudio streaming parameters:" << std::endl;
    std::cout << "  Chunk size: " << chunk_size_ms << " ms" << std::endl;
    std::cout << "  Sample rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Processing duration: " << processing_duration_sec << " seconds" << std::endl;

    std::cout << "\nStarting audio capture..." << std::endl;
    std::cout << "Speak into your microphone. Press Ctrl+C to stop." << std::endl;
    std::cout << "Make sure your microphone is working and 'arecord' is available." << std::endl;
    std::cout << "\n" << std::string(50, '=') << std::endl;

    // Start the audio streamer
    streamer.start();

    if (!streamer.isRunning()) {
        std::cerr << "Error: Failed to start audio streaming. Check if your microphone is available." << std::endl;
        whisper_free(ctx);
        return 1;
    }

    std::cout << "✓ Audio streaming started!" << std::endl;
    std::cout << "Listening for speech..." << std::endl;

    int chunk_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Main processing loop
    while (g_running && streamer.isRunning()) {
        std::vector<short> audio_chunk;

        // Get audio chunk (this will block until data is available)
        if (streamer.popChunk(audio_chunk)) {
            chunk_count++;

            // Show progress indicator
            if (chunk_count % 5 == 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                std::cout << "Processing... (" << elapsed << "s, " << accumulator.current_size() << " samples accumulated)" << std::endl;
            }

            // Accumulate audio chunks
            if (accumulator.add_chunk(audio_chunk)) {
                // We have enough samples for processing
                std::vector<float> audio_data = accumulator.get_and_reset();

                std::cout << "\n--- Processing " << audio_data.size() << " samples ---" << std::endl;

                // Process with Whisper
                const int result = whisper_full(ctx, params, audio_data.data(), static_cast<int>(audio_data.size()));

                if (result == 0) {
                    // Get transcription results
                    const int n_segments = whisper_full_n_segments(ctx);

                    bool has_speech = false;
                    for (int i = 0; i < n_segments; ++i) {
                        const char* text = whisper_full_get_segment_text(ctx, i);

                        // Skip if text is empty or just whitespace
                        if (text && strlen(text) > 0) {
                            std::string text_str(text);
                            // Remove leading/trailing whitespace
                            text_str.erase(0, text_str.find_first_not_of(" \t\n\r"));
                            text_str.erase(text_str.find_last_not_of(" \t\n\r") + 1);

                            if (!text_str.empty()) {
                                has_speech = true;
                                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                                // Convert timestamps to seconds
                                const float t0_sec = static_cast<float>(t0) / 100.0f;
                                const float t1_sec = static_cast<float>(t1) / 100.0f;

                                printf("[%08.3f --> %08.3f] %s\n", t0_sec, t1_sec, text_str.c_str());
                            }
                        }
                    }

                    if (!has_speech) {
                        std::cout << "[No speech detected]" << std::endl;
                    }
                } else {
                    std::cerr << "Warning: Failed to process audio chunk (error code: " << result << ")" << std::endl;
                }

                std::cout << std::string(50, '-') << std::endl;
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

    // Process any remaining accumulated audio
    if (accumulator.current_size() > 0) {
        std::cout << "Processing remaining audio samples..." << std::endl;
        std::vector<float> remaining_audio = accumulator.get_and_reset();

        const int result = whisper_full(ctx, params, remaining_audio.data(), static_cast<int>(remaining_audio.size()));
        if (result == 0) {
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char* text = whisper_full_get_segment_text(ctx, i);
                if (text && strlen(text) > 0) {
                    std::string text_str(text);
                    text_str.erase(0, text_str.find_first_not_of(" \t\n\r"));
                    text_str.erase(text_str.find_last_not_of(" \t\n\r") + 1);

                    if (!text_str.empty()) {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                        const float t0_sec = static_cast<float>(t0) / 100.0f;
                        const float t1_sec = static_cast<float>(t1) / 100.0f;

                        printf("[%08.3f --> %08.3f] %s\n", t0_sec, t1_sec, text_str.c_str());
                    }
                }
            }
        }
    }

    // Print statistics
    std::cout << "\n=== SESSION STATISTICS ===" << std::endl;
    std::cout << "Total chunks processed: " << chunk_count << std::endl;

    auto end_time = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Total duration: " << total_duration << " seconds" << std::endl;

    const auto* timings = whisper_get_timings(ctx);
    printf("Sample time:   %8.2f ms\n", timings->sample_ms);
    printf("Encode time:   %8.2f ms\n", timings->encode_ms);
    printf("Decode time:   %8.2f ms\n", timings->decode_ms);

    // Cleanup
    whisper_free(ctx);
    std::cout << "\n✓ Cleanup completed!" << std::endl;
    std::cout << "\n=== Livestream ASR test completed! ===" << std::endl;

    return 0;
}
