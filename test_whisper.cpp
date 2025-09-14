#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdio>

// Include whisper header
#include "whisper.h"

// Simple WAV file reader for 16-bit mono files
std::vector<float> read_wav_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open audio file: " << filename << std::endl;
        return {};
    }

    // Skip WAV header (44 bytes for standard WAV)
    file.seekg(44);

    // Read audio data
    std::vector<int16_t> audio_data_int16;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        audio_data_int16.push_back(sample);
    }

    file.close();

    // Convert to float and normalize
    std::vector<float> audio_data_float;
    audio_data_float.reserve(audio_data_int16.size());
    for (int16_t sample : audio_data_int16) {
        audio_data_float.push_back(static_cast<float>(sample) / 32768.0f);
    }

    std::cout << "Loaded " << audio_data_float.size() << " audio samples from " << filename << std::endl;
    return audio_data_float;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Whisper.cpp Test Application ===" << std::endl;

    // Default paths - try local files first, then parent directory
    std::string model_path = "ggml-base.en.bin";
    std::string audio_path = "jfk.wav";

    // Check if local files exist, otherwise use parent directory paths
    std::ifstream model_check(model_path);
    if (!model_check.good()) {
        model_path = "../whisper.cpp-1.7.6/models/ggml-base.en.bin";
    }
    model_check.close();

    std::ifstream audio_check(audio_path);
    if (!audio_check.good()) {
        audio_path = "../whisper.cpp-1.7.6/samples/jfk.wav";
    }
    audio_check.close();

    // Check command line arguments
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        audio_path = argv[2];
    }

    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Audio path: " << audio_path << std::endl;

    // Initialize whisper context
    std::cout << "\nInitializing Whisper model..." << std::endl;
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Error: Failed to initialize whisper context from " << model_path << std::endl;
        return 1;
    }
    std::cout << "✓ Whisper model loaded successfully!" << std::endl;

    // Load audio file
    std::cout << "\nLoading audio file..." << std::endl;
    std::vector<float> audio_data = read_wav_file(audio_path);
    if (audio_data.empty()) {
        std::cerr << "Error: Failed to load audio file" << std::endl;
        whisper_free(ctx);
        return 1;
    }
    std::cout << "✓ Audio file loaded successfully!" << std::endl;

    // Set up whisper parameters
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = "en";
    params.translate = false;
    params.print_realtime = false;
    params.print_progress = true;
    params.print_timestamps = true;
    params.print_special = false;
    params.no_context = true;
    params.single_segment = false;

    std::cout << "\nTranscription parameters:" << std::endl;
    std::cout << "  Language: " << (params.language ? params.language : "auto") << std::endl;
    std::cout << "  Translate: " << (params.translate ? "yes" : "no") << std::endl;
    std::cout << "  Print timestamps: " << (params.print_timestamps ? "yes" : "no") << std::endl;

    // Process audio
    std::cout << "\nProcessing audio..." << std::endl;
    const int result = whisper_full(ctx, params, audio_data.data(), static_cast<int>(audio_data.size()));
    if (result != 0) {
        std::cerr << "Error: Failed to process audio (error code: " << result << ")" << std::endl;
        whisper_free(ctx);
        return 1;
    }
    std::cout << "✓ Audio processing completed!" << std::endl;

    // Get and display transcription results
    std::cout << "\n=== TRANSCRIPTION RESULTS ===" << std::endl;
    const int n_segments = whisper_full_n_segments(ctx);
    std::cout << "Number of segments: " << n_segments << std::endl;
    std::cout << "\nTranscription:" << std::endl;

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

        // Convert timestamps to seconds
        const float t0_sec = static_cast<float>(t0) / 100.0f;
        const float t1_sec = static_cast<float>(t1) / 100.0f;

        printf("[%08.3f --> %08.3f] %s\n", t0_sec, t1_sec, text);
    }

    // Print some statistics
    std::cout << "\n=== STATISTICS ===" << std::endl;
    const auto* timings = whisper_get_timings(ctx);
    printf("Sample time:   %8.2f ms\n", timings->sample_ms);
    printf("Encode time:   %8.2f ms\n", timings->encode_ms);
    printf("Decode time:   %8.2f ms\n", timings->decode_ms);
    printf("Batch decode:  %8.2f ms\n", timings->batchd_ms);
    printf("Prompt time:   %8.2f ms\n", timings->prompt_ms);

    // Cleanup
    whisper_free(ctx);
    std::cout << "\n✓ Cleanup completed!" << std::endl;
    std::cout << "\n=== Test completed successfully! ===" << std::endl;

    return 0;
}
