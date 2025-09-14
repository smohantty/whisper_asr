#include "WhisperBackend.h"
#include "whisper.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstring>

namespace asr {

// Internal implementation class using Pimpl pattern
class WhisperBackend::Impl {
public:
    Impl(const std::string& baseModelPath, Language language, AsrEventCallback callback)
        : base_model_path_(baseModelPath)
        , current_language_(language)
        , callback_(std::move(callback))
        , running_(false)
        , ctx_(nullptr)
        , processing_duration_sec_(3)  // Process in 3-second chunks
        , sample_rate_(16000)
    {
        initializeWhisper();
    }

    ~Impl() {
        stop();
        if (ctx_) {
            whisper_free(ctx_);
        }
    }

    void processAudio(const std::vector<float>& audio, SpeechTag speechTag) {
        if (!running_ || !ctx_) {
            return;
        }

        // Lock and add audio to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);

            // Handle speech tag logic
            if (speechTag == SpeechTag::Start) {
                // Clear any accumulated audio for a fresh start
                accumulated_audio_.clear();
            }

            // Add new audio to accumulation buffer
            accumulated_audio_.insert(accumulated_audio_.end(), audio.begin(), audio.end());

            // Check if we have enough audio to process
            size_t target_samples = sample_rate_ * processing_duration_sec_;
            bool should_process = false;

            if (speechTag == SpeechTag::End) {
                // Process whatever we have when speech ends
                should_process = !accumulated_audio_.empty();
            } else if (accumulated_audio_.size() >= target_samples) {
                // Process when we have enough samples
                should_process = true;
            }

            if (should_process) {
                // Move accumulated audio to processing queue
                audio_queue_.push({accumulated_audio_, speechTag});
                accumulated_audio_.clear();

                // Notify worker thread
                queue_condition_.notify_one();
            }
        }
    }

    void start() {
        if (running_ || !ctx_) {
            return;
        }

        running_ = true;
        worker_thread_ = std::thread(&Impl::workerLoop, this);
    }

    void stop() {
        running_ = false;

        // Notify worker thread to wake up and exit
        queue_condition_.notify_all();

        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    bool setLanguage(Language language) {
        if (language == current_language_) {
            return true;  // Already using the requested language
        }

        // Stop processing temporarily
        bool was_running = running_;
        if (was_running) {
            stop();
        }

        // Unload current model
        if (ctx_) {
            whisper_free(ctx_);
            ctx_ = nullptr;
        }

        // Update language and reload model
        current_language_ = language;
        if (!initializeWhisper()) {
            std::cerr << "Error: Failed to switch to " << languageToString(language) << " model" << std::endl;
            return false;
        }

        // Restart if it was running before
        if (was_running) {
            start();
        }

        std::cout << "✓ Successfully switched to " << languageToString(language) << " model" << std::endl;
        return true;
    }

private:
    struct AudioChunk {
        std::vector<float> audio;
        SpeechTag speechTag;
    };

    std::string languageToString(Language language) const {
        switch (language) {
            case Language::English: return "English";
            case Language::Korean: return "Korean";
            default: return "Unknown";
        }
    }

    std::string languageToCode(Language language) const {
        switch (language) {
            case Language::English: return "en";
            case Language::Korean: return "ko";
            default: return "en";
        }
    }

    std::string buildModelPath(Language language) const {
        std::string lang_suffix;
        switch (language) {
            case Language::English:
                lang_suffix = ".en";
                break;
            case Language::Korean:
                // Korean models typically don't have language suffix in whisper.cpp
                lang_suffix = "";
                break;
            default:
                lang_suffix = ".en";
                break;
        }

        // Replace .bin with language-specific suffix + .bin
        std::string model_path = base_model_path_;
        size_t bin_pos = model_path.find(".bin");
        if (bin_pos != std::string::npos) {
            model_path.replace(bin_pos, 4, lang_suffix + ".bin");
        } else {
            model_path += lang_suffix + ".bin";
        }

        return model_path;
    }

    bool initializeWhisper() {
        // Build the language-specific model path
        std::string model_path = buildModelPath(current_language_);

        // Check if model file exists
        std::ifstream model_check(model_path);
        if (!model_check.good()) {
            std::cerr << "Error: Could not find " << languageToString(current_language_)
                      << " model file: " << model_path << std::endl;
            return false;
        }
        model_check.close();

        // Initialize whisper context
        struct whisper_context_params cparams = whisper_context_default_params();
        ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);

        if (!ctx_) {
            std::cerr << "Error: Failed to initialize whisper context from " << model_path << std::endl;
            return false;
        }

        // Set up whisper parameters for streaming
        params_ = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        current_language_code_ = languageToCode(current_language_);
        params_.language = current_language_code_.c_str();
        params_.translate = false;
        params_.print_realtime = false;
        params_.print_progress = false;
        params_.print_timestamps = false;
        params_.print_special = false;
        params_.no_context = true;       // Process each chunk independently
        params_.single_segment = false;
        params_.suppress_blank = true;   // Suppress blank outputs
        params_.suppress_nst = true;

        std::cout << "✓ WhisperBackend initialized successfully!" << std::endl;
        return true;
    }

    void workerLoop() {
        while (running_) {
            AudioChunk chunk;

            // Wait for audio data or shutdown signal
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_condition_.wait(lock, [this] {
                    return !running_ || !audio_queue_.empty();
                });

                if (!running_) {
                    break;
                }

                if (audio_queue_.empty()) {
                    continue;
                }

                chunk = std::move(audio_queue_.front());
                audio_queue_.pop();
            }

            // Process the audio chunk
            processAudioChunk(chunk);
        }
    }

    void processAudioChunk(const AudioChunk& chunk) {
        if (!ctx_ || chunk.audio.empty()) {
            return;
        }

        // Process with Whisper
        const int result = whisper_full(ctx_, params_, chunk.audio.data(), static_cast<int>(chunk.audio.size()));

        if (result == 0) {
            // Get transcription results
            const int n_segments = whisper_full_n_segments(ctx_);
            bool has_text = false;
            std::string combined_text;

            for (int i = 0; i < n_segments; ++i) {
                const char* text = whisper_full_get_segment_text(ctx_, i);

                if (text && strlen(text) > 0) {
                    std::string text_str(text);
                    // Remove leading/trailing whitespace
                    text_str.erase(0, text_str.find_first_not_of(" \t\n\r"));
                    text_str.erase(text_str.find_last_not_of(" \t\n\r") + 1);

                    if (!text_str.empty()) {
                        has_text = true;
                        if (!combined_text.empty()) {
                            combined_text += " ";
                        }
                        combined_text += text_str;
                    }
                }
            }

            // Call callback with results
            if (has_text) {
                ResultTag resultTag = (chunk.speechTag == SpeechTag::End) ? ResultTag::Final : ResultTag::Partial;
                callback_(resultTag, combined_text);
            } else if (chunk.speechTag == SpeechTag::End) {
                // Even if no text, call final callback to signal end
                callback_(ResultTag::Final, "");
            }
        } else {
            // Error in processing
            callback_(ResultTag::Error, "Failed to process audio chunk");
        }
    }

    std::string base_model_path_;
    Language current_language_;
    std::string current_language_code_;
    AsrEventCallback callback_;
    std::atomic<bool> running_;
    struct whisper_context* ctx_;
    struct whisper_full_params params_;

    // Audio processing parameters
    const size_t processing_duration_sec_;
    const int sample_rate_;

    // Worker thread and synchronization
    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::queue<AudioChunk> audio_queue_;

    // Audio accumulation
    std::vector<float> accumulated_audio_;
};

// WhisperBackend public interface implementation
WhisperBackend::WhisperBackend(const std::string& baseModelPath, Language language, AsrEventCallback asrEventCallback)
    : mImpl(std::make_unique<Impl>(baseModelPath, language, std::move(asrEventCallback)))
{
    mImpl->start();
}

WhisperBackend::~WhisperBackend() = default;

void WhisperBackend::processAudio(const std::vector<float>& audio, SpeechTag speechTag) {
    if (mImpl) {
        mImpl->processAudio(audio, speechTag);
    }
}

bool WhisperBackend::setLanguage(Language language) {
    if (mImpl) {
        return mImpl->setLanguage(language);
    }
    return false;
}

} // namespace asr
