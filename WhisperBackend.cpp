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
#include <map>
#include <stdexcept>

namespace asr {

// Internal implementation class using Pimpl pattern
class WhisperBackend::Impl {
public:
    Impl(const std::string& baseModelPath, Language language, AsrEventCallback callback)
        : mBaseModelPath(baseModelPath)
        , mUseCustomModels(false)
        , mCurrentLanguage(language)
        , mCallback(std::move(callback))
        , mRunning(false)
        , mCtx(nullptr)
        , mSampleRate(16000)
        , mLastPartialResult("")
        , mKeepSamples((mSampleRate * 200) / 1000)  // 200ms worth of samples
        , mInSpeechSequence(false)
    {
        initializeWhisper();
    }

    Impl(const std::map<Language, std::string>& languageModels, Language language, AsrEventCallback callback)
        : mCustomLanguageModels(languageModels)
        , mUseCustomModels(true)
        , mCurrentLanguage(language)
        , mCallback(std::move(callback))
        , mRunning(false)
        , mCtx(nullptr)
        , mSampleRate(16000)
        , mLastPartialResult("")
        , mKeepSamples((mSampleRate * 200) / 1000)  // 200ms worth of samples
        , mInSpeechSequence(false)
    {
        initializeWhisper();
    }

    ~Impl() {
        stop();
        if (mCtx) {
            whisper_free(mCtx);
        }
    }

    void processAudio(const std::vector<float>& audio, SpeechTag speechTag) {
        if (!mRunning || !mCtx) {
            return;
        }

        // Process each chunk immediately - no accumulation
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);

            // Always queue the chunk for immediate processing
            if (!audio.empty()) {
                mAudioQueue.push({audio, speechTag});
                mQueueCondition.notify_one();
            } else if (speechTag == SpeechTag::End) {
                // Even with empty audio, process End tag to signal completion
                mAudioQueue.push({audio, speechTag});
                mQueueCondition.notify_one();
            }
        }
    }

    void start() {
        if (mRunning || !mCtx) {
            return;
        }

        mRunning = true;
        mWorkerThread = std::thread(&Impl::workerLoop, this);
    }

    void stop() {
        mRunning = false;

        // Notify worker thread to wake up and exit
        mQueueCondition.notify_all();

        if (mWorkerThread.joinable()) {
            mWorkerThread.join();
        }
    }

    bool setLanguage(Language language) {
        if (language == mCurrentLanguage) {
            return true;  // Already using the requested language
        }

        // Stop processing temporarily
        bool was_running = mRunning;
        if (was_running) {
            stop();
        }

        // Unload current model
        if (mCtx) {
            whisper_free(mCtx);
            mCtx = nullptr;
        }

        // Update language and reload model
        mCurrentLanguage = language;
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
        // If using custom models, return the specific path for this language
        if (mUseCustomModels) {
            auto it = mCustomLanguageModels.find(language);
            if (it != mCustomLanguageModels.end()) {
                return it->second;
            } else {
                throw std::runtime_error("No model configured for language: " + languageToString(language));
            }
        }

        // Use base model path with automatic suffixes
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
        std::string model_path = mBaseModelPath;
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
        std::string model_path = buildModelPath(mCurrentLanguage);

        // Check if model file exists
        std::ifstream model_check(model_path);
        if (!model_check.good()) {
            std::cerr << "Error: Could not find " << languageToString(mCurrentLanguage)
                      << " model file: " << model_path << std::endl;
            return false;
        }
        model_check.close();

        // Initialize whisper context
        struct whisper_context_params cparams = whisper_context_default_params();
        mCtx = whisper_init_from_file_with_params(model_path.c_str(), cparams);

        if (!mCtx) {
            std::cerr << "Error: Failed to initialize whisper context from " << model_path << std::endl;
            return false;
        }

        // Set up whisper parameters for streaming
        mParams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        mCurrentLanguageCode = languageToCode(mCurrentLanguage);
        mParams.language = mCurrentLanguageCode.c_str();
        mParams.translate = false;
        mParams.print_realtime = false;
        mParams.print_progress = false;
        mParams.print_timestamps = false;
        mParams.print_special = false;
        mParams.no_context = true;       // Will be set dynamically based on speech tag
        mParams.single_segment = false;
        mParams.suppress_blank = true;   // Suppress blank outputs
        mParams.suppress_nst = true;
        mParams.prompt_tokens = nullptr; // Will be set dynamically for context
        mParams.prompt_n_tokens = 0;

        std::cout << "✓ WhisperBackend initialized successfully!" << std::endl;
        return true;
    }

    void workerLoop() {
        while (mRunning) {
            AudioChunk chunk;

            // Wait for audio data or shutdown signal
            {
                std::unique_lock<std::mutex> lock(mQueueMutex);
                mQueueCondition.wait(lock, [this] {
                    return !mRunning || !mAudioQueue.empty();
                });

                if (!mRunning) {
                    break;
                }

                if (mAudioQueue.empty()) {
                    continue;
                }

                chunk = std::move(mAudioQueue.front());
                mAudioQueue.pop();
            }

            // Process the audio chunk
            processAudioChunk(chunk);
        }
    }

    void processAudioChunk(const AudioChunk& chunk) {
        if (!mCtx) {
            return;
        }

        // Skip processing empty audio unless it's an End tag
        if (chunk.audio.empty() && chunk.speechTag != SpeechTag::End) {
            return;
        }

        std::string combined_text;
        std::vector<float> processAudio;

        // Process audio if we have samples
        if (!chunk.audio.empty()) {
            // Handle speech sequence state and audio buffering
            processAudio = prepareAudioWithContext(chunk.audio, chunk.speechTag);

            // Set context behavior based on speech tag (following stream.cpp logic)
            setupContextForSpeechTag(chunk.speechTag);

            const int result = whisper_full(mCtx, mParams, processAudio.data(), static_cast<int>(processAudio.size()));

            if (result == 0) {
                // Get transcription results and update context tokens
                combined_text = extractTextAndUpdateContext();
            } else {
                // Error in processing
                mCallback(ResultTag::Error, "Failed to process audio chunk");
                return;
            }
        }

        // Handle context management and callbacks based on speech tag
        handleSpeechTagContext(chunk.speechTag, combined_text);
    }

    // Helper method to prepare audio with context overlap (following stream.cpp logic)
    std::vector<float> prepareAudioWithContext(const std::vector<float>& newAudio, SpeechTag speechTag) {
        std::vector<float> processAudio;

        if (speechTag == SpeechTag::Start) {
            // For Start tag, clear any previous audio buffer and start fresh
            mAudioBuffer.clear();
            mInSpeechSequence = true;
            processAudio = newAudio;
        } else if (speechTag == SpeechTag::Continue && mInSpeechSequence) {
            // For Continue tag, combine overlap from previous audio with new audio
            if (!mAudioBuffer.empty()) {
                // Take up to mKeepSamples from the end of previous audio
                int samples_to_take = std::min(static_cast<int>(mAudioBuffer.size()), mKeepSamples);

                processAudio.reserve(samples_to_take + newAudio.size());

                // Add overlap samples from previous audio
                auto start_it = mAudioBuffer.end() - samples_to_take;
                processAudio.insert(processAudio.end(), start_it, mAudioBuffer.end());

                // Add new audio
                processAudio.insert(processAudio.end(), newAudio.begin(), newAudio.end());
            } else {
                // No previous audio buffer, just use new audio
                processAudio = newAudio;
            }
        } else {
            // For End tag or if not in speech sequence, just use new audio
            processAudio = newAudio;
        }

        // Update audio buffer for next iteration (only if in speech sequence)
        if (mInSpeechSequence && !processAudio.empty()) {
            mAudioBuffer = processAudio;
        }

        return processAudio;
    }

    // Setup whisper context parameters based on speech tag
    void setupContextForSpeechTag(SpeechTag speechTag) {
        if (speechTag == SpeechTag::Start) {
            // For Start tag, reset context (fresh start)
            mParams.no_context = true;
            mParams.prompt_tokens = nullptr;
            mParams.prompt_n_tokens = 0;
        } else if (speechTag == SpeechTag::Continue && mInSpeechSequence) {
            // For Continue tags, use accumulated prompt tokens for context
            if (!mPromptTokens.empty()) {
                mParams.no_context = false;
                mParams.prompt_tokens = mPromptTokens.data();
                mParams.prompt_n_tokens = static_cast<int>(mPromptTokens.size());
            } else {
                // No context tokens yet, start without context
                mParams.no_context = true;
                mParams.prompt_tokens = nullptr;
                mParams.prompt_n_tokens = 0;
            }
        } else {
            // For End tag, use context if available
            if (!mPromptTokens.empty() && mInSpeechSequence) {
                mParams.no_context = false;
                mParams.prompt_tokens = mPromptTokens.data();
                mParams.prompt_n_tokens = static_cast<int>(mPromptTokens.size());
            } else {
                mParams.no_context = true;
                mParams.prompt_tokens = nullptr;
                mParams.prompt_n_tokens = 0;
            }
        }
    }

    // Extract text from segments and update context tokens (following stream.cpp approach)
    std::string extractTextAndUpdateContext() {
        std::string combined_text;
        const int n_segments = whisper_full_n_segments(mCtx);

        // Collect text from all segments
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(mCtx, i);

            if (text && strlen(text) > 0) {
                std::string text_str(text);
                // Remove leading/trailing whitespace
                text_str.erase(0, text_str.find_first_not_of(" \t\n\r"));
                text_str.erase(text_str.find_last_not_of(" \t\n\r") + 1);

                if (!text_str.empty()) {
                    if (!combined_text.empty()) {
                        combined_text += " ";
                    }
                    combined_text += text_str;
                }
            }
        }

        // Update prompt tokens for context (following stream.cpp logic)
        if (mInSpeechSequence && n_segments > 0) {
            // Add tokens from all segments to prompt tokens for next iteration
            for (int i = 0; i < n_segments; ++i) {
                const int token_count = whisper_full_n_tokens(mCtx, i);
                for (int j = 0; j < token_count; ++j) {
                    mPromptTokens.push_back(whisper_full_get_token_id(mCtx, i, j));
                }
            }
        }

        return combined_text;
    }

    // Handle speech tag context and callbacks
    void handleSpeechTagContext(SpeechTag speechTag, const std::string& combined_text) {
        switch (speechTag) {
            case SpeechTag::Start:
                // Clear context for new speech sequence
                mPromptTokens.clear();
                mLastPartialResult = "";
                mInSpeechSequence = true;

                // Provide partial result only if not empty
                if (!combined_text.empty()) {
                    mLastPartialResult = combined_text;
                    mCallback(ResultTag::Partial, combined_text);
                }
                break;

            case SpeechTag::Continue:
                // Provide partial result only if not empty and different from last
                if (!combined_text.empty() && combined_text != mLastPartialResult) {
                    mLastPartialResult = combined_text;
                    mCallback(ResultTag::Partial, combined_text);
                }
                break;

            case SpeechTag::End:
                // Always call final callback for End tag (even if empty)
                mCallback(ResultTag::Final, combined_text);

                // Reset context after speech sequence ends
                mPromptTokens.clear();
                mAudioBuffer.clear();
                mLastPartialResult = "";
                mInSpeechSequence = false;
                break;
        }
    }

    std::string mBaseModelPath;
    std::map<Language, std::string> mCustomLanguageModels;
    bool mUseCustomModels;
    Language mCurrentLanguage;
    std::string mCurrentLanguageCode;
    AsrEventCallback mCallback;
    std::atomic<bool> mRunning;
    struct whisper_context* mCtx;
    struct whisper_full_params mParams;

    // Audio processing parameters
    const int mSampleRate;

    // Worker thread and synchronization
    std::thread mWorkerThread;
    std::mutex mQueueMutex;
    std::condition_variable mQueueCondition;
    std::queue<AudioChunk> mAudioQueue;

    // Note: No longer using audio accumulation - processing chunks immediately

    // Track last partial result to avoid duplicates
    std::string mLastPartialResult;

    // Context management (following stream.cpp approach)
    std::vector<whisper_token> mPromptTokens;  // Tokens from previous segments for context
    std::vector<float> mAudioBuffer;           // Audio overlap buffer for continuity
    const int mKeepSamples;                    // Number of samples to keep for overlap (200ms worth)
    bool mInSpeechSequence;                    // Track if we're in a speech sequence (Start->Continue->End)
};

// WhisperBackend public interface implementation
WhisperBackend::WhisperBackend(const std::string& baseModelPath, Language language, AsrEventCallback asrEventCallback)
    : mImpl(std::make_unique<Impl>(baseModelPath, language, std::move(asrEventCallback)))
{
    mImpl->start();
}

WhisperBackend::WhisperBackend(const WhisperBackendBuilder& builder)
    : mImpl(std::make_unique<Impl>(builder.languageModels_, builder.initialLanguage_, builder.callback_))
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

// WhisperBackendBuilder implementation
WhisperBackendBuilder::WhisperBackendBuilder()
    : initialLanguage_(Language::English)
    , hasCallback_(false)
{
}

WhisperBackendBuilder& WhisperBackendBuilder::setCallback(AsrEventCallback callback) {
    callback_ = std::move(callback);
    hasCallback_ = true;
    return *this;
}

WhisperBackendBuilder& WhisperBackendBuilder::setInitialLanguage(Language language) {
    initialLanguage_ = language;
    return *this;
}

WhisperBackendBuilder& WhisperBackendBuilder::setModelForLanguage(Language language, const std::string& modelPath) {
    languageModels_[language] = modelPath;
    return *this;
}

WhisperBackendBuilder& WhisperBackendBuilder::setBaseModelPath(const std::string& baseModelPath) {
    // Clear any existing custom models
    languageModels_.clear();

    // Set up automatic model paths
    languageModels_[Language::English] = baseModelPath + ".en.bin";
    languageModels_[Language::Korean] = baseModelPath + ".bin";

    return *this;
}

std::unique_ptr<WhisperBackend> WhisperBackendBuilder::build() const {
    if (!hasCallback_) {
        throw std::runtime_error("Callback must be set before building WhisperBackend");
    }

    if (languageModels_.empty()) {
        throw std::runtime_error("At least one model must be configured before building WhisperBackend");
    }

    // Check that initial language has a model
    if (languageModels_.find(initialLanguage_) == languageModels_.end()) {
        throw std::runtime_error("No model configured for initial language");
    }

    return std::unique_ptr<WhisperBackend>(new WhisperBackend(*this));
}

} // namespace asr
