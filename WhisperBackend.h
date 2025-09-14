#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <map>

namespace asr
{
    enum class Language
    {
        English,    // "en"
        Korean      // "ko"
    };

    enum class SpeechTag
    {
        Start,
        Continue,
        End,
    };

    enum class ResultTag
    {
        Partial,
        Final,
        Error
    };

    using AsrEventCallback = std::function<void(ResultTag resultTag, const std::string& text)>;

// Forward declaration for builder
class WhisperBackendBuilder;

class WhisperBackend {
    public:
        // Traditional constructor
        WhisperBackend(const std::string& baseModelPath, Language language, AsrEventCallback asrEventCallback);

        // Builder constructor
        WhisperBackend(const WhisperBackendBuilder& builder);

        ~WhisperBackend();

        void processAudio(const std::vector<float>& audio, SpeechTag speechTag);

        // Language switching API
        bool setLanguage(Language language);

        private:
        class Impl;                  // Forward declaration
        std::unique_ptr<Impl> mImpl;  // Pimpl pointer
};

// Builder class for configuring WhisperBackend
class WhisperBackendBuilder {
    public:
        WhisperBackendBuilder();

        // Set callback function
        WhisperBackendBuilder& setCallback(AsrEventCallback callback);

        // Set initial language
        WhisperBackendBuilder& setInitialLanguage(Language language);

        // Configure model for specific language
        WhisperBackendBuilder& setModelForLanguage(Language language, const std::string& modelPath);

        // Set models using base path (convenience method)
        WhisperBackendBuilder& setBaseModelPath(const std::string& baseModelPath);

        // Build the WhisperBackend instance
        std::unique_ptr<WhisperBackend> build() const;

        // Allow WhisperBackend to access private members
        friend class WhisperBackend;

    private:
        AsrEventCallback callback_;
        Language initialLanguage_;
        std::map<Language, std::string> languageModels_;
        bool hasCallback_;
};
} // namespace asr

