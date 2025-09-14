#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <string>

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


class WhisperBackend {
    public:
        WhisperBackend(const std::string& baseModelPath, Language language, AsrEventCallback asrEventCallback);
        ~WhisperBackend();

        void processAudio(const std::vector<float>& audio, SpeechTag speechTag);

        // Language switching API
        bool setLanguage(Language language);

        private:
        class Impl;                  // Forward declaration
        std::unique_ptr<Impl> mImpl;  // Pimpl pointer
};
} // namespace asr

