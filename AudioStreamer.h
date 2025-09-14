#pragma once
#include <vector>
#include <memory>

class AudioStreamer {
public:
    AudioStreamer(size_t chunkSizeMs = 10, int sampleRate = 16000, int channels = 1);
    ~AudioStreamer();

    void start();
    void stop();

    // Blocking: waits until data is available or stopped
    bool popChunk(std::vector<short>& outChunk);

    // Check if the streamer is currently running
    bool isRunning() const;

private:
    class Impl;                  // Forward declaration
    std::unique_ptr<Impl> mImpl;  // Pimpl pointer
};
