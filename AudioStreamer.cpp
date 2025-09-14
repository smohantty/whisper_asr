#include "AudioStreamer.h"
#include <iostream>
#include <cstdio>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>

class AudioStreamer::Impl {
public:
    Impl(size_t chunkSizeMs, int sampleRate, int channels)
        : mChunkSizeBytes(0),
          mBytesPerSample(sizeof(short)),
          mRunning(false)
    {
        size_t samplesPerChunk = (sampleRate * chunkSizeMs) / 1000;
        mChunkSizeBytes = samplesPerChunk * channels * mBytesPerSample;
    }

    ~Impl() {
        stop();
    }

    void start() {
        if (mRunning) return;
        mRunning = true;
        mWorker = std::thread(&Impl::captureLoop, this);
    }

    void stop() {
        if (!mRunning) return;
        mRunning = false;
        mCv.notify_all();
        if (mWorker.joinable()) {
            mWorker.join();
        }
        if (mPipe) {
            pclose(mPipe);
            mPipe = nullptr;
        }
    }

    bool popChunk(std::vector<short>& outChunk) {
        std::unique_lock<std::mutex> lock(mMtx);
        mCv.wait(lock, [this] { return !mQueue.empty() || !mRunning; });
        if (!mQueue.empty()) {
            outChunk = std::move(mQueue.front());
            mQueue.pop();
            return true;
        }
        return false;
    }

    bool isRunning() const {
        return mRunning;
    }

private:
    void captureLoop() {
        const char* cmd = "arecord -f S16_LE -c1 -r16000 -t raw";
        mPipe = popen(cmd, "r");
        if (!mPipe) {
            std::cerr << "Failed to start arecord\n";
            mRunning = false;
            mCv.notify_all(); // Notify waiting threads that we're done
            return;
        }

        int fd = fileno(mPipe);
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        std::vector<char> buffer(mChunkSizeBytes);

        while (mRunning) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(fd, &readfds);

            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 10000; // 10 ms

            int ret = select(fd + 1, &readfds, nullptr, nullptr, &tv);
            if (ret > 0 && FD_ISSET(fd, &readfds)) {
                size_t bytesRead = fread(buffer.data(), 1, mChunkSizeBytes, mPipe);
                if (bytesRead > 0) {
                    size_t sampleCount = bytesRead / mBytesPerSample;
                    std::vector<short> chunk(sampleCount);
                    std::memcpy(chunk.data(), buffer.data(), bytesRead);

                    {
                        std::lock_guard<std::mutex> lock(mMtx);
                        mQueue.push(std::move(chunk));
                    }
                    mCv.notify_one();
                } else if (feof(mPipe) || ferror(mPipe)) {
                    // Command terminated or encountered an error
                    std::cerr << "Audio capture command terminated unexpectedly\n";
                    mRunning = false;
                    mCv.notify_all(); // Notify waiting threads that we're done
                    break;
                }
            }
        }
    }

    size_t mChunkSizeBytes;
    size_t mBytesPerSample;

    FILE* mPipe = nullptr;
    std::thread mWorker;
    std::mutex mMtx;
    std::condition_variable mCv;
    std::queue<std::vector<short>> mQueue;
    std::atomic<bool> mRunning;
};

AudioStreamer::AudioStreamer(size_t chunkSizeMs, int sampleRate, int channels)
    : mImpl(std::make_unique<Impl>(chunkSizeMs, sampleRate, channels)) {}

AudioStreamer::~AudioStreamer() = default;
void AudioStreamer::start() { mImpl->start(); }
void AudioStreamer::stop() { mImpl->stop(); }
bool AudioStreamer::popChunk(std::vector<short>& outChunk) { return mImpl->popChunk(outChunk); }
bool AudioStreamer::isRunning() const { return mImpl->isRunning(); }
