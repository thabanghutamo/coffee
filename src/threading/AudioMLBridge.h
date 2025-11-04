#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <atomic>
#include <vector>

// Include necessary headers for complete types
#include "audio/PitchDetector.h"
#include "audio/RhythmAnalyzer.h"
#include "ml/ModelInference.h"

/**
 * Lock-free FIFO queue for audio thread communication
 * Single producer, single consumer (SPSC)
 * Thread-safe without mutexes for real-time safety
 */
template<typename T>
class LockFreeFIFO
{
public:
    explicit LockFreeFIFO(size_t capacity)
        : buffer(capacity + 1), // +1 for sentinel
          capacity(capacity + 1),
          head(0),
          tail(0)
    {
    }
    
    ~LockFreeFIFO() = default;
    
    /**
     * Push item to queue (call from producer thread only)
     * @return true if successful, false if queue is full
     */
    bool push(const T& item)
    {
        size_t currentTail = tail.load(std::memory_order_relaxed);
        size_t nextTail = increment(currentTail);
        
        if (nextTail == head.load(std::memory_order_acquire))
        {
            // Queue is full
            return false;
        }
        
        buffer[currentTail] = item;
        tail.store(nextTail, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Push item with move semantics
     */
    bool push(T&& item)
    {
        size_t currentTail = tail.load(std::memory_order_relaxed);
        size_t nextTail = increment(currentTail);
        
        if (nextTail == head.load(std::memory_order_acquire))
        {
            return false;
        }
        
        buffer[currentTail] = std::move(item);
        tail.store(nextTail, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Pop item from queue (call from consumer thread only)
     * @return true if successful, false if queue is empty
     */
    bool pop(T& item)
    {
        size_t currentHead = head.load(std::memory_order_relaxed);
        
        if (currentHead == tail.load(std::memory_order_acquire))
        {
            // Queue is empty
            return false;
        }
        
        item = std::move(buffer[currentHead]);
        head.store(increment(currentHead), std::memory_order_release);
        
        return true;
    }
    
    /**
     * Check if queue is empty
     */
    bool isEmpty() const
    {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }
    
    /**
     * Check if queue is full
     */
    bool isFull() const
    {
        size_t currentTail = tail.load(std::memory_order_acquire);
        size_t nextTail = increment(currentTail);
        return nextTail == head.load(std::memory_order_acquire);
    }
    
    /**
     * Get approximate size (may not be exact due to concurrent access)
     */
    size_t size() const
    {
        size_t currentHead = head.load(std::memory_order_acquire);
        size_t currentTail = tail.load(std::memory_order_acquire);
        
        if (currentTail >= currentHead)
        {
            return currentTail - currentHead;
        }
        else
        {
            return capacity - (currentHead - currentTail);
        }
    }
    
    /**
     * Get capacity
     */
    size_t getCapacity() const
    {
        return capacity - 1; // -1 for sentinel
    }
    
private:
    std::vector<T> buffer;
    size_t capacity;
    
    std::atomic<size_t> head;
    std::atomic<size_t> tail;
    
    size_t increment(size_t idx) const
    {
        return (idx + 1) % capacity;
    }
    
    // Prevent copying
    LockFreeFIFO(const LockFreeFIFO&) = delete;
    LockFreeFIFO& operator=(const LockFreeFIFO&) = delete;
};

/**
 * Audio data packet for communication between threads
 */
struct AudioDataPacket
{
    std::vector<std::vector<float>> melSpectrogram;
    struct PitchInfo pitchInfo;
    struct RhythmInfo rhythmInfo;
    int64_t samplePosition;
    double timestamp;
    
    AudioDataPacket() = default;
    
    // Move constructor
    AudioDataPacket(AudioDataPacket&& other) noexcept
        : melSpectrogram(std::move(other.melSpectrogram)),
          pitchInfo(std::move(other.pitchInfo)),
          rhythmInfo(std::move(other.rhythmInfo)),
          samplePosition(other.samplePosition),
          timestamp(other.timestamp)
    {
    }
    
    // Move assignment
    AudioDataPacket& operator=(AudioDataPacket&& other) noexcept
    {
        if (this != &other)
        {
            melSpectrogram = std::move(other.melSpectrogram);
            pitchInfo = std::move(other.pitchInfo);
            rhythmInfo = std::move(other.rhythmInfo);
            samplePosition = other.samplePosition;
            timestamp = other.timestamp;
        }
        return *this;
    }
};

/**
 * MIDI result packet from ML inference
 */
struct MIDIResultPacket
{
    struct GeneratedMIDI midiData;
    int64_t samplePosition;
    double timestamp;
    
    MIDIResultPacket() = default;
    
    MIDIResultPacket(MIDIResultPacket&& other) noexcept
        : midiData(std::move(other.midiData)),
          samplePosition(other.samplePosition),
          timestamp(other.timestamp)
    {
    }
    
    MIDIResultPacket& operator=(MIDIResultPacket&& other) noexcept
    {
        if (this != &other)
        {
            midiData = std::move(other.midiData);
            samplePosition = other.samplePosition;
            timestamp = other.timestamp;
        }
        return *this;
    }
};

/**
 * Manages communication between audio thread and ML inference thread
 * Uses lock-free queues to prevent priority inversions
 */
class AudioMLBridge
{
public:
    AudioMLBridge();
    ~AudioMLBridge();
    
    /**
     * Send audio data from audio thread to ML thread (non-blocking)
     * @return true if successful, false if queue is full
     */
    bool sendAudioData(AudioDataPacket&& packet);
    
    /**
     * Receive audio data in ML thread (non-blocking)
     * @return true if data was available, false if queue was empty
     */
    bool receiveAudioData(AudioDataPacket& packet);
    
    /**
     * Send MIDI results from ML thread to audio thread (non-blocking)
     * @return true if successful, false if queue is full
     */
    bool sendMIDIResult(MIDIResultPacket&& packet);
    
    /**
     * Receive MIDI results in audio thread (non-blocking)
     * @return true if data was available, false if queue was empty
     */
    bool receiveMIDIResult(MIDIResultPacket& packet);
    
    /**
     * Get queue statistics
     */
    struct QueueStats
    {
        size_t audioQueueSize;
        size_t midiQueueSize;
        size_t audioQueueCapacity;
        size_t midiQueueCapacity;
        int audioQueueOverflows;
        int midiQueueOverflows;
    };
    
    QueueStats getStats() const;
    
    /**
     * Reset overflow counters
     */
    void resetStats();
    
private:
    // Lock-free queues
    LockFreeFIFO<AudioDataPacket> audioQueue;
    LockFreeFIFO<MIDIResultPacket> midiQueue;
    
    // Overflow counters (for diagnostics)
    std::atomic<int> audioOverflows{0};
    std::atomic<int> midiOverflows{0};
    
    static constexpr size_t AUDIO_QUEUE_SIZE = 16;  // Small queue for low latency
    static constexpr size_t MIDI_QUEUE_SIZE = 32;   // Larger queue for results
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioMLBridge)
};

/**
 * Background ML inference thread
 * Runs at lower priority than audio thread to prevent audio dropouts
 */
class MLInferenceThread : public juce::Thread
{
public:
    MLInferenceThread(AudioMLBridge& bridge, class ModelInference& modelInference);
    ~MLInferenceThread() override;
    
    void run() override;
    
    void setEnabled(bool enabled) { isEnabled.store(enabled); }
    bool getEnabled() const { return isEnabled.load(); }
    
    // Configuration
    void setGenre(const juce::String& genre) { currentGenre = genre; }
    void setBPM(float bpm) { currentBPM = bpm; }
    void setInstrument(const juce::String& instrument) { currentInstrument = instrument; }
    
private:
    AudioMLBridge& bridge;
    ModelInference& modelInference;
    
    std::atomic<bool> isEnabled{true};
    
    // Current settings
    juce::String currentGenre{"Trap"};
    float currentBPM{120.0f};
    juce::String currentInstrument{"Piano"};
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MLInferenceThread)
};
