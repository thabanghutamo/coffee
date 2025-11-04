#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <atomic>

/**
 * Performance profiler for real-time audio processing
 * Tracks latency, CPU usage, and thread synchronization metrics
 */
class PerformanceProfiler
{
public:
    PerformanceProfiler();
    ~PerformanceProfiler();
    
    // Scoped timing utility
    class ScopedTimer
    {
    public:
        ScopedTimer(PerformanceProfiler& profiler, const juce::String& name)
            : profiler(profiler), name(name)
        {
            startTime = std::chrono::high_resolution_clock::now();
        }
        
        ~ScopedTimer()
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                endTime - startTime
            ).count();
            profiler.recordTiming(name, duration);
        }
        
    private:
        PerformanceProfiler& profiler;
        juce::String name;
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    };
    
    // Enable/disable profiling
    void setEnabled(bool enabled) { isEnabled = enabled; }
    bool getEnabled() const { return isEnabled; }
    
    // Record timing for a named section
    void recordTiming(const juce::String& name, int64_t microseconds);
    
    // Get statistics
    struct Stats
    {
        double mean;
        double min;
        double max;
        double stddev;
        int64_t count;
    };
    
    Stats getStats(const juce::String& name) const;
    juce::StringArray getAllTimingNames() const;
    
    // Audio thread specific metrics
    void recordBufferUnderrun();
    void recordMLInferenceLatency(int64_t microseconds);
    void recordAudioThreadLatency(int64_t microseconds);
    
    int getBufferUnderrunCount() const { return bufferUnderruns.load(); }
    
    // CPU usage tracking
    void updateCPUUsage(float usage);
    float getAverageCPUUsage() const;
    float getPeakCPUUsage() const;
    
    // Generate report
    juce::String generateReport() const;
    void saveReportToFile(const juce::File& outputFile) const;
    
    // Reset all statistics
    void reset();
    
    // Helper macro for easy profiling
    #define PROFILE_SCOPE(profiler, name) \
        PerformanceProfiler::ScopedTimer _scopedTimer_##__LINE__(profiler, name)
    
private:
    struct TimingData
    {
        std::vector<int64_t> samples;
        int64_t totalTime = 0;
        int64_t minTime = INT64_MAX;
        int64_t maxTime = 0;
        size_t count = 0;
        
        void addSample(int64_t microseconds);
        Stats getStats() const;
    };
    
    mutable juce::CriticalSection lock;
    std::map<juce::String, TimingData> timings;
    
    std::atomic<bool> isEnabled{true};
    std::atomic<int> bufferUnderruns{0};
    
    // CPU usage tracking
    std::vector<float> cpuSamples;
    float peakCPU = 0.0f;
    
    // Maximum samples to keep (rolling window)
    static constexpr size_t MAX_SAMPLES = 1000;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PerformanceProfiler)
};

/**
 * Thread synchronization analyzer
 * Detects lock contention, priority inversions, and thread bottlenecks
 */
class ThreadAnalyzer
{
public:
    ThreadAnalyzer();
    ~ThreadAnalyzer();
    
    // Track lock acquisition
    void recordLockAcquisition(const juce::String& lockName, int64_t waitTimeMicros);
    
    // Track thread execution
    void recordThreadExecution(const juce::String& threadName, int64_t executionTimeMicros);
    
    // Detect priority inversion
    struct PriorityInversion
    {
        juce::String highPriorityThread;
        juce::String lowPriorityThread;
        int64_t blockedTimeMicros;
        juce::Time timestamp;
    };
    
    std::vector<PriorityInversion> detectPriorityInversions() const;
    
    // Generate thread analysis report
    juce::String generateReport() const;
    
private:
    struct LockStats
    {
        int64_t totalWaitTime = 0;
        int64_t maxWaitTime = 0;
        size_t acquisitionCount = 0;
        std::vector<int64_t> waitTimes;
    };
    
    struct ThreadStats
    {
        int64_t totalExecutionTime = 0;
        size_t executionCount = 0;
        std::vector<int64_t> executionTimes;
    };
    
    mutable juce::CriticalSection lock;
    std::map<juce::String, LockStats> lockStats;
    std::map<juce::String, ThreadStats> threadStats;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ThreadAnalyzer)
};

/**
 * Real-time latency monitor
 * Measures end-to-end latency from audio input to MIDI output
 */
class LatencyMonitor
{
public:
    LatencyMonitor();
    ~LatencyMonitor();
    
    // Mark start of audio processing pipeline
    void startProcessing(int64_t samplePosition);
    
    // Mark end of processing and record latency
    void endProcessing(int64_t samplePosition);
    
    // Get latency statistics (in milliseconds)
    double getAverageLatencyMs() const;
    double getMinLatencyMs() const;
    double getMaxLatencyMs() const;
    double get95thPercentileMs() const;
    
    // Check if latency target is met
    bool meetsTarget(double targetMs) const;
    
    // Generate latency report
    juce::String generateReport() const;
    
    void reset();
    
private:
    struct LatencySample
    {
        int64_t startSample;
        int64_t endSample;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };
    
    std::vector<LatencySample> samples;
    mutable juce::CriticalSection lock;
    
    int64_t currentStartSample = -1;
    double sampleRate = 44100.0;
    
    static constexpr size_t MAX_SAMPLES = 1000;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatencyMonitor)
};
