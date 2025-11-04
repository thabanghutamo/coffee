#include "PerformanceProfiler.h"
#include <algorithm>
#include <cmath>
#include <sstream>

// ==================== PerformanceProfiler ====================

PerformanceProfiler::PerformanceProfiler()
{
}

PerformanceProfiler::~PerformanceProfiler()
{
}

void PerformanceProfiler::recordTiming(const juce::String& name, int64_t microseconds)
{
    if (!isEnabled)
        return;
    
    const juce::ScopedLock sl(lock);
    timings[name].addSample(microseconds);
}

void PerformanceProfiler::TimingData::addSample(int64_t microseconds)
{
    if (samples.size() >= MAX_SAMPLES)
    {
        // Remove oldest sample
        samples.erase(samples.begin());
    }
    
    samples.push_back(microseconds);
    totalTime += microseconds;
    count++;
    
    if (microseconds < minTime)
        minTime = microseconds;
    if (microseconds > maxTime)
        maxTime = microseconds;
}

PerformanceProfiler::Stats PerformanceProfiler::getStats(const juce::String& name) const
{
    const juce::ScopedLock sl(lock);
    
    auto it = timings.find(name);
    if (it == timings.end())
        return {0, 0, 0, 0, 0};
    
    return it->second.getStats();
}

PerformanceProfiler::Stats PerformanceProfiler::TimingData::getStats() const
{
    Stats stats;
    stats.count = count;
    
    if (count == 0)
        return stats;
    
    stats.mean = static_cast<double>(totalTime) / count;
    stats.min = static_cast<double>(minTime);
    stats.max = static_cast<double>(maxTime);
    
    // Calculate standard deviation
    double variance = 0.0;
    for (auto sample : samples)
    {
        double diff = sample - stats.mean;
        variance += diff * diff;
    }
    variance /= samples.size();
    stats.stddev = std::sqrt(variance);
    
    return stats;
}

juce::StringArray PerformanceProfiler::getAllTimingNames() const
{
    const juce::ScopedLock sl(lock);
    
    juce::StringArray names;
    for (const auto& pair : timings)
    {
        names.add(pair.first);
    }
    return names;
}

void PerformanceProfiler::recordBufferUnderrun()
{
    bufferUnderruns++;
}

void PerformanceProfiler::recordMLInferenceLatency(int64_t microseconds)
{
    recordTiming("ML_Inference", microseconds);
}

void PerformanceProfiler::recordAudioThreadLatency(int64_t microseconds)
{
    recordTiming("Audio_Thread", microseconds);
}

void PerformanceProfiler::updateCPUUsage(float usage)
{
    const juce::ScopedLock sl(lock);
    
    if (cpuSamples.size() >= MAX_SAMPLES)
    {
        cpuSamples.erase(cpuSamples.begin());
    }
    
    cpuSamples.push_back(usage);
    
    if (usage > peakCPU)
        peakCPU = usage;
}

float PerformanceProfiler::getAverageCPUUsage() const
{
    const juce::ScopedLock sl(lock);
    
    if (cpuSamples.empty())
        return 0.0f;
    
    float sum = 0.0f;
    for (float sample : cpuSamples)
    {
        sum += sample;
    }
    
    return sum / cpuSamples.size();
}

float PerformanceProfiler::getPeakCPUUsage() const
{
    return peakCPU;
}

juce::String PerformanceProfiler::generateReport() const
{
    std::ostringstream oss;
    
    oss << "=== Performance Profiling Report ===" << std::endl;
    oss << std::endl;
    
    // CPU Usage
    oss << "CPU Usage:" << std::endl;
    oss << "  Average: " << getAverageCPUUsage() << "%" << std::endl;
    oss << "  Peak:    " << getPeakCPUUsage() << "%" << std::endl;
    oss << std::endl;
    
    // Buffer underruns
    oss << "Audio Buffer:" << std::endl;
    oss << "  Underruns: " << bufferUnderruns.load() << std::endl;
    oss << std::endl;
    
    // Timing statistics
    oss << "Timing Statistics (microseconds):" << std::endl;
    oss << std::endl;
    
    auto names = getAllTimingNames();
    for (const auto& name : names)
    {
        auto stats = getStats(name);
        
        oss << "  " << name.toStdString() << ":" << std::endl;
        oss << "    Count:  " << stats.count << std::endl;
        oss << "    Mean:   " << stats.mean << " µs (" << (stats.mean / 1000.0) << " ms)" << std::endl;
        oss << "    Min:    " << stats.min << " µs" << std::endl;
        oss << "    Max:    " << stats.max << " µs" << std::endl;
        oss << "    StdDev: " << stats.stddev << " µs" << std::endl;
        oss << std::endl;
    }
    
    // Latency warnings
    oss << "Latency Warnings:" << std::endl;
    const double TARGET_LATENCY_MS = 10.0; // 10ms target
    
    for (const auto& name : names)
    {
        auto stats = getStats(name);
        double meanMs = stats.mean / 1000.0;
        
        if (meanMs > TARGET_LATENCY_MS)
        {
            oss << "  ⚠️  " << name.toStdString() << " exceeds target (" 
                << meanMs << " ms > " << TARGET_LATENCY_MS << " ms)" << std::endl;
        }
    }
    
    return juce::String(oss.str());
}

void PerformanceProfiler::saveReportToFile(const juce::File& outputFile) const
{
    auto report = generateReport();
    outputFile.replaceWithText(report);
    DBG("Performance report saved to: " + outputFile.getFullPathName());
}

void PerformanceProfiler::reset()
{
    const juce::ScopedLock sl(lock);
    
    timings.clear();
    cpuSamples.clear();
    bufferUnderruns = 0;
    peakCPU = 0.0f;
}

// ==================== ThreadAnalyzer ====================

ThreadAnalyzer::ThreadAnalyzer()
{
}

ThreadAnalyzer::~ThreadAnalyzer()
{
}

void ThreadAnalyzer::recordLockAcquisition(const juce::String& lockName, int64_t waitTimeMicros)
{
    const juce::ScopedLock sl(lock);
    
    auto& stats = lockStats[lockName];
    stats.totalWaitTime += waitTimeMicros;
    stats.acquisitionCount++;
    
    if (waitTimeMicros > stats.maxWaitTime)
        stats.maxWaitTime = waitTimeMicros;
    
    stats.waitTimes.push_back(waitTimeMicros);
    
    if (stats.waitTimes.size() > 1000)
    {
        stats.waitTimes.erase(stats.waitTimes.begin());
    }
}

void ThreadAnalyzer::recordThreadExecution(const juce::String& threadName, int64_t executionTimeMicros)
{
    const juce::ScopedLock sl(lock);
    
    auto& stats = threadStats[threadName];
    stats.totalExecutionTime += executionTimeMicros;
    stats.executionCount++;
    stats.executionTimes.push_back(executionTimeMicros);
    
    if (stats.executionTimes.size() > 1000)
    {
        stats.executionTimes.erase(stats.executionTimes.begin());
    }
}

std::vector<ThreadAnalyzer::PriorityInversion> ThreadAnalyzer::detectPriorityInversions() const
{
    // Simplified priority inversion detection
    // In production, this would analyze thread scheduling and blocking patterns
    return {};
}

juce::String ThreadAnalyzer::generateReport() const
{
    const juce::ScopedLock sl(lock);
    
    std::ostringstream oss;
    
    oss << "=== Thread Analysis Report ===" << std::endl;
    oss << std::endl;
    
    // Lock contention
    oss << "Lock Contention:" << std::endl;
    for (const auto& pair : lockStats)
    {
        const auto& name = pair.first;
        const auto& stats = pair.second;
        
        double avgWait = stats.acquisitionCount > 0 
            ? static_cast<double>(stats.totalWaitTime) / stats.acquisitionCount 
            : 0.0;
        
        oss << "  " << name.toStdString() << ":" << std::endl;
        oss << "    Acquisitions: " << stats.acquisitionCount << std::endl;
        oss << "    Avg Wait:     " << avgWait << " µs" << std::endl;
        oss << "    Max Wait:     " << stats.maxWaitTime << " µs" << std::endl;
        oss << std::endl;
    }
    
    // Thread execution
    oss << "Thread Execution:" << std::endl;
    for (const auto& pair : threadStats)
    {
        const auto& name = pair.first;
        const auto& stats = pair.second;
        
        double avgExec = stats.executionCount > 0
            ? static_cast<double>(stats.totalExecutionTime) / stats.executionCount
            : 0.0;
        
        oss << "  " << name.toStdString() << ":" << std::endl;
        oss << "    Executions:   " << stats.executionCount << std::endl;
        oss << "    Avg Duration: " << avgExec << " µs" << std::endl;
        oss << std::endl;
    }
    
    return juce::String(oss.str());
}

// ==================== LatencyMonitor ====================

LatencyMonitor::LatencyMonitor()
{
}

LatencyMonitor::~LatencyMonitor()
{
}

void LatencyMonitor::startProcessing(int64_t samplePosition)
{
    currentStartSample = samplePosition;
}

void LatencyMonitor::endProcessing(int64_t samplePosition)
{
    if (currentStartSample < 0)
        return;
    
    const juce::ScopedLock sl(lock);
    
    LatencySample sample;
    sample.startSample = currentStartSample;
    sample.endSample = samplePosition;
    sample.timestamp = std::chrono::high_resolution_clock::now();
    
    samples.push_back(sample);
    
    if (samples.size() > MAX_SAMPLES)
    {
        samples.erase(samples.begin());
    }
    
    currentStartSample = -1;
}

double LatencyMonitor::getAverageLatencyMs() const
{
    const juce::ScopedLock sl(lock);
    
    if (samples.empty())
        return 0.0;
    
    int64_t totalSamples = 0;
    for (const auto& sample : samples)
    {
        totalSamples += (sample.endSample - sample.startSample);
    }
    
    double avgSamples = static_cast<double>(totalSamples) / samples.size();
    return (avgSamples / sampleRate) * 1000.0; // Convert to ms
}

double LatencyMonitor::getMinLatencyMs() const
{
    const juce::ScopedLock sl(lock);
    
    if (samples.empty())
        return 0.0;
    
    int64_t minSamples = INT64_MAX;
    for (const auto& sample : samples)
    {
        int64_t latency = sample.endSample - sample.startSample;
        if (latency < minSamples)
            minSamples = latency;
    }
    
    return (static_cast<double>(minSamples) / sampleRate) * 1000.0;
}

double LatencyMonitor::getMaxLatencyMs() const
{
    const juce::ScopedLock sl(lock);
    
    if (samples.empty())
        return 0.0;
    
    int64_t maxSamples = 0;
    for (const auto& sample : samples)
    {
        int64_t latency = sample.endSample - sample.startSample;
        if (latency > maxSamples)
            maxSamples = latency;
    }
    
    return (static_cast<double>(maxSamples) / sampleRate) * 1000.0;
}

double LatencyMonitor::get95thPercentileMs() const
{
    const juce::ScopedLock sl(lock);
    
    if (samples.empty())
        return 0.0;
    
    std::vector<int64_t> latencies;
    for (const auto& sample : samples)
    {
        latencies.push_back(sample.endSample - sample.startSample);
    }
    
    std::sort(latencies.begin(), latencies.end());
    size_t index = static_cast<size_t>(latencies.size() * 0.95);
    
    return (static_cast<double>(latencies[index]) / sampleRate) * 1000.0;
}

bool LatencyMonitor::meetsTarget(double targetMs) const
{
    return get95thPercentileMs() <= targetMs;
}

juce::String LatencyMonitor::generateReport() const
{
    std::ostringstream oss;
    
    oss << "=== Latency Monitor Report ===" << std::endl;
    oss << std::endl;
    oss << "Samples:            " << samples.size() << std::endl;
    oss << "Average Latency:    " << getAverageLatencyMs() << " ms" << std::endl;
    oss << "Min Latency:        " << getMinLatencyMs() << " ms" << std::endl;
    oss << "Max Latency:        " << getMaxLatencyMs() << " ms" << std::endl;
    oss << "95th Percentile:    " << get95thPercentileMs() << " ms" << std::endl;
    oss << std::endl;
    oss << "Target (10ms):      " << (meetsTarget(10.0) ? "✓ MET" : "✗ NOT MET") << std::endl;
    
    return juce::String(oss.str());
}

void LatencyMonitor::reset()
{
    const juce::ScopedLock sl(lock);
    samples.clear();
    currentStartSample = -1;
}
