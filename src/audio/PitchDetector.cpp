#include "PitchDetector.h"
#include <cmath>
#include <algorithm>

PitchDetector::PitchDetector()
    : sampleRate(44100.0), bufferSize(2048), minFrequency(60.0f), maxFrequency(1500.0f)
{
    yinBuffer.resize(bufferSize / 2);
    autocorrelationBuffer.resize(bufferSize);
}

PitchDetector::~PitchDetector()
{
}

void PitchDetector::prepare(double newSampleRate)
{
    sampleRate = newSampleRate;
}

PitchInfo PitchDetector::detectPitch(const juce::AudioBuffer<float>& buffer)
{
    PitchInfo info;
    info.timestamp = 0.0; // Should be set by caller
    
    // Use YIN algorithm as primary method
    float confidence = 0.0f;
    info.frequency = detectPitchYIN(buffer, confidence);
    info.confidence = confidence;
    
    if (info.frequency > 0.0f && confidence > 0.7f)
    {
        info.midiNote = frequencyToMIDINote(info.frequency);
        info.cents = getCentsDeviation(info.frequency, info.midiNote);
        info.clarity = confidence;
    }
    else
    {
        info.frequency = 0.0f;
        info.midiNote = 0;
        info.cents = 0.0f;
        info.clarity = 0.0f;
    }
    
    return info;
}

float PitchDetector::detectPitchYIN(const juce::AudioBuffer<float>& buffer, float& confidence)
{
    const int numSamples = buffer.getNumSamples();
    const int halfSize = numSamples / 2;
    
    // Step 1: Difference function
    for (int tau = 0; tau < halfSize; ++tau)
    {
        float sum = 0.0f;
        for (int j = 0; j < halfSize; ++j)
        {
            float delta = buffer.getSample(0, j) - buffer.getSample(0, j + tau);
            sum += delta * delta;
        }
        yinBuffer[tau] = sum;
    }
    
    // Step 2: Cumulative mean normalized difference function
    yinBuffer[0] = 1.0f;
    float runningSum = 0.0f;
    
    for (int tau = 1; tau < halfSize; ++tau)
    {
        runningSum += yinBuffer[tau];
        yinBuffer[tau] *= tau / runningSum;
    }
    
    // Step 3: Absolute threshold
    const float threshold = 0.15f;
    int tau = -1;
    
    for (int t = 2; t < halfSize; ++t)
    {
        if (yinBuffer[t] < threshold)
        {
            // Find local minimum
            while (t + 1 < halfSize && yinBuffer[t + 1] < yinBuffer[t])
                t++;
            tau = t;
            break;
        }
    }
    
    if (tau == -1)
    {
        confidence = 0.0f;
        return 0.0f;
    }
    
    // Step 4: Parabolic interpolation
    float betterTau = tau;
    if (tau > 0 && tau < halfSize - 1)
    {
        float s0 = yinBuffer[tau - 1];
        float s1 = yinBuffer[tau];
        float s2 = yinBuffer[tau + 1];
        betterTau = tau + (s2 - s0) / (2.0f * (2.0f * s1 - s2 - s0));
    }
    
    float frequency = static_cast<float>(sampleRate) / betterTau;
    confidence = 1.0f - yinBuffer[tau];
    
    // Validate frequency range
    if (frequency < minFrequency || frequency > maxFrequency)
    {
        confidence = 0.0f;
        return 0.0f;
    }
    
    return frequency;
}

float PitchDetector::detectPitchAutocorrelation(const juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    
    // Compute autocorrelation
    for (int lag = 0; lag < numSamples / 2; ++lag)
    {
        float sum = 0.0f;
        for (int i = 0; i < numSamples / 2; ++i)
        {
            sum += buffer.getSample(0, i) * buffer.getSample(0, i + lag);
        }
        autocorrelationBuffer[lag] = sum;
    }
    
    // Find peak (excluding zero lag)
    int maxLag = 0;
    float maxValue = 0.0f;
    
    int minLag = static_cast<int>(sampleRate / maxFrequency);
    int maxSearchLag = static_cast<int>(sampleRate / minFrequency);
    
    for (int lag = minLag; lag < maxSearchLag && lag < numSamples / 2; ++lag)
    {
        if (autocorrelationBuffer[lag] > maxValue)
        {
            maxValue = autocorrelationBuffer[lag];
            maxLag = lag;
        }
    }
    
    if (maxLag > 0)
        return static_cast<float>(sampleRate) / maxLag;
    
    return 0.0f;
}

float PitchDetector::detectPitchCepstrum(const juce::AudioBuffer<float>& buffer)
{
    // Simplified cepstrum analysis
    // In production, use proper FFT-based cepstrum
    return 0.0f;
}

int PitchDetector::frequencyToMIDINote(float frequency)
{
    return static_cast<int>(std::round(69.0f + 12.0f * std::log2(frequency / 440.0f)));
}

float PitchDetector::midiNoteToFrequency(int midiNote)
{
    return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
}

float PitchDetector::getCentsDeviation(float frequency, int midiNote)
{
    float targetFreq = midiNoteToFrequency(midiNote);
    return 1200.0f * std::log2(frequency / targetFreq);
}
