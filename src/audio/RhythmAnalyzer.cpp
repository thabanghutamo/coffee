#include "RhythmAnalyzer.h"
#include <cmath>
#include <algorithm>

RhythmAnalyzer::RhythmAnalyzer()
    : sampleRate(44100.0), currentBPM(120.0f), hopSize(512), frameSize(2048)
{
    previousSpectrum.resize(frameSize / 2, 0.0f);
}

RhythmAnalyzer::~RhythmAnalyzer()
{
}

void RhythmAnalyzer::prepare(double newSampleRate, float initialBPM)
{
    sampleRate = newSampleRate;
    currentBPM = initialBPM;
}

void RhythmAnalyzer::setBPM(float bpm)
{
    currentBPM = bpm;
}

RhythmInfo RhythmAnalyzer::analyzeRhythm(const juce::AudioBuffer<float>& buffer)
{
    RhythmInfo info;
    
    // Detect onsets
    info.onsetEnvelope = detectOnsets(buffer);
    
    // Estimate tempo if needed
    if (currentBPM <= 0.0f)
    {
        info.detectedTempo = estimateTempo(info.onsetEnvelope);
    }
    else
    {
        info.detectedTempo = currentBPM;
    }
    
    // Track beats
    info.beatTimes = trackBeats(info.onsetEnvelope, info.detectedTempo);
    
    // Calculate beat strengths
    info.beatStrengths.resize(info.beatTimes.size(), 1.0f);
    
    // Calculate rhythmic clarity
    float sumStrength = 0.0f;
    for (float strength : info.beatStrengths)
        sumStrength += strength;
    
    info.rhythmicClarity = sumStrength / std::max(1.0f, static_cast<float>(info.beatStrengths.size()));
    info.isStable = info.rhythmicClarity > 0.5f;
    
    return info;
}

std::vector<float> RhythmAnalyzer::detectOnsets(const juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    const int numFrames = (numSamples - frameSize) / hopSize + 1;
    
    std::vector<float> onsetStrengths(numFrames, 0.0f);
    
    juce::dsp::FFT fft(11); // 2^11 = 2048
    std::vector<float> fftData(frameSize * 2, 0.0f);
    std::vector<float> spectrum(frameSize / 2);
    
    for (int frame = 0; frame < numFrames; ++frame)
    {
        int startSample = frame * hopSize;
        
        // Copy frame
        for (int i = 0; i < frameSize; ++i)
        {
            int sampleIndex = startSample + i;
            if (sampleIndex < numSamples)
                fftData[i] = buffer.getSample(0, sampleIndex);
            else
                fftData[i] = 0.0f;
        }
        
        // Compute FFT
        fft.performRealOnlyForwardTransform(fftData.data(), true);
        
        // Compute magnitude spectrum
        for (int i = 0; i < frameSize / 2; ++i)
        {
            float real = fftData[i * 2];
            float imag = fftData[i * 2 + 1];
            spectrum[i] = std::sqrt(real * real + imag * imag);
        }
        
        // Spectral flux (onset detection)
        float flux = 0.0f;
        for (int i = 0; i < frameSize / 2; ++i)
        {
            float diff = spectrum[i] - previousSpectrum[i];
            if (diff > 0.0f)
                flux += diff;
        }
        
        onsetStrengths[frame] = flux;
        previousSpectrum = spectrum;
    }
    
    // Normalize onset strengths
    float maxOnset = *std::max_element(onsetStrengths.begin(), onsetStrengths.end());
    if (maxOnset > 0.0f)
    {
        for (float& onset : onsetStrengths)
            onset /= maxOnset;
    }
    
    return onsetStrengths;
}

float RhythmAnalyzer::estimateTempo(const std::vector<float>& onsetStrengths)
{
    // Simple autocorrelation-based tempo estimation
    const int minBPM = 60;
    const int maxBPM = 180;
    
    const int numFrames = onsetStrengths.size();
    const float secondsPerFrame = hopSize / static_cast<float>(sampleRate);
    
    const int minLag = static_cast<int>((60.0f / maxBPM) / secondsPerFrame);
    const int maxLag = static_cast<int>((60.0f / minBPM) / secondsPerFrame);
    
    float maxCorrelation = 0.0f;
    int bestLag = minLag;
    
    for (int lag = minLag; lag < maxLag && lag < numFrames / 2; ++lag)
    {
        float correlation = 0.0f;
        int count = 0;
        
        for (int i = 0; i < numFrames - lag; ++i)
        {
            correlation += onsetStrengths[i] * onsetStrengths[i + lag];
            count++;
        }
        
        if (count > 0)
            correlation /= count;
        
        if (correlation > maxCorrelation)
        {
            maxCorrelation = correlation;
            bestLag = lag;
        }
    }
    
    float estimatedTempo = 60.0f / (bestLag * secondsPerFrame);
    
    // Clamp to reasonable range
    return std::clamp(estimatedTempo, static_cast<float>(minBPM), static_cast<float>(maxBPM));
}

std::vector<double> RhythmAnalyzer::trackBeats(const std::vector<float>& onsetEnvelope, float tempo)
{
    std::vector<double> beatTimes;
    
    const float beatInterval = 60.0f / tempo; // seconds
    const float secondsPerFrame = hopSize / static_cast<float>(sampleRate);
    
    // Find peaks in onset envelope
    std::vector<int> peakIndices;
    for (size_t i = 1; i < onsetEnvelope.size() - 1; ++i)
    {
        if (onsetEnvelope[i] > onsetEnvelope[i - 1] &&
            onsetEnvelope[i] > onsetEnvelope[i + 1] &&
            onsetEnvelope[i] > 0.3f) // Threshold
        {
            peakIndices.push_back(i);
        }
    }
    
    // Align peaks to beat grid
    for (int peakIdx : peakIndices)
    {
        double time = peakIdx * secondsPerFrame;
        beatTimes.push_back(time);
    }
    
    return beatTimes;
}
