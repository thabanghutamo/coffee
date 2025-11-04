#include "FeatureExtraction.h"
#include <cmath>

FeatureExtraction::FeatureExtraction()
    : sampleRate(44100.0), fftSize(2048), hopSize(512), numMelBands(128), numMFCC(13)
{
    fft = std::make_unique<juce::dsp::FFT>(11); // 2^11 = 2048
    window = std::make_unique<juce::dsp::WindowingFunction<float>>(
        fftSize, juce::dsp::WindowingFunction<float>::hann);
    
    // Initialize mel filterbank
    melFilterbank.resize(numMelBands);
    for (auto& band : melFilterbank)
        band.resize(fftSize / 2);
    
    // Compute mel filterbank (simplified triangular filters)
    for (int i = 0; i < numMelBands; ++i)
    {
        // This is a simplified implementation
        // In production, use proper mel scale conversion
        float centerFreq = (i + 1) * (sampleRate / 2) / (numMelBands + 1);
        int centerBin = static_cast<int>(centerFreq * fftSize / sampleRate);
        
        for (int j = 0; j < fftSize / 2; ++j)
        {
            melFilterbank[i][j] = std::max(0.0f, 
                1.0f - std::abs(static_cast<float>(j - centerBin)) / (fftSize / (2 * numMelBands)));
        }
    }
}

FeatureExtraction::~FeatureExtraction()
{
}

void FeatureExtraction::prepare(double newSampleRate)
{
    sampleRate = newSampleRate;
}

MelSpectrogramFeatures FeatureExtraction::extractMelSpectrogram(const juce::AudioBuffer<float>& buffer)
{
    MelSpectrogramFeatures features;
    features.hopSize = hopSize;
    
    const int numSamples = buffer.getNumSamples();
    const int numFrames = (numSamples - fftSize) / hopSize + 1;
    features.numFrames = numFrames;
    
    features.melSpectrogram.resize(numFrames);
    
    std::vector<float> fftData(fftSize * 2, 0.0f);
    std::vector<float> magnitudes(fftSize / 2);
    std::vector<float> windowData(fftSize);
    
    // Fill window data
    window->fillWindowingTables(windowData.data(), static_cast<size_t>(fftSize), juce::dsp::WindowingFunction<float>::hann, false);
    
    for (int frame = 0; frame < numFrames; ++frame)
    {
        int startSample = frame * hopSize;
        
        // Copy and window the input
        for (int i = 0; i < fftSize; ++i)
        {
            int sampleIndex = startSample + i;
            if (sampleIndex < numSamples)
            {
                float sample = buffer.getSample(0, sampleIndex);
                fftData[i] = sample * windowData[i];
            }
            else
            {
                fftData[i] = 0.0f;
            }
        }
        
        // Perform FFT
        fft->performRealOnlyForwardTransform(fftData.data(), true);
        
        // Compute magnitudes
        for (int i = 0; i < fftSize / 2; ++i)
        {
            float real = fftData[i * 2];
            float imag = fftData[i * 2 + 1];
            magnitudes[i] = std::sqrt(real * real + imag * imag);
        }
        
        // Apply mel filterbank
        std::vector<float> melSpectrum(numMelBands);
        applyMelFilterbank(magnitudes, melSpectrum);
        
        features.melSpectrogram[frame] = melSpectrum;
    }
    
    return features;
}

std::vector<float> FeatureExtraction::extractMFCC(const juce::AudioBuffer<float>& buffer)
{
    auto melFeatures = extractMelSpectrogram(buffer);
    
    std::vector<float> mfcc(numMFCC, 0.0f);
    
    if (!melFeatures.melSpectrogram.empty())
    {
        // Average MFCC across frames
        for (const auto& frame : melFeatures.melSpectrogram)
        {
            std::vector<float> frameMFCC(numMFCC);
            applyDCT(frame, frameMFCC);
            
            for (int i = 0; i < numMFCC; ++i)
                mfcc[i] += frameMFCC[i];
        }
        
        for (int i = 0; i < numMFCC; ++i)
            mfcc[i] /= melFeatures.melSpectrogram.size();
    }
    
    return mfcc;
}

float FeatureExtraction::getSpectralCentroid(const std::vector<float>& spectrum)
{
    float weightedSum = 0.0f;
    float sum = 0.0f;
    
    for (size_t i = 0; i < spectrum.size(); ++i)
    {
        weightedSum += i * spectrum[i];
        sum += spectrum[i];
    }
    
    return sum > 0.0f ? weightedSum / sum : 0.0f;
}

float FeatureExtraction::getSpectralRolloff(const std::vector<float>& spectrum, float rolloffThreshold)
{
    float totalEnergy = 0.0f;
    for (float mag : spectrum)
        totalEnergy += mag;
    
    float targetEnergy = totalEnergy * rolloffThreshold;
    float cumulativeEnergy = 0.0f;
    
    for (size_t i = 0; i < spectrum.size(); ++i)
    {
        cumulativeEnergy += spectrum[i];
        if (cumulativeEnergy >= targetEnergy)
            return static_cast<float>(i);
    }
    
    return static_cast<float>(spectrum.size() - 1);
}

float FeatureExtraction::getZeroCrossingRate(const juce::AudioBuffer<float>& buffer)
{
    int zeroCrossings = 0;
    const int numSamples = buffer.getNumSamples();
    
    for (int i = 1; i < numSamples; ++i)
    {
        if ((buffer.getSample(0, i - 1) >= 0.0f && buffer.getSample(0, i) < 0.0f) ||
            (buffer.getSample(0, i - 1) < 0.0f && buffer.getSample(0, i) >= 0.0f))
        {
            zeroCrossings++;
        }
    }
    
    return static_cast<float>(zeroCrossings) / numSamples;
}

float FeatureExtraction::getRMS(const juce::AudioBuffer<float>& buffer)
{
    float sum = 0.0f;
    const int numSamples = buffer.getNumSamples();
    
    for (int i = 0; i < numSamples; ++i)
    {
        float sample = buffer.getSample(0, i);
        sum += sample * sample;
    }
    
    return std::sqrt(sum / numSamples);
}

void FeatureExtraction::applyMelFilterbank(const std::vector<float>& spectrum, std::vector<float>& melSpectrum)
{
    for (int i = 0; i < numMelBands; ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < spectrum.size() && j < melFilterbank[i].size(); ++j)
        {
            sum += spectrum[j] * melFilterbank[i][j];
        }
        melSpectrum[i] = std::log(sum + 1e-10f); // Log mel
    }
}

void FeatureExtraction::applyDCT(const std::vector<float>& melSpectrum, std::vector<float>& mfcc)
{
    const float pi = juce::MathConstants<float>::pi;
    const int N = melSpectrum.size();
    
    for (int k = 0; k < numMFCC; ++k)
    {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n)
        {
            sum += melSpectrum[n] * std::cos(pi * k * (n + 0.5f) / N);
        }
        mfcc[k] = sum;
    }
}
