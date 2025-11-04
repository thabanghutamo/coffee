#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

struct MelSpectrogramFeatures
{
    std::vector<std::vector<float>> melSpectrogram;
    std::vector<float> mfcc;
    std::vector<float> spectralCentroid;
    std::vector<float> spectralRolloff;
    std::vector<float> zeroCrossingRate;
    int hopSize;
    int numFrames;
};

class FeatureExtraction
{
public:
    FeatureExtraction();
    ~FeatureExtraction();
    
    void prepare(double sampleRate);
    
    // Extract mel-spectrogram for ML model input
    MelSpectrogramFeatures extractMelSpectrogram(const juce::AudioBuffer<float>& buffer);
    
    // Extract MFCC coefficients
    std::vector<float> extractMFCC(const juce::AudioBuffer<float>& buffer);
    
    // Extract spectral features
    float getSpectralCentroid(const std::vector<float>& spectrum);
    float getSpectralRolloff(const std::vector<float>& spectrum, float rolloffThreshold = 0.85f);
    
    // Extract temporal features
    float getZeroCrossingRate(const juce::AudioBuffer<float>& buffer);
    float getRMS(const juce::AudioBuffer<float>& buffer);
    
private:
    void computeFFT(const float* inputData, int numSamples, std::vector<float>& magnitudes);
    void applyMelFilterbank(const std::vector<float>& spectrum, std::vector<float>& melSpectrum);
    void applyDCT(const std::vector<float>& melSpectrum, std::vector<float>& mfcc);
    
    double sampleRate;
    int fftSize;
    int hopSize;
    int numMelBands;
    int numMFCC;
    
    std::unique_ptr<juce::dsp::FFT> fft;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> window;
    
    std::vector<std::vector<float>> melFilterbank;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(FeatureExtraction)
};
