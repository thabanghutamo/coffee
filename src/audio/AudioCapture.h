#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

class AudioCapture
{
public:
    AudioCapture();
    ~AudioCapture();
    
    void prepare(double sampleRate, int samplesPerBlock);
    void reset();
    void process(const juce::AudioBuffer<float>& buffer);
    
    // Get captured audio for analysis
    const juce::AudioBuffer<float>& getCapturedBuffer() const { return capturedBuffer; }
    
    // Get recent audio window (for ML processing)
    void getRecentAudio(juce::AudioBuffer<float>& destination, int numSamples);
    
private:
    juce::AudioBuffer<float> capturedBuffer;
    int writePosition;
    int bufferSize;
    double currentSampleRate;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioCapture)
};
