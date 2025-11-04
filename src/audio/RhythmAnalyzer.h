#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

struct RhythmInfo
{
    float detectedTempo;      // BPM
    std::vector<double> beatTimes;
    std::vector<float> beatStrengths;
    float rhythmicClarity;
    std::vector<float> onsetEnvelope;
    bool isStable;
};

class RhythmAnalyzer
{
public:
    RhythmAnalyzer();
    ~RhythmAnalyzer();
    
    void prepare(double sampleRate, float initialBPM);
    void setBPM(float bpm);
    
    // Analyze rhythm and detect beats
    RhythmInfo analyzeRhythm(const juce::AudioBuffer<float>& buffer);
    
    // Onset detection
    std::vector<float> detectOnsets(const juce::AudioBuffer<float>& buffer);
    
    // Tempo estimation
    float estimateTempo(const std::vector<float>& onsetStrengths);
    
    // Beat tracking
    std::vector<double> trackBeats(const std::vector<float>& onsetEnvelope, float tempo);
    
private:
    double sampleRate;
    float currentBPM;
    int hopSize;
    int frameSize;
    
    // Onset detection state
    std::vector<float> previousSpectrum;
    std::vector<float> onsetBuffer;
    
    // Beat tracking state
    std::vector<double> previousBeats;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RhythmAnalyzer)
};
