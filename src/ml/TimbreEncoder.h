#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

class TimbreEncoder
{
public:
    TimbreEncoder();
    ~TimbreEncoder();
    
    // Extract timbre embedding from audio features
    std::vector<float> extractTimbreEmbedding(
        const std::vector<std::vector<float>>& melSpectrogram);
    
    // Find closest matching instrument
    std::string findClosestInstrument(const std::vector<float>& embedding);
    
    // Map voice characteristics to instrument type
    std::string mapVoiceToInstrument(
        const std::vector<float>& features,
        const std::string& userPreference = "");
    
private:
    std::map<std::string, std::vector<float>> instrumentEmbeddings;
    
    void initializeInstrumentEmbeddings();
    float computeSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TimbreEncoder)
};
