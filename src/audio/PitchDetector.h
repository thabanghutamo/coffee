#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

struct PitchInfo
{
    float frequency;        // Detected frequency in Hz
    float confidence;       // Confidence of detection (0-1)
    int midiNote;          // Closest MIDI note number
    float cents;           // Cents deviation from MIDI note
    float clarity;         // Pitch clarity measure
    double timestamp;      // Time in seconds
};

class PitchDetector
{
public:
    PitchDetector();
    ~PitchDetector();
    
    void prepare(double sampleRate);
    
    // Detect pitch using multiple algorithms
    PitchInfo detectPitch(const juce::AudioBuffer<float>& buffer);
    
    // Individual pitch detection methods
    float detectPitchYIN(const juce::AudioBuffer<float>& buffer, float& confidence);
    float detectPitchAutocorrelation(const juce::AudioBuffer<float>& buffer);
    float detectPitchCepstrum(const juce::AudioBuffer<float>& buffer);
    
    // Utility functions
    int frequencyToMIDINote(float frequency);
    float midiNoteToFrequency(int midiNote);
    float getCentsDeviation(float frequency, int midiNote);
    
private:
    double sampleRate;
    int bufferSize;
    float minFrequency;
    float maxFrequency;
    
    std::vector<float> yinBuffer;
    std::vector<float> autocorrelationBuffer;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PitchDetector)
};
