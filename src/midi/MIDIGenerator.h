#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "MIDITrack.h"
#include "ml/ModelInference.h"
#include <vector>
#include <memory>

class MIDIGenerator
{
public:
    MIDIGenerator();
    ~MIDIGenerator();
    
    // Generate multi-track MIDI from ML model output
    void updateTracks(const GeneratedMIDI& generatedData, 
                      std::vector<std::shared_ptr<MIDITrack>>& tracks);
    
    // Create initial track structure
    void createInitialTracks(std::vector<std::shared_ptr<MIDITrack>>& tracks,
                            const juce::String& genre);
    
    // Regenerate specific track with variation
    void regenerateTrack(std::shared_ptr<MIDITrack> track, ModelInference* inference);
    
    // Convert ML model output to MIDI notes
    std::vector<MIDINote> convertToMIDINotes(
        const std::vector<std::vector<float>>& pitchProbs,
        const std::vector<std::vector<float>>& velocities,
        const std::vector<float>& onsets,
        float bpm
    );
    
    // Export MIDI file
    void exportToFile(const std::vector<std::shared_ptr<MIDITrack>>& tracks,
                     const juce::File& outputFile);
    
    // Quantization utilities
    void quantizeToGrid(std::vector<MIDINote>& notes, float gridSize);
    void applySwing(std::vector<MIDINote>& notes, float swingAmount);
    
private:
    // Convert model outputs to notes
    MIDINote createNoteFromProbabilities(
        const std::vector<float>& pitchProb,
        float velocity,
        double timestamp,
        float bpm
    );
    
    // Track templates for different genres
    void applyGenreTemplate(std::shared_ptr<MIDITrack> track, const juce::String& genre);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MIDIGenerator)
};
