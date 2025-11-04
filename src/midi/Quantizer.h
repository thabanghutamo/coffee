#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "MIDITrack.h"

class Quantizer
{
public:
    enum class GridSize
    {
        None,
        Whole,      // 1/1
        Half,       // 1/2
        Quarter,    // 1/4
        Eighth,     // 1/8
        Sixteenth,  // 1/16
        ThirtySecond // 1/32
    };
    
    Quantizer();
    ~Quantizer();
    
    // Quantize notes to grid
    void quantize(std::vector<MIDINote>& notes, GridSize gridSize, float strength = 1.0f);
    
    // Get grid size in seconds
    static float getGridSizeInSeconds(GridSize gridSize, float bpm);
    
    // Triplet quantization
    void quantizeToTriplets(std::vector<MIDINote>& notes, float bpm);
    
private:
    float calculateQuantizedTime(float originalTime, float gridSize, float strength);
};
