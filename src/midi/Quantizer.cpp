#include "Quantizer.h"
#include <cmath>

Quantizer::Quantizer()
{
}

Quantizer::~Quantizer()
{
}

void Quantizer::quantize(std::vector<MIDINote>& notes, GridSize gridSize, float strength)
{
    if (gridSize == GridSize::None || strength <= 0.0f)
        return;
    
    float gridTime = 1.0f; // Default to whole note
    
    switch (gridSize)
    {
        case GridSize::Whole: gridTime = 1.0f; break;
        case GridSize::Half: gridTime = 0.5f; break;
        case GridSize::Quarter: gridTime = 0.25f; break;
        case GridSize::Eighth: gridTime = 0.125f; break;
        case GridSize::Sixteenth: gridTime = 0.0625f; break;
        case GridSize::ThirtySecond: gridTime = 0.03125f; break;
        default: break;
    }
    
    for (auto& note : notes)
    {
        note.startTime = calculateQuantizedTime(note.startTime, gridTime, strength);
    }
}

float Quantizer::getGridSizeInSeconds(GridSize gridSize, float bpm)
{
    float beatsPerSecond = bpm / 60.0f;
    float wholeNoteSeconds = 4.0f / beatsPerSecond;
    
    switch (gridSize)
    {
        case GridSize::Whole: return wholeNoteSeconds;
        case GridSize::Half: return wholeNoteSeconds / 2.0f;
        case GridSize::Quarter: return wholeNoteSeconds / 4.0f;
        case GridSize::Eighth: return wholeNoteSeconds / 8.0f;
        case GridSize::Sixteenth: return wholeNoteSeconds / 16.0f;
        case GridSize::ThirtySecond: return wholeNoteSeconds / 32.0f;
        default: return wholeNoteSeconds / 4.0f;
    }
}

void Quantizer::quantizeToTriplets(std::vector<MIDINote>& notes, float bpm)
{
    float tripletGrid = (60.0f / bpm) / 3.0f; // Eighth note triplets
    
    for (auto& note : notes)
    {
        note.startTime = std::round(note.startTime / tripletGrid) * tripletGrid;
    }
}

float Quantizer::calculateQuantizedTime(float originalTime, float gridSize, float strength)
{
    float quantizedTime = std::round(originalTime / gridSize) * gridSize;
    return originalTime + (quantizedTime - originalTime) * strength;
}
