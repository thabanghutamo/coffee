#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "PluginProcessor.h"
#include "TrackLaneComponent.h"

class PianoRollComponent : public juce::Component,
                           private juce::Timer
{
public:
    PianoRollComponent(VocalMIDIAudioProcessor& processor);
    ~PianoRollComponent() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override;
    
    // View controls
    void setZoom(float newZoom);
    void setVerticalZoom(float newVerticalZoom);
    void scrollTo(double time);
    
private:
    void timerCallback() override;
    
    void drawPianoKeys(juce::Graphics& g, juce::Rectangle<int> area);
    void drawGrid(juce::Graphics& g, juce::Rectangle<int> area);
    void drawNotes(juce::Graphics& g, juce::Rectangle<int> area);
    
    // Convert between screen and musical time/pitch
    float timeToX(double time) const;
    double xToTime(float x) const;
    int pitchToY(int pitch) const;
    int yToPitch(int y) const;
    
    VocalMIDIAudioProcessor& audioProcessor;
    
    // Track lanes
    std::vector<std::unique_ptr<TrackLaneComponent>> trackLanes;
    
    // View state
    float zoom;
    float verticalZoom;
    double scrollPosition;
    int noteHeight;
    
    // Piano keys area
    int pianoKeysWidth;
    
    // Selection and editing
    int selectedTrack;
    int selectedNote;
    bool isDraggingNote;
    juce::Point<int> dragStartPosition;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRollComponent)
};
