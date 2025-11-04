#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "midi/MIDITrack.h"

class TrackLaneComponent : public juce::Component
{
public:
    TrackLaneComponent(std::shared_ptr<MIDITrack> track);
    ~TrackLaneComponent() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    void setTrack(std::shared_ptr<MIDITrack> newTrack);
    std::shared_ptr<MIDITrack> getTrack() const { return track; }
    
private:
    std::shared_ptr<MIDITrack> track;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TrackLaneComponent)
};
