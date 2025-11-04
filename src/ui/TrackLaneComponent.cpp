#include "TrackLaneComponent.h"

TrackLaneComponent::TrackLaneComponent(std::shared_ptr<MIDITrack> track)
    : track(track)
{
}

TrackLaneComponent::~TrackLaneComponent()
{
}

void TrackLaneComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    g.setColour(juce::Colour(0xFF2A2A2A));
    g.fillRect(bounds);
    
    if (track)
    {
        g.setColour(juce::Colours::white);
        g.drawText(track->getName(), bounds, juce::Justification::centredLeft, true);
    }
}

void TrackLaneComponent::resized()
{
}

void TrackLaneComponent::setTrack(std::shared_ptr<MIDITrack> newTrack)
{
    track = newTrack;
    repaint();
}
