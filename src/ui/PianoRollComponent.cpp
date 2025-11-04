#include "PianoRollComponent.h"

PianoRollComponent::PianoRollComponent(VocalMIDIAudioProcessor& processor)
    : audioProcessor(processor),
      zoom(1.0f),
      verticalZoom(1.0f),
      scrollPosition(0.0),
      noteHeight(10),
      pianoKeysWidth(80),
      selectedTrack(-1),
      selectedNote(-1),
      isDraggingNote(false)
{
    setSize(800, 400);
    startTimer(30); // 30 FPS refresh
}

PianoRollComponent::~PianoRollComponent()
{
    stopTimer();
}

void PianoRollComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Piano keys area
    auto pianoArea = bounds.removeFromLeft(pianoKeysWidth);
    drawPianoKeys(g, pianoArea);
    
    // Grid and notes area
    auto gridArea = bounds;
    g.saveState();
    g.reduceClipRegion(gridArea);
    
    drawGrid(g, gridArea);
    drawNotes(g, gridArea);
    
    g.restoreState();
}

void PianoRollComponent::resized()
{
    auto area = getLocalBounds();
    area.removeFromLeft(pianoKeysWidth); // Leave space for piano keys
    
    // Layout track lanes
    int laneHeight = area.getHeight() / juce::jmax(1, (int)trackLanes.size());
    for (auto& lane : trackLanes)
    {
        lane->setBounds(area.removeFromTop(laneHeight));
    }
}

void PianoRollComponent::mouseDown(const juce::MouseEvent& e)
{
    auto pos = e.getPosition();
    
    if (pos.x < pianoKeysWidth)
        return; // Clicked on piano keys
    
    // Check if clicking on a note
    int pitch = yToPitch(pos.y);
    double time = xToTime(pos.x - pianoKeysWidth);
    
    // Find note at this position
    auto& tracks = audioProcessor.getTracks();
    for (int trackIdx = 0; trackIdx < tracks.size(); ++trackIdx)
    {
        auto& notes = tracks[trackIdx]->getNotes();
        for (int noteIdx = 0; noteIdx < notes.size(); ++noteIdx)
        {
            auto& note = notes[noteIdx];
            if (note.pitch == pitch &&
                time >= note.startTime &&
                time <= note.startTime + note.duration)
            {
                selectedTrack = trackIdx;
                selectedNote = noteIdx;
                isDraggingNote = true;
                dragStartPosition = pos;
                return;
            }
        }
    }
    
    selectedTrack = -1;
    selectedNote = -1;
}

void PianoRollComponent::mouseDrag(const juce::MouseEvent& e)
{
    if (!isDraggingNote || selectedTrack < 0 || selectedNote < 0)
        return;
    
    auto pos = e.getPosition();
    auto delta = pos - dragStartPosition;
    
    // Update note position
    auto& tracks = audioProcessor.getTracks();
    if (selectedTrack < tracks.size())
    {
        auto& notes = tracks[selectedTrack]->getNotes();
        if (selectedNote < notes.size())
        {
            // Vertical drag = pitch change
            int pitchDelta = (dragStartPosition.y - pos.y) / noteHeight;
            
            // Horizontal drag = time shift
            double timeDelta = xToTime(delta.x) - xToTime(0);
            
            // Apply changes (simplified - in production, store original values)
            repaint();
        }
    }
}

void PianoRollComponent::mouseUp(const juce::MouseEvent& e)
{
    isDraggingNote = false;
}

void PianoRollComponent::mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel)
{
    if (e.mods.isCtrlDown() || e.mods.isCommandDown())
    {
        // Zoom
        float zoomDelta = wheel.deltaY * 0.1f;
        setZoom(zoom + zoomDelta);
    }
    else
    {
        // Scroll
        scrollPosition -= wheel.deltaY * 100.0;
        scrollPosition = juce::jmax(0.0, scrollPosition);
        repaint();
    }
}

void PianoRollComponent::setZoom(float newZoom)
{
    zoom = juce::jlimit(0.1f, 10.0f, newZoom);
    repaint();
}

void PianoRollComponent::setVerticalZoom(float newVerticalZoom)
{
    verticalZoom = juce::jlimit(0.5f, 3.0f, newVerticalZoom);
    repaint();
}

void PianoRollComponent::scrollTo(double time)
{
    scrollPosition = time;
    repaint();
}

void PianoRollComponent::timerCallback()
{
    // Update view if needed
}

void PianoRollComponent::drawPianoKeys(juce::Graphics& g, juce::Rectangle<int> area)
{
    g.setColour(juce::Colour(0xFF2D2D2D));
    g.fillRect(area);
    
    // Draw piano keys (simplified)
    for (int note = 0; note < 128; ++note)
    {
        int y = pitchToY(note);
        bool isBlackKey = false;
        int noteInOctave = note % 12;
        
        if (noteInOctave == 1 || noteInOctave == 3 || noteInOctave == 6 ||
            noteInOctave == 8 || noteInOctave == 10)
        {
            isBlackKey = true;
        }
        
        g.setColour(isBlackKey ? juce::Colours::black : juce::Colours::white);
        g.fillRect(area.getX(), y, area.getWidth(), noteHeight - 1);
        
        // Draw note name for C notes
        if (noteInOctave == 0)
        {
            g.setColour(juce::Colours::grey);
            g.drawText("C" + juce::String(note / 12 - 1), 
                      area.getX() + 5, y, area.getWidth() - 10, noteHeight,
                      juce::Justification::centredLeft, false);
        }
    }
}

void PianoRollComponent::drawGrid(juce::Graphics& g, juce::Rectangle<int> area)
{
    g.setColour(juce::Colour(0xFF1A1A1A));
    g.fillRect(area);
    
    // Vertical grid lines (time)
    g.setColour(juce::Colour(0xFF303030));
    float beatWidth = 50.0f * zoom;
    
    for (int beat = 0; beat < 100; ++beat)
    {
        float x = beat * beatWidth - scrollPosition * zoom;
        if (x >= 0 && x < area.getWidth())
        {
            g.drawVerticalLine(area.getX() + static_cast<int>(x), 
                             area.getY(), area.getBottom());
        }
    }
    
    // Horizontal grid lines (pitches)
    for (int note = 0; note < 128; ++note)
    {
        int y = pitchToY(note);
        g.setColour(juce::Colour(0xFF252525));
        g.drawHorizontalLine(y, area.getX(), area.getRight());
    }
}

void PianoRollComponent::drawNotes(juce::Graphics& g, juce::Rectangle<int> area)
{
    auto& tracks = audioProcessor.getTracks();
    
    juce::Colour trackColors[] = {
        juce::Colours::red,
        juce::Colours::blue,
        juce::Colours::green,
        juce::Colours::yellow
    };
    
    for (int trackIdx = 0; trackIdx < tracks.size(); ++trackIdx)
    {
        if (tracks[trackIdx]->isMuted())
            continue;
        
        g.setColour(trackColors[trackIdx % 4].withAlpha(0.7f));
        
        for (const auto& note : tracks[trackIdx]->getNotes())
        {
            float x = timeToX(note.startTime);
            float width = timeToX(note.duration);
            int y = pitchToY(note.pitch);
            
            g.fillRect(area.getX() + static_cast<int>(x), y, static_cast<int>(width), noteHeight - 2);
            
            // Draw velocity indicator
            float velocityAlpha = note.velocity / 127.0f;
            g.setColour(juce::Colours::white.withAlpha(velocityAlpha * 0.3f));
            g.fillRect(area.getX() + static_cast<int>(x), y, static_cast<int>(width), 2);
        }
    }
}

float PianoRollComponent::timeToX(double time) const
{
    return static_cast<float>((time - scrollPosition) * zoom * 50.0);
}

double PianoRollComponent::xToTime(float x) const
{
    return (x / (zoom * 50.0)) + scrollPosition;
}

int PianoRollComponent::pitchToY(int pitch) const
{
    return getHeight() - ((pitch + 1) * noteHeight);
}

int PianoRollComponent::yToPitch(int y) const
{
    return (getHeight() - y) / noteHeight;
}
