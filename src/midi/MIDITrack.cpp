#include "MIDITrack.h"
#include <algorithm>

MIDITrack::MIDITrack(const juce::String& name, const juce::String& instrumentType)
    : trackName(name), instrumentType(instrumentType), midiChannel(0),
      muted(false), solo(false)
{
}

MIDITrack::~MIDITrack()
{
}

void MIDITrack::addNote(const MIDINote& note)
{
    notes.push_back(note);
    
    // Keep notes sorted by start time
    std::sort(notes.begin(), notes.end(), [](const MIDINote& a, const MIDINote& b) {
        return a.startTime < b.startTime;
    });
}

void MIDITrack::removeNote(int index)
{
    if (index >= 0 && index < notes.size())
    {
        notes.erase(notes.begin() + index);
    }
}

void MIDITrack::clearNotes()
{
    notes.clear();
}

void MIDITrack::quantizeNotes(float quantizeGrid)
{
    for (auto& note : notes)
    {
        // Quantize start time to nearest grid point
        double gridTime = quantizeGrid;
        note.startTime = std::round(note.startTime / gridTime) * gridTime;
        
        // Optionally quantize duration
        note.duration = std::round(note.duration / gridTime) * gridTime;
        if (note.duration < gridTime)
            note.duration = gridTime;
    }
}

void MIDITrack::humanizeNotes(float amount)
{
    juce::Random random;
    
    for (auto& note : notes)
    {
        // Add random timing variation
        double timingVariation = (random.nextFloat() - 0.5f) * amount * 0.05; // ±2.5% max
        note.startTime += timingVariation;
        
        // Add random velocity variation
        int velocityVariation = static_cast<int>((random.nextFloat() - 0.5f) * amount * 20);
        note.velocity = juce::jlimit(1, 127, note.velocity + velocityVariation);
    }
}

void MIDITrack::transposeNotes(int semitones)
{
    for (auto& note : notes)
    {
        note.pitch = juce::jlimit(0, 127, note.pitch + semitones);
    }
}

void MIDITrack::scaleVelocities(float scale)
{
    for (auto& note : notes)
    {
        note.velocity = juce::jlimit(1, 127, static_cast<int>(note.velocity * scale));
    }
}

void MIDITrack::setNoteDuration(int noteIndex, double newDuration)
{
    if (noteIndex >= 0 && noteIndex < notes.size())
    {
        notes[noteIndex].duration = juce::jmax(0.01, newDuration);
    }
}

void MIDITrack::addToMidiBuffer(juce::MidiBuffer& buffer) const
{
    if (muted)
        return;
    
    for (const auto& note : notes)
    {
        // Note on
        juce::MidiMessage noteOn = juce::MidiMessage::noteOn(
            midiChannel + 1,
            note.pitch,
            static_cast<juce::uint8>(note.velocity)
        );
        buffer.addEvent(noteOn, static_cast<int>(note.startTime * 44100)); // Assuming 44.1kHz
        
        // Note off
        juce::MidiMessage noteOff = juce::MidiMessage::noteOff(
            midiChannel + 1,
            note.pitch,
            static_cast<juce::uint8>(0)
        );
        buffer.addEvent(noteOff, static_cast<int>((note.startTime + note.duration) * 44100));
    }
}

void MIDITrack::addToMidiBuffer(juce::MidiBuffer& buffer, double startTime, double endTime) const
{
    if (muted)
        return;
    
    for (const auto& note : notes)
    {
        if (note.startTime >= startTime && note.startTime < endTime)
        {
            juce::MidiMessage noteOn = juce::MidiMessage::noteOn(
                midiChannel + 1,
                note.pitch,
                static_cast<juce::uint8>(note.velocity)
            );
            buffer.addEvent(noteOn, static_cast<int>((note.startTime - startTime) * 44100));
            
            if (note.startTime + note.duration <= endTime)
            {
                juce::MidiMessage noteOff = juce::MidiMessage::noteOff(
                    midiChannel + 1,
                    note.pitch,
                    static_cast<juce::uint8>(0)
                );
                buffer.addEvent(noteOff, static_cast<int>((note.startTime + note.duration - startTime) * 44100));
            }
        }
    }
}

juce::ValueTree MIDITrack::toValueTree() const
{
    juce::ValueTree tree("MIDITrack");
    tree.setProperty("name", trackName, nullptr);
    tree.setProperty("instrumentType", instrumentType, nullptr);
    tree.setProperty("channel", midiChannel, nullptr);
    tree.setProperty("muted", muted, nullptr);
    tree.setProperty("solo", solo, nullptr);
    
    for (const auto& note : notes)
    {
        juce::ValueTree noteTree("Note");
        noteTree.setProperty("pitch", note.pitch, nullptr);
        noteTree.setProperty("velocity", note.velocity, nullptr);
        noteTree.setProperty("startTime", note.startTime, nullptr);
        noteTree.setProperty("duration", note.duration, nullptr);
        noteTree.setProperty("channel", note.channel, nullptr);
        tree.appendChild(noteTree, nullptr);
    }
    
    return tree;
}

void MIDITrack::fromValueTree(const juce::ValueTree& tree)
{
    trackName = tree.getProperty("name", "Untitled");
    instrumentType = tree.getProperty("instrumentType", "Unknown");
    midiChannel = tree.getProperty("channel", 0);
    muted = tree.getProperty("muted", false);
    solo = tree.getProperty("solo", false);
    
    notes.clear();
    
    for (int i = 0; i < tree.getNumChildren(); ++i)
    {
        auto noteTree = tree.getChild(i);
        
        MIDINote note;
        note.pitch = noteTree.getProperty("pitch", 60);
        note.velocity = noteTree.getProperty("velocity", 80);
        note.startTime = noteTree.getProperty("startTime", 0.0);
        note.duration = noteTree.getProperty("duration", 0.5);
        note.channel = noteTree.getProperty("channel", 0);
        
        notes.push_back(note);
    }
}
