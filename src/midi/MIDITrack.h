#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <memory>

struct MIDINote
{
    int pitch;           // 0-127
    int velocity;        // 0-127
    double startTime;    // In seconds
    double duration;     // In seconds
    int channel;         // MIDI channel
};

class MIDITrack
{
public:
    MIDITrack(const juce::String& name, const juce::String& instrumentType);
    ~MIDITrack();
    
    // Track properties
    void setName(const juce::String& name) { trackName = name; }
    juce::String getName() const { return trackName; }
    
    void setInstrumentType(const juce::String& type) { instrumentType = type; }
    juce::String getInstrumentType() const { return instrumentType; }
    
    void setChannel(int ch) { midiChannel = ch; }
    int getChannel() const { return midiChannel; }
    
    // Track state
    void setMuted(bool shouldBeMuted) { muted = shouldBeMuted; }
    bool isMuted() const { return muted; }
    
    void setSolo(bool shouldBeSolo) { solo = shouldBeSolo; }
    bool isSolo() const { return solo; }
    
    // Note management
    void addNote(const MIDINote& note);
    void removeNote(int index);
    void clearNotes();
    
    std::vector<MIDINote>& getNotes() { return notes; }
    const std::vector<MIDINote>& getNotes() const { return notes; }
    
    // Quantization
    void quantizeNotes(float quantizeGrid); // Grid in beats (0.25 = 16th notes)
    void humanizeNotes(float amount);        // Add subtle timing variations
    
    // Note editing
    void transposeNotes(int semitones);
    void scaleVelocities(float scale);
    void setNoteDuration(int noteIndex, double newDuration);
    
    // MIDI output
    void addToMidiBuffer(juce::MidiBuffer& buffer) const;
    void addToMidiBuffer(juce::MidiBuffer& buffer, double startTime, double endTime) const;
    
    // Serialization
    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& tree);
    
private:
    juce::String trackName;
    juce::String instrumentType;
    int midiChannel;
    
    bool muted;
    bool solo;
    
    std::vector<MIDINote> notes;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MIDITrack)
};
