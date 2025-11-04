#include "MIDIGenerator.h"
#include <algorithm>

MIDIGenerator::MIDIGenerator()
{
}

MIDIGenerator::~MIDIGenerator()
{
}

void MIDIGenerator::updateTracks(const GeneratedMIDI& generatedData, 
                                  std::vector<std::shared_ptr<MIDITrack>>& tracks)
{
    if (tracks.empty())
    {
        createInitialTracks(tracks, generatedData.genre);
    }
    
    // Convert generated data to MIDI notes
    auto notes = convertToMIDINotes(
        generatedData.pitch,
        generatedData.velocity,
        generatedData.onset,
        generatedData.bpm
    );
    
    // Distribute notes to appropriate tracks based on pitch range and instrument
    if (!tracks.empty())
    {
        // For now, add all notes to the first track (melody)
        // In production, use intelligent track assignment
        tracks[0]->clearNotes();
        for (const auto& note : notes)
        {
            tracks[0]->addNote(note);
        }
    }
}

void MIDIGenerator::createInitialTracks(std::vector<std::shared_ptr<MIDITrack>>& tracks,
                                        const juce::String& genre)
{
    tracks.clear();
    
    // Create standard multi-track setup
    auto drumsTrack = std::make_shared<MIDITrack>("Drums", "Drums");
    drumsTrack->setChannel(9); // MIDI channel 10 (index 9) for drums
    
    auto bassTrack = std::make_shared<MIDITrack>("Bass", "Bass");
    bassTrack->setChannel(1);
    
    auto chordsTrack = std::make_shared<MIDITrack>("Chords", "Chords");
    chordsTrack->setChannel(2);
    
    auto melodyTrack = std::make_shared<MIDITrack>("Melody", "Melody");
    melodyTrack->setChannel(3);
    
    tracks.push_back(drumsTrack);
    tracks.push_back(bassTrack);
    tracks.push_back(chordsTrack);
    tracks.push_back(melodyTrack);
    
    // Apply genre-specific templates
    for (auto& track : tracks)
    {
        applyGenreTemplate(track, genre);
    }
}

void MIDIGenerator::regenerateTrack(std::shared_ptr<MIDITrack> track, ModelInference* inference)
{
    if (!track || !inference)
        return;
    
    // Get context from existing notes
    std::vector<std::vector<float>> contextFeatures;
    // In production, extract features from existing track
    
    // Generate new variation
    auto newData = inference->regenerateTrack(track->getInstrumentType(), contextFeatures);
    
    // Update track with new notes
    auto notes = convertToMIDINotes(newData.pitch, newData.velocity, newData.onset, newData.bpm);
    
    track->clearNotes();
    for (const auto& note : notes)
    {
        track->addNote(note);
    }
}

std::vector<MIDINote> MIDIGenerator::convertToMIDINotes(
    const std::vector<std::vector<float>>& pitchProbs,
    const std::vector<std::vector<float>>& velocities,
    const std::vector<float>& onsets,
    float bpm)
{
    std::vector<MIDINote> notes;
    
    if (pitchProbs.empty())
        return notes;
    
    const float secondsPerBeat = 60.0f / bpm;
    const int numFrames = static_cast<int>(pitchProbs.size());
    
    int currentNoteIndex = -1;
    MIDINote currentNote;
    
    for (int frame = 0; frame < numFrames; ++frame)
    {
        double timestamp = frame * 0.01; // Assuming 10ms hop size
        
        // Check for onset
        bool isOnset = (onsets.size() > frame && onsets[frame] > 0.5f);
        
        if (isOnset || currentNoteIndex == -1)
        {
            // End previous note if exists
            if (currentNoteIndex != -1)
            {
                currentNote.duration = timestamp - currentNote.startTime;
                notes.push_back(currentNote);
            }
            
            // Find dominant pitch
            int dominantPitch = 0;
            float maxProb = 0.0f;
            
            for (int pitch = 0; pitch < pitchProbs[frame].size(); ++pitch)
            {
                if (pitchProbs[frame][pitch] > maxProb)
                {
                    maxProb = pitchProbs[frame][pitch];
                    dominantPitch = pitch;
                }
            }
            
            // Create new note if probability is high enough
            if (maxProb > 0.3f)
            {
                currentNote.pitch = dominantPitch;
                currentNote.velocity = velocities.size() > frame && velocities[frame].size() > dominantPitch
                    ? static_cast<int>(velocities[frame][dominantPitch])
                    : 80;
                currentNote.startTime = timestamp;
                currentNote.duration = 0.1; // Will be updated
                currentNote.channel = 0;
                currentNoteIndex = dominantPitch;
            }
            else
            {
                currentNoteIndex = -1;
            }
        }
    }
    
    // Close final note
    if (currentNoteIndex != -1)
    {
        currentNote.duration = numFrames * 0.01 - currentNote.startTime;
        notes.push_back(currentNote);
    }
    
    return notes;
}

void MIDIGenerator::exportToFile(const std::vector<std::shared_ptr<MIDITrack>>& tracks,
                                 const juce::File& outputFile)
{
    juce::MidiFile midiFile;
    
    for (int i = 0; i < tracks.size(); ++i)
    {
        juce::MidiMessageSequence sequence;
        
        for (const auto& note : tracks[i]->getNotes())
        {
            // Note on
            juce::MidiMessage noteOn = juce::MidiMessage::noteOn(
                tracks[i]->getChannel() + 1,
                note.pitch,
                static_cast<juce::uint8>(note.velocity)
            );
            sequence.addEvent(noteOn, note.startTime);
            
            // Note off
            juce::MidiMessage noteOff = juce::MidiMessage::noteOff(
                tracks[i]->getChannel() + 1,
                note.pitch
            );
            sequence.addEvent(noteOff, note.startTime + note.duration);
        }
        
        sequence.updateMatchedPairs();
        midiFile.addTrack(sequence);
    }
    
    // Set time format (480 ticks per quarter note)
    midiFile.setTicksPerQuarterNote(480);
    
    // Write to file
    juce::FileOutputStream stream(outputFile);
    if (stream.openedOk())
    {
        midiFile.writeTo(stream);
    }
}

void MIDIGenerator::quantizeToGrid(std::vector<MIDINote>& notes, float gridSize)
{
    for (auto& note : notes)
    {
        note.startTime = std::round(note.startTime / gridSize) * gridSize;
        note.duration = std::round(note.duration / gridSize) * gridSize;
        
        if (note.duration < gridSize)
            note.duration = gridSize;
    }
}

void MIDIGenerator::applySwing(std::vector<MIDINote>& notes, float swingAmount)
{
    // Apply swing to 16th notes
    const float sixteenthNote = 0.25f; // Quarter note = 1.0
    
    for (auto& note : notes)
    {
        float beatPosition = note.startTime - std::floor(note.startTime);
        float sixteenthPosition = std::fmod(beatPosition, sixteenthNote);
        
        // Apply swing to off-beat 16th notes
        if (sixteenthPosition > sixteenthNote * 0.5f)
        {
            note.startTime += sixteenthNote * swingAmount * 0.5f;
        }
    }
}

MIDINote MIDIGenerator::createNoteFromProbabilities(
    const std::vector<float>& pitchProb,
    float velocity,
    double timestamp,
    float bpm)
{
    MIDINote note;
    
    // Find pitch with highest probability
    int bestPitch = 0;
    float maxProb = 0.0f;
    
    for (int i = 0; i < pitchProb.size(); ++i)
    {
        if (pitchProb[i] > maxProb)
        {
            maxProb = pitchProb[i];
            bestPitch = i;
        }
    }
    
    note.pitch = bestPitch;
    note.velocity = static_cast<int>(velocity);
    note.startTime = timestamp;
    note.duration = 60.0 / bpm / 4.0; // Default to 16th note
    note.channel = 0;
    
    return note;
}

void MIDIGenerator::applyGenreTemplate(std::shared_ptr<MIDITrack> track, const juce::String& genre)
{
    // Genre-specific customization would go here
    // E.g., different drum patterns, bassline styles, chord voicings
    
    if (genre == "Trap" && track->getInstrumentType() == "Drums")
    {
        // Trap-specific drum programming
    }
    else if (genre == "Afrobeat" && track->getInstrumentType() == "Bass")
    {
        // Afrobeat-specific bass patterns
    }
    // ... more genre templates
}
