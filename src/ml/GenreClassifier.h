#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>

class GenreClassifier
{
public:
    GenreClassifier();
    ~GenreClassifier();
    
    // Classify genre from audio features
    std::string classifyGenre(const std::vector<std::vector<float>>& features);
    
    // Get genre embedding for conditioning
    std::vector<float> getGenreEmbedding(const std::string& genreName);
    
    // Supported genres
    std::vector<std::string> getSupportedGenres() const;
    
private:
    std::map<std::string, std::vector<float>> genreEmbeddings;
    
    void initializeGenreEmbeddings();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GenreClassifier)
};
