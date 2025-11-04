#include "TimbreEncoder.h"
#include <cmath>

TimbreEncoder::TimbreEncoder()
{
    initializeInstrumentEmbeddings();
}

TimbreEncoder::~TimbreEncoder()
{
}

std::vector<float> TimbreEncoder::extractTimbreEmbedding(
    const std::vector<std::vector<float>>& melSpectrogram)
{
    // TODO: Use trained autoencoder to extract timbre features
    // For now, compute simplified spectral features
    
    std::vector<float> embedding(64, 0.0f);
    
    if (melSpectrogram.empty())
        return embedding;
    
    // Average over time
    for (const auto& frame : melSpectrogram)
    {
        for (size_t i = 0; i < std::min(frame.size(), embedding.size()); ++i)
        {
            embedding[i] += frame[i];
        }
    }
    
    for (float& val : embedding)
    {
        val /= melSpectrogram.size();
    }
    
    return embedding;
}

std::string TimbreEncoder::findClosestInstrument(const std::vector<float>& embedding)
{
    std::string closestInstrument = "Piano";
    float maxSimilarity = -1.0f;
    
    for (const auto& [instrument, instEmbedding] : instrumentEmbeddings)
    {
        float similarity = computeSimilarity(embedding, instEmbedding);
        if (similarity > maxSimilarity)
        {
            maxSimilarity = similarity;
            closestInstrument = instrument;
        }
    }
    
    return closestInstrument;
}

std::string TimbreEncoder::mapVoiceToInstrument(
    const std::vector<float>& features,
    const std::string& userPreference)
{
    if (!userPreference.empty())
        return userPreference;
    
    return findClosestInstrument(features);
}

void TimbreEncoder::initializeInstrumentEmbeddings()
{
    // Initialize with placeholder embeddings
    // In production, load from trained model
    
    std::vector<std::string> instruments = {
        "Piano", "Synth", "Guitar", "Bass", "Drums",
        "Strings", "Brass", "Woodwinds", "Organ", "Pad"
    };
    
    juce::Random random;
    
    for (const auto& instrument : instruments)
    {
        std::vector<float> embedding(64);
        for (int i = 0; i < 64; ++i)
        {
            embedding[i] = random.nextFloat();
        }
        instrumentEmbeddings[instrument] = embedding;
    }
}

float TimbreEncoder::computeSimilarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        return 0.0f;
    
    // Cosine similarity
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i)
    {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    normA = std::sqrt(normA);
    normB = std::sqrt(normB);
    
    if (normA < 1e-8f || normB < 1e-8f)
        return 0.0f;
    
    return dotProduct / (normA * normB);
}
