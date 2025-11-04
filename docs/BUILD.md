# Build Instructions

## Prerequisites

### All Platforms

- CMake 3.22 or higher
- Git
- C++17 compatible compiler

### Windows

- Visual Studio 2019 or later (with C++ desktop development)
- Windows 10 SDK

### macOS

- Xcode 12.0 or later
- macOS 10.15 or later

### Linux

- GCC 11+ or Clang 14+
- ALSA development files: `sudo apt-get install libasound2-dev`
- X11 development files: `sudo apt-get install libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev`

## Step-by-Step Build

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vocal-midi-generator.git
cd vocal-midi-generator
```

### 2. Get JUCE Framework

```bash
git clone --depth 1 --branch 7.0.9 https://github.com/juce-framework/JUCE.git external/JUCE
```

### 3. Install ONNX Runtime

**Windows:**
1. Download from https://github.com/microsoft/onnxruntime/releases
2. Extract to `C:\Program Files\onnxruntime`
3. Set environment variable: `ONNXRUNTIME_DIR=C:\Program Files\onnxruntime`

**macOS:**
```bash
brew install onnxruntime
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev

# Or build from source
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
sudo cmake --install build/Linux/Release
```

### 4. Configure CMake

**Windows (Visual Studio):**
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
```

**macOS/Linux:**
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### 5. Build

**Windows:**
```bash
cmake --build . --config Release
```

**macOS/Linux:**
```bash
cmake --build . -j8
```

The `-j8` flag uses 8 parallel jobs (adjust based on your CPU cores).

### 6. Install

**Windows:**
Built plugins are in: `build/VocalMIDI_artefacts/Release/VST3/`

Manually copy to: `C:\Program Files\Common Files\VST3\`

**macOS:**
```bash
sudo cmake --install .
```

Plugins installed to:
- VST3: `/Library/Audio/Plug-Ins/VST3/`
- AU: `/Library/Audio/Plug-Ins/Components/`

**Linux:**
```bash
sudo cmake --install .
```

Plugins installed to:
- VST3: `~/.vst3/` or `/usr/local/lib/vst3/`

## Building for Development

### Debug Build

```bash
mkdir build-debug
cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

### With Address Sanitizer (Linux/macOS)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address"
cmake --build .
```

### VSCode Integration

1. Install CMake Tools extension
2. Open folder in VSCode
3. Press `Ctrl+Shift+P` → "CMake: Configure"
4. Press `F7` to build

## Troubleshooting

### JUCE Not Found

Ensure JUCE is in `external/JUCE`:
```bash
ls external/JUCE/modules
```

Should show: `juce_audio_basics`, `juce_audio_devices`, etc.

### ONNX Runtime Not Found

**Linux:** Check if library is in system paths:
```bash
ldconfig -p | grep onnxruntime
```

If not found, add to CMakeLists.txt:
```cmake
set(ONNXRUNTIME_ROOT_DIR "/path/to/onnxruntime")
```

**macOS:** If Homebrew install failed:
```bash
brew update
brew upgrade
brew install onnxruntime
```

### Linking Errors

Clean and rebuild:
```bash
rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
```

### Missing Dependencies (Linux)

Install all at once:
```bash
sudo apt-get install \
    build-essential \
    cmake \
    libasound2-dev \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libfreetype6-dev \
    libwebkit2gtk-4.0-dev \
    libcurl4-openssl-dev
```

## Testing the Build

### Load in DAW

1. Open your DAW (Ableton, Logic, FL Studio, etc.)
2. Scan for new plugins
3. Insert "VocalMIDI" on a track
4. Check console/log for errors

### Standalone Application

The standalone app is also built:
- Windows: `build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI.exe`
- macOS: `build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI.app`
- Linux: `build/VocalMIDI_artefacts/Release/Standalone/VocalMIDI`

Run it directly to test without a DAW.

## Building ML Models

See separate guide: [ML Training Guide](ML_TRAINING.md)

## Performance Optimization

### Release Build Flags

For maximum performance, edit `CMakeLists.txt`:

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    endif()
endif()
```

### Link-Time Optimization

```cmake
set_target_properties(VocalMIDI PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)
```

## Cross-Compilation

### Windows → macOS (not recommended)

Use GitHub Actions or CI/CD for cross-platform builds.

### Linux → Windows

Use MinGW:
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=toolchain-mingw.cmake
```

## Continuous Integration

See `.github/workflows/build.yml` for automated builds on:
- Ubuntu 22.04
- macOS 12
- Windows Server 2022

## Next Steps

- [Development Guide](DEVELOPMENT.md)
- [ML Training](ML_TRAINING.md)
- [Contributing](CONTRIBUTING.md)
