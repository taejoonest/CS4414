# Installing FAISS for C++ on macOS

## Option 1: Using Homebrew (Easiest - Recommended)

```bash
# Install FAISS via Homebrew
brew install faiss

# This installs:
# - Headers in: /opt/homebrew/include/faiss (or /usr/local/include/faiss on Intel Mac)
# - Libraries in: /opt/homebrew/lib (or /usr/local/lib on Intel Mac)
```

## Option 2: Using Conda

```bash
# Install conda if you don't have it
# Then install FAISS
conda install -c pytorch faiss-cpu

# Or for GPU support:
conda install -c pytorch faiss-gpu
```

## Option 3: Build from Source

```bash
# Clone FAISS repository
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Build with CMake
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF .
cmake --build build -j

# Install (optional)
sudo cmake --install build
```

## Using FAISS in Your C++ Project

After installation, you'll need to link against FAISS in your CMakeLists.txt or Makefile.

### Example CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.10)
project(YourProject)

set(CMAKE_CXX_STANDARD 17)

# Find FAISS
find_path(FAISS_INCLUDE_DIR faiss/Index.h
    PATHS
    /opt/homebrew/include
    /usr/local/include
    ${CMAKE_INSTALL_PREFIX}/include
)

find_library(FAISS_LIB faiss
    PATHS
    /opt/homebrew/lib
    /usr/local/lib
    ${CMAKE_INSTALL_PREFIX}/lib
)

if(FAISS_INCLUDE_DIR AND FAISS_LIB)
    message(STATUS "Found FAISS: ${FAISS_LIB}")
    include_directories(${FAISS_INCLUDE_DIR})
else()
    message(FATAL_ERROR "FAISS not found. Please install it first.")
endif()

# Your executable
add_executable(your_program your_source.cpp)
target_link_libraries(your_program ${FAISS_LIB})
```

### Example Makefile:

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -O2
INCLUDES = -I/opt/homebrew/include  # or -I/usr/local/include for Intel Mac
LIBS = -L/opt/homebrew/lib -lfaiss  # or -L/usr/local/lib for Intel Mac

your_program: your_source.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)
```

## Verify Installation

Create a test file `test_faiss.cpp`:

```cpp
#include <faiss/IndexFlat.h>
#include <iostream>

int main() {
    int d = 64;      // dimension
    int nb = 100;    // database size
    
    faiss::IndexFlatL2 index(d);
    
    std::cout << "FAISS installed successfully!" << std::endl;
    std::cout << "Index is trained: " << index.is_trained << std::endl;
    std::cout << "Index total: " << index.ntotal << std::endl;
    
    return 0;
}
```

Compile and run:
```bash
g++ -std=c++17 -I/opt/homebrew/include test_faiss.cpp -L/opt/homebrew/lib -lfaiss -o test_faiss
./test_faiss
```

