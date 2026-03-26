# Building FAISS from Source - Step by Step

Follow these commands exactly as shown in the official FAISS documentation.

## Prerequisites Check

First, make sure you have the required tools:
```bash
# Check if you have git, cmake, and make
git --version
cmake --version
make --version
```

If any are missing:
- **cmake**: `brew install cmake`
- **git**: Usually pre-installed on macOS
- **make**: Usually pre-installed on macOS

## Step 1: Clone the Repository

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
```

## Step 2: Invoking CMake

Run CMake to configure the build. For C++ only (no GPU, no Python):

```bash
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON
```

**Optional optimization flags** (choose one based on your CPU):

For AVX2 (most modern Intel/AMD CPUs):
```bash
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx2 \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON
```

For AVX512 (newer Intel CPUs):
```bash
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx512 \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON
```

For generic (works on all CPUs, but slower):
```bash
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=generic \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON
```

## Step 3: Building the C++ Library

After CMake configuration, build the library:

**For generic build:**
```bash
make -C build -j faiss
```

**For AVX2 optimization:**
```bash
make -C build -j faiss_avx2
```

**For AVX512 optimization:**
```bash
make -C build -j faiss_avx512
```

The `-j` flag enables parallel compilation. You can specify the number of jobs:
```bash
make -C build -j4 faiss  # Use 4 parallel jobs
```

Or use all CPU cores:
```bash
make -C build -j$(sysctl -n hw.ncpu) faiss
```

## Step 4: Installing the C++ Library and Headers (Optional)

To install system-wide (requires sudo):
```bash
make -C build install
```

**Note:** This step is optional. You can also use FAISS directly from the build directory without installing.

## Step 5: Verify the Build

Check that the library was built:
```bash
ls -lh build/faiss/libfaiss*
```

Check that headers are present:
```bash
ls build/faiss/*.h | head -5
```

## Using FAISS in Your Project

After building, you can use FAISS in your C++ code:

**Headers location:** `faiss/build/faiss/`
**Library location:** `faiss/build/faiss/libfaiss.a` or `libfaiss.so`

### Example Compilation:

```bash
g++ -std=c++17 \
    -I./faiss/build/faiss \
    your_program.cpp \
    -L./faiss/build/faiss \
    -lfaiss \
    -o your_program
```

### Example CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.10)
project(YourProject)

set(CMAKE_CXX_STANDARD 17)

# FAISS paths
set(FAISS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/faiss")
set(FAISS_BUILD "${FAISS_ROOT}/build")

include_directories(${FAISS_BUILD}/faiss)
link_directories(${FAISS_BUILD}/faiss)

add_executable(your_program your_source.cpp)
target_link_libraries(your_program faiss)
```

