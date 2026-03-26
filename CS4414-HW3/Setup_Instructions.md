You cannot just run ./main directly. Follow these steps:

Step 0: Download Required Files
The following files should be in your project directory:
- preprocessed_documents.json (provided by instructor)
- bge-base-en-v1.5-f32.gguf (BGE embedding model)
- qwen2-1_5b-instruct-q4_0.gguf (Qwen2 LLM model)

To download the models:
wget https://huggingface.co/second-state/BGE-base-en-v1.5-Embedding-GGUF/resolve/main/bge-base-en-v1.5-f32.gguf
wget "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_0.gguf" -O qwen2-1_5b-instruct-q4_0.gguf

Step 1: Install OpenMP
brew install libomp

Step 2: Build llama.cpp
cd ..  # Go to parent directory (one level up)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release

Step 3: Build FAISS
cd ../..  # Back to parent directory
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build .
make -C build -j faiss

Step 4: Return to Project Directory
cd ../CS4414-HW3  # Adjust this path to your project folder name

Step 5: Compile
make clean
make all

Step 6: Set Library Path (REQUIRED!)
export DYLD_LIBRARY_PATH=../llama.cpp/build/bin:$DYLD_LIBRARY_PATH

Step 7: Run
./main
