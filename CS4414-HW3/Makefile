CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall
ARCH = arch -x86_64

FAISS_DIR = ../faiss
FAISS_INCLUDE = $(FAISS_DIR)
FAISS_LIB = $(FAISS_DIR)/build/faiss
FAISS_LIBS = $(FAISS_LIB)/libfaiss.a

LLAMA_DIR = ../llama.cpp
LLAMA_INCLUDE = $(LLAMA_DIR)/include
LLAMA_LIB = $(LLAMA_DIR)/build/bin
LLAMA_LIBS = -L$(LLAMA_LIB) -lllama -lggml-base -lggml-cpu -lggml
LLAMA_RPATH = -Wl,-rpath,$(LLAMA_LIB)

OPENMP_INCLUDE = /usr/local/opt/libomp/include
OPENMP_LIB = /usr/local/opt/libomp/lib
OPENMP_FLAGS = -Xpreprocessor -fopenmp -I$(OPENMP_INCLUDE)
OPENMP_LIBS = -L$(OPENMP_LIB) -lomp

BLAS_LIBS = -framework Accelerate

INCLUDES_FAISS = -I$(FAISS_INCLUDE) $(OPENMP_FLAGS)
INCLUDES_LLAMA = -I$(LLAMA_INCLUDE) -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src $(OPENMP_FLAGS)

LDFLAGS_FAISS = $(FAISS_LIBS) $(OPENMP_LIBS) $(BLAS_LIBS) -pthread
LDFLAGS_LLAMA = $(LLAMA_LIBS) $(LLAMA_RPATH) $(OPENMP_LIBS) $(BLAS_LIBS) -pthread


VECTOR_DB_SRC = vector_db.cpp
ENCODE_SRC = encode.cpp
LLM_GENERATION_SRC = llm_generation.cpp
MAIN_SRC = main.cpp


VECTOR_DB_OBJ = $(VECTOR_DB_SRC:.cpp=.o)
ENCODE_OBJ = $(ENCODE_SRC:.cpp=.o)
LLM_GENERATION_OBJ = $(LLM_GENERATION_SRC:.cpp=.o)
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)


VECTOR_DB = vector_db
ENCODE = encode
MAIN = main
BENCHMARK = benchmark
BATCH_BENCHMARK = batch_benchmark
INDEX_COMPARE = compare_index_types

.PHONY: all clean benchmarks

all: $(VECTOR_DB) $(ENCODE) $(MAIN)

benchmarks: $(BENCHMARK) $(BATCH_BENCHMARK) $(INDEX_COMPARE)


$(VECTOR_DB): vector_db.cpp vector_db.h
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_FAISS) -DVECTOR_DB_STANDALONE -o $@ vector_db.cpp $(LDFLAGS_FAISS)

$(ENCODE): encode.cpp encode.h
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_LLAMA) -DENCODE_STANDALONE -o $@ encode.cpp -L$(LLAMA_LIB) $(LDFLAGS_LLAMA)


$(MAIN): $(MAIN_OBJ) $(VECTOR_DB_OBJ) $(ENCODE_OBJ) $(LLM_GENERATION_OBJ)
	$(ARCH) $(CXX) $(CXXFLAGS) -o $@ $(MAIN_OBJ) $(VECTOR_DB_OBJ) $(ENCODE_OBJ) $(LLM_GENERATION_OBJ) -L$(LLAMA_LIB) $(LDFLAGS_FAISS) $(LDFLAGS_LLAMA)

$(BENCHMARK): benchmark.cpp $(VECTOR_DB_OBJ) $(ENCODE_OBJ) $(LLM_GENERATION_OBJ)
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_FAISS) $(INCLUDES_LLAMA) -o $@ benchmark.cpp $(VECTOR_DB_OBJ) $(ENCODE_OBJ) $(LLM_GENERATION_OBJ) -L$(LLAMA_LIB) $(LDFLAGS_FAISS) $(LDFLAGS_LLAMA)

$(BATCH_BENCHMARK): batch_benchmark.cpp $(VECTOR_DB_OBJ) $(ENCODE_OBJ)
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_FAISS) $(INCLUDES_LLAMA) -o $@ batch_benchmark.cpp $(VECTOR_DB_OBJ) $(ENCODE_OBJ) -L$(LLAMA_LIB) $(LDFLAGS_FAISS) $(LDFLAGS_LLAMA)

$(INDEX_COMPARE): compare_index_types.cpp $(VECTOR_DB_OBJ) $(ENCODE_OBJ)
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_FAISS) $(INCLUDES_LLAMA) -o $@ compare_index_types.cpp $(VECTOR_DB_OBJ) $(ENCODE_OBJ) -L$(LLAMA_LIB) $(LDFLAGS_FAISS) $(LDFLAGS_LLAMA)


vector_db.o: vector_db.cpp vector_db.h
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_FAISS) -c $< -o $@

encode.o: encode.cpp encode.h
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_LLAMA) -c $< -o $@

llm_generation.o: llm_generation.cpp llm_generation.h
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_LLAMA) -c $< -o $@

main.o: main.cpp vector_db.h encode.h llm_generation.h
	$(ARCH) $(CXX) $(CXXFLAGS) $(INCLUDES_FAISS) $(INCLUDES_LLAMA) -c $< -o $@

clean:
	rm -f $(VECTOR_DB) $(ENCODE) $(MAIN) $(BENCHMARK) $(BATCH_BENCHMARK) $(INDEX_COMPARE) $(VECTOR_DB_OBJ) $(ENCODE_OBJ) $(LLM_GENERATION_OBJ) $(MAIN_OBJ) benchmark.o batch_benchmark.o compare_index_types.o
