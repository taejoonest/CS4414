#pragma once

#include <map>
#include <string>
#include <vector>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

enum class IndexType {
    FLAT,
    IVF_FLAT
};

struct Document {
    std::string id;
    std::string text;
    std::vector<float> embedding;
};

struct SearchResult {
    std::vector<float> distances;
    std::vector<faiss::idx_t> indices;
    size_t batch_size;
    size_t k;
};

class VectorDB {
public:
    VectorDB(const std::string& json_path, IndexType type = IndexType::FLAT, size_t nlist = 100);
    ~VectorDB();

    void load_embeddings();

    void build_index();
    
    SearchResult search(const std::vector<float>& query_embedding, size_t k);
    
    SearchResult batch_search(const std::vector<std::vector<float>>& query_embeddings, size_t k);

    // Flat array version - no vector overhead
    SearchResult batch_search_flat(const float* query_data, size_t num_queries, size_t dim, size_t k);

    const Document& get_document_by_index(size_t index) const;
    const Document& get_document_by_id(const std::string& id) const;

    size_t size() const;
    
    IndexType get_index_type() const { return index_type_; }

private:
    void parse_json(const std::string& content);
    size_t skip_whitespace(const std::string& s, size_t pos) const;
    size_t find_char(const std::string& s, size_t pos, char ch) const;
    std::string parse_string(const std::string& s, size_t& pos) const;
    std::vector<float> parse_array(const std::string& s, size_t& pos) const;

    std::string json_path_;
    std::map<std::string, Document> documents_;
    std::vector<std::string> index_to_id_;
    faiss::Index* index_;
    IndexType index_type_;
    size_t nlist_;
    size_t dimension_;
};
