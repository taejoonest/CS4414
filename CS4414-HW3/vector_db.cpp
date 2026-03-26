#include "vector_db.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

VectorDB::VectorDB(const std::string& json_path, IndexType type, size_t nlist) 
    : json_path_(json_path), index_type_(type), nlist_(nlist), dimension_(768) {
    index_ = nullptr;
}

VectorDB::~VectorDB() {
    if (index_) {
        delete index_;
    }
}

void VectorDB::load_embeddings() {
    std::cout << "Loading embeddings from " << json_path_ << "..." << std::endl;

    std::ifstream ifs(json_path_);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + json_path_);
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
    ifs.close();

    parse_json(content);

    if (documents_.empty()) {
        throw std::runtime_error("No documents found in JSON file");
    }

    std::cout << "Loaded " << documents_.size() << " documents" << std::endl;

    for (const auto& [doc_id, doc] : documents_) {
        if (doc.embedding.size() != dimension_) {
            throw std::runtime_error(
                "Document ID '" + doc_id + "' has embedding dimension " +
                std::to_string(doc.embedding.size()) + ", expected " +
                std::to_string(dimension_)
            );
        }
    }

    std::cout << "Embedding dimension: " << dimension_ << std::endl;
}

void VectorDB::build_index() {
    if (documents_.empty()) {
        throw std::runtime_error("Must load embeddings before building index");
    }

    size_t num_docs = index_to_id_.size();
    std::vector<float> embeddings_matrix(num_docs * dimension_);

    for (size_t i = 0; i < num_docs; ++i) {
        const std::string& doc_id = index_to_id_[i];
        const Document& doc = documents_.at(doc_id);
        for (size_t j = 0; j < dimension_; ++j) {
            embeddings_matrix[i * dimension_ + j] = doc.embedding[j];
        }
    }

    if (index_type_ == IndexType::FLAT) {
        std::cout << "Building FAISS IndexFlatL2 index..." << std::endl;
        index_ = new faiss::IndexFlatL2(dimension_);
        index_->add(num_docs, embeddings_matrix.data());
    } else if (index_type_ == IndexType::IVF_FLAT) {
        std::cout << "Building FAISS IndexIVFFlat index (nlist=" << nlist_ << ")..." << std::endl;
        
        // Create quantizer and IVF index
        faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dimension_);
        faiss::IndexIVFFlat* ivf_index = new faiss::IndexIVFFlat(quantizer, dimension_, nlist_);
        
        // Train the index
        std::cout << "Training IVF index..." << std::endl;
        ivf_index->train(num_docs, embeddings_matrix.data());
        
        // Add vectors
        ivf_index->add(num_docs, embeddings_matrix.data());
        
        // Set nprobe for search quality (search in more clusters)
        ivf_index->nprobe = std::min(nlist_, size_t(10));
        
        index_ = ivf_index;
    }

    std::cout << "Index built with " << index_->ntotal << " vectors" << std::endl;
    std::cout << "Index is trained: " << (index_->is_trained ? "true" : "false")  << std::endl;
}

SearchResult VectorDB::search(const std::vector<float>& query_embedding, size_t k) {
    if (!index_) {
        throw std::runtime_error("Must build index before searching");
    }

    if (k <= 0) {
        throw std::runtime_error("k must be positive");
    }
    if (k > index_->ntotal) {
        k = index_->ntotal;
    }

    if (query_embedding.size() != dimension_) {
        throw std::runtime_error(
            "Query embedding dimension " + std::to_string(query_embedding.size()) +
            " does not match index dimension " + std::to_string(dimension_)
        );
    }

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);

    index_->search(1, query_embedding.data(), k, distances.data(), indices.data());

    SearchResult result;
    result.distances = std::move(distances);
    result.indices = std::move(indices);
    result.batch_size = 1;
    result.k = k;

    return result;
}

SearchResult VectorDB::batch_search(const std::vector<std::vector<float>>& query_embeddings, size_t k) {
    if (!index_) {
        throw std::runtime_error("Must build index before searching");
    }

    if (query_embeddings.empty()) {
        throw std::runtime_error("Query embeddings cannot be empty");
    }

    if (k <= 0) {
        throw std::runtime_error("k must be positive");
    }
    if (k > index_->ntotal) {
        k = index_->ntotal;
    }

    size_t batch_size = query_embeddings.size();
    
    // Flatten the query embeddings into a single matrix
    std::vector<float> query_matrix(batch_size * dimension_);
    for (size_t i = 0; i < batch_size; ++i) {
        if (query_embeddings[i].size() != dimension_) {
            throw std::runtime_error(
                "Query embedding " + std::to_string(i) + " has dimension " +
                std::to_string(query_embeddings[i].size()) + ", expected " +
                std::to_string(dimension_)
            );
        }
        for (size_t j = 0; j < dimension_; ++j) {
            query_matrix[i * dimension_ + j] = query_embeddings[i][j];
        }
    }

    std::vector<float> distances(batch_size * k);
    std::vector<faiss::idx_t> indices(batch_size * k);

    index_->search(batch_size, query_matrix.data(), k, distances.data(), indices.data());

    SearchResult result;
    result.distances = std::move(distances);
    result.indices = std::move(indices);
    result.batch_size = batch_size;
    result.k = k;

    return result;
}

SearchResult VectorDB::batch_search_flat(const float* query_data, size_t num_queries, size_t dim, size_t k) {
    if (!index_) {
        throw std::runtime_error("Must build index before searching");
    }
    if (dim != dimension_) {
        throw std::runtime_error("Query dimension mismatch");
    }
    if (k > index_->ntotal) {
        k = index_->ntotal;
    }

    std::vector<float> distances(num_queries * k);
    std::vector<faiss::idx_t> indices(num_queries * k);

    // Direct call to FAISS - no copying needed
    index_->search(num_queries, query_data, k, distances.data(), indices.data());

    SearchResult result;
    result.distances = std::move(distances);
    result.indices = std::move(indices);
    result.batch_size = num_queries;
    result.k = k;

    return result;
}

const Document& VectorDB::get_document_by_index(size_t index) const {
    if (index >= index_to_id_.size()) {
        throw std::runtime_error(
            "Index " + std::to_string(index) + " out of range [0, " +
            std::to_string(index_to_id_.size()) + ")"
        );
    }
    const std::string& doc_id = index_to_id_[index];
    return get_document_by_id(doc_id);
}

const Document& VectorDB::get_document_by_id(const std::string& id) const {
    auto it = documents_.find(id);
    if (it == documents_.end()) {
        throw std::runtime_error("Document ID not found: " + id);
    }
    return it->second;
}

size_t VectorDB::size() const {
    return documents_.size();
}

void VectorDB::parse_json(const std::string& content) {
    size_t pos = 0;

    pos = skip_whitespace(content, pos);

    if (pos >= content.size() || content[pos] != '[') {
        throw std::runtime_error("Expected '[' at start of JSON array");
    }
    pos++;

    while (pos < content.size()) {
        pos = skip_whitespace(content, pos);

        if (pos < content.size() && content[pos] == ']') {
            break;
        }

        if (content[pos] != '{') {
            throw std::runtime_error("Expected '{' for document object");
        }
        pos++;

        Document doc;
        bool has_id = false, has_text = false, has_embedding = false;

        while (pos < content.size()) {
            pos = skip_whitespace(content, pos);

            if (content[pos] == '}') {
                pos++;
                break;
            }

            if (content[pos] != '"') {
                throw std::runtime_error("Expected '\"' for field name");
            }
            pos++;

            size_t key_start = pos;
            while (pos < content.size() && content[pos] != '"') {
                pos++;
            }
            std::string key = content.substr(key_start, pos - key_start);
            pos++;

            pos = skip_whitespace(content, pos);
            if (pos >= content.size() || content[pos] != ':') {
                throw std::runtime_error("Expected ':' after field name");
            }
            pos++;
            pos = skip_whitespace(content, pos);

            if (key == "id") {

                pos = skip_whitespace(content, pos);
                if (pos < content.size() && content[pos] == '"') {
                    doc.id = parse_string(content, pos);
                } else {

                    size_t start = pos;
                    while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '-')) {
                        pos++;
                    }
                    doc.id = content.substr(start, pos - start);
                }
                has_id = true;
            } else if (key == "text") {
                doc.text = parse_string(content, pos);
                has_text = true;
            } else if (key == "embedding") {
                doc.embedding = parse_array(content, pos);
                has_embedding = true;
            } else {

                pos = find_char(content, pos, ',');
            }

            pos = skip_whitespace(content, pos);
            if (pos < content.size() && content[pos] == ',') {
                pos++;
            }
        }

        if (!has_id || !has_text || !has_embedding) {
            throw std::runtime_error("Document missing required fields");
        }

        std::string doc_id = doc.id;
        index_to_id_.push_back(doc_id);
        documents_[doc_id] = std::move(doc);

        pos = skip_whitespace(content, pos);

        if (pos < content.size() && content[pos] == ',') {
            pos++;
        } else if (pos < content.size() && content[pos] == ']') {
            break;
        }
    }
}

size_t VectorDB::skip_whitespace(const std::string& s, size_t pos) const {
    while (pos < s.size() && std::isspace(s[pos])) {
        pos++;
    }
    return pos;
}

size_t VectorDB::find_char(const std::string& s, size_t pos, char ch) const {
    int depth = 0;
    while (pos < s.size()) {
        if (s[pos] == '{' || s[pos] == '[') depth++;
        else if (s[pos] == '}' || s[pos] == ']') {
            if (depth > 0) depth--;
            else if (ch == '}' || ch == ']') return pos;
        }
        else if (depth == 0 && s[pos] == ch) return pos;
        pos++;
    }
    return pos;
}

std::string VectorDB::parse_string(const std::string& s, size_t& pos) const {
    pos = skip_whitespace(s, pos);

    if (pos >= s.size() || s[pos] != '"') {
        throw std::runtime_error("Expected '\"' for string value");
    }
    pos++;

    std::string result;
    bool escaped = false;
    while (pos < s.size()) {
        if (escaped) {
            if (s[pos] == 'n') result += '\n';
            else if (s[pos] == 't') result += '\t';
            else if (s[pos] == 'r') result += '\r';
            else if (s[pos] == '\\') result += '\\';
            else if (s[pos] == '"') result += '"';
            else result += s[pos];
            escaped = false;
        } else if (s[pos] == '\\') {
            escaped = true;
        } else if (s[pos] == '"') {
            pos++;
            break;
        } else {
            result += s[pos];
        }
        pos++;
    }

    return result;
}

std::vector<float> VectorDB::parse_array(const std::string& s, size_t& pos) const {
    pos = skip_whitespace(s, pos);

    if (pos >= s.size() || s[pos] != '[') {
        throw std::runtime_error("Expected '[' for array");
    }
    pos++;

    std::vector<float> result;
    pos = skip_whitespace(s, pos);

    while (pos < s.size()) {
        if (s[pos] == ']') {
            pos++;
            break;
        }

        size_t start = pos;
        while (pos < s.size() &&
               (std::isdigit(s[pos]) || s[pos] == '.' ||
                s[pos] == '-' || s[pos] == '+' ||
                s[pos] == 'e' || s[pos] == 'E')) {
            pos++;
        }

        std::string num_str = s.substr(start, pos - start);
        try {
            result.push_back(std::stof(num_str));
        } catch (const std::exception&) {
            throw std::runtime_error("Failed to parse float: " + num_str);
        }

        pos = skip_whitespace(s, pos);
        if (pos < s.size() && s[pos] == ',') {
            pos++;
        }
        pos = skip_whitespace(s, pos);
    }

    return result;
}

#ifdef VECTOR_DB_STANDALONE

std::vector<float> load_query_embedding(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open query embedding file: " + path);
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
    ifs.close();

    std::vector<float> embedding;
    size_t pos = 0;

    while (pos < content.size() && std::isspace(content[pos])) {
        pos++;
    }

    if (pos >= content.size() || content[pos] != '[') {
        throw std::runtime_error("Expected '[' at start of query embedding array");
    }
    pos++;

    while (pos < content.size()) {
        while (pos < content.size() && std::isspace(content[pos])) {
            pos++;
        }

        if (pos < content.size() && content[pos] == ']') {
            break;
        }

        size_t start = pos;
        while (pos < content.size() &&
               (std::isdigit(content[pos]) || content[pos] == '.' ||
                content[pos] == '-' || content[pos] == '+' ||
                content[pos] == 'e' || content[pos] == 'E')) {
            pos++;
        }

        std::string num_str = content.substr(start, pos - start);
        try {
            embedding.push_back(std::stof(num_str));
        } catch (const std::exception&) {
            throw std::runtime_error("Failed to parse float: " + num_str);
        }

        while (pos < content.size() && (std::isspace(content[pos]) || content[pos] == ',')) {
            pos++;
        }
    }

    if (embedding.size() != 768) {
        throw std::runtime_error(
            "Query embedding has " + std::to_string(embedding.size()) +
            " dimensions, expected 768"
        );
    }

    return embedding;
}

void print_search_results(const VectorDB& db, const SearchResult& result) {
    std::cout << "\nTop " << result.k << " Search Results:" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    for (size_t i = 0; i < result.k; ++i) {
        faiss::idx_t doc_idx = result.indices[i];
        float distance = result.distances[i];

        const Document& doc = db.get_document_by_index(doc_idx);

        std::cout << "\nRank " << (i + 1) << ":" << std::endl;
        std::cout << "  Document ID: " << doc.id << std::endl;
        std::cout << "  Distance: " << std::fixed << std::setprecision(6) << distance << std::endl;
        std::cout << "  Text: " << doc.text << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;
}

[[noreturn]] void usage_and_exit(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --input <path> --query-embedding <path> [options]\n"
        << "Options:\n"
        << "  --input <path>            Path to preprocessed_documents.json (default: preprocessed_documents.json)\n"
        << "  --query-embedding <path>  Path to query embedding file (JSON array of 768 floats)\n"
        << "  --top-k <n>               Number of results to return (default: 3)\n"
        << "  --help, -h                Show this help message\n";
    std::exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    try {
        std::string json_path = "preprocessed_documents.json";
        std::string query_embedding_path;
        size_t top_k = 3;

        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            auto need_value = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for " << flag << "\n";
                    usage_and_exit(argv[0]);
                }
                return argv[++i];
            };
            if (arg == "--input") {
                json_path = need_value("--input");
            } else if (arg == "--query-embedding") {
                query_embedding_path = need_value("--query-embedding");
            } else if (arg == "--top-k") {
                top_k = std::stoull(need_value("--top-k"));
            } else if (arg == "--help" || arg == "-h") {
                usage_and_exit(argv[0]);
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                usage_and_exit(argv[0]);
            }
        }

        if (query_embedding_path.empty()) {
            std::cerr << "Error: --query-embedding is required\n";
            usage_and_exit(argv[0]);
        }

        std::cout << "Vector Search - RAG Components 2 & 3" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Documents: " << json_path << std::endl;
        std::cout << "Query Embedding: " << query_embedding_path << std::endl;
        std::cout << "Top-K: " << top_k << std::endl;
        std::cout << std::string(70, '=') << std::endl << std::endl;

        std::cout << "Loading document database..." << std::endl;
        VectorDB db(json_path);
        db.load_embeddings();
        db.build_index();
        std::cout << std::endl;

        std::cout << "Loading query embedding..." << std::endl;
        std::vector<float> query_embedding = load_query_embedding(query_embedding_path);
        std::cout << "Query embedding loaded (768 dimensions)" << std::endl;
        std::cout << "  First 5 values: [";
        for (int i = 0; i < 5; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << query_embedding[i];
        }
        std::cout << ", ...]" << std::endl << std::endl;

        std::cout << "Performing vector search..." << std::endl;
        SearchResult result = db.search(query_embedding, top_k);

        print_search_results(db, result);

        std::cout << "\nVector search completed" << std::endl;
        return EXIT_SUCCESS;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}

#endif
