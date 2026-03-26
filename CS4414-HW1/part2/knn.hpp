#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>
#include <algorithm>


template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};


// dynamic vector: runtime-D (global, set once at startup)
inline size_t& runtime_dim() {
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }
    
    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};


// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}


// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;


/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    if (items.empty()) {
        return nullptr;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int d = Embedding_T<T>::Dim();
    int axis = depth % d;  
    int median_index = (items.size()-1 ) / 2;
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Partition around the median without fully sorting
    std::nth_element(items.begin(), items.begin() + median_index, items.end(),
    [axis, d](const auto& a, const auto& b) {
        float coord_a = getCoordinate(a.first, axis);
        float coord_b = getCoordinate(b.first, axis);

        if (coord_a != coord_b)
            return coord_a < coord_b;

        // tiebreakers to ensure deterministic build
        for (int i = 1; i < d; ++i) {
            int next_axis = (axis + i) % d;
            float a_next = getCoordinate(a.first, next_axis);
            float b_next = getCoordinate(b.first, next_axis);
            if (a_next != b_next)
                return a_next < b_next;
        }
        return false;
    });
    auto t3 = std::chrono::high_resolution_clock::now();

    Node<T>* root = new Node<T>;
    root->embedding = items[median_index].first;
    root->idx = items[median_index].second;
    auto t4 = std::chrono::high_resolution_clock::now();
    
    // split
    std::vector<std::pair<T, int>> left_items;
    std::vector<std::pair<T, int>> right_items;
    
    for (int i = 0; i < median_index; i++) {
        left_items.push_back(items[i]);
    }
    
    for (int i = median_index + 1; i < (int)items.size(); i++) {
        right_items.push_back(items[i]);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    
    // Recursively search subtrees
    root->left = buildKD(left_items, depth + 1);
    root->right = buildKD(right_items, depth + 1);
    auto t6 = std::chrono::high_resolution_clock::now();
    
    // Only print timing for the top-level call (depth 0)
    if (depth == 0) {
        auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        auto partition_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0;
        auto node_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0;
        auto split_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() / 1000.0;
        auto recursion_time = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() / 1000.0;
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t6 - start).count() / 1000.0;
        
        std::cout << "\n=== buildKD Timing Breakdown ===" << std::endl;
        std::cout << "  Setup (dim, axis, median_index): " << setup_time << " ms" << std::endl;
        std::cout << "  Partition (nth_element):         " << partition_time << " ms" << std::endl;
        std::cout << "  Node creation:                   " << node_creation_time << " ms" << std::endl;
        std::cout << "  Split into left/right:           " << split_time << " ms" << std::endl;
        std::cout << "  Recursive calls:                 " << recursion_time << " ms" << std::endl;
        std::cout << "  Total buildKD time:              " << total_time << " ms" << std::endl;
        std::cout << "=================================\n" << std::endl;
    }
    
    return root;
}

template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;


/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    if (!node) {
        return;
    }
    
    int d = Embedding_T<T>::Dim();
    int axis = depth % d;  
    
    T query = Node<T>::queryEmbedding;
    
    // we update the heap
    float dist = Embedding_T<T>::distance(query, node->embedding);
    
    if (heap.size() < K) {
        heap.push(std::make_pair(dist, node->idx));
    } else if (dist < heap.top().first) {
        heap.pop(); 
        heap.push(std::make_pair(dist, node->idx));  
    }
    
    //  Search the near subtree
    bool goLeft = getCoordinate(query, axis) < getCoordinate(node->embedding, axis);
    
    if (goLeft) {
        knnSearch(node->left, depth + 1, K, heap);
    } else {
        knnSearch(node->right, depth + 1, K, heap);
    }
    
    // Decide whether to search the far subtree
    float split_distance = std::abs(getCoordinate(query, axis) - getCoordinate(node->embedding, axis));
    
    if (heap.size() < K) {
        if (goLeft) {
            knnSearch(node->right, depth + 1, K, heap);
        } else {
            knnSearch(node->left, depth + 1, K, heap);
        }
    } else {
        float farthest_distance = heap.top().first;
        if (split_distance < farthest_distance) {
            if (goLeft) {
                knnSearch(node->right, depth + 1, K, heap);
            } else {
                knnSearch(node->left, depth + 1, K, heap);
            }
        }
    }
}