/**
 * @file mnist_kmeans.cpp - K-means clustering for MNIST using MPI
 */
#include "MnistKMeansMPI.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include "mpi.h"

// Number of clusters
const int K = 10;

// Number of images to use (for faster testing)
const int NUM_IMAGES_TO_USE = 60000;  // Use fewer images initially for faster testing

// Function to read MNIST images
std::vector<std::array<u_char, 784>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    // Read the magic number and metadata
    int magic_number, num_images, rows, cols;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::cout << "Reading " << num_images << " images of size " << rows << "x" << cols << std::endl;

    // Limit the number of images for faster testing
    num_images = std::min(num_images, NUM_IMAGES_TO_USE);

    // Read the image data
    std::vector<std::array<u_char, 784>> images(num_images);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)images[i].data(), 784);
        
        // Threshold the images to create more distinct features
        // This helps with clustering by making the digits more distinct
        for (int j = 0; j < 784; ++j) {
            // Apply thresholding: pixels below 127 become 0, above become 255
            images[i][j] = (images[i][j] > 127) ? 255 : 0;
        }
    }

    return images;
}

// Function to read MNIST labels
std::vector<u_char> read_mnist_labels(const std::string& path, int num_to_read = NUM_IMAGES_TO_USE) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    // Read the magic number and metadata
    int magic_number, num_labels;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_labels, sizeof(num_labels));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    std::cout << "Reading " << num_labels << " labels" << std::endl;

    // Limit the number of labels
    num_labels = std::min(num_labels, num_to_read);

    // Read the label data
    std::vector<u_char> labels(num_labels);
    file.read((char*)labels.data(), num_labels);

    return labels;
}

int main() {
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Load MNIST data
    std::vector<std::array<u_char, 784>> images;
    std::vector<u_char> labels;
    
    if (rank == 0) {
        try {
            std::cout << "Loading MNIST data..." << std::endl;
            images = read_mnist_images("mnist_data/train-images-idx3-ubyte");
            labels = read_mnist_labels("mnist_data/train-labels-idx1-ubyte", images.size());
            
            if (images.size() != labels.size()) {
                std::cerr << "Error: Number of images (" << images.size() 
                          << ") doesn't match number of labels (" << labels.size() << ")" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            
            std::cout << "Successfully loaded " << images.size() << " images and labels" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading MNIST data: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Set up k-means
    MnistKMeansMPI<K> kMeans;
    
    if (rank == 0) {
        std::cout << "Starting K-means clustering with K=" << K << std::endl;
        kMeans.fit(images.data(), images.size());
        std::cout << "K-means clustering completed" << std::endl;
    } else {
        kMeans.fitWork(rank);
        MPI_Finalize();
        return 0;
    }

    // Get the clusters
    auto clusters = kMeans.getClusters();

    // Evaluate the clustering results
    if (rank == 0) {
        std::cout << "\nClustering Results:\n" << std::endl;
        
        // For each cluster, count occurrences of each digit
        for (int i = 0; i < K; ++i) {
            std::vector<int> digit_counts(10, 0);
            int total_elements = clusters[i].elements.size();
            
            if (total_elements == 0) {
                std::cout << "Cluster " << i << ": Empty" << std::endl;
                continue;
            }
            
            for (int idx : clusters[i].elements) {
                int digit = labels[idx];
                digit_counts[digit]++;
            }
            
            // Find the most common digit in this cluster
            int most_common_digit = std::distance(
                digit_counts.begin(),
                std::max_element(digit_counts.begin(), digit_counts.end())
            );
            
            double purity = (double)digit_counts[most_common_digit] / total_elements * 100.0;
            
            std::cout << "Cluster " << i << ": Mostly digit " << most_common_digit 
                      << " (" << purity << "% purity, " << total_elements << " elements)" << std::endl;
            
            // Print digit distribution
            std::cout << "  Distribution: ";
            for (int digit = 0; digit < 10; ++digit) {
                std::cout << digit << ":" << digit_counts[digit] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}