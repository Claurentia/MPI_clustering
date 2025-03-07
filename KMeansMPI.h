/**
 * @file KMeansMPI.h - implementation of k-means clustering algorithm using MPI
 * @see "Seattle University, CPSC5600, Winter 2023"
 */
#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <array>
#include <cstring>
#include <numeric>
#include "mpi.h"

template <int kClusters, int dDimensions>
class KMeansMPI {
public:
    // some type definitions to make things easier
    typedef std::array<u_char,dDimensions> Element;
    class Cluster;
    typedef std::array<Cluster,kClusters> Clusters;
    const int MAX_FIT_STEPS = 300;
    const int ROOT = 0;

    // debugging
    const bool VERBOSE = false;  // set to true for debugging output
#define V(stuff) if(VERBOSE) {using namespace std; stuff}

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()
     */
    virtual const Clusters& getClusters() {
        return clusters;
    }

    /**
     * fit() is the main k-means algorithm entry point for ROOT process
     * Called by the ROOT process to supply the data elements.
     * Other processes call fitWork directly.
     */
    virtual void fit(const Element *data, int data_n) {
        elements = data;
        n = data_n;
        dist.resize(n);
        fitWork(ROOT);
    }

    /**
     * This is the per-process work for the fitting.
     * @param rank within MPI_COMM_WORLD
     * @pre n and elements are set in ROOT process; all processes call fitWork simultaneously
     * @post clusters are now stable (or we gave up after MAX_FIT_STEPS)
     */
    virtual void fitWork(int rank) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        V(std::cout << "Process " << rank << " starting fitWork with " << size << " processes" << std::endl;)
        
        int totalSize = 0;
        if (rank == ROOT) {
            totalSize = n;
            reseedClusters();
        }
        
        MPI_Bcast(&totalSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        
        int localStart = 0, localSize = 0;
        calculatePartition(rank, size, totalSize, &localStart, &localSize);
        V(std::cout << "Process " << rank << " has partition [" << localStart << ".." << (localStart + localSize - 1) << "]" << std::endl;)
        
        Clusters prior;
        prior[0].centroid[0]++;
        int generation = 0;
        
        auto *partition = new Element[localSize];
        std::vector<std::array<double, kClusters>> localDist(localSize);
        
        while (generation++ < MAX_FIT_STEPS && prior != clusters) {
            V(std::cout << "Process " << rank << " working on generation " << generation << std::endl;)
            
            bcastCentroids(rank);
            distributeData(rank, localStart, localSize, partition);
            updateDistances(rank, localSize, partition, localDist);
            
            prior = clusters;
            
            Clusters localClusters;
            for (int j = 0; j < kClusters; j++) {
                localClusters[j].centroid = Element{};
                localClusters[j].elements.clear();
            }
            
            for (int i = 0; i < localSize; i++) {
                int closestCluster = 0;
                for (int j = 1; j < kClusters; j++) {
                    if (localDist[i][j] < localDist[i][closestCluster]) {
                        closestCluster = j;
                    }
                }
                
                if (localClusters[closestCluster].elements.empty()) {
                    localClusters[closestCluster].centroid = partition[i];
                } else {
                    accum(localClusters[closestCluster].centroid, localClusters[closestCluster].elements.size(), partition[i], 1);
                }
                localClusters[closestCluster].elements.push_back(localStart + i);
            }
            
            mergeClusters(rank, localClusters);
        }
        delete[] partition;
        V(std::cout << "Process " << rank << " completed fitWork after " << generation << " generations" << std::endl;)
    }

    /**
     * The algorithm constructs k clusters and attempts to populate them with like neighbors.
     * This inner class, Cluster, holds each cluster's centroid (mean) and the index of the objects
     * belonging to this cluster.
     */
    struct Cluster {
        Element centroid;  // the current center (mean) of the elements in the cluster
        std::vector<int> elements;

        // equality is just the centroids, regardless of elements
        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;  // equality means the same centroid, regardless of elements
        }
    };

protected:
    const Element *elements = nullptr;                  // set of elements to classify into k categories (supplied to latest call to fit())
    int n = 0;                                          // number of elements in this->elements
    Clusters clusters;                                  // k clusters resulting from latest call to fit()
    std::vector<std::array<double,kClusters>> dist;     // dist[i][j] is the distance from elements[i] to clusters[j].centroid

    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the element
     * set
     * @return list of clusters made by using k random elements as the initial centroids
     */
    virtual void reseedClusters() {
        std::vector<int> seeds;
        std::vector<int> candidates(n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto random = std::mt19937{std::random_device{}()};
        // Note that we need C++20 for std::sample
        std::sample(candidates.begin(), candidates.end(), back_inserter(seeds), kClusters, random);
        for (int i = 0; i < kClusters; i++) {
            clusters[i].centroid = elements[seeds[i]];
            clusters[i].elements.clear();
        }
    }
    
    /**
     * Calculate the partition for each process based on rank and size
     * @param rank current process rank
     * @param size total number of processes
     * @param totalSize total number of elements
     * @param start output parameter for starting index
     * @param count output parameter for count of elements
     */
    virtual void calculatePartition(int rank, int size, int totalSize, int *start, int *count) {
        int baseSize = totalSize / size;
        int remainder = totalSize % size;
        
        if (rank < remainder) {
            *count = baseSize + 1;
            *start = rank * (*count);
        } else {
            *count = baseSize;
            *start = remainder * (baseSize + 1) + (rank - remainder) * baseSize;
        }
    }

    /**
     * Broadcast the centroids from root process to all processes
     * @param rank current process rank
     */
    virtual void bcastCentroids(int rank) {
        V(std::cout << "  " << rank << " bcastCentroids" << std::endl;)
        int count = kClusters * dDimensions;
        auto *buffer = new u_char[count];
        
        if (rank == ROOT) {
            int i = 0;
            for (int j = 0; j < kClusters; j++) {
                for (int jd = 0; jd < dDimensions; jd++) {
                    buffer[i++] = clusters[j].centroid[jd];
                }
            }
            V(std::cout << "  " << rank << " sending centroids" << std::endl;)
        }
        
        MPI_Bcast(buffer, count, MPI_UNSIGNED_CHAR, ROOT, MPI_COMM_WORLD);
        
        if (rank != ROOT) {
            int i = 0;
            for (int j = 0; j < kClusters; j++) {
                for (int jd = 0; jd < dDimensions; jd++) {
                    clusters[j].centroid[jd] = buffer[i++];
                }
            }
            V(std::cout << "  " << rank << " received centroids" << std::endl;)
        }
        delete[] buffer;
    }
    
    /**
     * Distribute data from root process to all processes
     * @param rank current process rank
     * @param localStart starting index for local partition
     * @param localSize size of local partition
     * @param partition array to store local partition data
     */
    virtual void distributeData(int rank, int localStart, int localSize, Element *partition) {
        if (rank == ROOT) {
            for (int i = 0; i < localSize; i++) {
                partition[i] = elements[localStart + i];
            }
            
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            
            for (int p = 1; p < size; p++) {
                int pStart, pSize;
                calculatePartition(p, size, n, &pStart, &pSize);
                
                if (pSize > 0) {
                    MPI_Send(const_cast<Element*>(&elements[pStart]), pSize * dDimensions, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            if (localSize > 0) {
                MPI_Recv(partition, localSize * dDimensions, MPI_UNSIGNED_CHAR, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    
    /**
     * Update distances for the local partition
     * @param rank current process rank
     * @param localSize size of local partition
     * @param partition local partition data
     * @param localDist array to store local distance calculations
     */
    virtual void updateDistances(int rank, int localSize, const Element *partition, std::vector<std::array<double, kClusters>> &localDist) {
        for (int i = 0; i < localSize; i++) {
            for (int j = 0; j < kClusters; j++) {
                localDist[i][j] = distance(clusters[j].centroid, partition[i]);
            }
        }
    }
    
    /**
     * Merge clusters from all processes using MPI collective operations
     * @param rank current process rank
     * @param localClusters local cluster information
     */
    virtual void mergeClusters(int rank, const Clusters &localClusters) {
        struct ClusterData {
            std::array<double, dDimensions> weightedCentroid;
            int count;
        };
        
        auto *sendData = new ClusterData[kClusters];
        auto *recvData = new ClusterData[kClusters];
        
        for (int j = 0; j < kClusters; j++) {
            int localCount = localClusters[j].elements.size();
            
            for (int i = 0; i < dDimensions; i++) {
                sendData[j].weightedCentroid[i] = (double)localClusters[j].centroid[i] * localCount;
            }
            sendData[j].count = localCount;
        }
        
        MPI_Reduce(&sendData[0].weightedCentroid[0], &recvData[0].weightedCentroid[0], kClusters * dDimensions, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);        
        MPI_Reduce(&sendData[0].count, &recvData[0].count, kClusters, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
        
        if (rank == ROOT) {
            for (int j = 0; j < kClusters; j++) {
                clusters[j].elements.clear();
                
                if (recvData[j].count > 0) {
                    for (int i = 0; i < dDimensions; i++) {
                        double value = recvData[j].weightedCentroid[i] / recvData[j].count;
                        value = std::max(0.0, std::min(255.0, value));
                        clusters[j].centroid[i] = (u_char)value;
                    }
                }
            }
            
            assignElementsToClusters();
        }
        
        delete[] sendData;
        delete[] recvData;
    }
    
    /**
     * Assign all elements to the closest clusters (root process only)
     */
    virtual void assignElementsToClusters() {
        updateFullDistances();
        
        for (int i = 0; i < n; i++) {
            int closestCluster = 0;
            for (int j = 1; j < kClusters; j++) {
                if (dist[i][j] < dist[i][closestCluster]) {
                    closestCluster = j;
                }
            }            
            clusters[closestCluster].elements.push_back(i);
        }
    }
    
    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
     * This is only used by the root process for the final assignment.
     */
    virtual void updateFullDistances() {
        dist.resize(n);
        for (int i = 0; i < n; i++) {
            V(std::cout<<"distances for "<<i<<"(";for(int x=0;x<dDimensions;x++)printf("%02x",elements[i][x]);)
            for (int j = 0; j < kClusters; j++) {
                dist[i][j] = distance(clusters[j].centroid, elements[i]);
                V(std::cout<<" " << dist[i][j];)
            }
            V(std::cout<<std::endl;)
        }
    }
    
    /**
     * Method to update a centroid with an additional element(s)
     * @param centroid   accumulating mean of the elements in a cluster so far
     * @param centroid_n number of elements in the cluster so far
     * @param addend     another element(s) to be added; if multiple, addend is their mean
     * @param addend_n   number of addends represented in the addend argument
     */
    virtual void accum(Element& centroid, int centroid_n, const Element& addend, int addend_n) const {
        int new_n = centroid_n + addend_n;
        for (int i = 0; i < dDimensions; i++) {
            double new_total = (double) centroid[i] * centroid_n + (double) addend[i] * addend_n;
            centroid[i] = (u_char)(new_total / new_n);
        }
    }
    
    /**
     * Subclass-supplied method to calculate the distance between two elements
     * @param a one element
     * @param b another element
     * @return distance from a to b (or more abstract metric); distance(a,b) >= 0.0 always
     */
    virtual double distance(const Element& a, const Element& b) const = 0;
};