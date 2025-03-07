/**
 * @file MnistKMeansMPI.h - a subclass of KMeansMPI to cluster MNIST digits
 */
#pragma once
#include "KMeansMPI.h"
#include <cmath>

// MNIST images are 28x28 pixels, so d=784
template <int k>
class MnistKMeansMPI : public KMeansMPI<k, 784> {
public:
    typedef typename KMeansMPI<k, 784>::Element Element;
    
    // Method to fit the MNIST data
    void fit(const Element *data, int n) {
        KMeansMPI<k, 784>::fit(data, n);
    }

protected:
    /**
     * We supply the distance method to the abstract KMeansMPI class
     * For MNIST, using Manhattan distance can sometimes work better than Euclidean
     * for binary (thresholded) images
     */
    double distance(const Element& a, const Element& b) const override {
        // Manhattan distance
        double sum = 0.0;
        for (int i = 0; i < 784; i++) {
            sum += std::abs((double)a[i] - (double)b[i]);
        }
        return sum;
        
        // Alternatively, you could use Euclidean distance:
        /*
        double sum = 0.0;
        for (int i = 0; i < 784; i++) {
            double diff = (double)a[i] - (double)b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
        */
    }
};