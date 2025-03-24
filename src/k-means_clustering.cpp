#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    // Check if the user provided an image input argument
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_input>" << endl; // If not, show usage message
        return -1;
    }

    // Load the image
    Mat orig = imread(argv[1]);  // Read the image file provided as an argument
    if (orig.empty()) {  // If the image is not loaded successfully, exit
        cerr << "Error: Unable to load image " << argv[1] << endl;
        return -1;
    }

    // Preprocessing step: Apply Gaussian Blur to reduce noise and smooth the image
    Mat blurred;
    GaussianBlur(orig, blurred, Size(5, 5), 2.0);  // Kernel size = 5x5, standard deviation = 2.0

    // Convert the image to a set of points (for K-means clustering)
    Mat data;
    blurred.convertTo(data, CV_32F);  // Convert the image to a 32-bit floating point matrix
    data = data.reshape(1, data.total()); // Reshape to a 2D matrix (Nx3) for K-means (flatten image)

    // Apply K-means clustering algorithm
    int K = 4;  // Number of clusters (adjustable based on the use case)
    Mat labels, centers;
    // K-means algorithm to segment the image into K clusters
    kmeans(data, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Reconstruct the segmented image using the cluster centers
    Mat segmented = Mat::zeros(orig.size(), orig.type()); // Initialize the segmented image with zeros
    // For each pixel, assign it the color of its cluster center
    for (int i = 0; i < data.rows; i++) {
        int cluster_idx = labels.at<int>(i);  // Get the cluster index for the pixel
        // Assign the pixel the corresponding color (cluster center)
        segmented.at<Vec3b>(i / orig.cols, i % orig.cols) = centers.at<Vec3f>(cluster_idx);
    }

    // Display the original and segmented images
    imshow("Original", orig);  // Show the original image
    imshow("Segmented", segmented);  // Show the segmented image
    waitKey(0);  // Wait for a key press to close the images

    // Save the segmented result to an output file
    imwrite("segmented_result.png", segmented);  // Save the segmented image to disk

    return 0;  // Exit the program successfully
}

