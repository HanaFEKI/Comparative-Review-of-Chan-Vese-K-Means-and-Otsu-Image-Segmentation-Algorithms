#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <math.h>
using namespace cv;
using namespace std;

// Function to calculate the inter-class variance for Otsu's method
float var(int hist[], int level, float val, int pix_num) {
    long long total = pix_num * val;  // Total pixel intensity sum
    int n = 0;                        // Sum of the histogram up to the threshold
    long long m = 0;                  // Mean of the intensities up to the threshold
    
    // Calculate the mean and sum of pixel intensities up to the threshold level
    for(int i = 0; i < level; i++) {
        m += i * hist[i];
        n += hist[i];
    }

    long long rem = total - m;        // Remaining intensity sum
    int rempix = pix_num - n;         // Remaining pixel count
    
    // Calculate the weight for the two classes (foreground and background)
    float w0 = (1.0 * n) / (1.0 * pix_num);  
    float w1 = (1.0 * rem) / (1.0 * pix_num);  
    
    // Calculate the mean for the two classes (foreground and background)
    float u0 = (1.0 * m) / (1.0 * n);  
    float u1 = (1.0 * rem) / (1.0 * rempix);  
    
    // Return the inter-class variance
    return w0 * w1 * (u0 - u1) * (u0 - u1);
}

int main() {
    Mat img;
    string name = "/home/hana/Downloads/data.jpg";  // Image file path
    img = imread(name);  // Load the image
    cvtColor(img, img, cv::COLOR_RGB2GRAY);  // Convert the image to grayscale

    long long u = 0;  // Variable to accumulate the sum of pixel intensities
    int hist[256];    // Histogram array to store the frequency of each intensity level
    for(int i = 0; i < 256; i++)
        hist[i] = 0;  // Initialize the histogram

    int sz = img.cols * img.rows;  // Total number of pixels in the image
    
    // Calculate the histogram and accumulate the pixel intensities
    for (int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            int n = img.at<uchar>(i, j);  // Get the pixel intensity at (i, j)
            u += n;  // Accumulate the intensity sum
            hist[n]++;  // Increment the histogram bin corresponding to the intensity
        }
    }

    int pix_num = img.rows * img.cols;  // Total number of pixels in the image
    float val = (1.0 * u) / float(pix_num);  // Average intensity value of the image

    float max = 0;  // Variable to store the maximum inter-class variance
    int threshold = 0;  // Variable to store the optimal threshold value
    
    // Loop through all possible thresholds (1 to 254)
    for(int i = 1; i < 255; i++) {
        int x = var(hist, i, val, pix_num);  // Calculate the inter-class variance for the current threshold
        if(x > max) {  // If the current variance is greater than the maximum found so far
            max = x;    // Update the maximum variance
            threshold = i;  // Update the optimal threshold
        }
    }

    // Apply the optimal threshold to segment the image
    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            if(img.at<uchar>(i, j) > threshold) {
                img.at<uchar>(i, j) = 255;  // Set pixel to white if it is above the threshold
            } else {
                img.at<uchar>(i, j) = 0;  // Set pixel to black if it is below the threshold
            }
        }
    }

    // Save the segmented image to a file
    imwrite("/home/hana/Downloads/data_otsu.png", img);

    // Display the segmented image
    imshow("image", img);
    waitKey(0);  // Wait for a key press to close the window

    return 0;  // Exit the program
}
