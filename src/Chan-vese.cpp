#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

// Parameters for the Chan-Vese model
struct Parameters {
    double mu = 0.5;       // Weight for the smoothness of the curve
    double nu = 0;         // Weight for the penalty term (not used in this implementation)
    double lambda1 = 1.5;  // Weight for the inside region fitting term
    double lambda2 = 1;    // Weight for the outside region fitting term
    double epsilon = 0.5;  // Regularization parameter for the Heaviside function
};

// Class for managing the level set (phi function)
class LevelSet {
public:
    int width, height;              // Dimensions of the image
    std::vector<double> phi;        // Level set (phi) function

    // Constructor: initializes phi with a circular contour in the center
    LevelSet(int w, int h) : width(w), height(h), phi(w * h, 1.0) {
        initialize();  // Set the initial contour as a circle in the middle
    }

    // Initialize the contour (phi) as a circle at the center of the image
    void initialize() {
        int cx = width / 2, cy = height / 2, radius = std::min(width, height) / 4;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = y * width + x;
                // Set phi to -1 inside the circle and 1 outside the circle
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < radius * radius) {
                    phi[index] = -1.0;
                }
            }
        }
    }
};

// Helper functions
// Heaviside function used to classify pixels inside or outside the contour
inline double heaviside(double x, double epsilon) {
    return 0.5 * (1.0 + (2.0 / M_PI) * atan(x / epsilon));
}

// Dirac delta function used for computing derivatives of phi
inline double delta(double x, double epsilon) {
    return (epsilon / (M_PI * (epsilon * epsilon + x * x)));
}

// Function to compute the curvature of the level set at a given pixel (x, y)
double compute_curvature(LevelSet& levelSet, int x, int y) {
    auto index = [&](int x, int y) { return y * levelSet.width + x; };
    
    // Calculate the derivatives of phi in x and y directions
    double phi_x = (x > 0 && x < levelSet.width - 1) ? 
        (levelSet.phi[index(x + 1, y)] - levelSet.phi[index(x - 1, y)]) / 2 : 0;
    double phi_y = (y > 0 && y < levelSet.height - 1) ? 
        (levelSet.phi[index(x, y + 1)] - levelSet.phi[index(x, y - 1)]) / 2 : 0;
    
    // Calculate the gradient of phi
    double grad_phi = sqrt(phi_x * phi_x + phi_y * phi_y + 1e-8);
    return grad_phi != 0 ? (phi_x + phi_y) / grad_phi : 0;
}

// Chan-Vese segmentation algorithm
class ChanVeseSegmenter {
public:
    cv::Mat& image;         // Input image
    LevelSet& levelSet;     // Level set (phi) function
    Parameters params;      // Model parameters

    // Constructor to initialize the image, level set, and parameters
    ChanVeseSegmenter(cv::Mat& img, LevelSet& ls, Parameters p) : image(img), levelSet(ls), params(p) {}

    // Main method to evolve the level set over several iterations
    void evolve(int iterations, double dt) {
        for (int iter = 0; iter < iterations; ++iter) {
            double change = update_phi(dt); // Update phi at each iteration
            if (change / (image.rows * image.cols) < 1e-5) break; // Stop if convergence is reached
        }
    }

    // Method to update the phi function based on image intensities and curvature
    double update_phi(double dt) {
        double c1 = 0, c2 = 0, sum1 = 0, sum2 = 0;
        
        // Calculate the average intensities inside and outside the contour
        for (int i = 0; i < image.total(); ++i) {
            double H = heaviside(levelSet.phi[i], params.epsilon);  // Heaviside function
            c1 += image.data[i] * H;  // Inside region
            sum1 += H;
            c2 += image.data[i] * (1 - H);  // Outside region
            sum2 += (1 - H);
        }

        // Compute the average intensities
        c1 /= (sum1 + 1e-8);
        c2 /= (sum2 + 1e-8);

        double change = 0.0;
        
        // Update the phi function at each pixel
        for (int i = 0; i < image.total(); ++i) {
            double delta_phi = delta(levelSet.phi[i], params.epsilon);  // Delta function
            double curvature = compute_curvature(levelSet, i % image.cols, i / image.cols); // Curvature
            double force = -params.lambda1 * pow(image.data[i] - c1, 2) + params.lambda2 * pow(image.data[i] - c2, 2); // Force term
            double old_phi = levelSet.phi[i];
            levelSet.phi[i] += dt * delta_phi * (params.mu * curvature + force);  // Update phi
            change += fabs(levelSet.phi[i] - old_phi);  // Track the change in phi
        }

        return change;  // Return the total change
    }

    // Method to save the final segmentation result as an image
    void save_result(const std::string& filename) {
        cv::Mat result(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < image.total(); ++i) {
            result.data[i] = (levelSet.phi[i] > 0) ? 255 : 0;  // Set pixel to 255 inside contour and 0 outside
        }
        cv::imwrite(filename, result);  // Save the result image
    }
};

// Main function to run the segmentation
int main() {
    std::string input_file = "data.jpg";  // Input image file
    std::string output_file = "segmented.jpg";  // Output segmented image file

    // Check if the input file exists
    FILE *file = fopen(input_file.c_str(), "rb");
    if (!file) {
        std::cerr << "Error: The file " << input_file << " does not exist or is not accessible." << std::endl;
        return -1;
    }
    fclose(file);

    // Load the image with OpenCV
    cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);  // Read the image in color
    if (image.empty()) {
        std::cerr << "Error: Unable to read the image " << input_file << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully! Dimensions: " 
              << image.cols << "x" << image.rows << std::endl;

    // Convert the image to grayscale for segmentation
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Apply a Gaussian blur to smooth the image and reduce noise
    cv::Mat blurred_image;
    cv::GaussianBlur(gray_image, blurred_image, cv::Size(5, 5), 0);

    // Perform basic thresholding for initial segmentation
    cv::Mat segmented;
    cv::threshold(blurred_image, segmented, 128, 255, cv::THRESH_BINARY);

    // Save the segmented result to the output file
    if (!cv::imwrite(output_file, segmented)) {
        std::cerr << "Error: Unable to save the image " << output_file << std::endl;
        return -1;
    }

    std::cout << "Segmentation completed. Result saved in " << output_file << std::endl;
    return 0;
}
