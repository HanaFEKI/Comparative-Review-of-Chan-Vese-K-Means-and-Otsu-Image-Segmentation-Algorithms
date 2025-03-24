# Comparative-Review-of-Chan-Vese-K-Means-and-Otsu-Image-Segmentation-Algorithms
## Project Overview
This repository provides a **comparative review** of three popular **image segmentation algorithms**: **Chan-Vese**, **K-Means**, and **Otsu**. The objective of this project is to evaluate and compare the performance of these algorithms in terms of segmentation accuracy, efficiency, and robustness across different types of images.

### Algorithms Reviewed:
1. **Chan-Vese Algorithm**: A region-based active contour model that segments images based on intensity variation.
2. **K-Means Clustering**: A clustering-based algorithm that divides the image into different intensity levels using K clusters.
3. **Otsu's Method**: A thresholding-based technique that maximizes the inter-class variance to separate the image into two classes (foreground and background).

## Objectives
- Implement the three segmentation algorithms in C++.
- Compare the performance of the algorithms on various images.
- Assess the quality of segmentation in terms of accuracy, execution time, and computational efficiency.

## Report

I also added a **report** that provides a detailed comparative analysis of the three segmentation methods, outlining the theoretical mathematical foundations, advantages, and challenges associated with each algorithm.


### Programming Language:
The project is implemented using **C++**, leveraging libraries like **OpenCV** for image manipulation and processing.


## What’s Included

- **Source Code**: C++ implementations of Chan-Vese, K-Means, and Otsu algorithms.
- **Results**: Examples of segmented images produced by the algorithms.

## Requirements

To run the code, follow the instructions below:
- C++ compiler (e.g., GCC, Clang)
- OpenCV library


## How to Run the Code

1. Clone this repository.
2. Install OpenCV if you haven't already.
3. Compile the C++ code.
4. Run the executable, providing an image file for segmentation.


## Expected Output
Once the program runs successfully, the segmented images for each algorithm (Chan-Vese, K-Means, Otsu) will be displayed, and results will be saved as image files in the output directory.

## Future Work
- **Optimization**: Improve the runtime efficiency for larger images and optimize algorithmic performance.
- **Extension to 3D Images**: Extend the current 2D image segmentation to 3D medical images such as MRI scans.
- **Integration of Machine Learning**: Incorporate machine learning models to automatically adapt algorithm parameters for better segmentation.

## Contributors
- **Hana Feki** - [hana.feki@ensta.fr](mailto:hana.feki@ensta.fr)
- **Rayen Mansour** - [rayen.mansour@ensta.fr](mailto:rayen.mansour@ensta.fr)
- **Rayen Zargui** - [rayen.zargui@ensta.fr](mailto:rayen.zargui@ensta.fr)


