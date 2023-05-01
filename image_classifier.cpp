/*
Image classification deployment using NCNN
Author: Huili Yu
*/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"    // must include this file in order to use NCNN net.

int main(int argc, char** argv) {
    // Load image
    std::string imagepath("../images/horse.png");
    cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Unable to read image file " << imagepath << std::endl;
        return -1;
    }

    // Specify the names of all classes for image classification
    std::vector<std::string> classes = {"plane", "car",  "bird", "cat",
                                        "deer",  "dog",  "frog", "horse",
                                        "ship",  "truck"};

    // Load NCNN model
    ncnn::Net net;
    int ret = net.load_param("../models/image_classifier_opt.param");
    if (ret) std::cerr << "Failed to load model parameters" << std::endl;
    ret = net.load_model("../models/image_classifier_opt.bin");
    if (ret) std::cerr << "Failed to load model weights" << std::endl;

    // Convert image data to ncnn format
    // opencv image in bgr, model needs bgr
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, 
        ncnn::Mat::PIXEL_BGR, img.cols, img.rows);

    // Preprocessing of image data
    const float mean_vals[3] = {0.5f*255.f, 0.5f*255.f, 0.5f*255.f};
    const float norm_vals[3] = {1/0.5f/255.f, 1/0.5f/255.f, 1/0.5f/255.f};
    // In ncnn, substract_mean_normalize needs input pixels in [0, 255]
    input.substract_mean_normalize(mean_vals, norm_vals);

    // Inference
    ncnn::Extractor extractor = net.create_extractor();
    extractor.input("input", input);
    ncnn::Mat C2, C3, C4, C5, output;
    extractor.extract("output", output);

    // Flatten
    ncnn::Mat out_flatterned = output.reshape(output.w * output.h * output.c);
    std::vector<float> scores;
    scores.resize(out_flatterned.w);
    for (int j=0; j<out_flatterned.w; j++) {
        scores[j] = out_flatterned[j];
    }

    // Prediction based on scores
    std::string pred_class = 
        classes[std::max_element(scores.begin(), scores.end()) - scores.begin()];

    std::cout << "The predicted class for the input image is " << pred_class << "." << std::endl;

    // Save and visualize results
    cv::namedWindow("Input_image", cv::WINDOW_NORMAL);
    cv::imshow("Input_image", img);
    cv::waitKey(0);
    std::cout << "Completed" << std::endl;
    return 0;
}
