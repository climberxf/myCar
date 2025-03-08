#ifndef NCNN_SEG_H
#define NCNN_SEG_H

#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "layer.h"
#include "net.h"
#include "QString"
#define MAX_STRIDE 64

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class ModelWrapper
{
public:
    ModelWrapper(const char* param_path, const char* model_path);
    ~ModelWrapper();

    void inference(cv::Mat& bgr, ncnn::Mat& da_seg_mask, ncnn::Mat& ll_seg_mask);
private:
    static inline float intersection_area(const Object& a, const Object& b);
    static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
    static void qsort_descent_inplace(std::vector<Object>& faceobjects);
    static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
    static inline float sigmoid(float x);

    ncnn::Net yolopv2;
};
void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis);
void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out);
cv::Mat draw_objects(const cv::Mat& bgr, ncnn::Mat& da_seg_mask, ncnn::Mat& ll_seg_mask);
#endif // NCNN_SEG_H
