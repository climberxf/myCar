#ifndef NCNN_YOLO_H
#define NCNN_YOLO_H

#include <opencv2/core/core.hpp>
#include "layer.h"
#include "net.h"
struct Object1
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class ModelWrapper1
{
public:
    ModelWrapper1(const char* param_path, const char* model_path);
    ~ModelWrapper1();
    void inference1(const cv::Mat& bgr, std::vector<Object1>& objects);

private:
    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object1>& objects);
    float intersection_area(const Object1& a, const Object1& b);
    void qsort_descent_inplace(std::vector<Object1>& objects, int left, int right);
    void qsort_descent_inplace(std::vector<Object1>& objects);
    void nms_sorted_bboxes(const std::vector<Object1>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);
    float sigmoid(float x);

    ncnn::Net yolov7;
};

void draw_objects(const cv::Mat& bgr, const std::vector<Object1>& objects);

#endif // NCNN_YOLO_H
