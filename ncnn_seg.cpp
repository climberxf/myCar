#include "ncnn_seg.h"

ModelWrapper::ModelWrapper(const char* param_path, const char* model_path)
{
#if NCNN_VULKAN
    yolopv2.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN
    yolopv2.load_param(param_path);
    yolopv2.load_model(model_path);
}
static int picnum = 0;
ModelWrapper::~ModelWrapper()
{
    yolopv2.clear();
}

void ModelWrapper::inference(cv::Mat& bgr, ncnn::Mat& da_seg_mask, ncnn::Mat& ll_seg_mask)
{
    const int target_size = 640;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolopv2.create_extractor();
    ex.input("images", in_pad);
    ncnn::Mat da, ll;
    {
        ex.extract("677", da);
        ex.extract("769", ll);
        slice(da, da_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
        slice(ll, ll_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
        slice(da_seg_mask, da_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
        slice(ll_seg_mask, ll_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
        interp(da_seg_mask, 1 / scale, 0, 0, da_seg_mask);
        interp(ll_seg_mask, 1 / scale, 0, 0, ll_seg_mask);
    }
}

void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts); // start
    pd.set(10, ends); // end
    pd.set(11, axes); // axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);      // resize_type
    pd.set(1, scale);  // height_scale
    pd.set(2, scale);  // width_scale
    pd.set(3, out_h);  // height
    pd.set(4, out_w);  // width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

float intersection_area(const Object& a, const Object&b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

cv::Mat draw_objects(const cv::Mat& bgr, ncnn::Mat& da_seg_mask, ncnn::Mat& ll_seg_mask)
{
    cv::Mat image = bgr.clone();

    const float* da_ptr = (float*)da_seg_mask.data;
    const float* ll_ptr = (float*)ll_seg_mask.data;
    int w = da_seg_mask.w;
    int h = da_seg_mask.h;
    for (int i = 0; i < h; i++)
    {
        cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < w; j++)
        {
            if (da_ptr[i * w + j] < da_ptr[w * h + i * w + j])
            {
                image_ptr[j] = cv::Vec3b(0, 255, 0);
            }

            if (std::round(ll_ptr[i * w + j]) == 1.0)
            {
                image_ptr[j] = cv::Vec3b(255, 0, 0);
            }
        }
    }
//    picnum += 1;
//    QString picname = "image" + QString::number(picnum) + ".jpg";
    cv::imwrite("seg_img.jpg",image);
    return image;
}
