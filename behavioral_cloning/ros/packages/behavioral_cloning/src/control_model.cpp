#include "control_model.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

ControlModel::ControlModel() {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
}

ControlModel::~ControlModel() {
}

void ControlModel::init(const std::string& defPath, const std::string& weightsPath,
                        const std::string& inputLayer, const std::string& outputLayer) {

    mNet.reset(new caffe::Net<float>(defPath, caffe::TEST));
    mNet->CopyTrainedLayersFrom(weightsPath);

    mNetInputBlob = mNet->blob_by_name(inputLayer);
    mNetOutputBlob = mNet->blob_by_name(outputLayer);
    mNetNumChannels = mNetInputBlob->channels();
    mNetInputSize = cv::Size(mNetInputBlob->width(), mNetInputBlob->height());
    ROS_INFO("Input size: %dx%dx%d", mNetNumChannels, mNetInputSize.width, mNetInputSize.height);
}

float ControlModel::predictSteering(sensor_msgs::Image::ConstPtr img) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::TYPE_8UC1);
        cv::Mat srcImg(cv_ptr->image);

        std::vector<cv::Mat> inputChannels;
        wrapNetInputLayer(&inputChannels);
        preprocessNetInput(srcImg, &inputChannels);

        mNet->Forward();

        const float* prediction = mNetOutputBlob->cpu_data();
        return *prediction;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return 0.f;
    }
}


void ControlModel::wrapNetInputLayer(std::vector<cv::Mat>* inputChannels) {

    const int w = mNetInputSize.width;
    const int h = mNetInputSize.height;

    float* inputData = mNetInputBlob->mutable_cpu_data();
    for (int i = 0; i < mNetNumChannels; ++i) {
        cv::Mat channel(h, w, CV_32FC1, inputData);
        inputChannels->push_back(channel);
        inputData += w * h;
    }
}

void ControlModel::preprocessNetInput(const cv::Mat& img,
                                      std::vector<cv::Mat>* inputChannels) {


    // Incoming image must be 8-bit BGR
    cv::Mat imgResized;
    if (img.size() != mNetInputSize)
        cv::resize(img, imgResized, mNetInputSize);
    else
        imgResized = img;

    // scale from [0, 255] -> [0, 1] range
    cv::Mat imgFloat;
    imgResized.convertTo(imgFloat, CV_32FC3, 1.f/255.f);

    cv::split(imgFloat, *inputChannels);
}
