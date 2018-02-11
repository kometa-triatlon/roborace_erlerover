#ifndef CONTROL_MODEL_H
#define CONTROL_MODEL_H

#include <caffe/caffe.hpp>
#include <boost/weak_ptr.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ControlModel
{
  public:
    ControlModel();
    virtual ~ControlModel();

    void init(const std::string& defPath, const std::string& weightsPath,
              const std::string& inputLayer, const std::string& outputLayer);
    float predictSteering(sensor_msgs::Image::ConstPtr img);

  private:
    boost::shared_ptr<caffe::Net<float>> mNet;
    cv::Size mNetInputSize;
    int mNetNumChannels;
    boost::shared_ptr<caffe::Blob<float>> mNetInputBlob;
    boost::shared_ptr<caffe::Blob<float>> mNetOutputBlob;

    void wrapNetInputLayer(std::vector<cv::Mat>* inputChannels);
    void preprocessNetInput(const cv::Mat& img,
                            std::vector<cv::Mat>* inputChannels);
};


#endif /* CONTROL_MODEL_H */
