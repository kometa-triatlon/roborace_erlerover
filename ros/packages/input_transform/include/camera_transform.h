#ifndef CAMERA_TRANSFORM_H
#define CAMERA_TRANSFORM_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class CameraTransform
{
  public:
    CameraTransform();
    virtual ~CameraTransform();
    void spin();

  private:
    sensor_msgs::Image::ConstPtr mLastImage{nullptr};

    ros::NodeHandle mNodeHandle;
    image_transport::ImageTransport mImageTransport;
    image_transport::Subscriber mCameraSubscriber;
    image_transport::Publisher mOutputPublisher;
    float mProcessRate = 15.f;

    cv::Mat M;
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
};

#endif /* CAMERA_TRANSFORM_H */
