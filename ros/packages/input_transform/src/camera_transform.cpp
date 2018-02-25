#include "camera_transform.h"

#include <cv_bridge/cv_bridge.h>

CameraTransform::CameraTransform() : mImageTransport(mNodeHandle) {
    ROS_INFO("Starting camera transform node");

    std::string cameraTopic;
    mNodeHandle.param<std::string>("camera_topic",  cameraTopic, "/zed/right/image_rect_color");

    std::string outputTopic;
    mNodeHandle.param<std::string>("output_topic",  outputTopic, "/camera_transform/output");

    std::string transformParamPath;
    mNodeHandle.param<std::string>("transform_param", transformParamPath, "");


    M.create(3, 3, CV_32FC1);

    M.at<float>(0,0) = -3.39188f;
    M.at<float>(0,1) = -7.85469f;
    M.at<float>(0,2) = 749.3876f;
    M.at<float>(1,0) = 0.107479f;
    M.at<float>(1,1) = -12.3717f;
    M.at<float>(1,2) = 888.83401f;
    M.at<float>(2,0) = 0.0000375f;
    M.at<float>(2,1) = -0.0436753f;
    M.at<float>(2,2) = 1.0f;

    ROS_INFO("Camera topic: %s", cameraTopic.c_str());
    ROS_INFO("Output topic: %s", outputTopic.c_str());

    mCameraSubscriber = mImageTransport.subscribe(cameraTopic, 1, &CameraTransform::imageCallback, this);
    mOutputPublisher  = mImageTransport.advertise(outputTopic, 1);
}

CameraTransform::~CameraTransform() {
}

void CameraTransform::spin() {

    ros::Rate rate(mProcessRate);
    ros::spinOnce();
    while (ros::ok()) {
        if (mLastImage != nullptr) {
            try {
                cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mLastImage, sensor_msgs::image_encodings::TYPE_32FC3);
                cv::Mat srcImg(cv_ptr->image);
                cv::Mat eightBitImg;
                srcImg.convertTo(eightBitImg, CV_8UC3);

                cv::Mat resizedImg;
                cv::resize(eightBitImg, resizedImg, cv::Size(320, 180));

                cv::Mat grayscaleImg;
                cv::cvtColor(resizedImg, grayscaleImg, CV_RGB2GRAY);

                cv::Mat transformedImg;
                cv::warpPerspective(grayscaleImg, transformedImg, M, grayscaleImg.size());

                cv::Mat resultImg;
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->setClipLimit(4);
                clahe->apply(transformedImg, resultImg);
                mOutputPublisher.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", resultImg).toImageMsg());
            }
            catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
            }
        }
        ros::spinOnce();
        rate.sleep();
    }
}

void CameraTransform::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    auto img = *msg;
    ROS_DEBUG("imageCallback: %u, %u, %s", img.width, img.height, img.encoding.c_str());

    mLastImage = msg;
}
