#include "behavioral_cloning.h"

SteeringControl::SteeringControl() {

    ROS_INFO("Starting behavioral cloning steering control node");
    ros::NodeHandle nh("~");

    std::string cameraTopic;
    nh.param<std::string>("camera_topic",  cameraTopic, "/zed/rgb/image_rect_color");

    std::string outputTopic;
    nh.param<std::string>("output_topic",  outputTopic, "/behavioral_cloning/steering_control");

    nh.param("processing_rate", mProcessRate, 15.0f);

    std::string modelPath;
    nh.param<std::string>("model_path", modelPath, "");

    std::string modelWeights;
    nh.param<std::string>("model_weights", modelWeights, "");

    std::string inputLayer;
    nh.param<std::string>("model_input", inputLayer, "");

    std::string outputLayer;
    nh.param<std::string>("model_output", outputLayer, "");

    nh.param("steering_zero_value", mSteeringZeroValue, 1500.f);
    nh.param("steering_amplitude", mSteeringAmplitude, 700.f);
    nh.param("steering_channel", mSteeringChannel, 0);

    nh.param("throttle_zero_value", mThrottleZeroValue, 1494.f);
    nh.param("throttle_default_value", mThrottleDefaultValue, 1550.f);
    nh.param("throttle_shrink_value", mThrottleShrinkValue, 10.f);
    nh.param("throttle_channel", mThrottleChannel, 2);
    
    ROS_INFO("Camera topic: %s", cameraTopic.c_str());
    ROS_INFO("Output topic: %s", outputTopic.c_str());
    ROS_INFO("Rate: %.1f", mProcessRate);
    ROS_INFO("Model: %s", modelPath.c_str());
    ROS_INFO("Weights: %s", modelWeights.c_str());
    ROS_INFO("Model input: %s", inputLayer.c_str());
    ROS_INFO("Model output: %s", outputLayer.c_str());

    mModel.init(modelPath, modelWeights, inputLayer, outputLayer);

    mCameraSubscriber = nh.subscribe<sensor_msgs::Image>(cameraTopic, 1, &SteeringControl::imageCallback, this);
    mOutputPublisher  = nh.advertise<mavros_msgs::OverrideRCIn>(outputTopic, 1);
}

SteeringControl::~SteeringControl() {}

void SteeringControl::spinOnce() {
  if (mLastImage != nullptr) {
      float model_predicted_steering = mModel.predictSteering(mLastImage);
      int steering_value = static_cast<int>(mSteeringZeroValue + mSteeringAmplitude * model_predicted_steering);
      int throttle_value = static_cast<int>(mThrottleDefaultValue - mThrottleShrinkValue * fabs(model_predicted_steering));

      auto msg = boost::make_shared<mavros_msgs::OverrideRCIn>();
      msg->channels[mSteeringChannel] = steering_value;
      msg->channels[mThrottleChannel] = throttle_value;
      mOutputPublisher.publish(msg);
  }
}

void SteeringControl::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    auto img = *msg;
    ROS_DEBUG("imageCallback: %u, %u, %s", img.width, img.height, img.encoding.c_str());

    mLastImage = msg;
}


void SteeringControl::shutdown() {
    auto msg = boost::make_shared<mavros_msgs::OverrideRCIn>();
    msg->channels[mSteeringChannel] = mSteeringZeroValue;
    msg->channels[mThrottleChannel] = mThrottleZeroValue;
    mOutputPublisher.publish(msg);
}
