
#ifndef BEHAVIORAL_CLONING_H
#define BEHAVIORAL_CLONING_H

#include "control_model.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <mavros_msgs/OverrideRCIn.h>

class SteeringControl
{
  public:
    SteeringControl();
    virtual ~SteeringControl();
    void spinOnce();
    void shutdown();

  private:

    float mSteeringZeroValue;
    float mSteeringAmplitude;
    int mSteeringChannel;

    float mThrottleZeroValue;
    float mThrottleDefaultValue;
    float mThrottleShrinkValue;
    int mThrottleChannel;

    sensor_msgs::Image::ConstPtr mLastImage{nullptr};
    ros::Publisher mOutputPublisher;
    ros::Subscriber mCameraSubscriber;

    ControlModel mModel;
    float mProcessRate;
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
};


#endif /* BEHAVIORAL_CLONING_H */
