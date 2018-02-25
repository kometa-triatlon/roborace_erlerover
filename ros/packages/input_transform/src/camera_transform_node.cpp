#include "camera_transform.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "camera_transform_node");
    CameraTransform t;
    t.spin();

    return 0;
}
