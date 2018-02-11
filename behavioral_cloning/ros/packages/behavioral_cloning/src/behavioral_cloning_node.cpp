#include "behavioral_cloning.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "behavioral_cloning_node");
    SteeringControl control;
    control.spin();

    return 0;
}
