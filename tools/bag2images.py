#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument('output_log_file', help='Output log file')
    parser.add_argument("output_img_dir", help="Output image directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    with open(args.output_log_file, 'w') as log:
        log.write('time,img_path\n')
        count = 0
        for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            out_img_path = os.path.join(args.output_img_dir, "frame%06i.png" % count)
            cv2.imwrite(out_img_path, cv_img)
            log.write("%d,%s\n" % (t.to_nsec(), out_img_path))
            print count

            count += 1

        bag.close()


if __name__ == '__main__':
    main()
