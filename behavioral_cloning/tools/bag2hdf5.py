#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging

import numpy as np
import pandas as pd
import cv2
import h5py

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("camera_topic", help="Camera topic")
    parser.add_argument("rc_topic", help="RC topic")
    parser.add_argument("channel", type=int, help='Channel index')
    parser.add_argument('dest_file', help='Path to the output file')
    parser.add_argument('--interpolate', action='store_true', help='Interpolate')
    parser.add_argument('--zero_value', type=float, required=True)
    parser.add_argument('--amplitude', type=float, required=True)
    parser.add_argument('--img_width', type=int, default=1280)
    parser.add_argument('--img_height', type=int, default=720)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    timestamps = {args.rc_topic: [], args.camera_topic: []}
    values_rc = []
    logger.info('Start reading source data...')
    with rosbag.Bag(args.bag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[args.rc_topic, args.camera_topic]):
            timestamps[topic].append(t.to_nsec()/1000000)
            if topic == args.rc_topic:
                values_rc.append(msg.channels[args.channel])

    logger.info('Source: %d RC messages, %d camera messages',
                len(timestamps[args.rc_topic]),
                len(timestamps[args.camera_topic]))

    steering = pd.Series(values_rc, index=timestamps[args.rc_topic])
    steering_camera = steering.reindex(index=timestamps[args.camera_topic], method='nearest', tolerance=33)
    if args.interpolate:
        logger.info('Interpolating missing values')
        steering_camera = steering_camera.interpolate(method='spline', order=2)

    num_channels = 3
    num_samples = steering_camera.count()

    logger.info("Writing %d images of size %dx%dx%d", num_samples, num_channels, args.img_height, args.img_width)
    data = np.zeros((num_samples, num_channels, args.img_height, args.img_width), dtype=np.float32)
    label = np.zeros((num_samples, 1), dtype=np.float32)

    sample_id = 0
    bridge = CvBridge()
    with rosbag.Bag(args.bag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[args.camera_topic]):
            timestamp = t.to_nsec()/1000000
            value = steering_camera[timestamp]
            if np.isnan(value): continue
            img = cv2.resize(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"),
                             (args.img_width, args.img_height))
            if sample_id < 10:
                logger.debug('Image shape: [%d,%d,%d]', img.shape[0], img.shape[1], img.shape[2])
            data[sample_id, :, :, :] = img.transpose([2, 0, 1]) / 255.0
            label[sample_id] = (value - args.zero_value) / args.amplitude
            if label[sample_id] < -1.0 or label[sample_id] > 1.0:
                raise ValueError("Value = %.3f (%f) at %d" % (label[sample_id], value, sample_id))
            sample_id += 1


    if not os.path.isdir(os.path.dirname(args.dest_file)):
        os.makedirs(os.path.dirname(args.dest_file))

    with h5py.File(args.dest_file, 'w') as h5f:
        h5f.create_dataset('data', data=data, compression='gzip', compression_opts=4)
        h5f.create_dataset('label', data=label, compression='gzip', compression_opts=4)

if __name__ == '__main__':
    main()
