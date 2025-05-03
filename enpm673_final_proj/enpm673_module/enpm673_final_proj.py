#!/usr/bin/env python3


import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from paper_orientation_detection import Orientation
# from stop_sign_detection import StopSignDetector


class Perception(Node):
    def __init__(self) -> None:
        super().__init__("image_subscriber")

        # create class object based for the task

        # self._detect_stop = StopSignDetector()
        self._pose_est = Orientation()

        self.bridge = CvBridge()
        self._process_freq = 20
        self.cv_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self._process_img = Image()
        # np.ones((100, 100, 3), dtype=np.uint8) * 255

        self._process_img_pub = self.create_publisher(Image, "/process_img", 10)

        self._image_raw_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_raw_callback, 10
        )

        self._process_timer = self.create_timer(
            1 / self._process_freq, self._process_callback
        )

        self.get_logger().info("perception_node has started")

    def _image_raw_callback(self, msg: Image) -> None:
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _process_callback(self) -> None:
        process_frame = self._pose_est.get_orientation(self.cv_image)

        self._process_img = self.bridge.cv2_to_imgmsg(process_frame, encoding="bgr8")
        self._process_img_pub.publish(self._process_img)

        # cv2.imshow("Camera Frame", process_frame)  # self.cv_image)
        # cv2.waitKey(1)  # required to update imshow window


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Perception()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Spin error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
