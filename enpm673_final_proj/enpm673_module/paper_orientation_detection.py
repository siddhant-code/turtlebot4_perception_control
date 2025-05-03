#!/usr/bin/env python3

"""
Find orientation of paper
Note : Handle case if multiple pages with different orientation in frame.

Input: Preprocessed Frame/Masked frame/Frame from video (Decide what kind of frame would be useful).
Output:
    x,y,theta corresponding to the center of paper, theta being the angle subtented by longer edge of paper.(May need to use projection and homography)

"""

import cv2
import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image


class Orientation:
    def __init__(self) -> None:
        # Marker size in meters (for pose estimation)
        self.marker_length = 0.10  # 10 cm

        # ArUco dictionary and detector
        # self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        # self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def get_dummy_camera_params(self, width, height):
        camera_matrix = np.array(
            [[800, 0, width / 2], [0, 800, height / 2], [0, 0, 1]], dtype=np.float32
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return camera_matrix, dist_coeffs

    def rotation_vector_to_euler_angles(self, rvec) -> Tuple[float | int]:
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        # Convert radians to degrees
        return np.degrees([x, y, z])  # roll, pitch, yaw

    def get_orientation(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        # if ids is not None:
        if ids is not None and len(ids) > 0:
            # Estimate pose
            camera_matrix, dist_coeffs = self.get_dummy_camera_params(width, height)

            # Compute center Y for each marker
            marker_centers = [np.mean(corner[0], axis=0) for corner in corners]
            center_ys = [center[1] for center in marker_centers]

            # Find index of the marker with the largest Y (bottom-most)
            bottom_index = int(np.argmax(center_ys))

            # Extract the bottom marker
            corner = corners[bottom_index][0]
            marker_id = ids[bottom_index][0]

            pt1, pt2, pt3, pt4 = corner

            # Bounding box
            corner_pts = corner.astype(int)
            # cv2.polylines(
            #     frame_bgr, [corner_pts], isClosed=True, color=(0, 255, 255), thickness=4
            # )

            # Diagonals
            # cv2.line(
            #     frame_bgr,
            #     tuple(pt1.astype(int)),
            #     tuple(pt3.astype(int)),
            #     (255, 0, 0),
            #     1,
            # )
            # cv2.line(
            #     frame_bgr,
            #     tuple(pt2.astype(int)),
            #     tuple(pt4.astype(int)),
            #     (255, 0, 0),
            #     1,
            # )

            # Center
            center = np.mean(corner, axis=0).astype(int)
            cv2.circle(frame_bgr, tuple(center), 10, (255, 0, 255), -1)

            # ID
            # cv2.putText(
            #     frame_bgr,
            #     f"ID: {marker_id}",
            #     tuple(pt1.astype(int)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (255, 100, 0),
            #     2,
            # )

            # Pose estimation
            objp = np.array(
                [
                    [-self.marker_length / 2, self.marker_length / 2, 0],
                    [self.marker_length / 2, self.marker_length / 2, 0],
                    [self.marker_length / 2, -self.marker_length / 2, 0],
                    [-self.marker_length / 2, -self.marker_length / 2, 0],
                ],
                dtype=np.float64,
            )

            success, rvec, tvec = cv2.solvePnP(objp, corner, camera_matrix, dist_coeffs)
            if success:
                cv2.drawFrameAxes(
                    frame_bgr, camera_matrix, dist_coeffs, rvec, tvec, 0.03
                )

                # Get Euler angles
                roll, pitch, yaw = self.rotation_vector_to_euler_angles(rvec)

                yaw = -1 * yaw

                # Show yaw angle as text (Z rotation)
                cv2.putText(
                    frame_bgr,
                    f"Yaw: {yaw:.1f} deg",
                    # tuple(center + np.array([0, 20])),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    3,
                )

                # Heading arrow
                heading_length = 0.05
                origin_2D, _ = cv2.projectPoints(
                    np.array([[0, 0, 0]], dtype=np.float32),
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                target_2D, _ = cv2.projectPoints(
                    np.array([[heading_length, 0, 0]], dtype=np.float32),
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )

                pt1 = tuple(origin_2D[0][0].astype(int))
                pt2 = tuple(target_2D[0][0].astype(int))
                cv2.arrowedLine(frame_bgr, pt1, pt2, (255, 255, 0), 4, tipLength=0.5)

        return frame_bgr


class OrientationNode(Node):
    def __init__(self) -> None:
        super().__init__("orientation_node")

        self._pose = Orientation()

        self.bridge = CvBridge()
        self._process_freq = 20
        self.cv_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self._process_img = Image()

        self._process_img_pub = self.create_publisher(Image, "/process_img", 10)

        self._image_raw_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_raw_callback, 10
        )

        self._process_timer = self.create_timer(
            1 / self._process_freq, self._process_callback
        )

        self.get_logger().info("orientation_node has started")

    def _image_raw_callback(self, msg: Image) -> None:
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _process_callback(self) -> None:
        process_frame = self._pose.get_orientation(self.cv_image)

        self._process_img = self.bridge.cv2_to_imgmsg(process_frame, encoding="bgr8")
        self._process_img_pub.publish(self._process_img)

        # cv2.imshow("Camera Frame", process_frame)  # self.cv_image)
        # cv2.waitKey(1)  # required to update imshow window


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrientationNode()
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
