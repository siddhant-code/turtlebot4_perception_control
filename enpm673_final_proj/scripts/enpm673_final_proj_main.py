#!/usr/bin/env python3

# std modules
import cv2
import numpy as np
from cv_bridge import CvBridge

# ros modules
import rclpy
from rclpy.node import Node

# interface modules
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TwistStamped

# user modules
from enpm673_module import preprocessing
from enpm673_module.horizon_detection import detect_horizon
from enpm673_module.obstacle_detection import detect_obstacle
from enpm673_module.stop_sign_detection import detect_stop_sign
from enpm673_module.paper_orientation_detection import Orientation


"""

PUBLISHER:
    cmd_vel
    process_img

SUBSCRIBER:
    camera_info
    image





"""


class Controller(Node):
    def __init__(self) -> None:
        super().__init__("controller_node")
        self.get_logger().info("controller_node has started.")

        # Attributes
        self._height = 720
        self._width = 1280
        self._camera_matrix = None
        self._dist_coeff = None

        # topic name
        # self._image_topic = "/camera/image_raw"
        # self._compress_img_topic = "/tb4_1/compressxxx"
        # self._camera_info_topic = "/camera/camera_info"
        # self._cmd_vel_topic = "/tb4_2/cmd_vel"
        # self._process_img_topic = "/process_img"
        
        self._image_topic = "/tb4_2/oakd/rgb/image_raw"
        self._compress_img_topic = "/tb4_2/oakd/rgb/image_raw/compressed"
        self._camera_info_topic = "/tb4_2/oakd/rgb/camera_info"
        self._cmd_vel_topic = "/tb4_2/cmd_vel"
        self._process_img_topic = "/process_img"

        self._process_freq = 20
        self.bridge = CvBridge()

        # PUBLISHER
        self._process_img_pub = self.create_publisher(
            Image, self._process_img_topic, 10
        )

        ### ------COMMENT the following Line, if running simulation
        self.velocity_msg = TwistStamped()
        self.velocity_pub = self.create_publisher(TwistStamped, self._cmd_vel_topic, 10)

        # self.velocity_msg = Twist()
        # self.velocity_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)

        # SUBSCRIBER
        self.camera_subscriber = None
        self._compress_img_sub = None

        self._camera_info_sub = self.create_subscription(
            CameraInfo, self._camera_info_topic, self._camera_info_callback, 10
        )

        self.horizon_detected = False
        self.horizon_x, self.horizon_y = None, None

        self.aruco_orientation = Orientation()
        self.aruco_missing_count = 0

    def _camera_info_callback(self, msg: CameraInfo):
        self._height = msg.height
        self._width = msg.width

        # Camera matrix (3x3)
        self._camera_matrix = np.array(msg.k).reshape((3, 3))

        # Distortion coefficients (length 5 or more)
        self._dist_coeff = np.array(msg.d)

        ret = self.aruco_orientation.set_camera_param(
            height=self._height,
            width=self._width,
            cam_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeff,
        )

        if ret:
            # self.camera_subscriber = self.create_subscription(
            #     Image, self._image_topic, self.camera_callback, 10
            # )
            self.get_logger().info("I received image info")
            self._compress_img_sub = self.create_subscription(
                CompressedImage,
                self._compress_img_topic,
                self.camera_callback,
                10,
            )
            self.destroy_subscription(self._camera_info_sub)
            self._camera_info_sub = None

    def camera_callback(self, image_msg: CompressedImage) -> None:
    # def camera_callback(self, image_msg: Image) -> None:
    
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # RAW Image
        # raw_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        canvas = raw_image.copy()
        # height, width, channels = canvas.shape

        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        _, gray_thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

        # stop sign
        stop_sign_detected, stop_sign_bbox = detect_stop_sign(raw_image)

        # detect obstacle
        obstacle_detected, obstacle_bbox = detect_obstacle(raw_image)

        aruco_detected, _, aruco_corner_list, aruco_center, aruco_yaw, arrow_pt = (
            self.aruco_orientation.get_results(gray)
        )

        # (
        #     aruco_detected,
        #     _,
        #     aruco_corner_list,
        #     aruco_center,
        #     aruco_yaw,
        #     arrow_pt,
        # ) = self.aruco_orientation.get_results2(gray)

        # LOGIC
        if not self.horizon_detected:
            # self.kl = 0.0005
            self.kl = 1
            self.horizon_vp1, self.horizon_vp2, self.horizon_detected = detect_horizon(
                gray_thresh, attempt_by_aruco=False, corner_list=aruco_corner_list
            )
        else:
            # self.kl = 0.004
            self.kl = 1
            self.draw_horizon_line(canvas, self.horizon_vp1, self.horizon_vp2)
            self.get_logger().info(
                f"Horizon at detected {self.horizon_vp1} {self.horizon_vp2}"
            )

        if stop_sign_bbox:
            self.get_logger().info("Stop sign detected!")
            canvas = self.draw_bbox(canvas, stop_sign_bbox, "Stop Sign")
        if obstacle_bbox:
            self.get_logger().info("Obstacle detected!")
            canvas = self.draw_bbox(canvas, obstacle_bbox, "Dynamic obstacle")
        if aruco_detected:
            canvas = self.draw_point(canvas, aruco_center[0], aruco_center[1])
            cv2.arrowedLine(
                canvas, arrow_pt[0], arrow_pt[1], (255, 255, 0), 4, tipLength=0.5
            )

        if stop_sign_detected or obstacle_detected:
            self.publish_velocity(0.0, 0.0)
        else:
            if aruco_detected:
                self.aruco_missing_count = 0
                aruco_x, aruco_y = aruco_center
                angular_error = self._width / 2 - aruco_x
                linear_error = self._height - aruco_y
                angular_vel = 0.001 * angular_error
                linear_vel = self.kl * linear_error
                if abs(angular_error) > 3:
                    self.publish_velocity(0.01, angular_vel)
                else:
                    self.publish_velocity(linear_vel, 0.0)
            else:
                self.aruco_missing_count += 1
                if self.aruco_missing_count > 50:
                    self.get_logger().info("Looking for Aruco marker!")
                    if aruco_yaw > 0:
                        self.publish_velocity(0.0, 0.09)
                    else:
                        self.publish_velocity(0.0, -0.09)

        self.publish_image(canvas)

    def publish_image(self, image) -> None:
        processed_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self._process_img_pub.publish(processed_image)

    def publish_velocity(self, linear_velocity, angular_velocity) -> None:
        # Twist
        # self.velocity_msg.linear.x = float(linear_velocity)
        # self.velocity_msg.angular.z = float(angular_velocity)

        # TwistStamped
        self.velocity_msg.header.stamp = self.get_clock().now().to_msg()
        self.velocity_msg.header.frame_id = "odom"
        self.velocity_msg.twist.linear.x = float(linear_velocity)
        self.velocity_msg.twist.angular.z = float(angular_velocity)

        self.velocity_pub.publish(self.velocity_msg)

    def draw_bbox(self, image, bbox, text):
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        return image

    def draw_horizon_line(self, image, vp1, vp2):
        point1 = int(vp1[0]), int(vp1[1])
        point2 = int(vp2[0]), int(vp2[1])
        print(point1, point2)
        cv2.line(image, point1, point2, (255, 0, 0), 2)
        cv2.circle(image, point1, 3, (255, 0, 0), 3)
        cv2.circle(image, point2, 3, (255, 0, 0), 3)
        return image

    def draw_point(self, image, x, y):
        cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), 3)
        return image


def main():
    rclpy.init()
    node = Controller()
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
