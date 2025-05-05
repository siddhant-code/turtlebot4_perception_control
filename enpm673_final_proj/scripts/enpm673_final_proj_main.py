#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped,Twist
from enpm673_module.horizon_detection import detect_horizon
from enpm673_module.stop_sign_detection import detect_stop_sign
from enpm673_module.obstacle_detection import detect_obstacle
from enpm673_module.paper_orientation_detection import Orientation
from enpm673_module import preprocessing

class Controller(Node):
    def __init__(self) -> None:
        super().__init__("controller_node")
        self._process_freq = 20
        self.bridge = CvBridge()
        self.camera_subscriber = self.create_subscription(Image, "/camera/image_raw", self.camera_callback, 10)
        self._process_img_pub = self.create_publisher(Image, "/process_img", 10)
        self.velocity_pub = self.create_publisher(Twist,"/cmd_vel",10)
        self.horizon_detected = False
        self.horizon_x,self.horizon_y = None,None
        self.velocity_msg = Twist()
        self.aruco_orientation = Orientation()
        self.aruco_missing_count = 0
        
    def camera_callback(self,image_msg :Image) -> None:
        raw_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")       
        canvas = raw_image.copy()
        height,width,channels = canvas.shape
        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        if not self.horizon_detected:
            self.horizon_x,self.horizon_y,self.horizon_detected = detect_horizon(gray)
        else:
            self.draw_horizon_line(canvas,self.horizon_x,self.horizon_y)
        self.get_logger().info(f"Horizon at x: {self.horizon_x} y: {self.horizon_y}")
        
        stop_sign_detected,stop_sign_bbox = detect_stop_sign(raw_image)
        obstacle_detected,obstacle_bbox = detect_obstacle(raw_image)
        aruco_detected,aruco_id,aruco_corner_list,aruco_center,aruco_yaw,arrows = self.aruco_orientation.get_results(gray)
        if stop_sign_bbox:
            self.get_logger().info("Stop sign detected!")
            canvas = self.draw_bbox(canvas,stop_sign_bbox,"Stop Sign")
        if obstacle_bbox:
            self.get_logger().info("Obstacle detected!")
            canvas = self.draw_bbox(canvas,obstacle_bbox,"Dynamic obstacle")
        if aruco_detected:
            canvas = self.draw_point(canvas,aruco_center[0],aruco_center[1])        
        
        if stop_sign_detected or obstacle_detected:
            self.publish_velocity(0.0,0.0)           
        else:
            if aruco_detected:
                self.aruco_missing_count = 0
                aruco_x,aruco_y = aruco_center
                angular_error = width/2 - aruco_x
                linear_error = height - aruco_y
                angular_vel = 0.001 * angular_error
                linear_vel = 0.001 * linear_error
                if abs(angular_error) > 3:
                    self.publish_velocity(0.0,angular_vel)
                else:
                    self.publish_velocity(linear_vel,0.0)
            else:
                self.aruco_missing_count +=1
                if self.aruco_missing_count > 50:
                    self.get_logger().info("Looking for Aruco marker!")
                    self.publish_velocity(0.0,aruco_yaw/abs(aruco_yaw) * 0.01)
                        
        self.publish_image(canvas)
    
    def publish_image(self,image) -> None:
        processed_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self._process_img_pub.publish(processed_image)
        
    def publish_velocity(self,linear_velocity,angular_velocity)->None:
        self.velocity_msg.linear.x = linear_velocity
        self.velocity_msg.angular.z = angular_velocity
        self.velocity_pub.publish(self.velocity_msg)
    
    def draw_bbox(self,image,bbox,text):
        x1,y1,width,height = bbox
        x2 = x1 + width
        y2 = y1 + height
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
    
    def draw_horizon_line(self,image,x,y):
        cv2.line(image,(0,y),(2000,y),(255,0,0),1)
        return image
    
    def draw_point(self,image,x,y):
        cv2.circle(image,(int(x),int(y)),3,(0,0,255),3)
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


if __name__ == '__main__':
    main()
