import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav2_msgs.action import NavigateToPose
import cv2
import numpy as np
from cv_bridge import CvBridge

class Nav2Explorer(Node):
    def __init__(self):
        super().__init__('nav2_explorer')

        # Initialize Nav2 Action Client
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None

        # Initialize Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.bridge = CvBridge()

        # Initialize Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # State Machine Variables
        self.state = 'GO_TO_POINT'  # Initial state
        self.spin_start_time = None
        self.spin_duration = 4.0  # Time for a 360° spin

        # Box Detection Variables
        self.box_detected = False
        self.x_offset = 0.0
        self.box_area = 0.0

        # Approach Logic Variables
        self.forward_speed_approach = 0.15
        self.angular_kp = 0.003
        self.stop_distance = 1.0  # Stop if LIDAR detects box is 1m ahead
        self.offset_threshold = 40.0  # Threshold for pivoting vs driving

        # Target Positions
        self.target_positions = [
            (3.0, -8.27, -0.00143),  # Position 1
            (-7.73, -6.77, -0.00143),  # Position 2
            (7.27, 5.25, -0.00143)   # Position 3
        ]
        self.current_target_index = 0

        # Timer for Control Loop
        self.timer_period = 0.1
        self.create_timer(self.timer_period, self.control_loop)

        # Wait for Nav2 Action Server
        self.get_logger().info("Waiting for 'navigate_to_pose' action server...")
        self.action_client.wait_for_server()
        self.get_logger().info("Nav2 action server available.")

        # OpenCV Window for Live Camera Feed
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('camera_Feed', 320, 240)

    # -------------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        if self.state == 'STOPPED':
            return  

        try:
           
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert image to HSV for color detection
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Sensitivity for color detection
            sensitivity = 10

            # Define HSV ranges
            hsv_green_lower = np.array([60 - sensitivity, 100, 100])
            hsv_green_upper = np.array([60 + sensitivity, 255, 255])

            hsv_red_lower1 = np.array([0, 100, 100])
            hsv_red_upper1 = np.array([sensitivity, 255, 255])
            hsv_red_lower2 = np.array([180 - sensitivity, 100, 100])
            hsv_red_upper2 = np.array([180, 255, 255])

            hsv_blue_lower = np.array([110 - sensitivity, 100, 100])
            hsv_blue_upper = np.array([110 + sensitivity, 255, 255])

            # Masks for seperate colors
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
            red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

            # Combine masks 
            combined_mask = cv2.bitwise_or(green_mask, red_mask)
            combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

            # Filter irrelevant colors
            filtered_img = cv2.bitwise_and(cv_image, cv_image, mask=combined_mask)

            # Display the live camera feed
            cv2.imshow('camera_Feed', cv_image)
            cv2.waitKey(3)

            # Detect blue, green, and red boxes
            self.detect_box(blue_mask, 'blue')
            self.detect_box(green_mask, 'green')
            self.detect_box(red_mask, 'red')

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    def detect_box(self, mask, color):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # Skip if the area is too small
        if area < 30.0:
            return

        # Calculate the center of the contour
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
        else:
            cx = 0

        # Log detection and handle blue box
        if color == 'blue':
            self.box_detected = True
            self.box_area = area
            img_center = mask.shape[1] // 2
            self.x_offset = cx - img_center
            self.get_logger().info(f"[image_callback] Blue box detected. area={area:.1f}, offset={self.x_offset:.1f}")

            # Transition to APPROACH_BOX state if blue box is detected
            if self.state in ['GO_TO_POINT', 'LOOK_AROUND']:
                self.cancel_current_nav_goal()
                self.state = 'APPROACH_BOX'
        else:
            # Log green and red box detections
            self.get_logger().info(f"[image_callback] {color.capitalize()} box detected. area={area:.1f}")

    def scan_callback(self, msg):
        if self.state == 'STOPPED':
            return  

        # Calculate the distance ahead using LIDAR data
        ranges = np.array(msg.ranges)
        center_idx = 0
        window_size = 5
        front_ranges = ranges[center_idx : center_idx + window_size]
        valid = [r for r in front_ranges if 0.2 < r < 25.0]
        self.distance_ahead = sum(valid) / len(valid) if valid else None

    def control_loop(self):
        if self.state == 'STOPPED':
            return  

        # State Machine Logic
        if self.state == 'GO_TO_POINT':
            self.go_to_next_point()
        elif self.state == 'LOOK_AROUND':
            self.look_around()
        elif self.state == 'APPROACH_BOX':
            self.approach_box()
        else:
            self.get_logger().warn(f"Unknown state: {self.state}")

    def go_to_next_point(self):
        if self.current_target_index >= len(self.target_positions):
            self.get_logger().info("All target positions visited. Stopping.")
            self.state = 'STOPPED'
            return

        # Send Nav2 goal to the next target position
        target = self.target_positions[self.current_target_index]
        self.get_logger().info(f"[GO_TO_POINT] Going to position {self.current_target_index + 1}: {target}")
        self.send_nav2_goal(target[0], target[1], target[2])
        self.state = 'GO_TO_POINT'
        # 360° spin to look around the room once reached one of the three positions
    def look_around(self):
        if self.spin_start_time is None:
            self.spin_start_time = time.time()
            self.get_logger().info("[LOOK_AROUND] Spinning 360°")

        # Spin the robot for 360°
        twist = Twist()
        elapsed = time.time() - self.spin_start_time
        if elapsed < self.spin_duration:
            twist.angular.z = 1.57
            self.cmd_vel_pub.publish(twist)
        else:
            self.get_logger().info("[LOOK_AROUND] Done spinning => GO_TO_POINT")
            self.current_target_index += 1
            self.state = 'GO_TO_POINT'
            self.spin_start_time = None

    def approach_box(self):
        if self.distance_ahead is not None and self.distance_ahead < self.stop_distance:
            self.get_logger().info("[APPROACH_BOX] ~1m from box. Stopping.")
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            self.state = 'STOPPED'
            return

        # Adjust robot's movement based on the box's position
        offset_abs = abs(self.x_offset)
        twist = Twist()

        if offset_abs > self.offset_threshold:
            # Pivot in place to align with the box
            twist.angular.z = -self.angular_kp * self.x_offset * 2.0
            self.get_logger().info(f"[APPROACH_BOX] Pivoting. offset={self.x_offset:.1f}")
        else:
            # Drive forward while adjusting angle
            twist.linear.x = self.forward_speed_approach
            twist.angular.z = -self.angular_kp * self.x_offset
            self.get_logger().info(f"[APPROACH_BOX] Driving. offset={self.x_offset:.1f}")

        self.cmd_vel_pub.publish(twist)
    # Nav2 goal action function
    def send_nav2_goal(self, x, y, yaw):
        # Create a new goal message for the Nav2 action server
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map' # The goal is in the map frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        # Set the target position (x, y) and orientation (yaw) for the goal
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2)

        self.action_client.wait_for_server()
         # Send the goal asynchronously and set up feedback and response callbacks
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        # Handle the response from the Nav2 action server after sending the goal
        goal_handle = future.result()
        if not goal_handle.accepted:
            # If the goal was rejected, log the rejection and stay in the GO_TO_POINT state
            self.get_logger().info("[GO_TO_POINT] Goal was rejected => GO_TO_POINT")
            self.state = 'GO_TO_POINT'
            return
        # If the goal was rejected, log the rejection and stay in the GO_TO_POINT state
        self.get_logger().info("[GO_TO_POINT] Goal accepted.")
        self.current_goal_handle = goal_handle

        # Callback to handle the result when the goal is completed
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # Handle the result of the Nav2 action (success or failure)
        status = future.result().status
        self.current_goal_handle = None

        if status == 4:  # Success now look around
            self.get_logger().info("[GO_TO_POINT] Nav2 success now LOOK_AROUND")
            self.state = 'LOOK_AROUND'
            self.spin_start_time = None
        else:
            # If the goal failed, log the failure and stay in the GO_TO_POINT state
            self.get_logger().warn(f"[GO_TO_POINT] Nav2 ended status={status} => GO_TO_POINT")
            self.state = 'GO_TO_POINT'

    def cancel_current_nav_goal(self):
        # Cancel the current navigation goal and head towards the box
        if self.current_goal_handle is not None:
            self.get_logger().info("[GO_TO_POINT] Cancelling Nav2 goal => approach box")
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done)

    def cancel_done(self, future):
        self.get_logger().info("Nav2 goal canceled.")
        self.current_goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    node = Nav2Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()