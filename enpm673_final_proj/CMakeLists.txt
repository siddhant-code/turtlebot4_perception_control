cmake_minimum_required(VERSION 3.8)
project(enpm673_final_proj)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)


##############################
# C++ node
##############################
# Create Cpp executable
add_executable(cpp_enpm673_final_proj src/enpm673_final_proj.cpp)
ament_target_dependencies(cpp_enpm673_final_proj rclcpp)

# Include Cpp "include" directory
target_include_directories(cpp_enpm673_final_proj PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install Cpp executables
install(TARGETS cpp_enpm673_final_proj
  DESTINATION lib/${PROJECT_NAME})

##############################
# Python node
##############################
# Install Python modules
ament_python_install_package(enpm673_module)

# Install python executable
install(PROGRAMS
  scripts/enpm673_final_proj_main.py
  DESTINATION lib/${PROJECT_NAME}
  )

##############################
# ROS2 Launch 
##############################
# Install launch files
install(FILES
  launch/enpm673_world.launch.py
  DESTINATION share/${PROJECT_NAME}/launch/
  )



##############################
# Create ROS2 package
##############################
ament_package()
