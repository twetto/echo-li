from glob import glob
from setuptools import find_packages, setup


package_name = 'echo_li_ros2'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chen-Fu Yeh',
    maintainer_email='franky85@hotmail.com.tw',
    description='ROS 2 integration for real-time ECHO-LI VIO.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bag_time_relay = echo_li_ros2.bag_time_relay:main',
            'voxl2_vio_node = echo_li_ros2.voxl2_vio_node:main',
        ],
    },
)
