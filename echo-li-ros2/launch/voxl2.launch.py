from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
    share = Path(get_package_share_directory('echo_li_ros2'))
    internal_id = LaunchConfiguration('internal_id').perform(context)
    if internal_id not in ('1', '2'):
        raise RuntimeError('internal_id must be 1 or 2')

    calibration = share / 'config' / f'voxl2_internal_id_{internal_id}.yaml'
    eqvio = share / 'config' / 'eqvio_voxl2.yaml'
    return [Node(
        package='echo_li_ros2',
        executable='voxl2_vio_node',
        name='echo_li_voxl2',
        output='screen',
        parameters=[
            str(calibration),
            {
                'echo_config_path': str(eqvio),
                'imu_topic': LaunchConfiguration('imu_topic'),
                'image_topic': LaunchConfiguration('image_topic'),
            },
        ],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'internal_id',
            description='Physical VOXL2 internal ID from its sticker (1 or 2)'),
        DeclareLaunchArgument('imu_topic', default_value='/voxl/raw_imu'),
        DeclareLaunchArgument(
            'image_topic', default_value='/tracking_front/decoded'),
        OpaqueFunction(function=_launch_node),
    ])
