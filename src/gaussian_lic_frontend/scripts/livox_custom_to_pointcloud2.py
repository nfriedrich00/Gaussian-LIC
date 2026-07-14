#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import struct

from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField


# Keep the layout used by the ROS1-to-ROS2 bag converter and consumed by the
# continuous-time frontends.  In particular, offset_time is the Livox per-point
# timestamp in nanoseconds relative to the packet timebase; dropping it disables
# scan deskew even though the cloud still looks geometrically valid.
POINT_STEP = 20
MAX_POINT_COUNT = 0xFFFFFFFF // POINT_STEP


def make_fields():
    return [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
        PointField(name="tag", offset=13, datatype=PointField.UINT8, count=1),
        PointField(name="line", offset=14, datatype=PointField.UINT8, count=1),
        PointField(name="offset_time", offset=16, datatype=PointField.UINT32, count=1),
    ]


def bounded_u8(value):
    return max(0, min(255, int(value)))


def time_from_timebase_ns(timebase_ns):
    """Convert the Livox packet epoch (nanoseconds) to a ROS2 Time message."""
    nanoseconds = int(timebase_ns)
    if nanoseconds < 0:
        raise ValueError(f"Livox CustomMsg timebase must be non-negative: {nanoseconds}")
    sec, nanosec = divmod(nanoseconds, 1_000_000_000)
    if sec > 0x7FFFFFFF:
        raise ValueError(
            "Livox CustomMsg timebase is outside the ROS2 builtin Time range: "
            f"{nanoseconds} ns"
        )
    return Time(sec=sec, nanosec=nanosec)


def validate_point_count(point_count):
    count = int(point_count)
    if count < 0:
        raise ValueError(f"Livox CustomMsg point_num must be non-negative: {count}")
    if count > MAX_POINT_COUNT:
        raise ValueError(
            "Livox CustomMsg is too large for PointCloud2 uint32 row_step: "
            f"point_num={count} maximum={MAX_POINT_COUNT}"
        )
    return count


def convert_livox_custom_msg(msg, output_frame=""):
    points = list(getattr(msg, "points", ()))
    declared_count = int(getattr(msg, "point_num", len(points)))
    if declared_count != len(points):
        raise ValueError(
            "Livox CustomMsg point_num does not match points length: "
            f"declared={declared_count} actual={len(points)}"
        )
    point_count = validate_point_count(declared_count)
    if not hasattr(msg, "timebase"):
        raise ValueError("Livox CustomMsg is missing the required timebase field")
    stamp = time_from_timebase_ns(msg.timebase)
    data = bytearray(point_count * POINT_STEP)

    for index in range(point_count):
        point = points[index]
        offset = index * POINT_STEP
        struct.pack_into(
            "<fffBBBBI",
            data,
            offset,
            float(point.x),
            float(point.y),
            float(point.z),
            bounded_u8(getattr(point, "reflectivity", 0)),
            bounded_u8(getattr(point, "tag", 0)),
            bounded_u8(getattr(point, "line", 0)),
            0,
            max(0, min(0xFFFFFFFF, int(getattr(point, "offset_time", 0)))),
        )

    cloud = PointCloud2()
    # ROS1 livox_ros_driver and CocoLIC both use CustomMsg.timebase as the scan
    # epoch.  header.stamp is not authoritative and may differ in converted bags.
    cloud.header.stamp = stamp
    cloud.header.frame_id = output_frame or msg.header.frame_id
    cloud.height = 1
    cloud.width = point_count
    cloud.fields = make_fields()
    cloud.is_bigendian = False
    cloud.point_step = POINT_STEP
    cloud.row_step = point_count * POINT_STEP
    cloud.data = bytes(data)
    cloud.is_dense = False
    return cloud


def load_livox_custom_msg_type():
    try:
        from livox_ros_driver2.msg import CustomMsg

        return CustomMsg
    except ImportError:
        try:
            from livox_ros_driver.msg import CustomMsg

            return CustomMsg
        except ImportError as exc:
            raise RuntimeError(
                "livox_custom_to_pointcloud2 requires livox_ros_driver2/msg/CustomMsg "
                "or livox_ros_driver/msg/CustomMsg at runtime. Install the Livox ROS2 "
                "driver or replay a bag that has already been converted to PointCloud2."
            ) from exc


def make_qos(rclpy, reliability, depth):
    qos = rclpy.qos.QoSProfile(depth=max(1, int(depth)))
    reliability_token = str(reliability).strip().lower()
    if reliability_token == "reliable":
        qos.reliability = rclpy.qos.QoSReliabilityPolicy.RELIABLE
    else:
        qos.reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
    qos.history = rclpy.qos.QoSHistoryPolicy.KEEP_LAST
    qos.durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
    return qos


class LivoxCustomBridge:
    def __init__(self, rclpy, node_cls):
        self.node = node_cls("livox_custom_to_pointcloud2")
        self.node.declare_parameter("input_topic", "/livox/lidar")
        self.node.declare_parameter("output_topic", "/livox/lidar/points")
        self.node.declare_parameter("output_frame", "")
        self.node.declare_parameter("sensor_qos_reliability", "best_effort")
        self.node.declare_parameter("sensor_qos_depth", 5)
        self.node.declare_parameter("report_period_sec", 2.0)

        self.output_frame = str(self.node.get_parameter("output_frame").value)
        input_topic = str(self.node.get_parameter("input_topic").value)
        output_topic = str(self.node.get_parameter("output_topic").value)
        qos = make_qos(
            rclpy,
            self.node.get_parameter("sensor_qos_reliability").value,
            self.node.get_parameter("sensor_qos_depth").value,
        )
        custom_msg_type = load_livox_custom_msg_type()
        self.publisher = self.node.create_publisher(PointCloud2, output_topic, qos)
        self.subscription = self.node.create_subscription(
            custom_msg_type,
            input_topic,
            self.on_msg,
            qos,
        )
        self.received = 0
        self.published = 0
        self.rejected = 0
        report_period = float(self.node.get_parameter("report_period_sec").value)
        self.timer = None
        if report_period > 0.0:
            self.timer = self.node.create_timer(report_period, self.report)
        self.node.get_logger().info(
            f"Bridging Livox CustomMsg {input_topic} -> PointCloud2 {output_topic}"
        )

    def on_msg(self, msg):
        self.received += 1
        try:
            cloud = convert_livox_custom_msg(msg, self.output_frame)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            self.rejected += 1
            self.node.get_logger().error(f"Rejected malformed Livox CustomMsg: {exc}")
            return
        self.publisher.publish(cloud)
        self.published += 1

    def report(self):
        self.node.get_logger().info(
            f"forwarded livox_custom={self.received} pointcloud2={self.published} "
            f"rejected={self.rejected}"
        )


def run_self_test():
    from std_msgs.msg import Header

    class Point:
        def __init__(self, x, y, z, reflectivity, tag, line, offset_time):
            self.x = x
            self.y = y
            self.z = z
            self.reflectivity = reflectivity
            self.tag = tag
            self.line = line
            self.offset_time = offset_time

    class Msg:
        pass

    msg = Msg()
    msg.header = Header(stamp=Time(sec=1, nanosec=2), frame_id="livox_frame")
    msg.timebase = 9_876_543_210
    msg.point_num = 2
    msg.points = [
        Point(1.0, 2.0, 3.0, 7, 8, 9, 123456),
        Point(-1.0, -2.0, -3.0, 300, -1, 4, 0x1FFFFFFFF),
    ]
    cloud = convert_livox_custom_msg(msg, "camera")
    assert cloud.header.frame_id == "camera"
    assert msg.header.frame_id == "livox_frame"
    assert cloud.header.stamp.sec == 9
    assert cloud.header.stamp.nanosec == 876_543_210
    assert cloud.header.stamp != msg.header.stamp
    assert cloud.width == 2
    assert cloud.height == 1
    assert cloud.point_step == POINT_STEP
    assert cloud.row_step == 2 * POINT_STEP
    assert len(cloud.data) == 2 * POINT_STEP
    offset_time_field = next(field for field in cloud.fields if field.name == "offset_time")
    assert offset_time_field.offset == 16
    assert offset_time_field.datatype == PointField.UINT32
    first = struct.unpack_from("<fffBBBBI", cloud.data, 0)
    second = struct.unpack_from("<fffBBBBI", cloud.data, POINT_STEP)
    assert first == (1.0, 2.0, 3.0, 7, 8, 9, 0, 123456)
    assert second == (-1.0, -2.0, -3.0, 255, 0, 4, 0, 0xFFFFFFFF)

    msg.point_num = 3
    try:
        convert_livox_custom_msg(msg)
    except ValueError as exc:
        assert "point_num does not match" in str(exc)
    else:
        raise AssertionError("mismatched Livox point_num was not rejected")
    try:
        validate_point_count(MAX_POINT_COUNT + 1)
    except ValueError as exc:
        assert "uint32 row_step" in str(exc)
    else:
        raise AssertionError("overflowing Livox point count was not rejected")
    print("livox custom conversion self-test OK")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bridge Livox CustomMsg packets to sensor_msgs/PointCloud2."
    )
    parser.add_argument("--self-test", action="store_true", help="Run conversion self-test and exit")
    # launch_ros appends ROS arguments such as --ros-args, --params-file and
    # remaps.  Only consume this script's private flag here and leave the ROS
    # arguments for rclpy; argparse.parse_args() would otherwise make every
    # normal launch of the bridge exit before rclpy.init().
    args, _ = parser.parse_known_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    import rclpy
    from rclpy.node import Node

    rclpy.init(args=argv)
    try:
        bridge = LivoxCustomBridge(rclpy, Node)
        rclpy.spin(bridge.node)
    finally:
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
