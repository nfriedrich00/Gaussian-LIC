# SPDX-License-Identifier: GPL-3.0-or-later

from gaussian_lic_tools.bag_contract import check_alternative_group, check_topic_contract
import unittest


class BagContractTest(unittest.TestCase):
    def test_required_topic_contract_checks_type_and_count(self):
        topics = {
            "/points": {
                "type": "sensor_msgs/msg/PointCloud2",
                "serialization_format": "cdr",
                "message_count": 2,
            },
            "/image": {
                "type": "sensor_msgs/msg/CompressedImage",
                "serialization_format": "cdr",
                "message_count": 0,
            },
        }
        contract = {
            "/points": "sensor_msgs/msg/PointCloud2",
            "/image": "sensor_msgs/msg/Image",
            "/imu": "sensor_msgs/msg/Imu",
        }

        checks, errors = check_topic_contract(topics, contract, required=True)

        self.assertTrue(checks["/points"]["ok"])
        self.assertFalse(checks["/image"]["ok"])
        self.assertFalse(checks["/imu"]["ok"])
        self.assertTrue(
            any("type is sensor_msgs/msg/CompressedImage" in error for error in errors)
        )
        self.assertTrue(any("message_count is zero" in error for error in errors))
        self.assertIn("/imu: missing topic", errors)

    def test_alternative_group_accepts_one_valid_source(self):
        topics = {
            "/odom": {
                "type": "nav_msgs/msg/Odometry",
                "serialization_format": "cdr",
                "message_count": 10,
            }
        }
        group, errors = check_alternative_group(
            topics,
            {
                "/pose": "geometry_msgs/msg/PoseStamped",
                "/odom": "nav_msgs/msg/Odometry",
            },
            "pose_source",
        )

        self.assertTrue(group["ok"])
        self.assertFalse(errors)


if __name__ == "__main__":
    unittest.main()
