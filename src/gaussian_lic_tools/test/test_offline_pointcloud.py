# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace
import struct
import unittest

from gaussian_lic_tools.offline import iter_xyzrgb_points


def field(name, offset, datatype):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


class OfflinePointCloudTest(unittest.TestCase):
    def test_iter_xyzrgb_points_honors_row_stride_and_packed_rgb(self):
        point_step = 16
        row_step = 20
        data = bytearray(row_step * 2)
        struct.pack_into("<fffI", data, 0, 1.0, 2.0, 3.0, 0x00112233)
        struct.pack_into("<fffI", data, row_step, 4.0, 5.0, 6.0, 0x00A0B0C0)
        cloud = SimpleNamespace(
            fields=[
                field("x", 0, 7),
                field("y", 4, 7),
                field("z", 8, 7),
                field("rgb", 12, 6),
            ],
            data=bytes(data),
            width=1,
            height=2,
            point_step=point_step,
            row_step=row_step,
            is_bigendian=False,
        )

        self.assertEqual(
            list(iter_xyzrgb_points(cloud, max_points=10)),
            [
                (1.0, 2.0, 3.0, 0x11, 0x22, 0x33, True),
                (4.0, 5.0, 6.0, 0xA0, 0xB0, 0xC0, True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
