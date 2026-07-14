#!/usr/bin/env python3
"""Static guard for the mapper executor/finalization ownership contract."""

from pathlib import Path
import sys


def require(text: str, token: str, description: str) -> None:
    if token not in text:
        raise AssertionError(f"missing {description}: {token}")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: mapping_concurrency_contract.py MAPPING_NODE_CPP")
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    require(text, "rclcpp::executors::MultiThreadedExecutor", "standalone multi-threaded executor")
    require(text, "rclcpp::ExecutorOptions{}, 3U", "three-thread standalone executor")
    require(text, "mapping_callback_group_", "dedicated mapping callback group")
    require(text, "status_callback_group_", "dedicated status callback group")
    require(
        text,
        "rclcpp::ServicesQoS(),\n      mapping_callback_group_",
        "SaveMap service mapping-group wiring",
    )
    require(text, "}, status_callback_group_);", "status timer group wiring")
    require(text, "mapping_state_mutex_", "mapper state ownership mutex")
    for parameter in (
        "max_queue_size",
        "process_period_ms",
        "select_every_k_frame",
        "test_frame_stride",
    ):
        require(
            text,
            f'"{parameter} must be positive"',
            f"fail-fast positive validation for {parameter}",
        )
    require(text, "std::scoped_lock mapping_lock(mapping_state_mutex_);", "state lock at callbacks")
    require(text, "last_input_receive_steady_ns_", "all-input inactivity watermark")
    require(text, "last_converted_frame_steady_ns_", "successful-frame inactivity watermark")
    if text.count("mark_input_received();") < 6:
        raise AssertionError("all six frame/feedback input callbacks must refresh inactivity")
    for counter, queue in (
        ("pointcloud_count_", "point_buf_"),
        ("pose_count_", "pose_buf_"),
        ("image_count_", "image_buf_"),
        ("depth_count_", "depth_buf_"),
    ):
        require(text, f"++{counter};\n          push_bounded({queue}", f"{counter} buffer ownership")
    require(
        text,
        "std::scoped_lock lock(buffer_mutex_);\n        ++imu_count_;\n        last_imu_stamp_",
        "IMU diagnostics buffer ownership",
    )
    require(text, "uint64_t camera_info_count{0};", "intrinsics count snapshot")
    require(text, "intrinsics.camera_info_count", "locked camera-info status read")
    require(text, "msg.header.frame_id = world_frame_;", "configured status frame")
    record_begin = text.index("void record_converted_frame")
    record_end = text.index("void publish_rendered_preview_for_record", record_begin)
    record_body = text[record_begin:record_end]
    if "current_intrinsics()" in record_body:
        raise AssertionError("record_converted_frame must reuse its conversion-time intrinsics")
    require(record_body, "maybe_complete_depth(frame_data, frame_data.intrinsics);", "SPNet frame K")
    require(record_body, "record.intrinsics.fx", "TorchCamera frame K")
    require(text, "frame, frame.intrinsics.fx, frame.intrinsics.fy", "per-frame optimization/render K")
    for queue in (
        "point_buf_.clear();",
        "pose_buf_.clear();",
        "image_buf_.clear();",
        "depth_buf_.clear();",
        "feedback_pose_buf_.clear();",
        "feedback_image_buf_.clear();",
    ):
        require(text, queue, "orphan drain before finalization")
    require(text, "write_finalize_manifest", "automatic finalization manifest")
    require(text, "handle_save_map(request, response);", "automatic SaveMap path")
    require(text, "std_srvs::srv::Trigger", "end-of-input Trigger service")
    require(text, "end_of_input_requested_", "end-of-input completion state")
    require(
        text,
        "end of input reached without a complete mapping frame",
        "zero-frame terminal failure",
    )
    require(text, "if (!response->success)", "save failure propagation")
    require(text, "map saved but finalize manifest failed", "manifest failure propagation")
    require(text, "const bool terminal_failure", "standalone terminal status ownership")
    require(text, "return 1;", "standalone non-zero terminal failure code")
    require(text, "if (auto_finalize_exit_)", "explicit automatic-exit ownership")
    if "auto_finalize_exit_ || end_of_input_requested_" in text:
        raise AssertionError("end-of-input must not override explicit auto_finalize_exit=false")
    print("mapping_concurrency_contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
