# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import unittest

from gaussian_lic_bringup.profile_parameters import (
    MAPPING_OVERRIDE_FALLBACKS,
    PROFILE_INHERIT_SENTINEL,
    bag_play_command,
    load_mapping_profile,
    resolve_depth_completion_engine_path,
    resolve_lpips_model_path,
    resolve_mapping_overrides,
    resolve_pointcloud_coordinates,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


class BringupContractTest(unittest.TestCase):
    def test_generic_profile_is_safe_and_sensor_profiles_use_real_rasterizer(self):
        self.assertIn(
            "render_mode: debug_cpu",
            (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"),
        )
        for name in ("fastlivo", "fastlivo2", "m2dgr", "mcd", "r3live"):
            text = (PACKAGE_ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8")
            self.assertIn("render_mode: rasterizer", text, name)
            self.assertIn("save_map_render_evaluation: true", text, name)
            self.assertIn('lpips_model_path: ""', text, name)
            self.assertIn("test_frame_stride: 1", text, name)


    def test_run_bag_clock_and_livox_defaults_are_safe(self):
        text = (PACKAGE_ROOT / "launch" / "run_bag.launch.py").read_text(encoding="utf-8")
        self.assertIn("default_value=play_bag", text)
        self.assertIn('"stub_mode",\n            default_value="false"', text)
        self.assertIn('"raw_pointcloud_topic": effective_adapter_raw_pointcloud_topic', text)
        self.assertIn("OpaqueFunction(function=_resolve_profile_and_coordinates)", text)
        self.assertIn("OpaqueFunction(function=_validate_runtime_backend)", text)
        self.assertIn('executable="component_container_mt"', text)
        self.assertIn("TimerAction(\n            period=2.0", text)
        self.assertIn('"auto_finalize_on_inactivity": auto_finalize_on_inactivity', text)
        self.assertIn('"auto_finalize_output_path": auto_finalize_output_path', text)
        self.assertIn(
            '"enable_non_upstream_density_control": enable_non_upstream_density_control',
            text,
        )
        self.assertIn('"lpips_model_path": lpips_model_path', text)
        self.assertIn('"depth_completion": depth_completion', text)
        self.assertIn('"depth_completion_engine_path": depth_completion_engine_path', text)
        self.assertIn('capabilities.get("tensorrt", False)', text)
        self.assertIn("LPIPS model is not a regular file", text)
        self.assertIn("OnProcessExit(", text)
        self.assertIn("target_action=bag_play_once_action", text)
        self.assertIn('"std_srvs/srv/Trigger"', text)
        self.assertIn('"end_of_input_service": end_of_input_service', text)
        self.assertIn('"mapping node"', text)
        self.assertIn('"mapping container"', text)
        self.assertIn('"--foreground"', text)
        self.assertIn('"15s"', text)
        self.assertIn("_handle_bag_exit", text)
        self.assertIn("_handle_notifier_exit", text)

    def test_one_shot_bag_finalizes_and_exits_but_live_and_looping_inputs_do_not(self):
        inherited = {
            name: PROFILE_INHERIT_SENTINEL for name in MAPPING_OVERRIDE_FALLBACKS
        }

        one_shot = resolve_mapping_overrides(inherited, {}, "true", "false")
        self.assertEqual(one_shot["auto_finalize_on_inactivity"], "false")
        self.assertEqual(one_shot["auto_finalize_exit"], "true")

        looping = resolve_mapping_overrides(inherited, {}, "true", "true")
        self.assertEqual(looping["auto_finalize_on_inactivity"], "false")
        self.assertEqual(looping["auto_finalize_exit"], "false")

        live = resolve_mapping_overrides(inherited, {}, "false", "false")
        self.assertEqual(live["auto_finalize_on_inactivity"], "false")
        self.assertEqual(live["auto_finalize_exit"], "false")

        explicit = dict(inherited)
        explicit["auto_finalize_on_inactivity"] = "false"
        explicit["auto_finalize_exit"] = "false"
        overridden = resolve_mapping_overrides(explicit, {}, "true", "false")
        self.assertEqual(overridden["auto_finalize_on_inactivity"], "false")
        self.assertEqual(overridden["auto_finalize_exit"], "false")

    def test_real_gl2_profiles_are_inherited_and_cli_overrides_win(self):
        inherited = {
            name: PROFILE_INHERIT_SENTINEL for name in MAPPING_OVERRIDE_FALLBACKS
        }
        cases = {
            "cbd_mapper_moreopt.yaml": 200,
            "cbd_mapper_opt300.yaml": 300,
        }
        for filename, expected_steps in cases.items():
            parameters = load_mapping_profile(
                REPOSITORY_ROOT / "run_lio" / "config" / filename
            )
            resolved = resolve_mapping_overrides(inherited, parameters, "false")
            self.assertEqual(resolved["require_depth_topic"], "false", filename)
            self.assertEqual(
                resolved["torch_gaussian_optimization_steps"], str(expected_steps), filename
            )
            self.assertEqual(resolved["save_map_render_evaluation"], "true", filename)

        explicit = dict(inherited)
        explicit["require_depth_topic"] = "true"
        explicit["torch_gaussian_optimization_steps"] = "17"
        resolved = resolve_mapping_overrides(
            explicit,
            load_mapping_profile(
                REPOSITORY_ROOT / "run_lio" / "config" / "cbd_mapper_moreopt.yaml"
            ),
            "false",
        )
        self.assertEqual(resolved["require_depth_topic"], "true")
        self.assertEqual(resolved["torch_gaussian_optimization_steps"], "17")

    def test_sensor_qos_cli_and_profile_precedence(self):
        inherited = {
            name: PROFILE_INHERIT_SENTINEL for name in MAPPING_OVERRIDE_FALLBACKS
        }
        profile = {
            "sensor_qos_reliability": "best_effort",
            "sensor_qos_history": "keep_last",
            "sensor_qos_depth": 5,
            "pointcloud_qos_reliability": "best_effort",
            "pointcloud_qos_history": "keep_last",
            "pointcloud_qos_depth": 5,
            "image_qos_reliability": "best_effort",
            "image_qos_history": "keep_last",
            "image_qos_depth": 5,
            "imu_qos_reliability": "best_effort",
            "imu_qos_history": "keep_last",
            "imu_qos_depth": 5,
        }

        global_cli = dict(inherited)
        global_cli["sensor_qos_reliability"] = "reliable"
        global_cli["sensor_qos_history"] = "keep_all"
        global_cli["sensor_qos_depth"] = "17"
        resolved = resolve_mapping_overrides(global_cli, profile, "false")
        for stream in ("pointcloud", "image", "imu"):
            self.assertEqual(resolved[f"{stream}_qos_reliability"], "reliable")
            self.assertEqual(resolved[f"{stream}_qos_history"], "keep_all")
            self.assertEqual(resolved[f"{stream}_qos_depth"], "17")

        stream_cli = dict(global_cli)
        stream_cli["image_qos_reliability"] = "best_effort"
        stream_cli["image_qos_history"] = "keep_last"
        stream_cli["image_qos_depth"] = "3"
        resolved = resolve_mapping_overrides(stream_cli, profile, "false")
        self.assertEqual(resolved["image_qos_reliability"], "best_effort")
        self.assertEqual(resolved["image_qos_history"], "keep_last")
        self.assertEqual(resolved["image_qos_depth"], "3")
        self.assertEqual(resolved["pointcloud_qos_reliability"], "reliable")
        self.assertEqual(resolved["imu_qos_reliability"], "reliable")

        profile_only = {
            "sensor_qos_reliability": "reliable",
            "sensor_qos_history": "keep_all",
            "sensor_qos_depth": 17,
            "pointcloud_qos_reliability": "best_effort",
            "pointcloud_qos_history": "keep_last",
            "pointcloud_qos_depth": 3,
        }
        resolved = resolve_mapping_overrides(inherited, profile_only, "false")
        self.assertEqual(resolved["pointcloud_qos_reliability"], "best_effort")
        self.assertEqual(resolved["pointcloud_qos_history"], "keep_last")
        self.assertEqual(resolved["pointcloud_qos_depth"], "3")
        self.assertEqual(resolved["image_qos_reliability"], "reliable")
        self.assertEqual(resolved["image_qos_history"], "keep_all")
        self.assertEqual(resolved["image_qos_depth"], "17")

        resolved = resolve_mapping_overrides(inherited, {}, "false")
        self.assertEqual(resolved["image_qos_reliability"], "best_effort")
        self.assertEqual(resolved["image_qos_history"], "keep_last")
        self.assertEqual(resolved["image_qos_depth"], "5")

    def test_pointcloud_coordinate_auto_resolution(self):
        self.assertEqual(resolve_pointcloud_coordinates("auto", "false", "false", "true"), "world")
        self.assertEqual(resolve_pointcloud_coordinates("auto", "true", "false", "true"), "sensor")
        self.assertEqual(resolve_pointcloud_coordinates("auto", "true", "true", "true"), "world")
        self.assertEqual(resolve_pointcloud_coordinates("sensor", "false", "true", "true"), "sensor")
        with self.assertRaisesRegex(RuntimeError, "auto, world, or sensor"):
            resolve_pointcloud_coordinates("guess", "false", "false", "false")

    def test_depth_completion_engine_resolution_precedence(self):
        fake_home = Path("/tmp/gaussian_lic_missing_engine_home")
        self.assertEqual(
            resolve_depth_completion_engine_path(
                "/explicit/spnet.engine",
                640,
                512,
                environ={"GAUSSIAN_LIC_SPNET_ENGINE": "/env/spnet.engine"},
                home=fake_home,
            ),
            "/explicit/spnet.engine",
        )
        self.assertEqual(
            resolve_depth_completion_engine_path(
                "",
                640,
                512,
                environ={"GAUSSIAN_LIC_SPNET_ENGINE": "/env/spnet.engine"},
                home=fake_home,
            ),
            "/env/spnet.engine",
        )
        self.assertEqual(
            resolve_depth_completion_engine_path(
                "", 640, 512, environ={}, home=fake_home
            ),
            "",
        )

    def test_lpips_model_resolution_precedence(self):
        self.assertEqual(
            resolve_lpips_model_path(
                "/explicit/lpips.pt",
                environ={"GAUSSIAN_LIC_LPIPS_MODEL": "/env/lpips.pt"},
            ),
            "/explicit/lpips.pt",
        )
        self.assertEqual(
            resolve_lpips_model_path(
                "", environ={"GAUSSIAN_LIC_LPIPS_MODEL": "/env/lpips.pt"}
            ),
            "/env/lpips.pt",
        )

    def test_bag_play_command_is_delayed_by_launch_and_loop_is_explicit(self):
        bag = object()
        self.assertEqual(
            bag_play_command(bag),
            ["ros2", "bag", "play", bag, "--clock", "--read-ahead-queue-size", "100"],
        )
        self.assertEqual(bag_play_command(bag, loop=True)[-1], "--loop")

    def test_gl2_scripts_capture_exit_codes_and_guard_recursive_removal(self):
        for filename in ("gl2_reproduce.sh", "run_gl2_step1_psnr.sh"):
            text = (REPOSITORY_ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn("safe_remove", text, filename)
            self.assertIn("if wait \"${pid}\"", text, filename)
            self.assertIn("MAPPER_LAST_EXIT_CODE=\"${exit_code}\"", text, filename)
            self.assertNotIn('rm -rf "${OUTDIR}"', text, filename)
        reproduce = (REPOSITORY_ROOT / "scripts" / "gl2_reproduce.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('if ! kill -0 "${track_pid}"', reproduce)
        self.assertIn('track A exited with code ${track_exit_code}', reproduce)

    def test_native_tracking_report_uses_selected_install_and_omits_empty_offline_args(self):
        text = (
            REPOSITORY_ROOT / "scripts" / "run_native_tracking_bag_report.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'INSTALL_SETUP="${GAUSSIAN_LIC_INSTALL_SETUP:-${ROOT_DIR}/install/setup.bash}"',
            text,
        )
        self.assertIn('if [[ ! -r "${INSTALL_SETUP}" ]]', text)
        self.assertIn('source "${INSTALL_SETUP}"', text)
        self.assertIn("deterministic_launch_args=()", text)
        self.assertIn("mapper_depth_completion_args=()", text)
        self.assertIn('if [[ -n "${DETERMINISTIC_FEEDBACK_BAG:-}" ]]', text)
        self.assertIn('if wait "${launch_pid}" 2>/dev/null; then', text)
        self.assertIn("deterministic_launch_exit=$?", text)
        self.assertIn(
            "deterministic replay FAIL: tracking launch exited with code", text
        )
        self.assertNotIn('wait "${launch_pid}" 2>/dev/null || true', text)
        self.assertIn(
            'if [[ "${MAPPER_FEEDBACK_DEPTH_COMPLETION}" == "true" ]]', text
        )
        self.assertIn('"${deterministic_launch_args[@]}"', text)
        self.assertIn('"${mapper_depth_completion_args[@]}"', text)
        self.assertIn(
            '--mapper-feedback-enable-depth-completion requires '
            '--mapper-feedback-depth-completion-engine-path',
            text,
        )
        self.assertNotIn(
            'deterministic_bag_path:="${DETERMINISTIC_FEEDBACK_BAG:+${BAG_PATH}}"',
            text,
        )

    def test_required_depth_completion_never_silently_falls_back(self):
        text = (
            REPOSITORY_ROOT
            / "src"
            / "gaussian_lic_mapping"
            / "src"
            / "mapping_node.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("DepthCompletionFatalError", text)
        self.assertIn("required TensorRT depth completion failed; stopping", text)
        self.assertIn("mapping node terminated after a required pipeline failure", text)
        self.assertNotIn("keeping sparse/provided depth", text)
        constructor_start = text.index("if (backend_config_.depth_completion) {")
        constructor_end = text.index("#ifdef GAUSSIAN_LIC_ENABLE_TORCH", constructor_start)
        constructor_guard = text[constructor_start:constructor_end]
        self.assertIn("std::filesystem::is_regular_file", constructor_guard)
        self.assertNotIn("std::make_unique<gaussian_lic_mapping::DepthCompleter>", constructor_guard)
        self.assertEqual(
            text.count("std::make_unique<gaussian_lic_mapping::DepthCompleter>"), 1
        )
        self.assertIn('output_dir / "render"', text)
        self.assertIn("create_directory_symlink", text)
        self.assertIn("final Gaussian save/evaluation was requested", text)
        self.assertNotIn(
            '-p depth_completion_engine_path:="${MAPPER_FEEDBACK_DEPTH_COMPLETION_ENGINE_PATH}"',
            text,
        )

    def test_spnet_export_keeps_declared_torch_onnx_dependency_contract(self):
        text = (REPOSITORY_ROOT / "scripts" / "build_spnet_engine.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('for name in ("torch", "onnx")', text)
        self.assertIn("dynamo=False", text)
        self.assertIn('getattr(nvidia.cublas, "__file__", None)', text)
        self.assertIn('getattr(module, "__file__", None)', text)
        self.assertIn("if base.is_dir()", text)


if __name__ == "__main__":
    unittest.main()
