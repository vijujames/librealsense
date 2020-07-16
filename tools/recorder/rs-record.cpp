// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <stdio.h>
#include <memory>
#include <functional>
#include <thread>
#include <string.h>
#include <chrono>
#include <librealsense2/rs_advanced_mode.hpp>
#include "tclap/CmdLine.h"

using namespace TCLAP;

int main(int argc, char * argv[]) try
{
    // Parse command line arguments
    CmdLine cmd("librealsense rs-record example tool", ' ');
    ValueArg<int>    time("t", "Time", "Amount of time to record (in seconds)", false, 10, "");
    ValueArg<std::string> out_file("f", "FullFilePath", "the file where the data will be saved to", false, "test.bag", "");
    ValueArg<float>    visual_preset("p", "VisualPreset", "0-Custom/1-Default/2-Hand/3-HighAccuracy/4-HighDensity/5-MediumDensity/6-RemoveIrPattern", false, 4.0, "");

    cmd.add(time);
    cmd.add(out_file);
    cmd.add(visual_preset);
    cmd.parse(argc, argv);

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_record_to_file(out_file.getValue());

    std::mutex m;
    auto callback = [&](const rs2::frame& frame)
    {
        std::lock_guard<std::mutex> lock(m);
        auto t = std::chrono::system_clock::now();
        static auto tk = t;
        static auto t0 = t;
        if (t - tk >= std::chrono::seconds(1)) {
            std::cout << "\r" << std::setprecision(3) << std::fixed
                      << "Recording t = "  << std::chrono::duration_cast<std::chrono::seconds>(t-t0).count() << "s" << std::flush;
            tk = t;
        }
    };

    // Set the max Depth
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() > 0) {
        rs2::device dev = devices[0];
        rs400::advanced_mode advanced_mode_dev = dev.as<rs400::advanced_mode>();
        if (!advanced_mode_dev.is_enabled()) {
            advanced_mode_dev.toggle_advanced_mode(true);
            std::cout << "Turned on advanced mode!" << std::endl;
        }
        auto depth_table = advanced_mode_dev.get_depth_table();
        std::cout << "depthClampMin before : " << depth_table.depthClampMin << std::endl;
        std::cout << "depthClampMax before : " << depth_table.depthClampMax << std::endl;
        depth_table.depthClampMin = 100; // .1m000 if depth unit at 0.001
        depth_table.depthClampMax = 10000; // 10m000 if depth unit at 0.001
        advanced_mode_dev.set_depth_table(depth_table);
        std::cout << "depthClampMin changed to : " << depth_table.depthClampMin << std::endl;
        std::cout << "depthClampMax changed to : " << depth_table.depthClampMax << std::endl;
    }

    rs2::pipeline_profile profiles = pipe.start(cfg, callback);

    // Set the Visual Preset (High Accuracy, High Density etc.)
    rs2::sensor depthSensor = profiles.get_device().query_sensors()[0];
    if (depthSensor.supports(rs2_option::RS2_OPTION_VISUAL_PRESET)) {
        std::cout << "Visual Preset before : " << depthSensor.get_option(rs2_option::RS2_OPTION_VISUAL_PRESET) << std::endl;
        rs2_rs400_visual_preset rs400_visual_preset = static_cast<rs2_rs400_visual_preset>(visual_preset.getValue());
        depthSensor.set_option(rs2_option::RS2_OPTION_VISUAL_PRESET, rs400_visual_preset);
        std::cout << "Visual Preset changed to : " << depthSensor.get_option(rs2_option::RS2_OPTION_VISUAL_PRESET) << std::endl;
    }

    auto t = std::chrono::system_clock::now();
    auto t0 = t;
    while(t - t0 <= std::chrono::seconds(time.getValue())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        t = std::chrono::system_clock::now();
    }
    std::cout << "\nFinished" << std::endl;

    pipe.stop();

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
