// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#include <unistd.h>
#include <math.h>

#include <librealsense2/rs.hpp>   // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h> // rs2_deproject_pixel_to_point

#include "cv-helpers.hpp"         // frame_to_mat
#include <opencv2/core/utils/filesystem.hpp>   // glob
namespace fs = cv::utils::fs;

#include <rs-vino/object-detection.h>
#include <rs-vino/detected-object.h>

#include <easylogging++.h>
#ifdef BUILD_SHARED_LIBS
// With static linkage, ELPP is initialized by librealsense, so doing it here will
// create errors. When we're using the shared .so/.dll, the two are separate and we have
// to initialize ours if we want to use the APIs!
INITIALIZE_EASYLOGGINGPP
#endif

#include <rs-vino/openvino-helpers.h>
namespace openvino = InferenceEngine;

#include <chrono>
using namespace std::chrono;


/*
    Enable loading multiple detectors at once, so we can switch at runtime.
    Each detector has its associated labels.
*/
struct detector_and_labels
{
    std::shared_ptr< openvino_helpers::object_detection > detector;
    std::vector< std::string > labels;

    detector_and_labels( std::string const & path_to_xml )
        : detector( std::make_shared< openvino_helpers::object_detection >( path_to_xml, 0.5 ) )
    {
    }

    openvino_helpers::object_detection * operator->() { return detector.get(); }

    void load_labels()
    {
        try
        {
            labels = openvino_helpers::read_labels( openvino_helpers::remove_ext( detector->pathToModel ) + ".labels" );
        }
        catch( const std::exception & e )
        {
            // If we have no labels, warn and continue... we can continue without them
            LOG(WARNING) << "Failed to load labels: " << e.what();
        }
    }
};


/*
    Populate a collection of detectors from those we find on disk (*.xml), load
    their labels, add them to the engine & device, etc.

    The detectors are loaded with all default values.
*/
void load_detectors_into(
    std::vector< detector_and_labels > & detectors,
    openvino::Core & engine,
    std::string const & device_name
)
{
    std::vector< std::string > xmls;
    fs::glob_relative( ".", "*.xml", xmls );
    for( auto path_to_xml : xmls )
    {
        detector_and_labels detector { path_to_xml };
        try
        {
            detector->load_into( engine, device_name );  // May throw!
            detector.load_labels();
            detectors.push_back( detector );
            LOG(INFO) << "   ... press '" << char( '0' + detectors.size() ) << "' to switch to it";
        }
        catch( const std::exception & e )
        {
            // The model files should have been downloaded automatically by CMake into build/wrappers/openvino/dnn,
            // which is also where Visual Studio runs the sample from. However, you may need to copy these files:
            //     *.bin
            //     *.xml
            //     *.labels  [optional]
            // Into the local directory where you run from (or change the path given in the ctor above)
            LOG(ERROR) << "Failed to load model: " << e.what();
        }
    }
}


/*
    Main detection code:

    Detected objects are placed into 'objects'. Each new object is assigned 'next_id', which is then incremented.
    The 'labels' are optional, and used to give labels to each object.
    
    Some basic effort is made to keep the creation of new objects to a minimum: previous objects (passed in via
    'objects') are compared with new detections to see if the new are simply new positions for the old. An
    "intersection over union" (IoU) quotient is calculated and, if over a threshold, an existing object is moved
    rather than a new one created.
*/
void detect_objects(
    cv::Mat const & image,
    std::vector< openvino_helpers::object_detection::Result > const & results,
    std::vector< std::string > & labels,
    size_t & next_id,
    openvino_helpers::detected_objects & objects
)
{
    openvino_helpers::detected_objects prev_objects{ std::move( objects ) };
    objects.clear();
    for( auto const & result : results )
    {
        if( result.label <= 0 )
            continue;  // ignore "background", though not clear why we'd get it
        cv::Rect rect = result.location;
        rect = rect & cv::Rect( 0, 0, image.cols, image.rows );
        auto object_ptr = openvino_helpers::find_object( rect, prev_objects );
        if( ! object_ptr )
        {
            // New object
            std::string label;
            if( result.label < labels.size() )
                label = labels[result.label];
            object_ptr = std::make_shared< openvino_helpers::detected_object >( next_id++, label, rect );
        }
        else
        {
            // Existing face; just update its parameters
            object_ptr->move( rect );
        }
        objects.push_back( object_ptr );
    }
}


/*
    Draws the detected objects with a distance calculated at the center pixel of each face
*/
void draw_objects(
    cv::Mat & image,
    rs2::depth_frame depth_frame,
    openvino_helpers::detected_objects const & objects
)
{
    cv::Scalar const green( 0, 255, 0 );  // BGR
    cv::Scalar const white( 255, 255, 255 );  // BGR
    
    for( auto && object : objects )
    {
        auto r = object->get_location();
        cv::rectangle( image, r, green );

        // Output the distance to the center
        auto center_x = r.x + r.width / 2;
        auto center_y = r.y + r.height / 2;
        auto d = depth_frame.get_distance( center_x, center_y );
        if( d )
        {
            std::ostringstream ss;
            ss << object->get_label() << " ";
            ss << std::setprecision( 2 ) << d;
            ss << " meters away";
            cv::putText( image, ss.str(), cv::Point( r.x + 5, r.y + r.height - 5 ), cv::FONT_HERSHEY_SIMPLEX, 0.4, white );
        }
    }
}

double get_actual_height(double angled_height, double angled_distance) {
    double angle_rad = 15 * 0.0174533; // 20 * 0.0174533

    double yb = angled_height / cos(angle_rad);
    double z2 = sin(angle_rad) * yb;
    double z1 = angled_distance - z2;
    double yt = sin(angle_rad) * z1;

    return (yb + yt);
}

double get_actual_distance(double angled_height, double angled_distance, double actual_height) {
    double angle_rad = 15 * 0.0174533; // 20 * 0.0174533

    double actual_angle_rad = atan(angled_height / angled_distance) + angle_rad;
    double actual_distance = actual_height / tan (actual_angle_rad);


    return actual_distance;
}

/**
 * @brief Calculate the actual vertical height of the given point from the camera and the horizontal
 * distance of the point from the camera
 * @param image_x x coordinate of the 2D point in the depth frame
 * @param image_y y cordinate of the 2D point in the depth frame
 * @param depth_frame The depth frame
 * @param DepthIntrinsics
 * @param actual_height address of the double value to return
 * @param actual_distance address of the double value to return
 */
void get_actual_height_and_distance(float image_x, float image_y, rs2::depth_frame depth_frame,
                                    rs2_intrinsics DepthIntrinsics, double *actual_height, double *actual_distance) {
     float ResultVector[3];
     float InputPixelAsFloat[2] = {image_x, image_y};

     auto angled_distance = depth_frame.get_distance( image_x, image_y );
     if (angled_distance) {
         rs2_deproject_pixel_to_point(ResultVector, &DepthIntrinsics, InputPixelAsFloat, angled_distance);

         *actual_height = get_actual_height(ResultVector[1], angled_distance);
         *actual_distance = get_actual_distance(ResultVector[1], angled_distance, *actual_height);
     }
}

cv::Point get_curb_point(int height, double image_x, rs2::depth_frame depth_frame, rs2_intrinsics DepthIntrinsics, cv::Mat& image) {
    const int IMAGE_X_STEP_SIZE = 15;
    const int CAMERA_BOTTOM_OFFSET = 60;
    const double GROUD_HEIGHT = 2.0; // 1.7
    const double DISTANCE_MAX = 6.0;
    const double DISTANCE_MIN = 4.2;
    const double CURB_HEIGHT_MAX = 0.172;
    const double CURB_HEIGHT_MIN = 0.132;
    cv::Scalar const white( 255, 255, 255 );  // BGR

    double prev_height = 0.0;
    double actual_height = 0.0;
    double actual_distance = 0.0;
    for (int image_y = height-CAMERA_BOTTOM_OFFSET; image_y > 0; image_y-=IMAGE_X_STEP_SIZE) {
        get_actual_height_and_distance(image_x, image_y, depth_frame, DepthIntrinsics, &actual_height, &actual_distance);
        if (actual_distance > DISTANCE_MAX || actual_distance < DISTANCE_MIN) continue;
/*
        if (image_x > 310 && image_x < 330) {
            std::ostringstream ss;
            ss << " : " << std::setprecision( 2 ) << actual_height;
            ss << " : " << std::setprecision( 2 ) << actual_distance;
            cv::putText( image, ss.str(), cv::Point( image_x, image_y ), cv::FONT_HERSHEY_SIMPLEX, 0.4, white );
        }
*/

        if (prev_height != 0.0 && actual_height > 0
                && (prev_height-actual_height < (CURB_HEIGHT_MAX))
                && (prev_height-actual_height > (CURB_HEIGHT_MIN))
                && actual_height < (GROUD_HEIGHT-CURB_HEIGHT_MIN)
                ) {
/*
            // Check next point as well so as to remove erroneous points
            if (image_y-13 > 0) {
                double next_actual_height, next_actual_distance;
                get_actual_height_and_distance(image_x, image_y-13, depth_frame, DepthIntrinsics, &next_actual_height, &next_actual_distance);

                if (abs(next_actual_height - actual_height) > 0.08) break;
            }
*/
            cv::Point pt( image_x, image_y );
            return pt;
        }

        prev_height = actual_height;
    }

    cv::Point pt_origin( 0, 0);
    return pt_origin;

}

void draw_fitted_line(cv::Mat& image, std::vector<cv::Point> curb_points, int width, int height) {
    cv::Point pt1, pt2;
    cv::Vec4f line;
    float d, t;

    if (curb_points.size() >= 2) {
        // find the optimal line
        cv::fitLine(curb_points, line, cv::DIST_L1, 1, 0.001, 0.001);

        pt1.x = curb_points[0].x;
        pt1.y = line[3];
        pt2.x = curb_points[curb_points.size()-1].x;
        pt2.y = line[3];
        cv::line(image, pt1, pt2, cv::Scalar(0, 255, 0), 3);
    }
}

/*
    Draws the edge of the curb
*/
void draw_curb(
    cv::Mat & image,
    rs2::depth_frame depth_frame,
    rs2::points points,
    rs2_intrinsics DepthIntrinsics
)
{
    cv::Scalar const green( 0, 255, 0 );  // BGR
    cv::Scalar const red( 0, 0, 255 );  // BGR
    cv::Scalar const white( 255, 255, 255 );  // BGR

    std::vector<cv::Point> curb_points;

    //auto depth_image = frame_to_mat( depth_frame );
    int width = image.cols;
    int height = image.rows;
    LOG(INFO) << "width=" << width << " height=" << height;

    for (auto image_x = 0; image_x < width; image_x+=10) {
        cv::Point pt = get_curb_point(height, image_x, depth_frame, DepthIntrinsics, image);
        if (pt.x == 0 &&pt.y == 0) continue;
        curb_points.push_back(pt);
        cv::circle(image, pt, 3, red, 2);
    }
    draw_fitted_line(image, curb_points, width, height);
}

/*
    When the user switches betweem models we show the detector number for 1 second as an
    overlay over the image, centered.
*/
void draw_detector_overlay(
    cv::Mat & image,
    size_t current_detector,
    high_resolution_clock::time_point switch_time
)
{
    auto ms_since_switch = duration_cast< milliseconds >( high_resolution_clock::now() - switch_time ).count();
    if( ms_since_switch > 1000 )
        ms_since_switch = 1000;
    double alpha = ( 1000 - ms_since_switch ) / 1000.;
    std::string str( 1, char( '1' + current_detector ) );
    auto size = cv::getTextSize( str, cv::FONT_HERSHEY_SIMPLEX, 3, 1, nullptr );
    cv::Point center{ image.cols / 2, image.rows / 2 };
    cv::Rect r{ center.x - size.width, center.y - size.height, size.width * 2, size.height * 2 };
    cv::Mat roi = image( r );
    cv::Mat overlay( roi.size(), CV_8UC3, cv::Scalar( 32, 32, 32 ) );
    cv::putText( overlay, str, cv::Point{ r.width / 2 - size.width / 2, r.height / 2 + size.height / 2 }, cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar{ 255, 255, 255 } );
    cv::addWeighted( overlay, alpha, roi, 1 - alpha, 0, roi );   // roi = overlay * alpha + roi * (1-alpha) + 0
}


int main(int argc, char * argv[]) try
{
    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    // Decimation filter reduces the amount of data (while preserving best samples)
    rs2::decimation_filter dec;
    // If the demo is too slow, make sure you run in Release (-DCMAKE_BUILD_TYPE=Release)
    // but you can also increase the following parameter to decimate depth more (reducing quality)
    dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    // Define transformations from and to Disparity domain
    rs2::disparity_transform depth2disparity;
    rs2::disparity_transform disparity2depth(false);
    // Define spatial filter (edge-preserving)
    rs2::spatial_filter spat;
    // Enable hole-filling
    // Hole filling is an aggressive heuristic and it gets the depth wrong many times
    // However, this demo is not built to handle holes
    // (the shortest-path will always prefer to "cut" through the holes since they have zero 3D distance)
    spat.set_option(RS2_OPTION_HOLES_FILL, 5); // 5 = fill all the zero pixels
    // Define temporal filter
    rs2::temporal_filter temp;
    // Spatially align all streams to depth viewport
    // We do this because:
    //   a. Usually depth has wider FOV, and we only really need depth for this demo
    //   b. We don't want to introduce new holes
//    rs2::align align_to(RS2_STREAM_DEPTH);


    el::Configurations conf;
    conf.set( el::Level::Global, el::ConfigurationType::Format, "[%level] %msg" );
    //conf.set( el::Level::Debug, el::ConfigurationType::Enabled, "false" );
    el::Loggers::reconfigureLogger( "default", conf );
    rs2::log_to_console( RS2_LOG_SEVERITY_WARN );    // only warnings (and above) should come through

    // Declare RealSense pipeline, encapsulating the actual device and sensors

    // Use streaming
    // rs2::pipeline pipe;
    // pipe.start();

    // Stream from recording // Viju
LOG(INFO) << "Before file load";
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::device device;
    rs2::pipeline_profile pipeline_profile;
    
    // enable file playback with playback repeat disabled    
    // realsense_20200427_171437.bag
    cfg.enable_device_from_file("/home/osboxes/rosbags/recordings/20200501_173833.bag", false);
    
    // start pipeline and get device
    pipeline_profile = pipe.start(cfg);
    device = pipeline_profile.get_device();

    // get & store depth intrinsics
    auto DepthStream = pipeline_profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2_intrinsics DepthIntrinsics = DepthStream.get_intrinsics();

    // get playback device and disable realtime mode
    auto playback = device.as<rs2::playback>();
    playback.set_real_time( false );
LOG(INFO) << "After file load";

    rs2::align align_to( RS2_STREAM_COLOR );

    // Start the inference engine, needed to accomplish anything. We also add a CPU extension, allowing
    // us to run the inference on the CPU. A GPU solution may be possible but, at least without a GPU,
    // a CPU-bound process is faster. To change to GPU, use "GPU" instead (and remove AddExtension()):
    openvino::Core engine;
    openvino_helpers::error_listener error_listener;
    engine.SetLogCallback( error_listener );
    std::string const device_name { "CPU" };
    engine.AddExtension( std::make_shared< openvino::Extensions::Cpu::CpuExtensions >(), device_name );

    std::vector< detector_and_labels > detectors;
    load_detectors_into( detectors, engine, device_name );
    if( detectors.empty() )
    {
        LOG(ERROR) << "No detectors available in: " << fs::getcwd();
        return EXIT_FAILURE;
    }
    // Look for the mobilenet-ssd so it always starts the same... otherwise default to the first detector we found
    size_t current_detector = 0;
    for( size_t i = 1; i < detectors.size(); ++i )
    {
        if( detectors[i]->pathToModel == "mobilenet-ssd.xml" )
        {
            current_detector = i;
            break;
        }
    }
    auto p_detector = detectors[current_detector].detector;
    LOG(INFO) << "Current detector set to (" << current_detector+1 << ") \"" << openvino_helpers::remove_ext( p_detector->pathToModel ) << "\"";
    auto p_labels = &detectors[current_detector].labels;

    const auto window_name = "OpenVINO DNN Color Image";
    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );

    cv::Mat prev_image;
    openvino_helpers::detected_objects objects;
    size_t id = 0;
    uint64 last_frame_number = 0;
    high_resolution_clock::time_point switch_time = high_resolution_clock::now();

	rs2::colorizer color_map;  //Viju
    while( cv::getWindowProperty( window_name, cv::WND_PROP_AUTOSIZE ) >= 0 )
    {
        // Wait for the next set of frames
        //auto frames = pipe.wait_for_frames();

	rs2::frameset frames;
	if (!pipe.poll_for_frames(&frames)) {
		usleep(10000);
		continue;
	}

        // Make sure the frames are spatially aligned (ORIGINAL)
 //       frames = align_to.process( frames );
        // First make the frames spatially aligned
        frames = frames.apply_filter(align_to);
/*
        // Decimation will reduce the resultion of the depth image,
        // closing small holes and speeding-up the algorithm
//        frames = dec.process(frames);
        // To make sure far-away objects are filtered proportionally
        // we try to switch to disparity domain
        frames = depth2disparity.process(frames);
        // Apply spatial filtering
        frames = spat.process(frames);
        // Apply temporal filtering
        frames = temp.process(frames);
        // If we are in disparity domain, switch back to depth
        frames = disparity2depth.process(frames);
*/


        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();

        if( ! color_frame  ||  ! depth_frame )
            continue;





        // If we only received a new depth frame, but the color did not update, continue
        if( color_frame.get_frame_number() == last_frame_number )
            continue;
        last_frame_number = color_frame.get_frame_number();

        auto image = frame_to_mat( color_frame );


        // We process the previous frame so if this is our first then queue it and continue
        if( ! p_detector->_request )
        {
            p_detector->enqueue( image );
            p_detector->submit_request();
            prev_image = image;
            continue;
        }

        // Wait for the results of the previous frame we enqueued: we're going to process these
        p_detector->wait();
        auto const results = p_detector->fetch_results();

        // Enqueue the current frame so we'd get the results when the next frame comes along!
        p_detector->enqueue( image );
        p_detector->submit_request();

        // MAIN DETECTION
//        detect_objects( image, results, *p_labels, id, objects );

        // Keep it alive so we can actually process pieces of it once we have the results
        prev_image = image;

        // Tell pointcloud object to map to this color frame
        pc.map_to(color_frame);
        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth_frame);

        // Display the results (from the last frame) as rectangles on top (of the current frame)
//        draw_objects( image, depth_frame, objects );
//        draw_detector_overlay( image, current_detector, switch_time );
        draw_curb( image, depth_frame, points, DepthIntrinsics );
        imshow( window_name, image );

        // Handle the keyboard before moving to the next frame
        const int key = cv::waitKey( 1 );
        if( key == 27 )
            break;  // escape
        if( key >= '1'  &&  key < '1' + detectors.size() )
        {
            size_t detector_index = key - '1';
            if( detector_index != current_detector )
            {
                current_detector = detector_index;
                p_detector = detectors[current_detector].detector;
                p_labels = &detectors[current_detector].labels;
                objects.clear();
                LOG(INFO) << "Current detector set to (" << current_detector+1 << ") \"" << openvino_helpers::remove_ext( p_detector->pathToModel ) << "\"";
            }
            switch_time = high_resolution_clock::now();
        }
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    LOG(ERROR) << "Caught RealSense exception from " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what();
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    LOG(ERROR) << "Unknown exception caught: " << e.what();
    return EXIT_FAILURE;
}

