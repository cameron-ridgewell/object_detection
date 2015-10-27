#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/utility.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/photo/photo.hpp"

//CV_Bridge Variables
static const std::string OPENCV_WINDOW = "Image window";
static const std::string CAMERA_TOPIC = "/usb_cam/image_raw"; //"/rgb/image"; 

//Canny Variables
cv::Mat src, src_gray;
cv::Mat dst, detected_edges;
std::list<cv::Mat> old_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 150;
int ratio = 3;
int blur = 19;
int op = 3;
int kernel_size = 3;
std::string window_name = "Edge Map";

//HoughLines Variables
int rho, theta, hthresh, mll, mlg;
const int EDGE_HISTORY = 5;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

struct Line {
  cv::Point point1;
  cv::Point point2;
};
  
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(CAMERA_TOPIC, 1, 
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_AUTOSIZE);

  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    //get image cv::Pointer
    cv_bridge::CvImagePtr cv_ptr;

    //acquire image frame
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    const std::string filename =  
      "/home/cam/Documents/catkin_ws/src/object_detection/positive_images/wrench.png";

    //read in calibration image
    cv::Mat object = cv::imread(filename, 
      CV_LOAD_IMAGE_COLOR);
    
    //Canny detection
    edge_detection(cv_ptr->image, 0);

    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }

  void edge_detection(cv::Mat image, int threshold) {
    src = image;

    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );

    /// Convert the image to grayscale
    cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    /// Create a window
    cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    /// Create a Trackbar for user to enter threshold
    if (threshold == 0) {
      cv::createTrackbar( "Min Threshold:", window_name, &lowThreshold, 
        max_lowThreshold, Solver );
    } 
    else
    {
      lowThreshold = threshold;
    } 
    /*
    cv::createTrackbar( "op:", window_name, &ratio, 
      100, CannyThreshold );
    
    cv::createTrackbar( "blur:", window_name, &blur, 
      200, CannyThreshold );
    
    cv::createTrackbar( "maxLineGap:", window_name, &mlg, 
      50, CannyThreshold );
    */
    //45 seems to get just about everything

    /// Show the image
    //CannyThreshold(0, 0);
    Solver(0,0);
    /// Wait until user exit program by pressing a key
    cv::waitKey(3);
  }

  static void Solver(int, void*)
  {
    LineDetection(CannyThreshold(0,0));
  }

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
  static cv::Mat CannyThreshold(int, void*)
  {
    cv::Mat element5(5,5,CV_8U,cv::Scalar(1));
    cv::blur( src_gray, detected_edges, cv::Size(3,3) );
    cv::GaussianBlur(src_gray, detected_edges, cv::Size(13, 13), 0, 0 );
    //::medianBlur()

    cv::morphologyEx(detected_edges, detected_edges, cv::MORPH_OPEN, element5);
    //cv::morphologyEx(detected_edges, detected_edges, cv::MORPH_OPEN, element5);
    //cv::fastNlMeansDenoising(detected_edges, detected_edges);
    /// Canny detector
    cv::Canny( detected_edges, detected_edges, lowThreshold, 
      ratio*lowThreshold, kernel_size );

    
    //cv::morphologyEx(detected_edges, detected_edges, cv::MORPH_CLOSE, element5);
    //cv::morphologyEx(detected_edges, detected_edges, cv::MORPH_BLACKHAT, element5);
    //cv::morphologyEx(detected_edges, detected_edges, cv::MORPH_OPEN, element5);
    //cv::morphologyEx(detected_edges, detected_edges, cv::MORPH_CLOSE, element5);

    /// Using Canny's output as a mask, we display our result
    dst = cv::Scalar::all(0);
    src.copyTo(dst, detected_edges);

    //cv::Mat useful;
    //cv::cvtColor(dst, useful, CV_BGR2GRAY);
    
    //line detection
    return detected_edges;
  }

/**
 * @function LineDetection
 * @brief use edge history and Hough Line Detector to find image lines
 */
  static void LineDetection(cv::Mat edges) 
  {
    //Create edge history of EDGE_HISTORY number of frames combined
    cv::Mat filteredEdges = edges.clone();
    std::vector<cv::Mat> temps;
    if (old_edges.size() == EDGE_HISTORY)
    {
      for (int i = 0; i < old_edges.size(); i++)
      {
        cv::Mat temp;
        cv::threshold(*boost::next(old_edges.begin(),i), temp, 100, 255, cv::THRESH_BINARY);
        temps.push_back(temp);
      }
      cv::Mat binImage;
      cv::threshold(edges, binImage, 100, 255, cv::THRESH_BINARY);
      
      filteredEdges = binImage;
      for (int i = 0; i < temps.size(); i++) 
      {
        filteredEdges += temps.at(i);
      }

      old_edges.pop_front();
      old_edges.push_back(edges.clone());
    } 
    else 
    {
      for (int i = 0; i < EDGE_HISTORY; i++)
      {
        old_edges.push_back(edges.clone());
      }
    }

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(filteredEdges, lines, 1, CV_PI/180, 50, 50, 10);
    cv::Point prev1;
    cv::Point prev2;
    std::vector<Line> projectedlines;
    for (size_t i = 0; i < lines.size(); i++)
    {
      cv::Point point1 = cv::Point(lines[i][0], lines[i][1]);
      cv::Point point2 = cv::Point(lines[i][2], lines[i][3]);
      Line line = Line();
      line.point1 = point1;
      line.point2 = point2;
      projectedlines.push_back(line);
      if (i != 0) 
      {
        std::vector<cv::Point> ROI_Poly;
        std::vector<cv::Point> vertices;
        vertices.push_back(point1); 
        vertices.push_back(point2);
        vertices.push_back(prev1);
        vertices.push_back(prev2);
        //cv::approxPolyDP(vertices, ROI_Poly, 1.0, true);
        //cv::fillConvexPoly(src, &ROI_Poly[0], ROI_Poly.size(), 255-(i%2)*255, 8, 0);
      }
      prev1 = point2;
      prev2 = point1;      
    }
    
    for (size_t i = 0; i < projectedlines.size(); i++) {
      cv::line(src, projectedlines[i].point1, projectedlines[i].point2, cv::Scalar(0,0,255), 2, 8);
    }
    
    cv::imshow(window_name, src);
  }
}; 

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
