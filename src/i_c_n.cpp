#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

//CV_Bridge Variables
static const std::string OPENCV_WINDOW = "Image window";
static const std::string CAMERA_TOPIC = "/usb_cam/image_raw";

//Canny Variables
cv::Mat src, src_gray;
cv::Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
std::string window_name = "Edge Map";
const int CANNY_THRESHOLD = 30;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(CAMERA_TOPIC, 1, 
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
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
      CV_LOAD_IMAGE_GRAYSCALE);

    cv::namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);
    //SURF Detector, and descriptor parameters
    int minHess=2000;
    std::vector<cv::KeyPoint> kpObject, kpImage;
    cv::Mat desObject, desImage;

    //Display keypoints on training image
    cv::Mat interestPointObject=object;

    //SURF Detector, and descriptor parameters, match object initialization
    cv::SurfFeatureDetector detector(minHess);
    detector.detect(object, kpObject);
    cv::SurfDescriptorExtractor extractor;
    extractor.compute(object, kpObject, desObject);
    cv::FlannBasedMatcher matcher;

    //Object corner cv::Points for plotting box
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );

    double frameCount = 0;
    float thresholdMatchingNN=0.7;
    unsigned int thresholdGoodMatches=4;
    unsigned int thresholdGoodMatchesV[]={4,5,6,7,8,9,10};

    char escapeKey = 'k';

    for (int j=0; j<7;j++)
    {
      thresholdGoodMatches = thresholdGoodMatchesV[j];
      
      while (escapeKey != 'q')
      {
        frameCount++;
        cv::Mat image;
        cvtColor(cv_ptr->image, image, CV_RGB2GRAY);

        cv::Mat des_image, img_matches, H;
        std::vector<cv::KeyPoint> kp_image;
        std::vector<std::vector<cv::DMatch > > matches;
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        std::vector<cv::Point2f> scene_corners(4);

        detector.detect( image, kp_image );
        extractor.compute( image, kp_image, des_image );
        matcher.knnMatch(desObject, des_image, matches, 2);

        for(int i = 0; i < std::min(des_image.rows-1, (int) matches.size()); i++) 
        //THIS LOOP IS SENSITIVE TO SEGFAULTS
        {
          if((matches[i][0].distance < thresholdMatchingNN*(matches[i][1].distance)) 
            && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
            {
                good_matches.push_back(matches[i][0]);
            }
        }

        //Draw only "good" matches
        cv::drawMatches(object, kpObject, image, kp_image, good_matches, img_matches, 
          cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), 
          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        if (good_matches.size() >= thresholdGoodMatches)
        {

          //Display that the object is found
          cv::putText(img_matches, "Object Found", cvPoint(10,50), 0, 2, 
            cvScalar(0,0,250), 1, CV_AA);
            for(unsigned int i = 0; i < good_matches.size(); i++ )
            {
              //Get the keypoints from the good matches
              obj.push_back( kpObject[ good_matches[i].queryIdx ].pt );
              scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
            }

            H = findHomography( obj, scene, CV_RANSAC );

            perspectiveTransform( obj_corners, scene_corners, H);

            //Draw lines between the corners (the mapped object in the scene image )
            cv::line( img_matches, scene_corners[0] + cv::Point2f( object.cols, 0), 
              scene_corners[1] + cv::Point2f( object.cols, 0), cv::Scalar(0, 255, 0), 4 );
            cv::line( img_matches, scene_corners[1] + cv::Point2f( object.cols, 0), 
              scene_corners[2] + cv::Point2f( object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( img_matches, scene_corners[2] + cv::Point2f( object.cols, 0), 
              scene_corners[3] + cv::Point2f( object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            cv::line( img_matches, scene_corners[3] + cv::Point2f( object.cols, 0), 
              scene_corners[0] + cv::Point2f( object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        }
        else
        {
          putText(img_matches, "", cvPoint(10,50), 0, 3, cvScalar(0,0,250), 1, CV_AA);
        }

        //Show detected matches
        imshow("Good Matches", img_matches);
        
        escapeKey=cvWaitKey(10);

        if(frameCount>10)
        {
          escapeKey='q';
        }


      }

      frameCount=0;
      escapeKey='a';
    }

    // Update GUI Window
    //cv::namedWindow(OPENCV_WINDOW);
    //cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    //cv::waitKey(3);
    
    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
  static void CannyThreshold(int, void*)
  {
    /// Reduce noise with a kernel 3x3
    cv::blur( src_gray, detected_edges, cv::Size(3,3) );

    /// Canny detector
    //cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    cv::Canny( detected_edges, detected_edges, CANNY_THRESHOLD, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = cv::Scalar::all(0);

    src.copyTo( dst, detected_edges);
    cv::imshow( window_name, dst );
  }
}; 

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
