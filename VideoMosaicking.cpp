
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>   

#include "Logger.hpp"
#include "HistogramComparator.hpp"
#include "Stitcher.hpp"

cv::Mat stitch(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &mask, const cv::Mat &H)
{
    //Coordinates of the 4 corners of the image
    std::vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f(0, 0);
    corners[1] = cv::Point2f(0, img2.rows);
    corners[2] = cv::Point2f(img2.cols, 0);
    corners[3] = cv::Point2f(img2.cols, img2.rows);

    std::vector<cv::Point2f> cornersTransform(4);
    cv::perspectiveTransform(corners, cornersTransform, H);

    double offsetX = 0.0;
    double offsetY = 0.0;

    //Get max offset outside of the image
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "cornersTransform[" << i << "]=" << cornersTransform[i] << std::endl;
        if (cornersTransform[i].x < offsetX)
        {
            offsetX = cornersTransform[i].x;
        }

        if (cornersTransform[i].y < offsetY)
        {
            offsetY = cornersTransform[i].y;
        }
    }

    offsetX = -offsetX;
    offsetY = -offsetY;
    std::cout << "offsetX=" << offsetX << " ; offsetY=" << offsetY << std::endl;

    //Get max width and height for the new size of the panorama
    double maxX = std::max((double)img1.cols + offsetX, (double)std::max(cornersTransform[2].x, cornersTransform[3].x) + offsetX);
    double maxY = std::max((double)img1.rows + offsetY, (double)std::max(cornersTransform[1].y, cornersTransform[3].y) + offsetY);
    std::cout << "maxX=" << maxX << " ; maxY=" << maxY << std::endl;

    cv::Size size_warp(maxX, maxY);
    cv::Mat panorama(size_warp, CV_8UC3);

    //Create the transformation matrix to be able to have all the pixels
    cv::Mat H2 = cv::Mat::eye(3, 3, CV_64F);
    H2.at<double>(0, 2) = offsetX;
    H2.at<double>(1, 2) = offsetY;

    cv::warpPerspective(img2, panorama, H2*H, size_warp);

    //ROI for img1
    cv::Rect img1_rect(offsetX, offsetY, img1.cols, img1.rows);
    cv::Mat half;
    //First iteration
    if (mask.empty())
    {
        //Copy img1 in the panorama using the ROI
        cv::Mat half = cv::Mat(panorama, img1_rect);
        img1.copyTo(half);

        //Create the new mask matrix for the panorama
        mask = cv::Mat::ones(img2.size(), CV_8U) * 255;                                                                                             
        cv::warpPerspective(mask, mask, H2*H, size_warp);
        cv::rectangle(mask, img1_rect, cv::Scalar(255), -1);
    }
    else
    {
        //Create an image with the final size to paste img1
        cv::Mat maskTmp = cv::Mat::zeros(size_warp, img1.type());
        half = cv::Mat(maskTmp, img1_rect);
        img1.copyTo(half);

        //Copy img1 into panorama using a mask
        cv::Mat maskTmp2 = cv::Mat::zeros(size_warp, CV_8U);
        half = cv::Mat(maskTmp2, img1_rect);
        mask.copyTo(half);
        maskTmp.copyTo(panorama, maskTmp2);

        //Create a mask for the warped part
        maskTmp = cv::Mat::ones(img2.size(), CV_8U) * 255;
        cv::warpPerspective(maskTmp, maskTmp, H2*H, size_warp);

        maskTmp2 = cv::Mat::zeros(size_warp, CV_8U);
        half = cv::Mat(maskTmp2, img1_rect);
        //Copy the old mask in maskTmp2
        mask.copyTo(half);
        //Merge the old mask with the new one
        maskTmp += maskTmp2;
        maskTmp.copyTo(mask);
    }

    return panorama;
}

bool hasWhitePixels(const cv::Mat& img)
{
    cv::Mat grayFrame;
    cv::cvtColor(img, grayFrame, CV_RGB2GRAY);
    double min, max;
    cv::minMaxLoc(grayFrame, &min, &max);
    return max == 255;  // Skip images with no white pixel  and no black pixel
}

int main(int argc, char **argv)
{
    LOG_START("Video mosaic tool");
    std::string filename = "../winter.mp4";
    //std::string filename = "../foglab3.mov";
    cv::VideoCapture capture(filename);
    LOG_DEBUG("Reading file: " + filename);
    if (!capture.isOpened())
        throw "Error when reading steam_avi";
    int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
    int frameRate = capture.get(CV_CAP_PROP_FPS);
    LOG_DEBUG("Frame rate: " << frameRate);
    LOG_DEBUG("Frame count: " << frameCount);
    LOG_START("Reading the frames...");
    
    std::unique_ptr<IFrameComparator> frameComparator(new HistogramComparator);
    // Settings
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::BFMatcher();
    // End settings
    std::unique_ptr<Stitcher> stitcher(new Stitcher(detector, extractor, matcher));
    bool firstPass = true;
    int frameNumber = 0;
    cv::Mat prev, curr;
    while (true)
    {
        if (firstPass)
        {
            capture >> prev;
            frameNumber++;
            if (prev.empty()) break; // No valid image to stitch (i.e. all white/black)
            if (hasWhitePixels(prev))
            {
                LOG_DEBUG("prev = frame " << frameNumber);
                firstPass = false;
            }
        }
        capture >> curr;
        frameNumber++;
        if (curr.empty()) break; // No more frames

        // ------------------ Preprocessing
        if (!hasWhitePixels(curr))
        {
            continue;
        }
        // ------------------ End of preprocessing

        // ------------------ Compare
        if (frameNumber%(frameRate) != 0) continue;   // Sample per second
        // ------------------ End compare  
        LOG_DEBUG("curr = frame " << frameNumber);
        cv::imshow("prev", prev); 
        cv::imshow("curr", curr);
        LOG_START("Stitching the images");
        cv::waitKey(50);
        cv::Mat homography;
        stitcher->computeHomography(prev, curr, homography);
        cv::Mat mask;
        cv::Mat panorama = stitch(curr, prev, mask, homography);
        cv::imshow("panorama", panorama);
        LOG_FINISH("Stitching the images");  
        prev.release();
        prev = panorama;   // Update prev as the panorama
        curr.release();
        panorama.release();
        cv::waitKey(50);
    }

    LOG_FINISH("Reading the frames...");
    /*
    int i = 100;
    cv::Mat imgs1 = images[6], imgs2 = images[i];
    cv::imshow("imgs1", imgs1);
    cv::imshow("imgs2", imgs2);
    cv::waitKey(0);
    do
    {
        std::cout << "STITCHING " << i << std::endl;
        cv::Mat matH_1_to_2 = runStitcher(imgs1, imgs2);

        std::cout << "H Matrix found " << i << std::endl;
        cv::Mat mask;
        cv::Mat panorama = stitch(imgs2, imgs1, mask, matH_1_to_2);
        imgs1 = panorama;

        std::cout << "Displaying... " << i << std::endl;
        cv::imshow("panorama", panorama);
        cv::waitKey(0);
        i += 10;
        imgs2 = images[i];
        break;
        /// add adaptive change for i
    } while (i < frameCount);
    */
    LOG_FINISH("Video mosaic tool");
    system("pause");
    return 0;
}
