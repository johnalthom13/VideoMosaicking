#include <iostream>

#include <iostream>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
                                    

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

cv::Mat runStitcher(const cv::Mat image1, const cv::Mat image2)
{
    cv::Mat gray_image1, gray_image2;

    cv::cvtColor(image1, gray_image1, CV_RGB2GRAY);
    cv::cvtColor(image2, gray_image2, CV_RGB2GRAY);

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

    detector->detect(gray_image1, keypoints_object);
    detector->detect(gray_image2, keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();

    cv::Mat descriptors_object, descriptors_scene;

    extractor->compute(gray_image1, keypoints_object, descriptors_object);
    extractor->compute(gray_image2, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::BFMatcher();
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_object, descriptors_scene, matches);

    double max_dist = 0;
    double min_dist = 100;
    std::cout << "Descriptors object = " << descriptors_object.rows << std::endl;
    std::cout << "Descriptors scene = " << descriptors_scene.rows << std::endl;
    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_object.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector<cv::DMatch > good_matches;
    // Normalize distances
    cv::Mat distances;
    for (const auto& m : matches)
    {
        distances.push_back(m.distance);
    }

    std::cout << "Normalizing distances..." << std::endl;
    // Normalize distances to have a [0,1] range
    cv::normalize(distances, distances, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    std::cout << "Normalizing distances...DONE" << std::endl;
    // Store the normalized distance on corresponding pair
    for (int i = 0; i < distances.rows; ++i)
    {
        matches[i].distance = distances.at<float>(i, 0);
    }
                                                     // TODO remove duplicates
    std::cout << "Searching for good matches..." << std::endl;
    for (int i = 0; i < descriptors_object.rows; i++)
    {
        if (matches[i].distance < 0.4)     // TODO Set threshold parameter
        {
            good_matches.push_back(matches[i]);
        }
    }

    std::cout << "Good matches = " << good_matches.size() << std::endl;
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (int i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }
    // Find the Homography Matrix                                     
    std::cout << "Finding homography..." << std::endl;
    cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
    std::cout << "Find the Homography Matrix = \n" << H << std::endl;
    // Use the Homography Matrix to warp the images
    //cv::Mat result;
    //cv::warpPerspective(image1, result, H, cv::Size(image1.cols+image2.cols,image1.rows));
    //cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
    //image2.copyTo(half);
    //output = result;
    //cv::imshow("Result", result);
    //cv::waitKey(0);
    return H;
}

int main(int argc, char **argv)
{
    
    std::cout << "Program started." << std::endl;
    std::string filename = "../winter.mp4";
    //std::string filename = "../foglab3.mov";
    cv::VideoCapture capture(filename);
    std::cout << "File read..." << std::endl;
    std::vector<cv::Mat> images;
    if (!capture.isOpened())
        throw "Error when reading steam_avi";
    int frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
    std::cout << "Frame rate: " << capture.get(CV_CAP_PROP_FPS) << std::endl;
    std::cout << "Frame count: " << frameCount << std::endl;
    std::cout << "Reading the frames..." << std::endl;
    int ctr = 0;
    while (true)
    {
        cv::Mat frame;
        capture >> frame;
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, CV_RGB2GRAY);
        int xx = cv::countNonZero(grayFrame);
        std::cout << ctr << " - Non-zero: " << xx << std::endl;
        if (xx < 1)
        {
            continue;
        }
        // Compare
        images.push_back(frame);
        // end compare
        // TODO add scheme to compare similarity. if too similar, reject
        //images.push_back(frame);
        if (frame.empty() || ctr > 102)
            break;

        ++ctr;
    }

    std::cout << "Done reading the frames..." << std::endl;

    int i = 100;
    cv::Mat imgs1 = images[30], imgs2 = images[i];
    cv::imshow("imgs1", imgs1);
    cv::imshow("imgs2", imgs2);
    //cv::Mat matH_1_to_3 = runStitcher(imgs1, imgs2, temp);
    cv::waitKey(0);
    /*
    cv::Mat mask;
    cv::Mat panorama = stitch(imgs2, imgs1, mask, matH_1_to_3);
    cv::imshow("panorama", panorama);
    cv::waitKey(0);*/;
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

    /*
    std::vector<cv::Mat> images{img1, img2, img3};
    cv::Mat pano;
    cv::Stitcher stitcher = cv::Stitcher::createDefault(true);
    cv::Stitcher::Status stat = stitcher.stitch(images, pano);
    cv::imshow( "Result", pano );
    //cv::imshow("panorama", panorama);

    */
    return 0;
}
