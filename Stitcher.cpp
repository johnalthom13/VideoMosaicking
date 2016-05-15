#include "Stitcher.hpp"

Stitcher::Stitcher(cv::Ptr<cv::FeatureDetector> detector,
    cv::Ptr<cv::DescriptorExtractor> extractor,
    cv::Ptr<cv::DescriptorMatcher> matcher)
    : detector_(detector)
    , extractor_(extractor)
    , matcher_(matcher)
{
}

void Stitcher::computeHomography(const cv::Mat& image1, const cv::Mat& image2, cv::Mat& homography) const
{
    cv::Mat gray_image1, gray_image2;

    cv::cvtColor(image1, gray_image1, CV_RGB2GRAY);
    cv::cvtColor(image2, gray_image2, CV_RGB2GRAY);
                                                                            
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

    detector_->detect(gray_image1, keypoints_object);
    detector_->detect(gray_image2, keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)             
    cv::Mat descriptors_object, descriptors_scene;

    extractor_->compute(gray_image1, keypoints_object, descriptors_object);
    extractor_->compute(gray_image2, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    std::vector<cv::DMatch> matches;
    matcher_->match(descriptors_object, descriptors_scene, matches);

    
    std::cout << "Descriptors object = " << descriptors_object.rows << std::endl;
    std::cout << "Descriptors scene = " << descriptors_scene.rows << std::endl;
    
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
    // Remove duplicates
    auto comparator = [](const cv::DMatch& a, const cv::DMatch& b)
    {
        return (a.distance < b.distance);
    };
    std::sort(matches.begin(), matches.end(), comparator);

    const int MATR_SIZE = matches.size();
    for (int i = 0; i < MATR_SIZE - 1; ++i)
    {
        for (int j = i + 1; j < MATR_SIZE; ++j)
        {
            // If the left keypoint or right keypoint occurred again
            if (matches[i].queryIdx == matches[j].queryIdx ||
                matches[i].trainIdx == matches[j].trainIdx)
            {
                // Set to 1 since this is the highest possible value
                matches[j].distance = 1.0;
            }
        }
    }
    // end remove duplicates
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
    homography = cv::findHomography(obj, scene, CV_RANSAC);
    std::cout << "Find the Homography Matrix = \n" << homography << std::endl;
}

cv::Mat Stitcher::stitch(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &mask, const cv::Mat &H) const
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
