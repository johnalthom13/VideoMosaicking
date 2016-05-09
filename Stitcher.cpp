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
    homography = cv::findHomography(obj, scene, CV_RANSAC);
    std::cout << "Find the Homography Matrix = \n" << homography << std::endl;
}