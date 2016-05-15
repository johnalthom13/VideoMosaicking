#include "AlgorithmFactory.hpp"

FeatureDetectorPtr AlgorithmFactory::createDetector(const std::string& name)
{
    if (name == "FAST")
    {
        return cv::FastFeatureDetector::create();
    }
    if (name == "SIFT")
    {
        return cv::xfeatures2d::SiftFeatureDetector::create();
    }
    if (name == "SURF")
    {
        return cv::xfeatures2d::SurfFeatureDetector::create();
    }
    if (name == "HARRIS")
    {
        return cv::GFTTDetector::create(1000, 0.01, 1.0, 3, true);
    }
    if (name == "GFTT")
    {
        return cv::GFTTDetector::create();
    }
    if (name == "BRISK")
    {
        return cv::BRISK::create();
    }
    if (name == "ORB")
    {
        return cv::ORB::create();
    }
    return FeatureDetectorPtr();
}

DescriptorExtractorPtr AlgorithmFactory::createExtractor(const std::string& name)
{
    if (name == "SIFT")
    {
        return cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
    if (name == "SURF")
    {
        return cv::xfeatures2d::SurfDescriptorExtractor::create();
    }
    if (name == "ORB" || name == "BRIEF")
    {
        return cv::ORB::create();
    }
    if (name == "BRISK")
    {
        return cv::BRISK::create();
    }
    return DescriptorExtractorPtr();
}

DescriptorMatcherPtr AlgorithmFactory::createMatcher(const std::string& name)
{
    /// TODO Fix linking error
    //if (name == "FLANN")
    {
        //return new cv::FlannBasedMatcher();
    }
    if (name == "BRUTE")
    {
        return new cv::BFMatcher();
    }
    return DescriptorMatcherPtr();
}
