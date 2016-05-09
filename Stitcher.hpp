#ifndef STITCHER_HPP
#define STITCHER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

class Stitcher
{
public:
    Stitcher(cv::Ptr<cv::FeatureDetector>, cv::Ptr<cv::DescriptorExtractor>, cv::Ptr<cv::DescriptorMatcher>);
    void computeHomography(const cv::Mat&, const cv::Mat&, cv::Mat&) const;
    ~Stitcher() = default;
private:
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

#endif  // STITCHER_HPP