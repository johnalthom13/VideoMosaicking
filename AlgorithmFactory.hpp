#ifndef ALGORITHM_FACTORY_HPP
#define ALGORITHM_FACTORY_HPP

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using FeatureDetectorPtr = cv::Ptr<cv::FeatureDetector>;
using DescriptorExtractorPtr = cv::Ptr<cv::DescriptorExtractor>;
using DescriptorMatcherPtr = cv::Ptr<cv::DescriptorMatcher>;

class AlgorithmFactory
{
public:
    FeatureDetectorPtr createDetector(const std::string&);
    DescriptorExtractorPtr createExtractor(const std::string&);
    DescriptorMatcherPtr createMatcher(const std::string&);
};


#endif  // ALGORITHM_FACTORY_HPP