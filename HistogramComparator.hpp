#ifndef HISTOGRAMCOMPARATOR_HPP
#define HISTOGRAMCOMPARATOR_HPP

#include "IFrameComparator.hpp"
#include <opencv2/opencv.hpp>

class HistogramComparator : public IFrameComparator
{
public:
    HistogramComparator(int method = CV_COMP_BHATTACHARYYA);
    bool isSimilar(const cv::Mat&, const cv::Mat&) override;
private:
    int method_;
};

#endif // HISTOGRAMCOMPARATOR_HPP
