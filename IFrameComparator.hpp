#ifndef IFRAMECOMPARATOR_HPP
#define IFRAMECOMPARATOR_HPP

namespace cv
{
    class Mat;
}

class IFrameComparator
{
public:
    virtual bool isSimilar(const cv::Mat&, const cv::Mat&) = 0;
    virtual ~IFrameComparator() {}
};

#endif // IFRAMECOMPARATOR_HPP
