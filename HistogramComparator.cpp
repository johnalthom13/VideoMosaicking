#include "HistogramComparator.hpp"


HistogramComparator::HistogramComparator(int method)
: method_(method)
{
}

bool HistogramComparator::isSimilar(const cv::Mat& base, const cv::Mat& img)
{
    cv::Mat lastImg = img;
    cv::Mat hsv_base, hsv_test1;
    cv::cvtColor(lastImg, hsv_base, cv::COLOR_BGR2HSV);
    cv::cvtColor(base, hsv_test1, cv::COLOR_BGR2HSV);

    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    int channels[] = { 0, 1 };

    /// Histograms
    cv::MatND hist_base;
    cv::MatND hist_test1;

    /// Calculate the histograms for the HSV images
    cv::calcHist(&hsv_base, 1, channels, cv::Mat(), hist_base, 2, histSize, ranges, true, false);
    cv::normalize(hist_base, hist_base, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    cv::calcHist(&hsv_test1, 1, channels, cv::Mat(), hist_test1, 2, histSize, ranges, true, false);
    cv::normalize(hist_test1, hist_test1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    double ratio = cv::compareHist(hist_base, hist_test1, method_);
    std::cout << ratio << std::endl;
    return ratio == 0;
}
