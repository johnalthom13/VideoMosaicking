#include <chrono>
#include <memory>
#include <cstdint>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Logger.hpp"
#include "Stitcher.hpp"
#include "AlgorithmFactory.hpp"

bool hasWhitePixels(const cv::Mat& img)
{
    cv::Mat grayFrame;
    cv::cvtColor(img, grayFrame, CV_RGB2GRAY);
    double min, max;
    cv::minMaxLoc(grayFrame, &min, &max);
    return max == 255;  // Signifies that there is white pixel
}

// Removes all the black pixels found on frames collected
void crop(cv::Mat& image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, CV_RGB2GRAY);
    int minCol = gray.cols;
    int minRow = gray.rows;
    int maxCol = 0;
    int maxRow = 0;
    for (int i = 0; i < gray.rows - 3; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {
            if (gray.at<char>(i, j) != 0)
            {
                if (i < minRow) minRow = i;
                if (j < minCol) minCol = j;
                if (i > maxRow) maxRow = i;
                if (j > maxCol) maxCol = j;
            }
        }
    }
    cv::Rect cropRect(minCol, minRow, maxCol - minCol, maxRow - minRow);
    image = image(cropRect).clone();
}


int main(int argc, char **argv)
{
    LOG_START("Video mosaic tool");
    if (argc <= 1)
    {
        LOG_DEBUG("Error! Insufficient parameters provided.");
        return -1;
    }
    std::string filename(argv[1]);
    cv::VideoCapture capture(filename);
    LOG_DEBUG("Reading file: " + filename);
    if (!capture.isOpened())
        throw "Error when reading steam_avi";
    double frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
    double frameRate = capture.get(CV_CAP_PROP_FPS);
    LOG_DEBUG("Frame rate: " << frameRate);
    LOG_DEBUG("Frame count: " << frameCount);
    LOG_START("Reading the frames...");
    // Settings
    std::shared_ptr<AlgorithmFactory> factory(new AlgorithmFactory);
    FeatureDetectorPtr detector = factory->createDetector("SURF");
    DescriptorExtractorPtr extractor = factory->createExtractor("SIFT");
    DescriptorMatcherPtr matcher = factory->createMatcher("FLANN");

    // End settings
    std::shared_ptr<Stitcher> stitcher(new Stitcher(detector, extractor, matcher));
    bool firstPass = true;
    int frameNumber = 0; // Counts actual number of frames
    double relativeFrameNum = 0; // Count based on t
    double frameJumps = frameRate;
    cv::Mat prev, curr;
    double WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    const double W_H_ratio = WIDTH / HEIGHT;
    LOG_DEBUG("Dimension: " << WIDTH << " x " << HEIGHT);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (true)
    {
        if (firstPass)
        {
            frameNumber++;
            capture >> prev;
            if (prev.empty()) break; // No valid image to stitch (i.e. all white/black)
            if (hasWhitePixels(prev))
            {
                cv::resize(prev, prev, cv::Size(prev.size().width / 2, prev.size().height / 2));
                crop(prev);
                LOG_DEBUG("first frame = " << frameNumber);
                relativeFrameNum = capture.get(CV_CAP_PROP_POS_FRAMES);
                firstPass = false;
            }
        }

        frameNumber++;
        capture >> curr;
        if (curr.empty()) break; // No more frames
        if (capture.get(CV_CAP_PROP_POS_FRAMES) - relativeFrameNum < frameJumps)
        {
            continue; /// Skip
        }

        crop(curr);
        cv::resize(curr, curr, cv::Size(curr.size().width / 2, curr.size().height / 2));
        relativeFrameNum = capture.get(CV_CAP_PROP_POS_FRAMES);

        LOG_DEBUG("curr = frame " << frameNumber);
        LOG_START("Stitching the images");
        cv::Mat homography;

        stitcher->computeHomography(prev, curr, homography);
        double delta_y = homography.at<double>(1, 2);     // Tells the direction of motion
        double delta_x = homography.at<double>(0, 2);     // Tells the direction of motion
        // - means the video is translating from left to RIGHT
        // + means the video is translating from right to LEFT
        frameJumps = (0.75*frameRate + 0.25*std::abs(delta_x)) / (double)(2);
        std::cout << frameJumps << std::endl;
        cv::Mat mask;
        cv::Mat panorama = stitcher->stitch(curr, prev, mask, homography);
        // Crop panorama image
        crop(panorama);
        cv::imshow("panorama.jpg", panorama);
        cv::waitKey(0);
        LOG_FINISH("Stitching the images");
        prev.release();
        prev = panorama;
        curr.release();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::int64_t duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    LOG_DEBUG("Duration: " << duration);
    LOG_DEBUG("Rate: " << frameCount / duration);
    LOG_FINISH("Reading the frames...");
    LOG_FINISH("Video mosaic tool");
    system("pause");
    return 0;
}
