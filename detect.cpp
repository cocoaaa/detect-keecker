#include <opencv2/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
#include "opencv2/video/background_segm.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <thread>
#include <mutex>

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> detect_haar(Mat& frame);
void detectBlob(Mat& frame, Rect& ROI, SimpleBlobDetector::Params& params);
void test_detectBlobs();
Rect detect_movement(Mat& frame);
Rect refineSegments(const Mat& img, Mat& mask, Mat& dst);


/** Global variables */
mutex mtx;
Mat frame;
String keecker_cascade_name = "../classifier/cascade.xml";
CascadeClassifier keecker_cascade;
string window_name = "Capture - keecker detection";
string IMG_PATH = "/Users/hayley/opencv-haar-classifier-training/positive_images/";
int COUNT=0;
Mat PREV_FRAME, DIFF;
Mat H_debug = (Mat_<double>(3,3) << 
0.4489607764550727, 0.1351003751241841, -154.9639265640121,
 0.01191502411752055, 0.8548885289745943, -68.83102780474229,
 -2.816918868191362e-05, 0.001144330460324215, 1);


void fastLoopCode() {
    
    const string url = "rtsp://192.168.10.233:554/onvif1";
    VideoCapture camera(url);   
    double fps = camera.get(CV_CAP_PROP_FPS);
    cout << "Frame per seconds: " << fps << endl;
    while(1){
        mtx.lock();
        camera >> frame;
        mtx.unlock();
        }
}
vector<KeyPoint> filter_loc(int im_width, int im_height, vector<KeyPoint>& keypoints){ 
    /* Filter by y location
    * 1/6 top of the keecker body
    */
    int cutoff_y = im_height/6;
    int cutoff_x_min = im_width*2/10;
    int cutoff_x_max = im_width*8/10;
    vector<KeyPoint> filtered;
    for (int i=0; i<keypoints.size(); i++){
        if (keypoints[i].pt.y<cutoff_y && cutoff_x_min < keypoints[i].pt.x && keypoints[i].pt.x < cutoff_x_max ){
            //cout << "Pass loc filter" << endl;
            filtered.push_back(keypoints[i]);
        }
    }
    return filtered;
}
void test_detectBlobs(){
    cout << "Test detectBlobs....." << endl;
    string FILE_PATH = "/Users/hayley/opencv-haar-classifier-training/positives-short.txt";

    //--Read file names
    ifstream ifs(FILE_PATH);
    istream_iterator<string> start(ifs), end;
    vector<string> fnames(start, end);
    cout << "READ: " << fnames.size() << endl;
    for (int i=0; i<fnames.size(); i++){
        string fname = IMG_PATH + fnames[i];
        Mat im = imread(fname);
        
        //--Set parameters
        //-- Parmeters setting
        cv::SimpleBlobDetector::Params params; 
        params.filterByArea = true;
        params.minArea = 1.0;//todo: percentage of total pixel
        params.maxArea = 10.0;//todo

        params.filterByCircularity = true;
        params.minCircularity = 0.7;

        params.filterByConvexity = true;
        params.minConvexity = 0.95;
        //detectBlob(im,Rect(0,0,im.size().width, im.size().height),params);//
        waitKey(0);
    }
}

void test_detectBlobs2(){
    string fname = "diff106.png";
    Mat out;
    Mat im = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
   // refineSegments(im, out, )
    imshow("test", im);
}
int main(int argc, char** argv){

    thread fastLoop (fastLoopCode);
    namedWindow(window_name, WINDOW_AUTOSIZE); 
    int key;
    int count;
    string outname;
    
    //-- Load keecker cascade
    if( !keecker_cascade.load( keecker_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    //test_detectBlobs2();
    
    
    while((key = waitKey(30)) != 27){ 
        Mat m;
        mtx.lock();
        frame.copyTo(m);
        mtx.unlock();
        if (!m.empty()) {
            imshow(window_name, m);
            Mat blurred;
            GaussianBlur(m, blurred, Size(3,3),0,0 );
            //imshow("blurred", blurred);
            Rect ROI = detect_movement(blurred);
            
            //--Set blob detector parameters
            cv::SimpleBlobDetector::Params params; 
            params.filterByArea = true;
            params.minArea = 1.0;//todo: percentage of total pixel
            params.maxArea = 10.0;//todo

            params.filterByCircularity = true;
            params.minCircularity = 0.7;

            params.filterByConvexity = true;
            params.minConvexity = 0.95;
            detectBlob(blurred, ROI, params );
            /*
            //--Detect and track keecker
            vector<Point2f> srcs,dsts;
        
            //--Detect and track blobs within the keecker area
            vector<Rect> keeckers = detect_haar(m);
            for (int i =0; i<(int)keeckers.size(); i++){
                Point2f p(keeckers[i].x, keeckers[i].y );
                cout << "keecker idx: " << i << endl;
                cout << p << endl;
                srcs.push_back(p);
            }
            
            //--Get the world coordinates
            perspectiveTransform(srcs, dsts, H_debug);
            
            for (int i=0; i<(int)dsts.size(); i++){
                cout << "world coordinates: " << dsts[i] << endl;
            }
            */
        }
    }
    fastLoop.join();

    /**/
    
}

vector<Rect> detect_haar(Mat& frame){
    cout << "detect_haar called!" << endl;
    vector<Rect> keeckers;
    Mat gray;
    
    cvtColor(frame, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);
    
    //--Detect keeckers
    keecker_cascade.detectMultiScale(gray, keeckers, 1.1);
    for (int i=0; i<(int)keeckers.size(); i++){
        Point2f p1(keeckers[i].x, keeckers[i].y);
        Point2f p2(keeckers[i].x+keeckers[i].width, keeckers[i].y+keeckers[i].width);
        rectangle(frame, p1, p2, 200); //color is the last para
    }
    imshow(window_name, frame);
    return keeckers;
}

Rect detect_movement(Mat& frame){
    cout << "detect movement is called" << endl;
    Mat fgMaskKNN;
//    Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
//    Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
    Ptr<BackgroundSubtractorKNN> pKNN;
    //pMOG2 = createBackgroundSubtractorMOG2(100,16,false);//300,32,true); //MOG2 approach
    pKNN = createBackgroundSubtractorKNN(1,4000,false);
    //update the background model
    pKNN->apply(frame, fgMaskKNN);
//    pMOG2->apply(frame, fgMaskMOG2);
    //show the current frame and the fg masks
//    imshow("FG Mask MOG 2", fgMaskMOG2);
    
    Mat diff;
    if (PREV_FRAME.empty()){
        diff = fgMaskKNN;
    }else{
        diff = fgMaskKNN-PREV_FRAME;
    }
    
    Mat contoured;
    //imwrite("contour"+to_string(COUNT)+".png", diff);
    //imshow("KNN", fgMaskKNN);
    Rect boundingBox = refineSegments(frame,fgMaskKNN, contoured);
    imshow("contoured", contoured);
    PREV_FRAME = fgMaskKNN;

    COUNT ++;
    return boundingBox;
}
void detectBlob(Mat& frame, Rect& ROI, SimpleBlobDetector::Params& params){ 
    /* Input image should refer to the keecker (bounding boxed)
    Returns a vector of keypoints
    */
    
    cout << "detect blob is called" << endl;
    namedWindow("blobs", WINDOW_AUTOSIZE);
    
    Mat img(frame, ROI);
    imshow("roi selected", img);
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    //--Filter by ylocation 
    //cout << "Before loc filter: " << keypoints.size() << endl;
    keypoints = filter_loc(img.size().width, img.size().height, keypoints);
    //cout << "After loc filter: " << keypoints.size() <<endl;
    for (std:: vector<cv::KeyPoint>::iterator bIterator=keypoints.begin(); bIterator!=keypoints.end(); bIterator++){
        cout << "size of blob: " << bIterator->size << endl;
        cout << "   at: " << bIterator->pt.x << ", " << bIterator->pt.y << endl;
        cout << "" <<endl;
    }
    
    Mat im_out;
    //--Draw detected blobs as red circles
    drawKeypoints(img, keypoints, im_out, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    //--Show blobs
    imshow("blobs", im_out);
    return;
}
void getBBox(Mat& img){
    /*Get the bounding box of a white pixel area in the input img
    */
    for (int x=0; x<img.size().width; x++){
        for (int y=0; y<img.size().height; y++){
            
        }
    }
}

Rect refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 3;
    vector<vector<Point> > contours;
    
    vector<Vec4i> hierarchy;
    Mat temp;
    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);
    findContours( temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    
    //--Bounding box for the contour
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );

    dst = Mat::zeros(img.size(), CV_8UC3);
    if( contours.size() == 0 )
        return Rect(0,0,0,0);
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = 0;
    double maxArea = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    
    for( size_t i = 0; i < contours.size(); i++ ){
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
     }
    
    //--Draw contours
    Scalar color( 0, 0, 255 );
    ///drawContours( dst, contours, largestComp, color, FILLED, LINE_8, hierarchy );
    rectangle( dst, boundRect[largestComp].tl(), boundRect[largestComp].br(), color, 2, 8, 0 );
    return boundRect[largestComp];

}

