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
vector<Rect> detect_haar(const Mat& frame);
Point2f detect_blobs(const Mat& frame, const Rect& ROI, const SimpleBlobDetector::Params& params);
int myfilter(const Mat& im_src, const vector<KeyPoint>& keypoints);
int getBiggestIdx(const vector<KeyPoint>& keypoints);
Rect scaleRect(const Mat& im, const Rect& rect, double scale);

void test_detect_blobs();
Rect detect_movement(Mat& frame);
Rect getBbox(const Mat& img, Mat& mask, Mat& dst);

/** Global variables */
mutex mtx;
Mat frame;
String keecker_cascade_name = "../classifier/haarcascade_keecker2.xml";
CascadeClassifier keecker_cascade;
string IMG_PATH = "/Users/hayley/opencv-haar-classifier-training/positive_images/";
string IMG_PATH2 = "/Users/hayley/opencv-stuff/Playground/detect/images/test/";
int COUNT=0;
Mat PREV_FRAME, DIFF;
Rect PREV_BBOX;
Mat H_debug = (Mat_<double>(3,3) << 
0.4727188590049271, 0.2447096686753352, -195.0549601305597,
 -0.001365389089110007, 0.9716478026383405, 6.518203707308869,
 5.496273033415821e-06, 0.001633761706771382, 1);


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
void test_detect_blobs(){
    
    cout << "Test detect_blobss....." << endl;
    string FILE_PATH = "/Users/hayley/opencv-haar-classifier-training/positives-short.txt";

    //--Read file names
    ifstream ifs(FILE_PATH);
    istream_iterator<string> start(ifs), end;
    vector<string> fnames(start, end);
    cout << "READ: " << fnames.size() << endl;
    for (int i=0; i<fnames.size(); i++){
        string fname = IMG_PATH + fnames[i];
        Mat im = imread(fname);
        int total_pixels = im.size().width * im.size().height;
        
        //--Set parameters
        //-- Parmeters setting
        cv::SimpleBlobDetector::Params params; 
        params.filterByArea = true;
        params.minArea = 1;//todo: percentage of total pixel
        params.maxArea = round(total_pixels*0.01);//todo
        cout << "maxArea: " << params.maxArea << endl;

        params.filterByCircularity = true;
        params.minCircularity = 0.6;

        params.filterByConvexity = true;
        params.minConvexity = 0.95;
        //detect_blobs(im,Rect(0,0,im.size().width, im.size().height),params);//
        waitKey(0);
    }
}
void test_detect_blobs2(){
    
    cout << "Test detect_blobs2....." << endl;
    string FILE_PATH = "/Users/hayley/opencv-stuff/Playground/detect/imageslist.txt";

    //--Read file names
    ifstream ifs(FILE_PATH);
    istream_iterator<string> start(ifs), end;
    vector<string> fnames(start, end);
    cout << "READ: " << fnames.size() << endl;
    for (int i=0; i<fnames.size(); i++){
        string fname = IMG_PATH2 + fnames[i];
        cout << fname << endl;
        Mat im = imread(fname,0);
        Mat blurred, binary, adp_binary;            
        GaussianBlur(im, blurred, Size(3,3),0,0 );
        //threshold(blurred, binary, thresh, 255, THRESH_BINARY);
        adaptiveThreshold(blurred, adp_binary, 255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY,5,1 );   
        //imshow("blurred",blurred);
        //imshow("binary", binary);
        imshow("adp-threshed", adp_binary);
        

        int total_pixels = im.size().width * im.size().height;
        
//        cout << "blurred info: "<< endl;
//        cout << "type: " << blurred.type() << endl;
//        cout << "depth: " << blurred.depth() << endl;
//        cout << "channels: "<< blurred.channels() << endl;
        //--Set parameters
        //-- Parmeters setting
        cv::SimpleBlobDetector::Params params; 
        params.filterByArea = true;
        params.minArea = 3.0;//todo: percentage of total pixel
        params.maxArea = round(total_pixels*0.01);//todo
        cout << "maxArea: " << params.maxArea << endl;

        params.filterByCircularity = true;
        params.minCircularity = 0.78;

        params.filterByConvexity = true;
        params.minConvexity = 0.95;
        Rect all(0,0,im.size().width, im.size().height);
        detect_blobs(adp_binary,all,params);//
        waitKey(0);
    }
}
void test_scaleRect(){
    Mat im = Mat::zeros(100,100, CV_64FC3);
    Rect r0(50,50,30,30);
    Point2f p0(r0.x, r0.y);
    Point2f p1(r0.x+r0.width, r0.y+r0.height);
    rectangle(im, p0,p1,Scalar(255,0,0));//Blue
    
    Rect r1 = scaleRect(im, r0, 5);
    rectangle(im, Point2f(r1.x, r1.y), Point2f(r1.x+r1.width, r1.y+r1.height),Scalar(0,255,0));//Green
    imshow("test scale", im);
}

int main(int argc, char** argv){

    thread fastLoop (fastLoopCode);
    int key;
    int count;
    string outname;

    //-- Load keecker cascade
    if( !keecker_cascade.load( keecker_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    //test_detect_blobs2();
    //test_scaleRect();
    while((key = waitKey(30)) != 27){ 
        Mat m;
        mtx.lock();
        if (!frame.empty()){
            m = Mat::Mat(frame,Range(50,frame.size().height));
        }
        mtx.unlock();
        if (!m.empty()) {
            //--Preprocess the frame
            Mat gray, blurred, binary, adp_binary;     
            int total_pixels = m.size().width * m.size().height;
            GaussianBlur(m, blurred, Size(3,3),0,0 );
            cvtColor(blurred, gray, CV_BGR2GRAY);
            threshold( gray, binary, 122, 255,THRESH_BINARY);//todo: dynamic threshold selection?
//            Rect ROI = detect_movement(blurred);      
            
            //--Detect and track keecker using detect_haar (classifier)
            vector<Point2f> srcs,dsts;
        
            //--Detect and track blobs within the keecker area
                //--Set parameters for blob detector
                //-- Parmeters setting
            cv::SimpleBlobDetector::Params params; 
            params.filterByArea = true;
            params.minArea = 3.0;//todo: percentage of total pixel
//            params.maxArea = round(total_pixels*0.01);//todo
//            cout << "max area: " << params.maxArea << endl;
            params.filterByCircularity = true;
            params.minCircularity = 0.65;

            params.filterByConvexity = true;
            params.minConvexity = 0.95;
            vector<Rect> keeckers = detect_haar(m);
            for (int i =0; i<(int)keeckers.size(); i++){
                Point2f p(keeckers[i].x, keeckers[i].y );
                cout << "keecker idx: " << i << endl;
                cout << p << endl;
                srcs.push_back(p);
            }
            //--Draw keeckers
            for (int i=0; i<(int)keeckers.size(); i++){
                Point2f p1(keeckers[i].x, keeckers[i].y);
                Point2f p2(keeckers[i].x+keeckers[i].width, keeckers[i].y+keeckers[i].width);
                rectangle(m, p1, p2, Scalar(255,255,0),2); //color is the last para
            }
            
            Point2f blob;
            Point2f temp;
            Rect ROI;
            for (int i=0; i<keeckers.size(); i++){
                ROI = keeckers[i];
                ROI = scaleRect(m,ROI,4);//DEBUG
                
                //--Set maxArea parameter
                int roi_npixels = ROI.width*ROI.height;
                params.maxArea = round(roi_npixels*0.01);//todo
                cout << "max area: " << params.maxArea << endl;
                blob = detect_blobs(binary,ROI,params);
                
                int n_iter=0;
                int MAX_ITER = 3;
                double scale = 2.0;//todo: input parameter
                double step = 2;
                while(false){
                //while (n_iter<MAX_ITER && blob == Point2f(0,0)){
                    cout << "still no blob, scaling and then searching again" << endl;
                    //Debug for scaleRect
                    //rectangle(m, Point2f(ROI.x,ROI.y), Point2f(ROI.x+ROI.width, ROI.y+ROI.height),Scalar(0,0,0),1,8,0);

                    ROI = scaleRect(m, ROI, scale);
                    //Debug drawing for scaleRect
                    rectangle(m, Point2f(ROI.x,ROI.y), Point2f(ROI.x+ROI.width, ROI.y+ROI.height),Scalar(255,0,0),1,8,0);
                    //imshow("Debug", m);
                    blob = detect_blobs(binary,ROI,params);
                    scale *= step;
                    n_iter ++;
                }
                
                if (blob != Point2f(0,0)){
                    //--Draw a blob for a keecker
                    temp.x = blob.x+3; temp.y = blob.y+3;
                    Point2f dp(3,3);
                    Point2f blob_tl = blob - dp;
                    Point2f blob_br = blob + dp;
                    rectangle(m, blob_tl, blob_br, Scalar(0,255,0),2,8,0 );
                    cout << "Blob for keecker " << i << " :" << blob << endl;
                    
                    //Find the blob position at the world coord using H_debug
                    vector<Point2f> blob_vec = {blob};
                    vector<Point2f> blob_world;
                    perspectiveTransform(blob_vec, blob_world, H_debug);
                    cout << "   in world coord: " << blob_world[0] << endl;
                }else{
                    cout << "No blob detected for this keecker:(.." << endl;
                }
            }
            
            //--show
            imshow("blobs in the world", m);

//            vector<Point2f> srcs, dsts;
//            srcs = detect_blobs(blurred, ROI, params );
//            //--Get the world coordinates
//            if (srcs.size()>0){
//                perspectiveTransform(srcs, dsts, H_debug);
//
//                for (int i=0; i<(int)dsts.size(); i++){
//                    cout << "world coordinates: " << dsts[i] << endl;
//                }
//            }
//            
//            //Show rectangle positions
//            rectangle(blurred, ROI.tl(), ROI.br(), Scalar(255,0,0), 1, 8, 0 ); //BLUE

//            //--Get the world coordinates
//            perspectiveTransform(srcs, dsts, H_debug);
//            for (int i=0; i<(int)dsts.size(); i++){
//                cout << "world coordinates: " << dsts[i] << endl;
//                Point2f temp(dsts[i].x+3, dsts[i].y+3);
////                rectangle(blurred, dsts[i] , temp, Scalar(255,0,0), 2, 8, 0 );
//            }
//            //End of haar detection 
        }
    }
    fastLoop.join();
}

vector<Rect> detect_haar(const Mat& frame){
    Mat gray;
    
    cvtColor(frame, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);
    //--Detect keeckers
    //--Set parameters
    const float scale_factor(1.2f);
    const int min_neighbors(15);
    vector<int> reject_levels;
    vector<double> level_weights;
    
    vector<Rect> keeckers;
    //keecker_cascade.detectMultiScale(gray, keeckers, scale_factor, min_neighbors, 0,  Size(20, 20), Size(200,200));
    keecker_cascade.detectMultiScale(gray, keeckers, scale_factor, min_neighbors, 0,  Size(20, 20), gray.size());
//    keecker_cascade.detectMultiScale(gray, keeckers, reject_levels, level_weights, scale_factor, min_neighbors, 0, Size(60, 13), img.size(), true);

    return keeckers;
}

Rect detect_movement(Mat& frame){
    //cout << "detect movement is called" << endl;
    Mat fgMaskKNN;
    Ptr<BackgroundSubtractorKNN> pKNN;
    pKNN = createBackgroundSubtractorKNN(1,4000,false);
    //update the background model
    pKNN->apply(frame, fgMaskKNN);
    
    Mat diff;
    if (PREV_FRAME.empty()){
        diff = fgMaskKNN;
    }else{
        absdiff(fgMaskKNN,PREV_FRAME,diff);
    }
    
    Rect bbox_union;
    Mat bboxed;// = Mat::zeros( frame.size(), CV_32FC3 );
    Mat im_out = Mat::zeros( frame.size(), CV_32FC3 );
    
    Rect curr_bbox = getBbox(frame,fgMaskKNN, bboxed);
    bbox_union = (PREV_BBOX | curr_bbox);

    //--Update for next iter
    PREV_FRAME = fgMaskKNN;
    PREV_BBOX = curr_bbox;
    COUNT ++;
    return bbox_union;
}
Point2f detect_blobs(const Mat& frame, const Rect& ROI, const SimpleBlobDetector::Params& params){ 
    
    /* Assumes input of grayscale frame (entire)
    ROI is the keecker bounding box.
    */
    
    //cout << "detect blob is called" << endl;
    vector<Point2f> blob_positions;
    if (ROI.width ==0){return Point2f(0,0);}
    
    Mat img(frame, ROI);
    imshow("before",img);
    threshold(img, img, 110, 255,THRESH_BINARY);//todo: dynamic threshold selection?
    imshow("After", img);
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    //--Filter detected blobs
    //cout << "Before loc filter: " << keypoints.size() << endl;
    //keypoints = filter_loc(img.size().width, img.size().height, keypoints);//turn off the location filter
    //cout << "After loc filter: " << keypoints.size() <<endl;
    //cout << "Filtering... "<< endl;
    int id = myfilter(img,keypoints);
    //cout << "   best idx: " << id << endl;
    if (keypoints.size() > 0){//todo: negate and return 
        for (std:: vector<cv::KeyPoint>::iterator bIterator=keypoints.begin(); bIterator!=keypoints.end(); bIterator++){
//            cout << "here" << endl;
//            cout << "size of blob: " << bIterator->size << endl;
//            cout << "   at: " << bIterator->pt.x << ", " << bIterator->pt.y << endl;
//            cout << "" <<endl;
            blob_positions.push_back(Point(bIterator->pt.x, bIterator->pt.y));
        }

        //--Draw detected blobs as red circles
        Mat im_out;
        frame.copyTo(im_out);
        //for debug
        drawKeypoints(img, keypoints, im_out, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        Point2f best(keypoints[id].pt.x, keypoints[id].pt.y);
        Point2f temp(best.x+3, best.y+3);
        //rectangle( im_out, best, temp, Scalar(255,0,0), 2, 8, 0 );
        
        int biggest_idx = getBiggestIdx(keypoints);
        Point2f biggest = keypoints[biggest_idx].pt;
        Point2f temp2(biggest.x+3, biggest.y+3);
        rectangle( im_out, biggest, temp2, Scalar(0,255,0), 2, 8, 0 );

        imshow("detect blob", im_out);

        Point2f blob_world(biggest.x+ROI.x, biggest.y+ROI.y);
        return blob_world;
    }
}

Rect getBbox(const Mat& img, Mat& mask, Mat& dst){   
    int niters = 3;
    vector<vector<Point> > contours;
    
    vector<Vec4i> hierarchy;
    Mat temp;
    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);
    findContours( temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    
    //--Bounding box for all contours found
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );

    dst = Mat::zeros(img.size(), CV_8UC1);
    if( contours.size() == 0 )
        return Rect(0,0,0,0);
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = 0;
    double maxArea = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] ){
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea ){

            maxArea = area;
            largestComp = idx;
        }
    }
    
//    for( size_t i = 0; i < contours.size(); i++ ){
//        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
//        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
//     }
    approxPolyDP( Mat(contours[largestComp]), contours_poly[largestComp], 3, true );
    boundRect[largestComp] = boundingRect( Mat(contours_poly[largestComp]) );
    
    //cout << "number of contours found: " << contours.size() << endl;
    //--Draw contours
    if (maxArea!=0){
//        drawContours( img, contours, largestComp, Scalar(255), 2, LINE_8, hierarchy );        
        //rectangle( img, boundRect[largestComp].tl(), boundRect[largestComp].br(), Scalar(255), 2, 8, 0 );
    }
    //imshow("RECT", img);
    return boundRect[largestComp];
}

int getBiggestIdx(const vector<KeyPoint>& keypoints){
    /*assume im_src is a adpative-threshed image of the original frame
    *It's assumed to be of type 0, CV_8U3
    */
    if (keypoints.size()<=0){
        return -1;
    }
    
    double biggest_size = -1;
    int biggest_idx;
    for (int i=0; i<keypoints.size(); i++){
        if (keypoints[i].size > biggest_size){
            biggest_size = keypoints[i].size;
            biggest_idx = i;
        }
    }
        
    return biggest_idx;
}
//
Rect scaleRect(const Mat& im, const Rect& rect, double scale){
    /*
    im is passed in for checking the boundaries when enlarging the rects. 
    clip the new values.
    */
    if(rect.width==0){
        return Rect(0,0,0,0);
    }
    
    Point2f tl_0(rect.x, rect.y);
    Point2f d_0(rect.width/2, rect.height/2);
    Point2f c = tl_0 + d_0;
    
    Point2f d_1 = d_0*sqrt(scale);
    Point2f temp1 = c-d_1;
    //cout << "Temp1: " << temp1 << endl;
    float x,y;
    x = max(0.0f,temp1.x);
    y = max(0.0f,temp1.y);
    Point2f tl_1(x,y);
    
    Point2f temp2 = c+d_1;
    //cout << "Temp2: " << temp2 << endl;
    x = min((float)im.size().width-1, temp2.x);
    y = min((float)im.size().height-1, temp2.y);
    Point2f br_1(x,y);
    
    Rect scaled(tl_1,br_1);
    return scaled;
}

int myfilter(const Mat& im_src, const vector<KeyPoint>& keypoints){
    /*assume im_src is a blurred image of the original frame
    *It's assumed to be of type 0, CV_8U3
    */
    //cout << "here!!!!! " << im_src.depth() << endl;
    if (keypoints.size()<=0){
        return -1;
    }
    Mat im;
    im_src.convertTo(im, CV_64FC3);
    //cout << "again " << im.depth() << endl;
    vector<KeyPoint> filtered;
    int best_idx;
    double best_cost=1000000;
    cout << endl;
    for (int i=0; i<keypoints.size(); i++){
        int _x = keypoints[i].pt.x;
        int _y = keypoints[i].pt.y;
        int r = round((keypoints[i].size)/2);
        int R = 2*r;
        double alpha = 0.2; 
        double avg_center=0, avg_outer=0;
        int min_xc = max(0, _x-r);
        int max_xc = min(im_src.size().width-1, _x+r);
        int min_yc = max(0, _y-r);
        int max_yc = min(im_src.size().height-1, _y+r);
        int n_center = (max_yc-min_yc+1)*(max_xc-min_xc+1);
        
        int min_xo = max(0, _x-2*R);
        int max_xo = min(im_src.size().width-1, _x+R);
        int min_yo = max(0, _y-R);
        int max_yo = min(im_src.size().height-1, _y+R);
        int n_outer = ((max_yo-min_yo+1)*(max_xo-min_xo+1)) - n_center;

        double cost;
        for (int j=min_yc; j<=max_yc; j++){
            for (int i = min_xc; i<=max_xc; i++){
                avg_center += im.at<double>(j,i);
            }
        }
        for (int j=min_yo; j<=max_yo; j++){
            for (int i = min_xo; i<=max_xo; i++){
                avg_outer += im.at<double>(j,i);
//                cout << im.at<double>(j,i) << endl;

            }
        }
        avg_outer -= avg_center;
        //Normalize
        avg_center /= n_center;
        avg_outer /= n_outer;
        cost = ((255-avg_outer) + (avg_center))*alpha*r; //penalize it by the number at center portion (we prefer smaller blobs);        
        
        //Report for debug
//        cout << "new X,y: " << _x << ", " << _y << endl;
//        cout << "Npixel CENTER: " << n_center << endl;
//        cout << "Avg_center: " << avg_center << endl;
//        cout << "Npixel OUTER: " << n_outer << endl;
//        cout << "Avg_outer: " << avg_outer << endl;
//        cout << "cost: " << cost << endl;
//        cout << endl;
        
        if (cost<best_cost){
            best_idx=i;
            best_cost=cost;
        }
    }
    //cout << "best: " << best_idx << endl;
    return best_idx;
}