#include<iostream>
#include<fstream>
#include<cmath>
#include "opencv2/opencv.hpp"
#include "mycv.h"

using namespace std;
using namespace cv;
const int maxlongint=2147483647;

int main(int argc, char *argv[])
{
  IplImage* src=NULL;
  src=cvLoadImage(argv[1],CV_LOAD_IMAGE_ANYCOLOR);
  if(!src)
    cout<<"Could not load image file: "<<argv[1]<<endl;
  IplImage* dst=cvCreateImage(cvGetSize(src),src->depth,1);
  IplImage* logp=cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
  IplImage* dctp=cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
  cvScale(src,logp);
  cvLog(logp,logp);
  cvDCT(logp,dctp,CV_DXT_FORWARD);
  cvNamedWindow("DCT",CV_WINDOW_AUTOSIZE);
  cvMoveWindow("DCT",100,100);
  cvShowImage("DCT",logp);
  if (cvWaitKey(0)>=0)
  {
    cvReleaseImage(&dst);
    cvReleaseImage(&logp);
    cvReleaseImage(&dctp);
    cvDestroyAllWindows();
  }
  return 0;
}

  
  
