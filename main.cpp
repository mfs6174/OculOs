#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<cstring>
#include<algorithm>
#include<cmath>
#include "opencv2/opencv.hpp"
#include "mycv.h"
#include "IC.h"
#include "CL.h"
using namespace std;
using namespace cv;
const int maxlongint=2147483647;

int main(int argc, char *argv[])
{
  IplImage* src=NULL;
  src=cvLoadImage(argv[1],CV_LOAD_IMAGE_ANYCOLOR);
  if(!src)
    cout<<"Could not load image file: "<<argv[1]<<endl;
  IplImage* icp=cvCreateImage(cvGetSize(src),src->depth,1);
  IplImage* cpp=cvCreateImage(cvGetSize(src),src->depth,1);
  OI10nC10n(src,icp);
  OCoarsePoints(icp,cpp);
  // cvNamedWindow("DCT",CV_WINDOW_AUTOSIZE);
  // cvMoveWindow("DCT",100,100);
  // cvShowImage("DCT",icp);
  if (cvWaitKey(0)>=0)
  {
    cvReleaseImage(&icp);
    cvReleaseImage(&cpp);
    cvDestroyAllWindows();
  }
  return 0;
}
