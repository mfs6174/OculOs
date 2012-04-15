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
  IplImage* src=0,dst=0,logp=0,dctp=0; 
  img=cvLoadImage(argv[1]);
  if(!img)
    cout<<"Could not load image file: "<<argv[1]<<endl;
  dst=cvCreateImage(cvGetSize(src),src->depth,1);
  logp=cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
  dctp=cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
  cvLog(src,logp);
  cvDCT(logp,dctp,CV_DXT_FORWARD);
  
  
