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
  cvAddS(logp,cvScalar(1.0),logp);
  cvLog(logp,logp);
  cvDCT(logp,dctp,CV_DXT_FORWARD);
  cvNamedWindow("DCT",CV_WINDOW_AUTOSIZE);
  cvMoveWindow("DCT",100,100);
  BwImageFloat fsh(dctp);
  double Rf=min(dctp->height,dctp->width)/64.0;
  Rf=Rf*Rf;
  double rr=0,conv=5.54518*sqrt(dctp->height*dctp->width);
  for (int i=0;i<dctp->height;i++)
  {
    if (i*i>Rf)
      break;
    for (int j=0;j<dctp->width;j++)
    {
      rr=i*i+j*j;
      if (rr>Rf)
        break;
      fsh[i][j]=0;
    }
  }
  cvDCT(dctp,logp,CV_DXT_INVERSE);
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

  
  
