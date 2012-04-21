#include<cmath>
#include "opencv2/opencv.hpp"
#include "mycv.h"
#include "IC.h"
using namespace std;
using namespace cv;

void OI10nC10n(IplImage *src,IplImage *dst)
{
  IplImage* logp=cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
  IplImage* dctp=cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,1);
  cvScale(src,logp);
  cvAddS(logp,cvScalar(1.0),logp);
  cvLog(logp,logp);
  cvDCT(logp,dctp,CV_DXT_FORWARD);
  BwImageFloat fsh(dctp);
  //double Rf=min(dctp->height,dctp->width)/96.0;
  //Rf=Rf*Rf;
  int ddis=(dctp->height+dctp->width)/64;
  double conv=log(120)*sqrt(dctp->height*dctp->width);
  for (int i=0;i<dctp->height;i++)
  {
    if (i>ddis)
      break;
    for (int j=0;j<dctp->width;j++)
    {
      if (i+j+1>ddis)
        break;
      if (i==0&&j==0)
        fsh[i][j]=conv;
      else
        fsh[i][j]=0;
    }
  }
  cvDCT(dctp,logp,CV_DXT_INVERSE);
  cvNormalize(logp,logp,1,0,CV_MINMAX);
  cvScale(logp,dst,255);
  cvReleaseImage(&logp);
  cvReleaseImage(&dctp);  
}

  
  
