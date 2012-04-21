#include<iostream>
#include<cstring>
#include<vector>
#include<cmath>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

#include "mycv.h"
#include "FL.h"

CascadeClassifier cas;

void OFLInit()
{
  cas.load("eyes.xml");
}


void OFineLocate(IplImage *src,IplImage *dst,IplImage *mask)
{
  cvCopy(src,dst);
  BwImage sh0(mask);
  Mat sROI;
  vector<Rect> leyes,reyes;
  Point ctr,lrp,rrp;
  Scalar color=Scalar(255);
  leyes.clear();
  reyes.clear();
  int lp[5],rp[5];
  for (int i=0;i<src->height/3;i++)
    for (int j=0;j<src->width/2;j++)
    {
      
    }
  cas.detectMultiScale( sROI, leyes,
                        1.25, 3, 0
                        //|CV_HAAR_FIND_BIGGEST_OBJECT
                        //|CV_HAAR_DO_ROUGH_SEARCH
                        //|CV_HAAR_DO_CANNY_PRUNING
                        |CV_HAAR_SCALE_IMAGE
                        ,
                        Size(15, 15) );
  for (int i=0;i<src->height/3;i++)
    for (int j=src->width/2+1;j<src->width;j++)
    {
    }
  cas.detectMultiScale( sROI, reyes,
                        1.25, 3, 0
                        //|CV_HAAR_FIND_BIGGEST_OBJECT
                        //|CV_HAAR_DO_ROUGH_SEARCH
                        //|CV_HAAR_DO_CANNY_PRUNING
                        |CV_HAAR_SCALE_IMAGE
                        ,
                        Size(15, 15) );
  
  
}
