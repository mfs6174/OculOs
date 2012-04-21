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
  cas.load("eyes0.xml");
  //cvNamedWindow("lo",CV_WINDOW_AUTOSIZE);
  //cvMoveWindow("lo",500,100);
}


void OFineLocate(IplImage *src,IplImage *dst,IplImage *mask,bool flag)
{
  int w,h,x,y;
  int minw=(int)(src->width/2.35),minh=(int)(minw*0.65);
  cvCopy(src,dst);
  BwImage sh0(mask);
  Mat sROI;
  vector<Rect> leyes,reyes;
  Point ctr,lrp,rrp;
  Scalar color=Scalar(0);
  CvRect eal,ear;
  leyes.clear();
  reyes.clear();
  int lp[5]={10000,-1,10000,-1},rp[5]={10000,-1,10000,-1};
  for (int i=0;i<src->height/3;i++)
    for (int j=0;j<src->width/2;j++)
      if (sh0[i][j])
      {
        lp[0]=min(lp[0],i);
        lp[1]=max(lp[1],i);
        lp[2]=min(lp[2],j);
        lp[3]=max(lp[3],j);
      }
  w=(int)((lp[3]-lp[2])*1.5);
  h=(int)((lp[1]-lp[0])*1.5);
  if (w<minw)
    w=minw;
  if (h<minh)
    h=minh;
  x=(lp[2]+lp[3])/2-w/2;
  if (x<0) x=0;
  y=(lp[0]+lp[1])/2-h/2;
  if (y<0) y=0;
  eal=cvRect(x,y,w,h);
  cvSetImageROI(src,eal);
  sROI=Mat(src);
  cas.detectMultiScale( sROI, leyes,
                        1.25, 1, 0
                        //|CV_HAAR_FIND_BIGGEST_OBJECT
                        //|CV_HAAR_DO_ROUGH_SEARCH
                        //|CV_HAAR_DO_CANNY_PRUNING
                        |CV_HAAR_SCALE_IMAGE
                        ,
                        Size(15, 15) );
  for (int i=0;i<src->height/3;i++)
    for (int j=src->width/2+1;j<src->width;j++)
      if (sh0[i][j])
      {
        rp[0]=min(rp[0],i);
        rp[1]=max(rp[1],i);
        rp[2]=min(rp[2],j);
        rp[3]=max(rp[3],j);
      }
  
  w=(int)((rp[3]-rp[2])*1.5);
  h=(int)((rp[1]-rp[0])*1.5);
  if (w<minw)
    w=minw;
  if (h<minh)
    h=minh;
  x=(rp[2]+rp[3])/2-w/2;
  if (x<0) x=0;
  y=(rp[0]+rp[1])/2-h/2;
  if (y<0) y=0;
  ear=cvRect(x,y,w,h);
  cvSetImageROI(src,cvRect(x,y,w,h));
  sROI=Mat(src);
  cas.detectMultiScale( sROI, reyes,
                        1.25, 1, 0
                        |CV_HAAR_FIND_BIGGEST_OBJECT
                        //|CV_HAAR_DO_ROUGH_SEARCH
                        //|CV_HAAR_DO_CANNY_PRUNING
                        |CV_HAAR_SCALE_IMAGE
                        ,
                        Size(15, 15) );
  cvResetImageROI(dst);
  cvRectangle(dst,cvPoint(eal.x,eal.y),cvPoint(eal.x+eal.width,eal.y+eal.height),color);
  cvRectangle(dst,cvPoint(ear.x,ear.y),cvPoint(ear.x+ear.width,ear.y+ear.height),color);
  if (leyes.size()==1)
  {
    cvLine(dst,cvPoint(eal.x+leyes[0].x,eal.y+leyes[0].y+leyes[0].height/2),cvPoint(eal.x+leyes[0].x+leyes[0].width,eal.y+leyes[0].y+leyes[0].height/2),color,2);
    cvLine(dst,cvPoint(eal.x+leyes[0].x+leyes[0].width/2,eal.y+leyes[0].y),cvPoint(eal.x+leyes[0].x+leyes[0].width/2,eal.y+leyes[0].y+leyes[0].height),color,1);
  }
  if (reyes.size()==1)
  {
    cvLine(dst,cvPoint(ear.x+reyes[0].x,ear.y+reyes[0].y+reyes[0].height/2),cvPoint(ear.x+reyes[0].x+reyes[0].width,ear.y+reyes[0].y+reyes[0].height/2),color,2);
    cvLine(dst,cvPoint(ear.x+reyes[0].x+reyes[0].width/2,ear.y+reyes[0].y),cvPoint(ear.x+reyes[0].x+reyes[0].width/2,ear.y+reyes[0].y+reyes[0].height),color,1);
  }
}
