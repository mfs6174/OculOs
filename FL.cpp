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


int OFineLocate(IplImage *src,IplImage *dst,IplImage *mask,bool flag,CrossPoint cplist[],int para1)
{
  int w,h,x,y;
  int ndiv,nheight,n0;
  if (src->height>src->width*1.7)
  {
    nheight=(int)(src->height);
    ndiv=2;
    n0=src->height/5;
  }
  else
  {
    nheight=(int)(src->height);
    ndiv=2;
    n0=src->height/5;
  }
  int ms=para1;
  int minw=(int)(src->width/2.2),minh=(int)(minw*0.65);
  cvCopy(src,dst);
  BwImage sh0(mask);
  Mat sROI;
  vector<Rect> leyes,reyes;
  Point ctr,lrp,rrp;
  Scalar color=Scalar(0);
  CvRect eal,ear;
  leyes.clear();
  reyes.clear();
  bool hei;
  int lp[5]={10000,-1,10000,-1},rp[5]={10000,-1,10000,-1};
  hei=false;
  for (int i=n0;i<nheight/ndiv;i++)
    for (int j=0;j<src->width/2;j++)
      if (sh0[i][j])
      {
        hei=true;
        lp[0]=min(lp[0],i);
        lp[1]=max(lp[1],i);
        lp[2]=min(lp[2],j);
        lp[3]=max(lp[3],j);
      }
  if (!hei)
    return 0;
  w=(int)((lp[3]-lp[2])*1.35);
  h=(int)((lp[1]-lp[0])*1.25);
  if (w<minw)
    w=minw;
  if (h<minh)
    h=minh;
  x=(lp[2]+lp[3])/2-w/2;
  if (x<0) x=0;
  if (x>=(src->width)) x=src->width-1;
  y=(lp[0]+lp[1])/2-h/2;
  if (y<0) y=0;
  if (y>=(src->height*0.37)) y=src->height*0.37-1;
  eal=cvRect(x,y,w,h);
  cvSetImageROI(src,eal);
  sROI=Mat(src);
  cas.detectMultiScale( sROI, leyes,
                        1.05, ms, 0
                        |CV_HAAR_FIND_BIGGEST_OBJECT
                        //|CV_HAAR_DO_ROUGH_SEARCH
                        //|CV_HAAR_DO_CANNY_PRUNING
                        //|CV_HAAR_SCALE_IMAGE
                        ,
                        Size(15, 15) );
  hei=false;
  for (int i=n0;i<nheight/ndiv;i++)
    for (int j=src->width/2+1;j<src->width;j++)
      if (sh0[i][j])
      {
        hei=true;
        rp[0]=min(rp[0],i);
        rp[1]=max(rp[1],i);
        rp[2]=min(rp[2],j);
        rp[3]=max(rp[3],j);
      }
  if (!hei)
    return 0;
  w=(int)((rp[3]-rp[2])*1.5);
  h=(int)((rp[1]-rp[0])*1.5);
  if (w<minw)
    w=minw;
  if (h<minh)
    h=minh;
  x=(rp[2]+rp[3])/2-w/2;
  if (x<0) x=0;
  if (x>=src->width) x=src->width-1;
  y=(rp[0]+rp[1])/2-h/2;
  if (y<0) y=0;
  if (y>=(src->height*0.37)) y=src->height*0.37-1;
  ear=cvRect(x,y,w,h);
  cvSetImageROI(src,cvRect(x,y,w,h));
  sROI=Mat(src);
  cas.detectMultiScale( sROI, reyes,
                        1.05, ms, 0
                        |CV_HAAR_FIND_BIGGEST_OBJECT
                        //|CV_HAAR_DO_ROUGH_SEARCH
                        //|CV_HAAR_DO_CANNY_PRUNING
                        //|CV_HAAR_SCALE_IMAGE
                        ,
                        Size(15, 15) );
  cvResetImageROI(dst);
  cvRectangle(dst,cvPoint(eal.x,eal.y),cvPoint(eal.x+eal.width,eal.y+eal.height),color);
  cvRectangle(dst,cvPoint(ear.x,ear.y),cvPoint(ear.x+ear.width,ear.y+ear.height),color);
  int hf=0;
  if (leyes.size())
  {
    hf--;
    cplist[0]=CrossPoint(cvPoint(eal.x+leyes[0].x,eal.y+leyes[0].y+leyes[0].height/2),cvPoint(eal.x+leyes[0].x+leyes[0].width,eal.y+leyes[0].y+leyes[0].height/2),
                         cvPoint(eal.x+leyes[0].x+leyes[0].width/2,eal.y+leyes[0].y),cvPoint(eal.x+leyes[0].x+leyes[0].width/2,eal.y+leyes[0].y+leyes[0].height) );
    cvLine(dst,cplist[0].h0,cplist[0].h1,color,2);
    cvLine(dst,cplist[0].v0,cplist[0].v1,color,1);
  }
  if (reyes.size())
  {
    hf+=2;
    cplist[1]=CrossPoint(cvPoint(ear.x+reyes[0].x,ear.y+reyes[0].y+reyes[0].height/2),cvPoint(ear.x+reyes[0].x+reyes[0].width,ear.y+reyes[0].y+reyes[0].height/2),
                         cvPoint(ear.x+reyes[0].x+reyes[0].width/2,ear.y+reyes[0].y),cvPoint(ear.x+reyes[0].x+reyes[0].width/2,ear.y+reyes[0].y+reyes[0].height) );
    cvLine(dst,cplist[1].h0,cplist[1].h1,color,2);
    cvLine(dst,cplist[1].v0,cplist[1].v1,color,1);
  }
  if (hf==1)
    return 2;
  if (hf==0)
  {
    if (!flag)
      return 0;
    int y=max(eal.y+eal.height/2,ear.y+ear.height/2),x1=eal.x+eal.width/2,x2=ear.x+ear.width/2;
    cplist[0]=CrossPoint(cvPoint(x1-8,y),cvPoint(x1+8,y),cvPoint(x1,y-8),cvPoint(x1,y+8));
    cplist[1]=CrossPoint(cvPoint(x2-8,y),cvPoint(x2+8,y),cvPoint(x2,y-8),cvPoint(x2,y+8));
    cvLine(dst,cvPoint(x1-8,y),cvPoint(x1+8,y),color,2);
    cvLine(dst,cvPoint(x1,y-8),cvPoint(x1,y+8),color,1);
    cvLine(dst,cvPoint(x2-8,y),cvPoint(x2+8,y),color,2);
    cvLine(dst,cvPoint(x2,y-8),cvPoint(x2,y+8),color,1);
    return 3;
  }
  if (hf==2)
  {
    int x=eal.x+eal.width/2,y=ear.y+reyes[0].y+reyes[0].height/2;
    cplist[0]=CrossPoint(cvPoint(x-8,y),cvPoint(x+8,y),cvPoint(x,y-8),cvPoint(x,y+8));
    cvLine(dst,cvPoint(x-8,y),cvPoint(x+8,y),color,2);
    cvLine(dst,cvPoint(x,y-8),cvPoint(x,y+8),color,1);
    return 1;
  }
  if (hf==-1)
  {
    int x=ear.x+ear.width/2,y=eal.y+leyes[0].y+leyes[0].height/2;
    cplist[1]=CrossPoint(cvPoint(x-8,y),cvPoint(x+8,y),cvPoint(x,y-8),cvPoint(x,y+8));
    cvLine(dst,cvPoint(x-8,y),cvPoint(x+8,y),color,2);
    cvLine(dst,cvPoint(x,y-8),cvPoint(x,y+8),color,1);
    return 1;
  }
}
