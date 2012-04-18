#include<cmath>
#include<cstring>
#include "opencv2/opencv.hpp"
#include "mycv.h"
#include "CL.h"
using namespace std;
using namespace cv;

const int wdsz=5/2;

inline uchar PeakThresh(int x,int y,IplImage *src)
{
  BwImage srcsh(src);
  if ( x-wdsz<0 || x+wdsz>src->height-1 || y-wdsz<0 || y+wdsz>src->width-1 )
    return 0;
  for (int i=-wdsz;i<=wdsz;i++)
    for (int j=-wdsz;j<=wdsz;j++)
      if (srcsh[x][y]>srcsh[x+i][y+j])
        return 0;
  return 255;
}

void OCoarsePoints(IplImage *src,IplImage *dst)
{
  IplImage* bh=cvCreateImage(cvGetSize(src),src->depth,1);
  //IplImage* th=cvCreateImage(cvGetSize(src),src->depth,1);
  //IplImage* wp0=cvCreateImage(cvGetSize(src),src->depth,1);
  IplConvKernel *cls=cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_RECT,NULL);
  cvMorphologyEx(src,bh,NULL,cls,CV_MOP_BLACKHAT,2);
  //cvMorphologyEx(src,th,NULL,cls,CV_MOP_TOPHAT,2);
  // cvNamedWindow("bh",CV_WINDOW_AUTOSIZE);
  // cvMoveWindow("bh",100,100);
  // cvShowImage("bh",bh);
  //cvAdd(bh,th,bh);
  cvSmooth(bh,bh,CV_GAUSSIAN);
  int histcnt[257];
  memset(histcnt,0,sizeof(histcnt));
  BwImage fsh(bh);
  for (int i=0;i<bh->height;i++)
    for (int j=0;j<bh->width;j++)
      histcnt[fsh[i][j]]++;
  int nc=(int)(bh->height*bh->width*0.027),hsum=0,trhd=0;
  for (int i=255;i>=0;i--)
  {
    hsum+=histcnt[i];
    if (hsum>nc)
    {
      trhd=i;
      break;
    }
  }
  cvThreshold(bh, bh, trhd, 255, CV_THRESH_BINARY);
  BwImage wpsh(dst);
  int bd=bh->height/3;
  for (int i=0;i<bh->height;i++)
    for (int j=0;j<bh->width;j++)
      if (fsh[i][j]>0 && i<=bd)
        wpsh[i][j]=PeakThresh(i,j,src);
      else
        wpsh[i][j]=0;
  cvNamedWindow("th",CV_WINDOW_AUTOSIZE);
  cvMoveWindow("th",200,100);
  cvShowImage("th",dst);
  cvReleaseStructuringElement(&cls);
  cvReleaseImage(&bh);
  //cvReleaseImage(&th);
  //cvReleaseImage(&wp0);
}

  
