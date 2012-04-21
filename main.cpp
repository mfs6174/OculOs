#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<vector>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

#include "mycv.h"
#include "IC.h"
#include "CL.h"
#include "FD.h"
const int maxlongint=2147483647;

int main(int argc, char *argv[])
{
  IplImage* src=NULL;
  if (argc<2)
  {
    CvCapture* capture = NULL;
	IplImage* frame = NULL;
    IplImage* facep=NULL;
    vector<CvRect> flist;
	capture = cvCaptureFromCAM(0);
    if (!cvGrabFrame(capture))
    {
      cout<<"can not capture"<<endl;
      exit(0);
    }
    frame=cvQueryFrame(capture);
    facep=cvCreateImage(cvGetSize(frame),frame->depth,3);
    OFDInit(frame);
    cvNamedWindow("cam0",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("cam0",0,100);
    cvNamedWindow("procam",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("procam",700,100);
    while (true)
    {
      frame=cvQueryFrame(capture);
      flist=OFaceDetect(frame,facep);
      for (int i=0;i<flist.size();i++)
        cvRectangle(facep,cvPoint(flist[i].x,flist[i].y),cvPoint(flist[i].x+flist[i].width,flist[i].y+flist[i].height),CV_RGB(255,0,0));
      cvShowImage("cam0",frame);
      cvShowImage("procam",facep);
      if (cvWaitKey(20)>=0)
        break;
	}
    cvReleaseCapture(&capture);
    OFDRelease();
  }
  else
  {
    src=cvLoadImage(argv[1],CV_LOAD_IMAGE_ANYCOLOR);
    if(!src)
      cout<<"Could not load image file: "<<argv[1]<<endl;
    IplImage* icp=cvCreateImage(cvGetSize(src),src->depth,1);
    IplImage* cpp=cvCreateImage(cvGetSize(src),src->depth,1);
    IplImage* dst=cvCreateImage(cvGetSize(src),src->depth,1);
    OI10nC10n(src,icp);
    OCoarsePoints(icp,cpp);
    OFineLocate(icp,dst,cpp);
    cvNamedWindow("DCT",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("DCT",300,100);
    cvShowImage("DCT",dst);
    if (cvWaitKey(0)>=0)
    {
      cvReleaseImage(&icp);
      cvReleaseImage(&cpp);
      cvReleaseImage(&dst);
    }
  }
  cvDestroyAllWindows();
  return 0;
}
