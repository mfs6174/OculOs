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
#include "FL.h"

const int maxlongint=2147483647;

int main(int argc, char *argv[])
{
  IplImage* src=NULL;
  OFLInit();
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
    // cvNamedWindow("cam0",CV_WINDOW_AUTOSIZE);
    // cvMoveWindow("cam0",0,100);
    cvNamedWindow("procam",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("procam",0,100);
    while (true)
    {
      frame=cvQueryFrame(capture);
      flist=OFaceDetect(frame,facep);
      for (int i=0;i<flist.size();i++)
        cvRectangle(facep,cvPoint(flist[i].x,flist[i].y),cvPoint(flist[i].x+flist[i].width,flist[i].y+flist[i].height),CV_RGB(255,0,0));
      //cvShowImage("cam0",frame);
      cvShowImage("procam",facep);
      if (cvWaitKey(1)>=0)
        break;
	}
    cvReleaseCapture(&capture);
    OFDRelease();
  }
  else
  {
    int waittime=0;
    vector<string> pics;
    pics.clear();
    cvNamedWindow("result",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("result",300,100); 
    if (argc==2)
    {
      pics.push_back(string(argv[1]));
      waittime=0;
    }
    else
    {
      ifstream inf(argv[2]);
      string is;
      while (inf>>is)
        pics.push_back(is);
      waittime=1;
    }
    int scnt[5]={0},acnt=0;
    for (int it=0;it<pics.size();it++)
    {
      src=cvLoadImage(pics[it].c_str(),CV_LOAD_IMAGE_ANYCOLOR);
      if(!src)
      {
        cout<<"Could not load image file: "<<argv[1]<<endl;
        continue;
      }
      acnt++;
      IplImage* icp=cvCreateImage(cvGetSize(src),src->depth,1);
      IplImage* cpp=cvCreateImage(cvGetSize(src),src->depth,1);
      IplImage* dst=cvCreateImage(cvGetSize(src),src->depth,1);
      OI10nC10n(src,icp);
      OCoarsePoints(icp,cpp);
      int rtr;
      rtr=OFineLocate(icp,dst,cpp,true);
      //cout<<rtr<<endl;
      scnt[rtr]++;
      cvShowImage("result",dst);
      if (cvWaitKey(waittime)>=0)
      {
        cvReleaseImage(&icp);
        cvReleaseImage(&cpp);
        cvReleaseImage(&dst);
      }
    }
    cout<<"Fail "<<scnt[3]+scnt[0]<<endl;
    cout<<"Success1 "<<scnt[1]<<endl;
    cout<<"Success2 "<<scnt[2]<<endl;
    cout<<"Success "<<scnt[1]+scnt[2]<<'/'<<acnt<<endl;
  }
  cvDestroyAllWindows();
  return 0;
}
