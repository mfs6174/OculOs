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
	IplImage* rframe = NULL;
    IplImage* facep=NULL;
    IplImage *frame=NULL;
    vector<CvRect> flist;
	capture = cvCaptureFromCAM(0);
    if (!cvGrabFrame(capture))
    {
      cout<<"can not capture"<<endl;
      exit(0);
    }
    rframe=cvQueryFrame(capture);
    frame=cvCreateImage(cvSize( (int)(rframe->width/1.4),(int)(rframe->height/1.4) ),rframe->depth,3);
    facep=cvCreateImage(cvGetSize(frame),frame->depth,3);
    OFDInit(frame);
    cvNamedWindow("face1",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("face1",700,100);
    cvNamedWindow("procam",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("procam",0,100);
    while (true)
    {
      rframe=cvQueryFrame(capture);
      cvResize(rframe,frame);
      flist=OFaceDetect(frame,facep);
      for (int i=0;i<flist.size();i++)
      {
        cvSetImageROI(facep,flist[i]);
        IplImage *sgrp=cvCreateImage(cvSize(flist[i].width,flist[i].height),frame->depth,1);
        IplImage *dgrp=cvCreateImage(cvSize(flist[i].width,flist[i].height),frame->depth,1);
        IplImage *cpp=cvCreateImage(cvSize(flist[i].width,flist[i].height),frame->depth,1);
        cvCvtColor(facep,sgrp,CV_BGR2GRAY);
        OI10nC10n(sgrp,sgrp);
        OCoarsePoints(sgrp,cpp);
        int rtr;
        rtr=OFineLocate(sgrp,dgrp,cpp,false);
        //cout<<rtr<<endl;
        cvResetImageROI(facep);
        if (rtr>0)
        {
          cvRectangle(facep,cvPoint(flist[i].x,flist[i].y),cvPoint(flist[i].x+flist[i].width,flist[i].y+flist[i].height),CV_RGB(0,255,0));
          cvShowImage("face1",dgrp);
        }
        else
           cvRectangle(facep,cvPoint(flist[i].x,flist[i].y),cvPoint(flist[i].x+flist[i].width,flist[i].y+flist[i].height),CV_RGB(255,0,0));
         
        cvReleaseImage(&sgrp);
        cvReleaseImage(&dgrp);
        cvReleaseImage(&cpp);
      }
      cvShowImage("procam",facep);
      if (cvWaitKey(1)>=0)
        break;
	}
    cvReleaseImage(&frame);
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
