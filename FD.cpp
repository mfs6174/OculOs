#include<iostream>
#include<cstring>
#include<vector>
#include<cmath>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

#include "mycv.h"
#include "FD.h"

const double a0=0.01;
CvMat* mr;
CvMat* mg;
CvMat* mb;
CvMat* mrt;
CvMat* mgt;
CvMat* mt;
CvMat* mtt;
CvMat* mi;
CvMat* mii;
CvMat* OZ;
CvMat* OE;
CvMat* OP;
double kr,kg,kb;
inline int trimin(uchar x,uchar y,uchar z)
{
  if (x>y)
    x=y;
  if (x<z)
    return x;
  else
    return z;
}

inline double sqr(double x)
{
  return x*x;
}
  


inline void GetE(IplImage *src)
{
  int i,j;
  double r=0,g=0,b=0;
  int rr=0,gg=0,bb=0,nn=src->height*src->width;
  int mm=(int)(nn*a0);
  RgbImage ssh(src);
  int cnt[257],rsum[257],gsum[257],bsum[257],t;
  memset(cnt,0,sizeof(cnt));
  memset(rsum,0,sizeof(rsum));
  memset(gsum,0,sizeof(gsum));
  memset(bsum,0,sizeof(bsum));
  for (i=0;i<src->height;i++)
    for (j=0;j<src->width;j++)
    {
      cnt[t=trimin(ssh[i][j].r,ssh[i][j].g,ssh[i][j].b)]++;
      rsum[t]+=ssh[i][j].r;
      gsum[t]+=ssh[i][j].g;
      bsum[t]+=ssh[i][j].b;
      rr+=ssh[i][j].r;
      gg+=ssh[i][j].g;
      bb+=ssh[i][j].b;
    }
  kr=rr/nn;kg=gg/nn;kb=bb/nn;
  double ag=(kr+kg+kb)/3;
  kr=ag/kr;kg=ag/kg;kb=ag/kb;
  int sum=0;
  for (i=255;i>0;i--)
  {
    sum+=cnt[i];
    if (sum>mm)
      break;
    r+=rsum[i];
    g+=gsum[i];
    b+=bsum[i];
  }
  //cout<<r/mm<<endl;
  CV_MAT_ELEM(*OE,float,0,0)=r/mm;
  CV_MAT_ELEM(*OE,float,0,1)=g/mm;
  CV_MAT_ELEM(*OE,float,0,2)=b/mm;
 
}

inline void WhiteBalance(IplImage *src,IplImage *dst,CvMat *mt,double b)
{
  RgbImage ssh(src);
  RgbImage dsh(dst);
  int i,j;
  for (i=0;i<src->height;i++)
    for (j=0;j<src->width;j++)
    {
      // CV_MAT_ELEM(*mii,float,0,0)=ssh[i][j].r;
      // CV_MAT_ELEM(*mii,float,0,1)=ssh[i][j].g;
      // CV_MAT_ELEM(*mii,float,0,2)=ssh[i][j].b;
      // cvMatMul(mii,mt,mi);
      // for (int k=0;k<3;k++)
      // {
      //   CV_MAT_ELEM(*mi,float,0,k)*=b;
      //   if (CV_MAT_ELEM(*mi,float,0,k)<0)
      //     CV_MAT_ELEM(*mi,float,0,k)=0;
      //   if (CV_MAT_ELEM(*mi,float,0,k)>255)
      //     CV_MAT_ELEM(*mi,float,0,k)=255;
      // }
      // dsh[i][j].r=CV_MAT_ELEM(*mi,float,0,0);
      // dsh[i][j].g=CV_MAT_ELEM(*mi,float,0,1);
      // dsh[i][j].b=CV_MAT_ELEM(*mi,float,0,2);
      int r,g,b;
      r=ssh[i][j].r*kr;
      g=ssh[i][j].g*kg;
      b=ssh[i][j].b*kb;
      if (r>255)
        dsh[i][j].r=255;
      else
        dsh[i][j].r=r;
      if (g>255)
        dsh[i][j].g=255;
      else
        dsh[i][j].g=g;
      if (b>255)
        dsh[i][j].b=255;
      else
        dsh[i][j].b=b;
      
    }
}

void OFDInit()
{
  mr=cvCreateMat(3,3,CV_32FC1);
  mg=cvCreateMat(3,3,CV_32FC1);
  mb=cvCreateMat(3,3,CV_32FC1);
  mrt=cvCreateMat(3,3,CV_32FC1);
  mgt=cvCreateMat(3,3,CV_32FC1);
  mt=cvCreateMat(3,3,CV_32FC1);
  mtt=cvCreateMat(3,3,CV_32FC1);
  mi=cvCreateMat(1,3,CV_32FC1);
  mii=cvCreateMat(1,3,CV_32FC1);
  OZ=cvCreateMat(1,3,CV_32FC1);
  OP=cvCreateMat(1,3,CV_32FC1);
  OE=cvCreateMat(1,3,CV_32FC1);
  CV_MAT_ELEM(*OP,float,0,0)=255;
  CV_MAT_ELEM(*OP,float,0,1)=255;
  CV_MAT_ELEM(*OP,float,0,2)=255;
  cvSetZero(mr);
  cvSetZero(mg);
  cvSetZero(mb);
  CV_MAT_ELEM(*mr,float,0,0)=1;
  CV_MAT_ELEM(*mg,float,1,1)=1;
  //唉话说我现在已经被迫有在打dota的舍友附近写代码的能力了
  CV_MAT_ELEM(*mb,float,2,2)=1;
}

void OFDRelease()
{
  cvReleaseMat(&mr);
  cvReleaseMat(&mg);
  cvReleaseMat(&mb);
  cvReleaseMat(&mrt);
  cvReleaseMat(&mgt);
  cvReleaseMat(&mt);
  cvReleaseMat(&mtt);
  cvReleaseMat(&mi);
  cvReleaseMat(&mii);
  cvReleaseMat(&OE);
  cvReleaseMat(&OP);
  cvReleaseMat(&OZ);
}

vector<CvRect> OFaceDetect(IplImage *src,IplImage *dst)
{
  //IplImage* wp0=cvCreateImage(cvGetSize(src),src->depth,3);
  vector<CvRect> list;
  list.clear();
  GetE(src);
  cvCrossProduct(OE,OP,OZ);
  double cos,sin,tt;
  //  话说我发现说话的时候如果有一句话打断了没说出来我会直接粘贴在正在写的代码里当注释,挺好玩的..好吧我蛋疼别管我
  tt=sqrt(sqr(CV_MAT_ELEM(*OZ,float,0,1))+sqr(CV_MAT_ELEM(*OZ,float,0,2)));
  cos=CV_MAT_ELEM(*OZ,float,0,2)/tt;
  sin=CV_MAT_ELEM(*OZ,float,0,1)/tt;
  CV_MAT_ELEM(*mr,float,1,1)=cos;
  CV_MAT_ELEM(*mr,float,1,2)=sin;
  CV_MAT_ELEM(*mr,float,2,1)=-sin;
  CV_MAT_ELEM(*mr,float,2,2)=cos;
  double ttoz=cvNorm(OZ);
  cos=tt/ttoz;
  sin=-CV_MAT_ELEM(*OZ,float,0,0)/ttoz;
  CV_MAT_ELEM(*mg,float,0,0)=cos;
  CV_MAT_ELEM(*mg,float,2,0)=sin;
  CV_MAT_ELEM(*mg,float,0,2)=-sin;
  CV_MAT_ELEM(*mg,float,2,2)=cos;
  double ttt=cvNorm(OE)*cvNorm(OP);
  cos=sqrt(cvDotProduct(OE,OP))/ttt;
  sin=ttoz/ttt;
  CV_MAT_ELEM(*mb,float,0,0)=cos;
  CV_MAT_ELEM(*mb,float,0,1)=sin;
  CV_MAT_ELEM(*mb,float,1,0)=-sin;
  CV_MAT_ELEM(*mb,float,1,1)=cos;
  cvT(mg,mgt);
  cvT(mr,mrt);
  cvMatMul(mr,mg,mt);
  cvMatMul(mt,mb,mtt);
  cvMatMul(mtt,mgt,mt);
  cvMatMul(mt,mrt,mtt);
  double beta=cvNorm(OP)/cvNorm(OE);
  WhiteBalance(src,dst,mtt,beta);
  
  return list;
}
