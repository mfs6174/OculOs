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
IplImage* wp0;
BwImage fa;

double kr,kg,kb;
//int fa[650][490];
int sdidx;
int cn[10000][4];
int ar[10000];
double qp[11000],qm[11000];
int W,H;
bool flag;
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
  //CV_MAT_ELEM(*OE,float,0,0)=r/mm;
  //CV_MAT_ELEM(*OE,float,0,1)=g/mm;
  //CV_MAT_ELEM(*OE,float,0,2)=b/mm;
 
}

inline void WhiteBalance(IplImage *src,IplImage *dst,CvMat *mt)
{
  BwImage stsh(wp0);
  RgbImage ssh(src);
  int i,j;
  double rr,gg;
  int r,g,b;
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
      r=ssh[i][j].r*kr;
      g=ssh[i][j].g*kg;
      b=ssh[i][j].b*kb;
      if (r>255)
        ssh[i][j].r=255;
      else
        ssh[i][j].r=r;
      if (g>255)
        ssh[i][j].g=255;
      else
        ssh[i][j].g=g;
      if (b>255)
        ssh[i][j].b=255;
      else
        ssh[i][j].b=b;
    }
  cvSmooth(src,dst,CV_MEDIAN);
  RgbImage dsh(dst);
  for (i=0;i<src->height;i++)
    for (j=0;j<src->width;j++)
    {
      double sum=dsh[i][j].r+dsh[i][j].b+dsh[i][j].g;
      rr=dsh[i][j].r/sum;
      gg=dsh[i][j].g/sum;
      stsh[i][j]=0;
      if (rr>0.4 && rr<0.6 && gg<qp[(int)(rr/0.0001)] && gg>qm[(int)(rr/0.0001)])
        if ( sqr(rr-0.33)+sqr(gg-0.33)>0.006 )
        {
          stsh[i][j]=255;
        }
    }
  cvDilate(wp0,wp0,NULL,2);      
}

void FloodFill(int x,int y)
{
  if (flag)
  {
    sdidx++;
    flag=false;
    cn[sdidx][0]=cn[sdidx][1]=x;
    cn[sdidx][2]=cn[sdidx][3]=y;
  }
  fa[x][y]=1;
  ar[sdidx]++;
  cn[sdidx][0]=min(cn[sdidx][0],x);
  cn[sdidx][1]=max(cn[sdidx][1],x);
  cn[sdidx][2]=min(cn[sdidx][2],y);
  cn[sdidx][3]=max(cn[sdidx][3],y);
  if ( fa[x+1][y]==255 && x+1<H )
    FloodFill(x+1,y);
  if (fa[x-1][y]==255 && x-1>=0 )
    FloodFill(x-1,y);
  if (fa[x][y+1]==255 && y+1<W)
    FloodFill(x,y+1);
  if (fa[x][y-1]==255 && y-1>=0)
    FloodFill(x,y-1);
  if ( fa[x+1][y+1]==255 && x+1<H &&y+1<W )
    FloodFill(x+1,y+1);
  if (fa[x-1][y-1]==255 && x-1>=0 && y-1>=0 )
    FloodFill(x-1,y-1);
  if (fa[x-1][y+1]==255 && y+1<W && x-1>=0)
    FloodFill(x-1,y+1);
  if (fa[x+1][y-1]==255 && y-1>=0 && x+1<H)
    FloodFill(x+1,y-1);
}

void OFDInit(IplImage *src)
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
  CV_MAT_ELEM(*mb,float,2,2)=1;
  wp0=cvCreateImage(cvGetSize(src),src->depth,1);
  fa=BwImage(wp0);
  W=src->width;
  H=src->height;
  for (int i=0;i<=10000;i++)
  {
    double rr=i*0.0001;
    qp[i]=(-1.3767*sqr(rr)+1.0743*rr+0.1452);
    qm[i]=(-0.776*sqr(rr)+0.5601*rr+0.1766);
  }
  //cvNamedWindow("procam",CV_WINDOW_AUTOSIZE);
  //cvMoveWindow("procam",700,100);
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
  cvReleaseImage(&wp0);
}

vector<CvRect> OFaceDetect(IplImage *src,IplImage *dst)
{
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
  //memset(fa,0,sizeof(fa));
  WhiteBalance(src,dst,mtt);
  //cvShowImage("procam",wp0);
  sdidx=0;
  memset(ar,0,sizeof(ar));
  //memset(cn,0,sizeof(cn));
  for (int i=0;i<src->height;i++)
    for (int j=0;j<src->width;j++)
      if (fa[i][j]==255)
      {
        flag=true;
        FloodFill(i,j);
      }
  for (int i=1;i<=sdidx;i++)
  {
    int w=(cn[i][3]-cn[i][2])&(~1),h=(cn[i][1]-cn[i][0])&(~1);
    if (ar[i]>=10000 && h<=2.5*w && h>=0.4*w) //&& w*h>=14400)
      list.push_back(cvRect(cn[i][2],cn[i][0],w,h));
  }
  return list;
}
