#ifndef MYCV_H
#define MYCV_H
typedef unsigned char uchar;
template<class T> class Image
{
  private:
  IplImage* imgp;
  public:
  Image(IplImage* img=0) {imgp=img;}
  ~Image(){imgp=0;}
  void operator=(IplImage* img) {imgp=img;}
  inline T* operator[](const int rowIndx) {
    return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));}
}; 
 
typedef struct{
  unsigned char b,g,r;
} RgbPixel; 
 
typedef struct{
  float b,g,r;
} RgbPixelFloat; 
 
typedef Image<RgbPixel>       RgbImage;
typedef Image<RgbPixelFloat>  RgbImageFloat;
typedef Image<unsigned char>  BwImage;
typedef Image<float>          BwImageFloat;

struct CrossPoint
{
  CvPoint v0,v1,h0,h1;
  CrossPoint (CvPoint p1,CvPoint p2,CvPoint p3,CvPoint p4)
  {
    v0=p1;v1=p2;h0=p3;h1=p4;
  }
  CrossPoint ()
  {
  }
  CrossPoint &operator+=(const CvPoint tr)
  {
    v0.x+=tr.x;v0.y+=tr.y;
    v1.x+=tr.x;v1.y+=tr.y;
    h0.x+=tr.x;h0.y+=tr.y;
    h1.x+=tr.x;h1.y+=tr.y;
    return *this;
  }
};

#endif
