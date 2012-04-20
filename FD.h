#ifndef FD_H
#define FD_H

vector<CvRect> OFaceDetect(IplImage *src,IplImage *dst);
void OFDRelease();
void OFDInit(IplImage *src);

#endif
