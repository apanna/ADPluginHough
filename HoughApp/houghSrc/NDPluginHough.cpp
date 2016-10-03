/*
 * NDPluginHough.cpp
 *
 * Detects circles or lines in images 
 * Author: Alireza Panna NIH/NHLBI/IPL
 *
 * Created September 25, 2016
 *
 * Change Log:
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <epicsString.h>
#include <epicsMutex.h>
#include <iocsh.h>

#include "NDArray.h"
#include "NDPluginHough.h"
#include <epicsExport.h>

#include <opencv2/opencv.hpp>

static const char *driverName = "NDPluginHough";
/* Enums to describe the types of transform */
typedef enum 
{
  None,
  Circles,
  Lines,
} NDPluginHoughType_;
/** Callback function that is called by the NDArray driver with new NDArray data.
  * Detects circles/lines via hough transform.
  * \param[in] pArray  The NDArray from the callback.
  */
void NDPluginHough::processCallbacks(NDArray *pArray)
{
  /* This function does array processing.
   * It is called with the mutex already locked.  It unlocks it during long calculations when private
   * structures don't need to be protected.
   */
  NDArray *pScratch=NULL;
  NDArrayInfo arrayInfo;

  int minRadius, maxRadius, detected, houghType, dataType, arrayCallbacks = 0;
  double minDistance, minLineLength, maxLineGap, param1, param2;
  unsigned int numRows, rowSize;
  unsigned char *inData, *outData;

  static const char* functionName = "processCallbacks";
  /* Call the base class method */
  NDPluginDriver::processCallbacks(pArray);
  // This plugin only works with 1-D or 2-D arrays
  if (pArray->ndims > 2 || pArray->ndims < 0)
  {
    asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
        "%s::%s: error, number of array dimensions must be 1 or 2\n",
        driverName, functionName);
    return;
  }
  /* Get user inputs */
  getDoubleParam(NDPluginHoughMinDistance,    &minDistance);
  getIntegerParam(NDPluginHoughMinRadius,     &minRadius);
  getIntegerParam(NDPluginHoughMaxRadius,     &maxRadius);
  getDoubleParam(NDPluginHoughParam1,         &param1);
  getDoubleParam(NDPluginHoughParam2,         &param2);
  getDoubleParam(NDPluginHoughMinLineLength,  &minLineLength);
  getDoubleParam(NDPluginHoughMaxLineGap,     &maxLineGap);
  getIntegerParam(NDPluginHoughHoughType_,    &houghType);
  /* Do the computationally expensive code with the lock released */
  this->unlock();
  /* Convert to UInt8 */
  this->pNDArrayPool->convert(pArray, &pScratch, NDUInt8);
  /* Get array info */
  pScratch->getInfo(&arrayInfo);
  rowSize = pScratch->dims[arrayInfo.xDim].size;
  numRows = pScratch->dims[arrayInfo.yDim].size;
  /* Opencv definitions */
  cv::Mat img = cv::Mat(numRows, rowSize, CV_8UC1);
  cv::Mat bImg;
  cv::Mat cImg;
  std::vector<cv::Vec3f> detected_circles;
  std::vector<cv::Vec4i> lines;
  // Initialize the output data array
  inData  = (unsigned char *)pScratch->pData;
  outData = (unsigned char *)img.data;
  memcpy(outData, inData, arrayInfo.nElements * sizeof(*inData));

  switch (houghType)
  {
    case(None):
      break;

    case(Circles):
      try 
      {
        // blur the image to avoid false circle detections
        cv::GaussianBlur(img, img, cv::Size(3, 3), 0, 0);
      }
      catch(cv::Exception &e)
      {
        const char* err_msg = e.what();
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR, "%s::%s cv::GaussianBlur exception: %s\n", 
                  driverName, functionName, err_msg);
        this->lock();
        return;
      }

      try 
      { 
        /*
          HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
          src_gray: Input image (grayscale)
          circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
          CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
          dp = 1: The inverse ratio of resolution
          min_dist = src_gray.rows/8: Minimum distance between detected centers
          param_1 = 200: Upper threshold for the internal Canny edge detector
          param_2 = 100*: Threshold for center detection.
          min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
          max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
        */
        cv::HoughCircles(img, detected_circles, CV_HOUGH_GRADIENT, 1, minDistance, param1, param2, minRadius, maxRadius);
        detected = detected_circles.size();
        setIntegerParam(NDPluginHoughDetected,     (int) detected);

      }
      catch(cv::Exception &e) 
      {
        const char* err_msg = e.what();
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR, "%s::%s cv::HoughCircles exception: %s\n", 
                  driverName, functionName, err_msg);
        this->lock();
        return;
      }
      // Draw the detected circles
      for(size_t i = 0; i < detected_circles.size(); i++)
      {
        cv::Point center(cvRound(detected_circles[i][0]), cvRound(detected_circles[i][1]));
        int radius = cvRound(detected_circles[i][2]);
        // cv::circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
        // cv::Scalar(b, g, r)
        cv::circle(img, center, 3, cv::Scalar(0, 0, 255), -1, 8, 0);      // draw circle center
        cv::circle(img, center, radius, cv::Scalar(0, 0, 0), 2, 8, 0);    // draw circle outline
      }
      break;

    case(Lines):
      try 
      {
        cv::Canny(img, cImg, 50, 150, 3);     
      }
      catch(cv::Exception &e)
      {
        const char* err_msg = e.what();
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR, "%s::%s cv::Canny exception:  %s\n", 
                  driverName, functionName, err_msg);
        this->lock();
        return;
      }

      try 
      { 
        /*
          cv::HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0)
          Parameters: 
          image – 8-bit, single-channel binary source image. The image may be modified by the function.
          lines – Output vector of lines. Each line is represented by a 4-element vector  (x_1, y_1, x_2, y_2) , where  (x_1,y_1) and  (x_2, y_2) are the ending points of each detected line segment.
          rho – Distance resolution of the accumulator in pixels.
          theta – Angle resolution of the accumulator in radians.
          threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
          minLineLength – Minimum line length. Line segments shorter than that are rejected.
          maxLineGap – Maximum allowed gap between points on the same line to link them.
        */
        cv::HoughLinesP(cImg, lines, 1, CV_PI/180, 50, minLineLength, maxLineGap);
        detected = lines.size();
        setIntegerParam(NDPluginHoughDetected,     (int) detected);
      }
      catch(cv::Exception &e) 
      {
        const char* err_msg = e.what();
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR, "%s::%s cv::HoughLinesP exception: %s\n", 
                  driverName, functionName, err_msg);
        this->lock();
        return;
      }

      for(size_t i = 0; i < lines.size(); i++)
      {
        cv::Vec4i l = lines[i];
        line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 0), 5, CV_AA);
      }
      break;

    default:
      break;
  }
  // Take the lock again since we are accessing the parameter library and 
  // these calculations are not time consuming
  this->lock();
  getIntegerParam(NDArrayCallbacks, &arrayCallbacks);
  getIntegerParam(NDDataType,     &dataType);
  if (arrayCallbacks == 1) {
    inData  = (unsigned char *)img.data;
    outData = (unsigned char *)pScratch->pData;
    memcpy(outData, inData, arrayInfo.nElements * sizeof( *inData));
    this->pNDArrayPool->convert(pScratch, &pScratch, (NDDataType_t)dataType);
    this->getAttributes(pScratch->pAttributeList);
    this->unlock();
    doCallbacksGenericPointer(pScratch, NDArrayData, 0);
    this->lock();
  }

  if (NULL != pScratch)
    pScratch->release();

  callParamCallbacks();
}


/** Constructor for NDPluginHough; most parameters are simply passed to NDPluginDriver::NDPluginDriver.
  * After calling the base class constructor this method sets reasonable default values for all of the
  * parameters.
  * \param[in] portName The name of the asyn port driver to be created.
  * \param[in] queueSize The number of NDArrays that the input queue for this plugin can hold when
  *            NDPluginDriverBlockingCallbacks=0.  Larger queues can decrease the number of dropped arrays,
  *            at the expense of more NDArray buffers being allocated from the underlying driver's NDArrayPool.
  * \param[in] blockingCallbacks Initial setting for the NDPluginDriverBlockingCallbacks flag.
  *            0=callbacks are queued and executed by the callback thread; 1 callbacks execute in the thread
  *            of the driver doing the callbacks.
  * \param[in] NDArrayPort Name of asyn port driver for initial source of NDArray callbacks.
  * \param[in] NDArrayAddr asyn port driver address for initial source of NDArray callbacks.
  * \param[in] maxBuffers The maximum number of NDArray buffers that the NDArrayPool for this driver is
  *            allowed to allocate. Set this to -1 to allow an unlimited number of buffers.
  * \param[in] maxMemory The maximum amount of memory that the NDArrayPool for this driver is
  *            allowed to allocate. Set this to -1 to allow an unlimited amount of memory.
  * \param[in] priority The thread priority for the asyn port driver thread if ASYN_CANBLOCK is set in asynFlags.
  * \param[in] stackSize The stack size for the asyn port driver thread if ASYN_CANBLOCK is set in asynFlags.
  */
NDPluginHough::NDPluginHough(const char *portName, int queueSize, int blockingCallbacks,
                         const char *NDArrayPort, int NDArrayAddr,
                         int maxBuffers, size_t maxMemory,
                         int priority, int stackSize)
    /* Invoke the base class constructor */
    : NDPluginDriver(portName, queueSize, blockingCallbacks,
                   NDArrayPort, NDArrayAddr, 1, NUM_NDPLUGIN_HOUGH_PARAMS, maxBuffers, maxMemory,
                   asynInt32ArrayMask | asynFloat64ArrayMask | asynGenericPointerMask,
                   asynInt32ArrayMask | asynFloat64ArrayMask | asynGenericPointerMask,
                   ASYN_MULTIDEVICE, 1, priority, stackSize)
{
  static const char *functionName = "NDPluginHough::NDPluginHough";
  // printf("In function: %s\n", functionName);
  createParam(NDPluginHoughMinDistanceString,   asynParamFloat64,  &NDPluginHoughMinDistance);
  createParam(NDPluginHoughMinRadiusString,     asynParamInt32,    &NDPluginHoughMinRadius);
  createParam(NDPluginHoughMaxRadiusString,     asynParamInt32,    &NDPluginHoughMaxRadius);
  createParam(NDPluginHoughParam1String,        asynParamFloat64,  &NDPluginHoughParam1);
  createParam(NDPluginHoughParam2String,        asynParamFloat64,  &NDPluginHoughParam2);
  createParam(NDPluginHoughDetectedString,      asynParamInt32,    &NDPluginHoughDetected);
  createParam(NDPluginHoughMinLineLengthString, asynParamFloat64,  &NDPluginHoughMinLineLength);
  createParam(NDPluginHoughMaxLineGapString,    asynParamFloat64,  &NDPluginHoughMaxLineGap);
  createParam(NDPluginHoughHoughTypeString,     asynParamInt32,    &NDPluginHoughHoughType_);
  /* Set the plugin type string */
  setStringParam(NDPluginDriverPluginType, "NDPluginHough");
  
  /* Try to connect to the array port */
  connectToArrayPort();
}

/** Configuration command */
extern "C" int NDHoughConfigure(const char *portName, int queueSize, int blockingCallbacks,
                                 const char *NDArrayPort, int NDArrayAddr,
                                 int maxBuffers, size_t maxMemory,
                                 int priority, int stackSize)
{
    new NDPluginHough(portName, queueSize, blockingCallbacks, NDArrayPort, NDArrayAddr,
                        maxBuffers, maxMemory, priority, stackSize);
    return(asynSuccess);
}

/* EPICS iocsh shell commands */
static const iocshArg initArg0 = { "portName",iocshArgString};
static const iocshArg initArg1 = { "frame queue size",iocshArgInt};
static const iocshArg initArg2 = { "blocking callbacks",iocshArgInt};
static const iocshArg initArg3 = { "NDArrayPort",iocshArgString};
static const iocshArg initArg4 = { "NDArrayAddr",iocshArgInt};
static const iocshArg initArg5 = { "maxBuffers",iocshArgInt};
static const iocshArg initArg6 = { "maxMemory",iocshArgInt};
static const iocshArg initArg7 = { "priority",iocshArgInt};
static const iocshArg initArg8 = { "stackSize",iocshArgInt};
static const iocshArg * const initArgs[] = {&initArg0,
                                            &initArg1,
                                            &initArg2,
                                            &initArg3,
                                            &initArg4,
                                            &initArg5,
                                            &initArg6,
                                            &initArg7,
                                            &initArg8};
static const iocshFuncDef initFuncDef = {"NDHoughConfigure",9,initArgs};
static void initCallFunc(const iocshArgBuf *args)
{
    NDHoughConfigure(args[0].sval, args[1].ival, args[2].ival,
                       args[3].sval, args[4].ival, args[5].ival,
                       args[6].ival, args[7].ival, args[8].ival);
}

extern "C" void NDHoughRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" {
epicsExportRegistrar(NDHoughRegister);
}