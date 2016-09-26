#ifndef NDPluginHough_H
#define NDPluginHough_H

#include "NDPluginDriver.h"

#define NDPluginHoughMinDistanceString       "MIN_DISTANCE"     /* (asynFloat64, r/w)  */
#define NDPluginHoughMinRadiusString         "MIN_RADIUS"       /* (asynInt32, r/w)  */
#define NDPluginHoughMaxRadiusString         "MAX_RADIUS"       /* (asynInt32, r/w)  */
#define NDPluginHoughParam1String            "PARAM1"           /* (asynFloat64, r/w)  */
#define NDPluginHoughParam2String            "PARAM2"           /* (asynFloat64, r/w)  */
#define NDPluginHoughDetectedString          "DETECTED"         /* (asynInt32, r)  */
#define NDPluginHoughMaxLineGapString        "MAX_LINE_GAP"     /* (asynFloat64, r/w)  */
#define NDPluginHoughMinLineLengthString     "MIN_LINE_LENGTH"  /* (asynFloat64, r/w)  */
#define NDPluginHoughHoughTypeString         "HOUGH_TYPE"       /* (asynInt32, r/w)  */

/** Does image processing operations.
 */
class NDPluginHough : public NDPluginDriver {
public:
    NDPluginHough(const char *portName, int queueSize, int blockingCallbacks, 
                 const char *NDArrayPort, int NDArrayAddr,
                 int maxBuffers, size_t maxMemory,
                 int priority, int stackSize);
    /* These methods override the virtual methods in the base class */
    void processCallbacks(NDArray *pArray);
    
protected:
    /* Min distance between the centers of detected circles */
    int NDPluginHoughMinDistance;
    #define FIRST_NDPLUGIN_HOUGH_PARAM NDPluginHoughMinDistance
    /* Min Radius of the hough circles */
    int NDPluginHoughMinRadius;
    /* Max Radius of the hough circles */
    int NDPluginHoughMaxRadius;
    int NDPluginHoughParam1;
    int NDPluginHoughParam2;
    int NDPluginHoughDetected;
    int NDPluginHoughMinLineLength;
    int NDPluginHoughMaxLineGap;
    /* Type of Hough Detection (None, lines, circles) */
    int NDPluginHoughHoughType_;
    #define LAST_NDPLUGIN_HOUGH_PARAM NDPluginHoughHoughType_
};
#define NUM_NDPLUGIN_HOUGH_PARAMS ((int)(&LAST_NDPLUGIN_HOUGH_PARAM - &FIRST_NDPLUGIN_HOUGH_PARAM + 1))
    
#endif