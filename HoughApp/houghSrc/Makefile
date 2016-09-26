TOP=../..
include $(TOP)/configure/CONFIG
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

# The following gets rid of the -fno-implicit-templates flag on vxWorks, 
# so we get automatic template instantiation.
# This is what we want for the source files that define template functions.
ifeq (vxWorks,$(findstring vxWorks, $(T_A)))
CODE_CXXFLAGS=
endif

DBD += NDPluginHough.dbd

INC += NDPluginHough.h

LIBRARY_IOC += NDPluginHough

NDPluginHough_SRCS += NDPluginHough.cpp

USR_INCLUDES += $(OPENCV_INCLUDE)
ifdef OPENCV_LIB
  NDPluginHough_DIR += $(OPENCV_LIB)
endif

include $(AREA_DETECTOR)/ADCore/ADApp/commonLibraryMakefile

include $(TOP)/configure/RULES
#----------------------------------------
#  ADD RULES AFTER THIS LINE

