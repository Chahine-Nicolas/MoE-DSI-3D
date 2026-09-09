FIND_LIBRARY(LAS_LIBRARIES liblas.so ./extern/LAStools/build/)

if ( LAS_LIBRARIES )
 MESSAGE(STATUS "LAS OK")
endif ( LAS_LIBRARIES )

