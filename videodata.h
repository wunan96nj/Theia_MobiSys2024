#ifndef _VIDEO_DATA_H_
#define _VIDEO_DATA_H_

#include "vivo_server.h"

struct CHUNK_INFO {
	long offset;
	int frameID;
	int cellID;
	int quality;
	int len;	//includes the 14-byte header and the framesize data, sent to the client through meta data message
	int points;
};

struct VIDEO_DATA {
	static const int MAX_QUALITIES = 6;

	static int nChunks;
	static int nTilesX;
	static int nTilesY;
	static int nTilesZ;
	static int nQualities;
	static int gopSize;		//in frames
	static int tileWidth;	//in pixels
	static int tileHeight;	//in pixels
		
	static void Init();

	//server only 
	static void UnloadVideo();
	static void LoadVideo(const char * name);
		
	static BYTE * GetChunk(int chunkID, int tileID, int quality, int * dataLen);
	static void GetEncodedData(int & encLen, BYTE * & encData);
	
	//client only
	static void DecodeMetaData(BYTE * pData, int metaLen);
	static void DecodeFrameSizes(BYTE * pData);

	static int GetChunkSize(int chunkID, int tileID, int quality);
	static int GetChunkPoints(int chunkID, int tileID, int quality);
	
	static void SwitchToKernelMemory();
public:
	static BYTE * pBatch;	

private:
	//server only 
	static CHUNK_INFO * pMeta;
	static BYTE * pBuf;
	static long bufSize;
	static BYTE * pEncodedMetaData;
	static int encodedMetaDataLen;
			
	static void EncodeMetaData();
	static char * SkipComment(FILE * ifs);
	static void EncodeFrameSizes(BYTE * pData, vector<int> & sizes);
	static void FillInFixedVideoDataHeaders();

	
	static float * bitrateUtilities;

private:
	static const int BITRATE_UTILITY_LINEAR = 1;
	static const int BITRATE_UTILITY_LOG = 2;
	static const int BITRATE_UTILITY_HD = 3;
	//static void ComputeBitrateUtility(int bitrateUtility, float * bu);
	static void ComputeBitrateUtility(int bitrateUtility, float * bu, vector<string> & bitrate);

	//client only
	static int * chunkSizes;


//emulation only
public:
	static void LoadVideo_Emulation(const char * name);
		
private:
	static void LoadVideo_Emulation(CHUNK_INFO * pInfo, int chunkID, int tileID, int quality);
	


};

#endif

