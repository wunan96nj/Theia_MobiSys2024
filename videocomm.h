#ifndef _VIDEO_COMM_H_
#define _VIDEO_COMM_H_

#include "theia_server.h"

struct VIDEO_MESSAGE {
	BYTE id;	
	int msgLen;
	BYTE * pData;
};

//for small messages
struct VIDEO_MESSAGE_STORAGE {
	static const int limit = 1024*1024;
	BYTE * data;

	int size;
	int head;
	int tail;

	VIDEO_MESSAGE_STORAGE();
	~VIDEO_MESSAGE_STORAGE();

	void ReleaseBlock(BYTE * pData, int len);
	BYTE * AllocateBlock(int len);
};

// Structure for a point in 3D space
struct Point {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  int r = 0, g = 0, b = 0;
  float point_size = 0.0f;
  int count = 0;
};

// Structure for the gaze position and direction
struct Gaze {
  Point position;
  Point direction;
};

//messages have a common header of 5 bytes (1B: ID + 4B: tot Len)

//over control channel
#define MSG_SELECT_VIDEO 1  //client->server
//ID (1 byte)                 0
//len (4B) = xx + 11          1
//bwID (2 byte)               5
//damp (2 byte)               7
//tileXMitMode (2 byte)       9
//the video name (xx bytes)   11 //including the trailing 0x0

#define MSG_REQUEST_CHUNK 2
//ID (1 byte)     0
//len (4B) = 15   1
//chunk_id (2B)   5
//tile_id (1B)    7
//quality (1B)    8
//class (1B)      9
//seqnumber (4B) 10
//flag (1B)      14 (reserved for future uses, 0 for now)

#define MSG_VIDEO_METADATA 3
//ID (1 byte)	 0
//len (4B) = xx	 1
//nChunks (4B)	 5
//nTilesX (4B)	 9
//nTilesZ (4B)	 13
//nQualities (4) 17
//gopSize (4)	 21
//tileWidth (4)  25
//tileHeight (4) 29
//encoded meta data 33

#define MSG_BATCH_REQUESTS 4
//ID (1 byte)	 0
//len (4B) = XX  1
//seqnumber (4B) 5
//lists of tiles (5B*x) 9
//{chunk_id 2B, tileID 1B, quality 1B, class 1B}
//{chunk_id 2B, tileID 2B, quality 1B}

//over data channel
#define MSG_VIDEO_DATA 5
//ID (1 byte)     0
//len (4B) = ??   1
//chunk_id (2B)   5
//tile_id (2B)    7
//quality (1B)    9
//seqnumber (4B) 10
//flag (1B)		 14 //fornow, 1=more data coming form the sending queue, 0=end of sending queue (for now)
//encoded frame size (4*gopSize) 15
//actual data    15+4*gopSize

//over data channel
#define MSG_GAZE_BATCH_REQUESTS 6
//ID (1 byte)     0
//len (4B) = ??   1
//chunk_id (2B)   5
//tile_id (2B)    7
//quality (1B)    9
//seqnumber (4B) 10
//flag (1B)		 14 //fornow, 1=more data coming form the sending queue, 0=end of sending queue (for now)
//encoded frame size (4*gopSize) 15
//actual data    15+4*gopSize

#define MSG_VIDEO_DATA_DYNAMIC 7

//for both request and response (payload)
#define CHUNK_HEADER_LEN 15
//#define CHUNK_HEADER_LEN 19

struct TRANSMITTED_TILE_BITMAP {
	static const int STATUS_NOTQUEUED = 0;
	static const int STATUS_QUEUED = 1;
	static const int STATUS_XMITTED = 2;

	BYTE * map;
	
	TRANSMITTED_TILE_BITMAP();
	~TRANSMITTED_TILE_BITMAP();
	
	void Init();
	void MarkTile(int chunkID, int tileID, int status);
	void UnmarkTile(int chunkID, int tileID);
	int TileStatus(int chunkID, int tileID);
};

struct MESSAGE_QUEUE {

	static const int ENQUEUE_ALWAYS	= 1;	//always enqueue 
	static const int ENQUEUE_ONCE = 2;		//enqueue once
	static const int ENQUEUE_UPDATE = 3;	//allow update if the tile is still in the queue

	int head;
	int tail;
	int size;

	VIDEO_MESSAGE * data;
	MESSAGE_QUEUE();
	~MESSAGE_QUEUE();

	void Init();
	void Enqueue(VIDEO_MESSAGE * pm);
	int GetSize();
	int Dequeue(VIDEO_MESSAGE * pm);
	VIDEO_MESSAGE * GetHead();
};

struct TCP_TUPLE {
	DWORD serverIP;
	DWORD clientIP;
	WORD serverPort;
	WORD clientPort;
};

struct VIDEO_COMM {
public:
	static TCP_TUPLE tcpTuple;
	static string clientIPStr;
	static int isKMLoaded;
	static int tileXMitMode;

        static int* d_classification;
        static Point* d_points;

public:
	static const int XMIT_QUEUE_UPDATE_ONCE = 0;
	static const int XMIT_QUEUE_UPDATE_REPLACE_USER = 1;
	static const int XMIT_QUEUE_UPDATE_REPLACE_KERNEL = 2;


	static float * replayGazeCenterX;
	static float * replayGazeCenterY;
	static float * replayGazeCenterZ;

	static float * replayGazeDirX;
	static float * replayGazeDirY;
	static float * replayGazeDirZ;

public:
	static int mode;
	static int bw_id;
	static bool skip;
	static bool augmentation;

	static struct pollfd peers[1];
	static int fds[1];	//fd[0] is used as control channel
	static VIDEO_MESSAGE_STORAGE vms;

	static void Init();
	static int ConnectionSetup(const char * remoteIP);
		
	static void TransmitContainers();	//KM only
	static int TransmitMessages();	
	static void ReceiveMessage();

	static void SendMessage_SelectVideo(const char * name);
	static void SendMessage_RequestRandomChunk();
	static void SendMessage_RequestBatch(int seqnum, int len, BYTE * reqData);
	static void SendMessage_RequestChunk(WORD chunkID, WORD tileID, BYTE quality, BYTE cls, int seqnum);
	static void SendMessage_Data(WORD chunkID, WORD tileID, BYTE quality, BYTE cls, int seqnum, BYTE * pData, int dataLen, int bTransmit);
	static void SendMessage_Data_dynamic(WORD chunkID, WORD tileID, BYTE quality, BYTE cls, int seqnum, BYTE * pData, int dataLen, int bTransmit);
	static void SendMessage_VideoMetaData();

	static void MainLoop();

private:
	static int sendingPri;
	static int sentBytes;	//how many bytes of the *current* (head) msg have been sent?
	static int rcvdBytes;	//how many bytes of the *current* (head) msg have been sent?
	
	static MESSAGE_QUEUE sendMsgQueue;
	static VIDEO_MESSAGE rcvdMsg;

	static void ProcessMessage(VIDEO_MESSAGE * pM);		
	static void UpdatePendingTiles(int nTiles, int seqnum, BYTE * pData);
	static void UpdatePendingTiles_start(int nTiles, int seqnum, BYTE * pData);

	static int reqCounter;
	static int xmitCounter;

	static int seqcounter;
	static bool previous_dynamic_skip;
	
public:	//for KM only
	static int curBatch;
	static int userOutBytes;
	static int userQueuedBytes;
	static void ProcessBatchRequests_KM(BYTE * pData);
	
public:		//for non-KM only
	static TRANSMITTED_TILE_BITMAP ttb;	
};


struct CHUNK_INFO;
struct KERNEL_INTERFACE {
private:
	KERNEL_INTERFACE();
	~KERNEL_INTERFACE();

private:
	static int fd;
	static int bLateBindingEnabled;
	static const int FAST_SHARE_SIZE = 16;

public:
	static BYTE * sperke_data_base;
	static CHUNK_INFO * pMeta;
	static BYTE * pBuf;
	static BYTE * pBatch;
	static int * pFastShare;		
				
public:
	static void DetectKernelModule();
	static void GlobalInit();
	static void VideoInit();
	static int OnNewBatch(int seqnum);
	static void EnableLateBinding();
	static void DisableLateBinding();
	static int IsLBEnabled();
};

struct TC_INTERFACE {
private:
	static void * ReplayThread(void * arg);
	static vector<int> bw;
	static int CheckFileForExec(const char * pathname);

public:
	static void CleanUpAndSystemCheck();
	static void LoadBWTrace(const char * filename, double dampFactor);
	static void InitReplay(const char * clientIP, int baseRTT);	//baseRTT in ms
	static void StartReplay(int interval);	//interval in ms
};



#endif

