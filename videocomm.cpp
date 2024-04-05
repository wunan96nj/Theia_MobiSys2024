#include "stdafx.h"
#include "videocomm.h"
#include "videodata.h"
#include "tools.h"
#include "settings.h"

#include "draco/point_cloud/point_cloud_builder.h"
#include "draco/compression/point_cloud/point_cloud_sequential_decoder.h"
#include "draco/compression/point_cloud/point_cloud_sequential_encoder.h"
#include "draco/compression/point_cloud/point_cloud_kd_tree_decoder.h"
#include "draco/compression/point_cloud/point_cloud_kd_tree_encoder.h"

#include "draco/compression/mesh/mesh_encoder.h"
#include "draco/mesh/mesh_are_equivalent.h"
#include "draco/compression/expert_encode.h"
#include "draco/compression/encode.h"
#include "draco/core/decoder_buffer.h"
#include "draco/io/obj_decoder.h"
#include "draco/io/mesh_io.h"

#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace draco;

int VIDEO_COMM::mode;

bool VIDEO_COMM::skip;
bool VIDEO_COMM::augmentation;
int VIDEO_COMM::bw_id;



struct pollfd VIDEO_COMM::peers[1];
int VIDEO_COMM::fds[1];

int VIDEO_COMM::sentBytes;
MESSAGE_QUEUE VIDEO_COMM::sendMsgQueue;
int VIDEO_COMM::sendingPri = -1;

int VIDEO_COMM::rcvdBytes;
VIDEO_MESSAGE VIDEO_COMM::rcvdMsg;

int VIDEO_COMM::reqCounter = 0;
int VIDEO_COMM::xmitCounter = 0;

int VIDEO_COMM::curBatch = -1;
int VIDEO_COMM::userOutBytes = 0;
int VIDEO_COMM::userQueuedBytes = 0;

int VIDEO_COMM::isKMLoaded = 0;
int VIDEO_COMM::tileXMitMode = -1;

TCP_TUPLE VIDEO_COMM::tcpTuple;
string VIDEO_COMM::clientIPStr;

int* VIDEO_COMM::d_classification;
Point* VIDEO_COMM::d_points;


int VIDEO_COMM::seqcounter=30;
bool VIDEO_COMM::previous_dynamic_skip = false;

float * VIDEO_COMM::replayGazeCenterX = new float[90];
float * VIDEO_COMM::replayGazeCenterY = new float[90];
float * VIDEO_COMM::replayGazeCenterZ = new float[90];

float * VIDEO_COMM::replayGazeDirX = new float[90];
float * VIDEO_COMM::replayGazeDirY = new float[90];
float * VIDEO_COMM::replayGazeDirZ = new float[90];


struct Id_Size {
  int id;
  float size;
};

extern "C"
//std::vector<Point> get_fovea(BYTE * pp_pc, int dataLen, Gaze gaze, Point* d_points);
void get_fovea(BYTE * pp_pc, int dataLen, Gaze gaze, Point* d_points, std::vector<Point> &selectedPoints_vector_inner, std::vector<Point> &selectedPoints_vector_outter, bool dynamic_skip, float rate_adapt, bool aug);

uint64_t timeSinceEpochMillisec_comm() {
	using namespace std::chrono;
	return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

uint64_t NDKGetTime_comm() {
	//struct timespec res;
	//clock_gettime(CLOCK_REALTIME, &res);
	//double t = res.tv_sec + (double)res.tv_nsec / 1e9f;

	//float t = FPlatformTime::Seconds()*1000;

	uint64_t t = timeSinceEpochMillisec_comm();
	return t;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//VIDEO_MESSAGE_STORAGE

VIDEO_MESSAGE_STORAGE VIDEO_COMM::vms;

VIDEO_MESSAGE_STORAGE::VIDEO_MESSAGE_STORAGE() {
	data = new BYTE[limit];
	head = tail = size = 0;
}

VIDEO_MESSAGE_STORAGE::~VIDEO_MESSAGE_STORAGE() {
	delete [] data;
}

BYTE* VIDEO_MESSAGE_STORAGE::AllocateBlock(int len) {
	BYTE * p = data + tail;
	tail += len;
	size += len;
	if (tail >= limit) tail = 0;
	MyAssert(size < limit - 1, 3532);
	return p;
}

void VIDEO_MESSAGE_STORAGE::ReleaseBlock(BYTE *pData, int len) {
	//assuming blocks are released in a FIFO
	MyAssert(pData == data + head && size >= len, 3533);
	head += len;
	size -= len;
	if (head >= limit) head = 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//TRANSMITTED_TILE_BITMAP

TRANSMITTED_TILE_BITMAP VIDEO_COMM::ttb;

TRANSMITTED_TILE_BITMAP::TRANSMITTED_TILE_BITMAP() {
	map = NULL;
}

TRANSMITTED_TILE_BITMAP::~TRANSMITTED_TILE_BITMAP() {
	if (map != NULL) delete [] map;
}

void TRANSMITTED_TILE_BITMAP::Init() {
	MyAssert(map == NULL, 3542);
	int n = VIDEO_DATA::nTilesX * VIDEO_DATA::nTilesY* VIDEO_DATA::nTilesZ * VIDEO_DATA::nChunks;
	map = new BYTE[n];
	memset(map, (BYTE)STATUS_NOTQUEUED, n);
}

void TRANSMITTED_TILE_BITMAP::MarkTile(int chunkID, int tileID, int status) {
	int k =  chunkID * VIDEO_DATA::nTilesX * VIDEO_DATA::nTilesY * VIDEO_DATA::nTilesZ + tileID;

	switch (status) {
		case STATUS_QUEUED:
			MyAssert(map[k] == STATUS_NOTQUEUED, 3544);
			break;

		case STATUS_XMITTED:
			MyAssert(map[k] == STATUS_QUEUED, 3545);
			break;

		default:
			MyAssert(0, 3553);
	}

	//InfoMessage("Mark Tile: %d %d %d", chunkID, tileID, status);
	map[k] = (BYTE)status;
}

void TRANSMITTED_TILE_BITMAP::UnmarkTile(int chunkID, int tileID) {
	int k =  chunkID * VIDEO_DATA::nTilesX * VIDEO_DATA::nTilesY * VIDEO_DATA::nTilesZ + tileID;
	MyAssert(map[k] != STATUS_NOTQUEUED, 3552);

	//InfoMessage("Unmark Tile: %d %d", chunkID, tileID);
	map[k] = STATUS_NOTQUEUED;
}


int TRANSMITTED_TILE_BITMAP::TileStatus(int chunkID, int tileID) {
	int k =  chunkID * VIDEO_DATA::nTilesX * VIDEO_DATA::nTilesY * VIDEO_DATA::nTilesZ + tileID;
	return map[k];
}

///////////////////////////////////////////////////////////////////////////////////////////////
//MESSAGE_QUEUE

MESSAGE_QUEUE::MESSAGE_QUEUE() {
	data = NULL;
}

void MESSAGE_QUEUE::Init() {
	data = new VIDEO_MESSAGE[SETTINGS::SEND_MESSAGE_QUEUE_SIZE];
	head = 0;	//point to the current spot
	tail = 0;	//point to the next empty spot
	size = 0;
}

MESSAGE_QUEUE::~MESSAGE_QUEUE() {
	delete [] data;
}

void MESSAGE_QUEUE::Enqueue(VIDEO_MESSAGE * pm) {
	data[tail++] = *pm;
	if (tail >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE) tail -= SETTINGS::SEND_MESSAGE_QUEUE_SIZE;
	if (++size >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE) {
		MyAssert(0, 3381);
	}

	if ((VIDEO_COMM::mode == MODE_SERVER && pm->id == MSG_VIDEO_DATA) || (VIDEO_COMM::mode == MODE_SERVER && pm->id == MSG_VIDEO_DATA_DYNAMIC)) {
		int chunkID = (int)ReadWORD(pm->pData + 5);
		int tileID = (int)ReadWORD(pm->pData + 7);
		//int tileID = (int)pm->pData[7];
		//VIDEO_COMM::ttb.MarkTile(chunkID, tileID, TRANSMITTED_TILE_BITMAP::STATUS_QUEUED);
	}
}

int MESSAGE_QUEUE::GetSize() {return size;}

int MESSAGE_QUEUE::Dequeue(VIDEO_MESSAGE * pm) {
	if (size == 0) return 0;
	if (pm != NULL) *pm = data[head];
	head++;
	if (head >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE) head -= SETTINGS::SEND_MESSAGE_QUEUE_SIZE;
	size--;
	return 1;
}

VIDEO_MESSAGE * MESSAGE_QUEUE::GetHead() {
	if (size == 0) return NULL; else return &data[head];
}

///////////////////////////////////////////////////////////////////////////////////////////////
//VIDEO_COMM

void VIDEO_COMM::Init() {
	sentBytes = 0;
	rcvdBytes = 0;

	rcvdMsg.pData = new BYTE[SETTINGS::RECV_BUFFER_SIZE];
	sendMsgQueue.Init();

	memset(&tcpTuple, 0, sizeof(TCP_TUPLE));
	// Set up the point cloud and gaze data
	const int maxPoints = 1500000;


	// Allocate memory on the GPU for the points and classification arrays

	cudaMalloc((void**)&d_points, maxPoints * sizeof(Point));

	//cudaMalloc((void**)&d_classification, maxPoints * sizeof(int));
}

void VIDEO_COMM::TransmitContainers() {
	static const int dummyBufSize = 1024*1024;
	static BYTE * dummyBuf = NULL;

	if (dummyBuf == NULL) {
		dummyBuf = new BYTE[dummyBufSize];
		memset(dummyBuf, 0xFF, dummyBufSize);
	}

	int fd = fds[0];
	MyAssert(tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_KERNEL && KERNEL_INTERFACE::IsLBEnabled(), 3707);

	while (userQueuedBytes > 0) {
		int nToWrite = userQueuedBytes;
		if (nToWrite > dummyBufSize) nToWrite = dummyBufSize;
		int n = write(fd, dummyBuf, nToWrite);

		if (n>=0) {
			userQueuedBytes -= n;
			userOutBytes += n;
			//sentBytes += n;
		}  else if (n < 0 && (errno == ECONNRESET || errno == EPIPE)) {
			ErrorMessage("Connection closed.");
			MyAssert(0, 3704);
		} else if (n < 0 && (errno == EWOULDBLOCK)) {
			//if (sentBytes == 0) sendingPri = -1;
			peers[0].events |= POLLWRNORM;
			return;
		} else {
			ErrorMessage("Unexpected send error %d: %s", errno, strerror(errno));
			MyAssert(0, 3705);
		}
	}

	if (userQueuedBytes == 0) {
		peers[0].events &= ~POLLWRNORM;
	}
}

//return 1 if fully transmitted for this channel
int VIDEO_COMM::TransmitMessages() {
	if (sendingPri != -1) MyAssert(sentBytes > 0, 3425);

	int fd = fds[0];

	while (1) {
		VIDEO_MESSAGE * pM = sendMsgQueue.GetHead();
		sendingPri = 0;

		if (pM == NULL) {
			peers[0].events &= ~POLLWRNORM;
			sendingPri = -1;
			//InfoMessage("### UNSET POLLWRNORM A, ch = %d ###", ch);
			return 1;
		}

		BYTE * pBase = pM->pData;
		MyAssert(pM->msgLen - sentBytes > 0 && pBase != NULL, 3383);

		//set the flag (bLast for now)
		if (pM->id == MSG_VIDEO_DATA) {
			MyAssert(tileXMitMode != XMIT_QUEUE_UPDATE_REPLACE_KERNEL, 3706);
			// if (sendMsgQueue.GetSize() > 1) {
			// 	pBase[14] = 0;
			// } else {
			// 	pBase[14] = 0; //Nan change for dynamic skip
			// }
		}

		while (pM->msgLen - sentBytes > 0) {
			int n = write(fd, pBase + sentBytes, pM->msgLen - sentBytes);
			if (n>=0) {
				sentBytes += n;
			}  else if (n < 0 && (errno == ECONNRESET || errno == EPIPE)) {
				ErrorMessage("Connection closed.");
				MyAssert(0, 3384);
			} else if (n < 0 && (errno == EWOULDBLOCK)) {
				if (sentBytes == 0) sendingPri = -1;
				peers[0].events |= POLLWRNORM;
				//InfoMessage("### SET POLLWRNORM, ch = %d ###", ch);
				return 0;
			} else {
				ErrorMessage("Unexpected send error %d: %s", errno, strerror(errno));
				MyAssert(0, 3385);
			}
		}

		if (sentBytes == pM->msgLen) {
			xmitCounter++;
			if (pM->id == MSG_REQUEST_CHUNK /*|| pM->id == MSG_BATCH_REQUESTS*/) {
				//we don't do this for MSG_BATCH_REQUESTS
				//becuase we are doing trace-driven emulation
				vms.ReleaseBlock(pM->pData, pM->msgLen);
			}

			int nPoints = 0;
			int chunkID = 0;
			int tileID = 0;
			if (mode == MODE_SERVER && pM->id == MSG_VIDEO_DATA) {
				chunkID = (int)ReadWORD(pM->pData + 5);
				tileID = (int)ReadWORD(pM->pData + 7);
				int quality = pM->pData[9];
				//ttb.MarkTile(chunkID, tileID, TRANSMITTED_TILE_BITMAP::STATUS_XMITTED);

				nPoints = VIDEO_DATA::GetChunkPoints(chunkID, tileID, quality);
								//nPoints = (int)ReadInt(pM->pData + 15);
			}

			InfoMessage("XMit #%d Size = %d Points = %d chunkID = %d tileID = %d", xmitCounter, sentBytes, nPoints, chunkID, tileID);
						InfoMessage("send timestamp: %lu\n", NDKGetTime_comm());
			
			std::ostringstream oss;
			std::string logMessage;
			
			oss << "CPU Sent Size:" << sentBytes << " chunkID = " << chunkID << " tileID = " << tileID << " send timestamp: " << NDKGetTime_comm() << std::endl;

			logMessage = oss.str();

			oss.str("");
			oss.clear();

			writeToLogFile(logMessage);

			sentBytes = 0;
			sendMsgQueue.Dequeue(NULL);


			/*
			if (bXmitOneMsg) { //transmit only one message
				peers[0].events &= ~POLLWRNORM;
				sendingPri = -1;
				//InfoMessage("### UNSET POLLWRNORM B, ch = %d ###", ch);
				return 1;
			}
			*/
		}
				delete [] pM->pData;
	}
}

void VIDEO_COMM::ProcessMessage(VIDEO_MESSAGE * pM) {
	BYTE * pData = pM->pData;
	MyAssert(pData != NULL, 3536);

	switch (pM->id) {
	case MSG_REQUEST_CHUNK:
	{
		MyAssert(mode == MODE_SERVER, 3391);

		//int cls = pData[9];
		int cls = 0;
		int chunkID = ReadWORD(pData+5);
		int tileID = ReadWORD(pData+7);
		int quality = pData[9];
		int seqnum = ReadInt(pData+10);

		reqCounter++;

		InfoMessage("REQ_CHUNK#%d cls=%d id=%d tile=%d quality=%d seq=%d",
			reqCounter, cls, chunkID, tileID, quality, seqnum
		);

		int dataLen;
		BYTE * pData = VIDEO_DATA::GetChunk(chunkID, tileID, quality, &dataLen);
		//InfoMessage("*** chunkID = %d, tileID = %d, dataLen = %d ***", chunkID, tileID, dataLen); //###
		SendMessage_Data(chunkID, tileID, quality, cls, seqnum, pData, dataLen, 1);
		break;
	}

	case MSG_SELECT_VIDEO:
	{
		MyAssert(mode == MODE_SERVER, 3392);
		int bw = ReadShort(pData + 5);
		int bwDamp = ReadWORD(pData + 7);
		tileXMitMode = ReadWORD(pData + 9);
		const char * name = (char *)pData+11;
		MyAssert(pM->msgLen == 11 + strlen(name) + 1, 3426);

		InfoMessage("SEL_VIDEO name=%s, bw=%d, damp=%d tileXMit=%d", name, bw, bwDamp, tileXMitMode);

		if (tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_KERNEL) {
			VIDEO_DATA::SwitchToKernelMemory();
		}

		VIDEO_DATA::LoadVideo(name);

		if (tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_KERNEL) {
			KERNEL_INTERFACE::VideoInit();
			curBatch = -1;
			userOutBytes = 0;
			userQueuedBytes = 0;
		} else {
			ttb.Init();
		}

		SendMessage_VideoMetaData();

		bw = bw_id; //NanWu

		if (bw != -1) {
			InfoMessage("Init bw trace replay. baseRTT = %d ms, id %d", SETTINGS::BW_BASE_RTT, bw_id);
			char filename[256];
			sprintf(filename, "bw/%d.txt", bw);
			TC_INTERFACE::LoadBWTrace(filename, bwDamp / (double)100.0f);
			TC_INTERFACE::InitReplay(clientIPStr.c_str(), SETTINGS::BW_BASE_RTT);
			TC_INTERFACE::StartReplay(1000);
		}

		break;
	}

	case MSG_VIDEO_DATA:
	{
		MyAssert(mode == MODE_CLIENT, 3393);

		//int cls = pData[9];
		int cls = 0;
		int chunkID = ReadWORD(pData+5);
		int tileID = ReadWORD(pData+7);
		//int tileID = pData[7];
		int quality = pData[9];
		int dataLen = ReadInt(pData+1) - 14;
		int seqnum = ReadInt(pData+10);

		InfoMessage("VIDEO_DATA cls=%d id=%d tile=%d quality=%d, data=%d, seqnum=%d",
			cls, chunkID, tileID, quality, dataLen, seqnum
		);
		break;
	}

	case MSG_VIDEO_METADATA:
	{
		MyAssert(mode == MODE_CLIENT, 3408);
		VIDEO_DATA::DecodeMetaData(pM->pData, pM->msgLen);
		break;
	}

	case MSG_BATCH_REQUESTS:
	{
		MyAssert(mode == MODE_SERVER, 3537);
		int nTilesToFetch = (ReadInt(pData+1) - 9) / 5;
		int seqnum = ReadInt(pData + 5);

		if (tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_KERNEL) {
			if (!KERNEL_INTERFACE::IsLBEnabled()) {
				KERNEL_INTERFACE::EnableLateBinding();
			}

			ProcessBatchRequests_KM(pData);
		} else {
			UpdatePendingTiles_start(nTilesToFetch, seqnum, pData + 9);
		}

		break;
	}
	case MSG_GAZE_BATCH_REQUESTS:
	{
		MyAssert(mode == MODE_SERVER, 3537);
		int nTilesToFetch = 1;
		int seqnum = ReadInt(pData + 5);

		//printf("MSG_GAZE_BATCH_REQUESTS: %lu, seqnum: %d\n", NDKGetTime_comm(), seqcounter);


		std::ostringstream oss;
		std::string logMessage;
		oss << "MSG_GAZE_BATCH_REQUESTS:" << NDKGetTime_comm() << " seqcounter: " << seqcounter << std::endl;

		logMessage = oss.str();

		oss.str("");
		oss.clear();

		writeToLogFile(logMessage);
		seqcounter++;
		if (tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_KERNEL) {
			if (!KERNEL_INTERFACE::IsLBEnabled()) {
				KERNEL_INTERFACE::EnableLateBinding();
			}

			ProcessBatchRequests_KM(pData);
		} else {
						//printf("UpdatePendingTiles\n");
			UpdatePendingTiles(nTilesToFetch, seqnum, pData + 9);
		}

		break;
	}

	default:
		MyAssert(0, 3389);
	}
}

void VIDEO_COMM::ProcessBatchRequests_KM(BYTE * pData) { //pData includes 9-byte header
	if (++curBatch >= SETTINGS::MAX_BATCHES) {
		curBatch = 0;
	}

	int len = ReadInt(pData + 1);
	MyAssert(len < SETTINGS::MAX_BATCH_SIZE, 3702);

	int * pSeqnum = (int *)(pData + 5);

	int seqnum = *pSeqnum;
	*pSeqnum = 0;	//first "hide" the seqnum
	BYTE * pDest = VIDEO_DATA::pBatch + curBatch * SETTINGS::MAX_BATCH_SIZE;
	memcpy(pDest, pData, len);

	//change > 0: need more bytes
	//change < 0: need less bytes
	int change = KERNEL_INTERFACE::OnNewBatch(seqnum);

	userQueuedBytes += change;
	if (userQueuedBytes <= 0)
		userQueuedBytes = 0;
	else
		TransmitContainers();
}

void LinearRegression(float* feature, float* label, int len, float & a, float & b) {
	float mean_x =0;
	float mean_y =0;
	for (int i=0; i<len; i++) {
		mean_x = mean_x + feature[i];
		mean_y = mean_y + label[i];
		//InfoMessage("label %d: %.3lf", i, label[i]);
	}
	mean_x = mean_x / float(len);
	mean_y = mean_y / float(len);

	float sum1 = 0;
	float sum2 = 0;
	for (int i=0; i<len; i++) {
		sum1 = sum1 + (feature[i]-mean_x)*(label[i]-mean_y);
		sum2 = sum2 + (feature[i]-mean_x)*(feature[i]-mean_x);
	}
	a = sum1 / sum2;
	b = mean_y - a * mean_x;
	//InfoMessage("%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf", a, b, sum1, sum2, mean_x, mean_y);
	//float test_label = a * test + b;
}

void Predict_Linear(int nextFrame, float * eyeX, float * eyeY, float * eyeZ, float * dirX, float * dirY, float * dirZ) {

	float a, b;
	int len = 9;

	if (len == 9) {
		float features[9];

		// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				features[i*3+j] = (float) (3 * (i + 25) + j);
			}
				
		}


		// //double t1 = NDKGetTime();
		// LinearRegression(features, &(VIDEO_COMM::replayGazeCenterX[nextFrame - 6 - 2]), 3, a, b);
		// //InfoMessage("%.3lf %.3lf", a, b);
		// *eyeX = 30.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeCenterY[nextFrame - 6 - 2]), 3, a, b);
		// *eyeY = 30.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeCenterZ[nextFrame - 6 - 2]), 3, a, b);
		// *eyeZ = 30.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeDirX[nextFrame - 6 - 2]), 3, a, b);
		// *dirX = 30.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeDirY[nextFrame - 6 - 2]), 3, a, b);
		// *dirY = 30.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeDirZ[nextFrame - 6 - 2]), 3, a, b);
		// *dirZ = 30.0f * a + b;
		// //double t2 = NDKGetTime();


		// //double t1 = NDKGetTime();
		// LinearRegression(features, &(VIDEO_COMM::replayGazeCenterX[3 * (nextFrame - 6 - 2)]), 9, a, b);
		// //InfoMessage("%.3lf %.3lf", a, b);
		// *eyeX = 90.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeCenterY[3 * (nextFrame - 6 - 2)]), 9, a, b);
		// *eyeY = 90.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeCenterZ[3 * (nextFrame - 6 - 2)]), 9, a, b);
		// *eyeZ = 90.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeDirX[3 * (nextFrame - 6 - 2)]), 9, a, b);
		// *dirX = 90.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeDirY[3 * (nextFrame - 6 - 2)]), 9, a, b);
		// *dirY = 90.0f * a + b;

		// LinearRegression(features, &(VIDEO_COMM::replayGazeDirZ[3 * (nextFrame - 6 - 2)]), 9, a, b);
		// *dirZ = 90.0f * a + b;


		//double t1 = NDKGetTime();
		LinearRegression(features, &(VIDEO_COMM::replayGazeCenterX[3 * (nextFrame - 3 - 2)]), 9, a, b);
		//InfoMessage("%.3lf %.3lf", a, b);
		*eyeX = 90.0f * a + b;

		LinearRegression(features, &(VIDEO_COMM::replayGazeCenterY[3 * (nextFrame - 3 - 2)]), 9, a, b);
		*eyeY = 90.0f * a + b;

		LinearRegression(features, &(VIDEO_COMM::replayGazeCenterZ[3 * (nextFrame - 3 - 2)]), 9, a, b);
		*eyeZ = 90.0f * a + b;

		LinearRegression(features, &(VIDEO_COMM::replayGazeDirX[3 * (nextFrame - 3 - 2)]), 9, a, b);
		*dirX = 90.0f * a + b;

		LinearRegression(features, &(VIDEO_COMM::replayGazeDirY[3 * (nextFrame - 3 - 2)]), 9, a, b);
		*dirY = 90.0f * a + b;

		LinearRegression(features, &(VIDEO_COMM::replayGazeDirZ[3 * (nextFrame - 3 - 2)]), 9, a, b);
		*dirZ = 90.0f * a + b;
	}
}

//only used for non-KM
void VIDEO_COMM::UpdatePendingTiles(int nTiles, int seqnum, BYTE * pData /* a batch of tiles */) {
	std::ostringstream oss;
	std::string logMessage;

	switch (tileXMitMode) {
		case XMIT_QUEUE_UPDATE_REPLACE_USER:
		case XMIT_QUEUE_UPDATE_ONCE:
			{
				MESSAGE_QUEUE & q = sendMsgQueue;
				/*
				if (q.size > 0 && tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_USER) {
					//first clear the queue
					int h = q.head;

					if (sendingPri != -1) {
						MyAssert(q.size > 0, 3557);
						h++;
						if (h >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE)
							h -= SETTINGS::SEND_MESSAGE_QUEUE_SIZE;
					}
					int tail = h;

					while (h != q.tail) {
						MyAssert(q.data[h].id == MSG_VIDEO_DATA, 3546);
						BYTE * pData = q.data[h].pData;
						int chunkID = ReadWORD(pData + 5);
						int tileID = pData[7];
						ttb.UnmarkTile(chunkID, tileID);
						h++;
						if (h >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE)
							h -= SETTINGS::SEND_MESSAGE_QUEUE_SIZE;
						q.size--;
					}

					MyAssert((sendingPri == -1 && q.size == 0) || q.size == 1, 3547);
					q.tail = tail;
				}
				*/

				//then pump in the data
				BYTE * p = pData;
				int n = 0;

				

				int x, y, z;
				int nTotalPointsNew = 0;
				uint64_t t_total = 0;
				int nCompressedSize = 0;

				int segments_x ,segments_y, segments_z;
				int minBox_x, minBox_y, minBox_z;
				int maxBox_x, maxBox_y, maxBox_z;
				int segmentation = 2;
				int step;
				if (segmentation == 1) step = 2000;
				else if (segmentation == 2) step = 1000;
				else if (segmentation == 4) step = 500;
				else if (segmentation == 8) step = 250;
				else if (segmentation == 10) step = 200;


				segments_x = segments_y = segments_z = segmentation;
				minBox_x = -1000;
				minBox_y = -50;
				minBox_z = -1050;

				maxBox_x = 1000;
				maxBox_y = 1950;
				maxBox_z = 950;

				segments_x = (maxBox_x - minBox_x) / step;
				segments_y = (maxBox_y - minBox_y) / step;
				segments_z = (maxBox_z - minBox_z) / step;


				//InfoMessage("nTiles = %d", nTiles);

				for (int i=0; i<nTiles; i++) {
					int chunkID = ReadWORD(p);
					int tileID = ReadWORD(p + 2);

					//InfoMessage("### Chunk=%d tile=%d", chunkID, (int)tileID);

					if (ttb.TileStatus(chunkID, tileID) == ttb.STATUS_NOTQUEUED) {
												//printf("### Chunk=%d tile=%d\n", chunkID, (int)tileID);
						reqCounter++;
						n++;
						//BYTE cls = p[4];
						BYTE cls = 0;
						BYTE quality = p[4];

						p += 5;

						float gazedata;
						for (int gaze_idx = 0; gaze_idx < 3; gaze_idx++){
							for (int gaze_90fps_idx = 0; gaze_90fps_idx < 3; gaze_90fps_idx++){
								int temp_idx = 3 * (30 - 3 - 2 + gaze_idx) + gaze_90fps_idx;
								gazedata = ReadFloat(p);
								replayGazeCenterX[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeCenterY[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeCenterZ[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeDirX[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeDirY[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeDirZ[temp_idx] = gazedata;
								p += 4;
							}

						}
						float eyeX, eyeY, eyeZ, Yaw, Pitch, Roll;
						Predict_Linear(30, &eyeX, &eyeY, &eyeZ, &Yaw, &Pitch, &Roll);

						//calculate rotation speed
						std::vector<float> angular_diff;
						// float diffX = replayGazeDirX[3 * (30 - 3 - 2 + 2) + 2] - replayGazeDirX[3 * (30 - 3 - 2 + 0) + 0];
						// angular_diff.push_back(diffX);
						// float diffY = replayGazeDirY[3 * (30 - 3 - 2 + 2) + 2] - replayGazeDirY[3 * (30 - 3 - 2 + 0) + 0];
						// angular_diff.push_back(diffY);
						float diffX = Yaw - replayGazeDirX[3 * (30 - 3 - 2 + 2) + 2];
						angular_diff.push_back(diffX);
						float diffY = Pitch - replayGazeDirY[3 * (30 - 3 - 2 + 2) + 2];
						angular_diff.push_back(diffY);

						// Calculate the total rotation speed as the magnitude of the difference vector
						float rotation_speed = std::sqrt(angular_diff[0] * angular_diff[0] + angular_diff[1] * angular_diff[1]) / (0.011f * 9);
						// InfoMessage("rotation_speed %f", rotation_speed);
						
						oss << "rotation_speed:" << rotation_speed << std::endl;

						logMessage = oss.str();


						oss.str("");
						oss.clear();
						bool dynamic_skip = false;

						if (rotation_speed > 10.0f){
							dynamic_skip = true;
						}

						if (previous_dynamic_skip){
							dynamic_skip = false;
						}
						previous_dynamic_skip = dynamic_skip;
						if (VIDEO_COMM::skip == false){
							dynamic_skip = false; 
						}
						//dynamic_skip = false; //NanWu
						writeToLogFile(logMessage);

						//InfoMessage("%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf", eyeX, eyeY, eyeZ, Yaw, Pitch, Roll);

						// Assuming temp_rotation is { Yaw, Pitch, 0.0f };
						std::vector<float> temp_rotation = { Yaw, Pitch, 0.0f };
						float YawRadians = temp_rotation[0] * (M_PI / 180.0f);
						float PitchRadians = temp_rotation[1] * (M_PI / 180.0f);
						// Roll is not used when converting to direction vector

						float dirX = cos(PitchRadians) * cos(YawRadians);
						float dirY = sin(PitchRadians);
						float dirZ = cos(PitchRadians) * sin(YawRadians);


						float length = std::sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
						dirX /= length;
						dirY /= length;
						dirZ /= length;
						//InfoMessage("%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf", eyeX, eyeY, eyeZ, dirX, dirY, dirZ);


						InfoMessage("REQ_BATCH#%d cls=%d chunkID=%d tileID=%d quality=%d seq=%d qlen=%d",
							reqCounter, (int)cls, (int)chunkID, (int)tileID, (int)quality, seqnum, q.size
						);

						Gaze gaze;
						gaze.position.x = (float)eyeX;
						gaze.position.y = (float)eyeY;
						gaze.position.z = (float)eyeZ;
						gaze.direction.x = (float)dirX;
						gaze.direction.y = (float)dirY;
						gaze.direction.z = (float)dirZ;
						
						//Gaze gaze;
						// gaze.position.x = 0.0;
						// gaze.position.y = 1.5;
						// gaze.position.z = 2.0;
						// gaze.direction.x = 0.0;
						// gaze.direction.y = 0.0;
						// gaze.direction.z = -1.0;
						// int temp_idx = 3 * (30 - 6) + 2;
						// gaze.position.x = replayGazeCenterX[temp_idx];
						// gaze.position.y = replayGazeCenterY[temp_idx];
						// gaze.position.z = replayGazeCenterZ[temp_idx];
						// gaze.direction.x = replayGazeDirX[temp_idx];
						// gaze.direction.y = replayGazeDirY[temp_idx];
						// gaze.direction.z = replayGazeDirZ[temp_idx];

						int dataLen;
						//BYTE * pp_pc = VIDEO_DATA::GetChunk(chunkID, tileID, quality, &dataLen);
						BYTE * pp_pc = VIDEO_DATA::GetChunk(chunkID, tileID, 0, &dataLen);

						//compress point cloud retrieved from pp
						int nPoints = (dataLen-CHUNK_HEADER_LEN) / (sizeof(short)+sizeof(char)) / 3;
						//printf("nPoints: %d\n", nPoints);

						// char * pointBuf = new char[nPoints * sizeof(short) * 3];
						// char * colorBuf = new char[nPoints * sizeof(char) * 3];

						// int pDataSize = sizeof(short) * nPoints * 3;
						// memcpy(pointBuf, pp_pc+CHUNK_HEADER_LEN, pDataSize);

						// int cDataSize = sizeof(char) * nPoints * 3;
						// memcpy(colorBuf, pp_pc+CHUNK_HEADER_LEN+pDataSize, cDataSize);



						//printf("get_fovea: %lu\n", NDKGetTime_comm());
						uint64_t t1 = NDKGetTime_comm();

						//std::vector<int> selectedPoints_vector = get_fovea(pp_pc, dataLen, gaze, d_points, d_classification);
						std::vector<Point> selectedPoints_vector_inner;
						std::vector<Point> selectedPoints_vector_outer;

						quality = 10;
						float rate_adapt = sqrt(10.0f / max(2.5f, float(quality)));
						printf("rate_adapt: %f, quality %d\n", rate_adapt, quality);

						get_fovea(pp_pc, dataLen, gaze, d_points, selectedPoints_vector_inner, selectedPoints_vector_outer, dynamic_skip, rate_adapt, augmentation);
						//printf("after get_fovea: %lu\n", NDKGetTime_comm());
						
						oss << "CPU get_fovea takes:" << NDKGetTime_comm()-t1 << std::endl;

						logMessage = oss.str();

						oss.str("");
						oss.clear();

						writeToLogFile(logMessage);

						nPoints = selectedPoints_vector_inner.size();
						float * pDataX = new float[nPoints];
						float * pDataY = new float[nPoints];
						float * pDataZ = new float[nPoints];

						char * cData1 = new char[nPoints];
						char * cData2 = new char[nPoints];
						char * cData3 = new char[nPoints];

						float * sData = new float[nPoints];


						int index = 0;


						//for(int select_idx : selectedPoints_vector) {
						for(Point select_point : selectedPoints_vector_inner) {
							//pOffset = select_idx.id * sizeof(short) * 3;
							//cOffset = select_idx.id * sizeof(char) * 3;
							//short * pp = (short *)(pointBuf + pOffset);
							pDataX[index] = float(select_point.x * 1000.0f);
							pDataY[index] = float(select_point.y * 1000.0f);
							pDataZ[index] = float(select_point.z * 1000.0f);

							//char * pc = (char *)(colorBuf + cOffset);
							cData1[index] = char(select_point.r);
							cData2[index] = char(select_point.g);
							cData3[index] = char(select_point.b);

							//sData[index] = 0.5f;
							sData[index] = select_point.point_size;
							index++;
						}


						//printf("start segmentation and compression: %lu\n", NDKGetTime_comm());
						uint64_t t_compression_start = NDKGetTime_comm();
						// minBox_x = pDataX[0]-1000;
						// //minBox_y = pDataY[0]-1000;
						// minBox_z = pDataZ[0]-1050;

						// maxBox_x = pDataX[0]+1000;
						// //maxBox_y = pDataY[0]+1000;
						// maxBox_z = pDataZ[0]+950;

						//int mod = 100;
						for (x = 0; x < segments_x; x++) {
							for (y = 0; y < segments_y; y++) {
								for (z = 0; z < segments_z; z++) {
									



									int cellID = z + y * segments_z + x * segments_z * segments_y;
									int nPointsNew = 0;

									int lowX = minBox_x + x * step;
									int lowY = minBox_y + y * step;
									int lowZ = minBox_z + z * step;
									int idx = 0;
									vector<int> selected;
									for (int index = 0; index < nPoints; index++) {
										if (pDataX[index] > lowX && pDataX[index] <= lowX + step && pDataY[index] > lowY &&
											pDataY[index] <= lowY + step && pDataZ[index] > lowZ && pDataZ[index] <= lowZ + step) {
											//if (rand() % 100 < mod) {

											selected.push_back(index);
											idx++;

										}
									}

									if (idx == 0){
										BYTE * p_send_0 = new BYTE[CHUNK_HEADER_LEN];
										memcpy(p_send_0, pp_pc, CHUNK_HEADER_LEN);
										seqnum = (int)(NDKGetTime_comm() - t1);

										if (dynamic_skip){
											SendMessage_Data_dynamic(chunkID, cellID, quality, cls, seqnum, p_send_0, CHUNK_HEADER_LEN, 0);
										} else{
											SendMessage_Data(chunkID, cellID, quality, cls, seqnum, p_send_0, CHUNK_HEADER_LEN, 0);
										}
										continue;
									}



									nPointsNew = idx;
									//printf("nPointsNew: %d\n", nPointsNew);
									PointCloudBuilder builder;
									builder.Start(nPointsNew);
									const int pos_att_id =
										builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
										//builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_INT32);
									const int color_att_id =
										builder.AddAttribute(GeometryAttribute::COLOR, 3, DT_INT8);
									const int size_att_id =
										builder.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);

									for (int index = 0; index < nPointsNew; index++) {
										std::array<float, 3> point;
										//point[0] = (pDataX[selected[index]] - lowX) / 1000.0f;
										//point[1] = (pDataY[selected[index]] - lowY) / 1000.0f;
										//point[2] = (pDataZ[selected[index]] - lowZ) / 1000.0f;
										//point[0] = pDataX[selected[index]];
										//point[1] = pDataY[selected[index]];
										//point[2] = pDataZ[selected[index]];
										point[0] = pDataX[selected[index]] / 1000.0f;
										point[1] = pDataY[selected[index]] / 1000.0f;
										point[2] = pDataZ[selected[index]] / 1000.0f;
										//std::array<int, 3> point;
										//point[0] = pDataX[selected[index]];
										//point[1] = pDataY[selected[index]];
										//point[2] = pDataZ[selected[index]];
										builder.SetAttributeValueForPoint(pos_att_id, PointIndex(index), &(point)[0]);

										std::array<uint8_t, 3> color;
										color[0] = cData1[selected[index]];
										color[1] = cData2[selected[index]];
										color[2] = cData3[selected[index]];
										builder.SetAttributeValueForPoint(color_att_id, PointIndex(index), &(color)[0]);

										std::array<float, 1> size;
										size[0] = sData[selected[index]];
										//size[0] = 0.5;
										builder.SetAttributeValueForPoint(size_att_id, PointIndex(index), &(size)[0]);
									}

									std::unique_ptr<PointCloud> pc = builder.Finalize(false);
									MyAssert(pc != nullptr, 1001);
									MyAssert(pc->num_points() == nPointsNew, 1002);

									int compression_level = 0;
									EncoderBuffer buffer;
									PointCloudSequentialEncoder encoder;
									EncoderOptions options = EncoderOptions::CreateDefaultOptions();
									int quantization_bits = 11;
									options.SetGlobalInt("quantization_bits", quantization_bits);
									options.SetSpeed(10- compression_level, 10 - compression_level);
									encoder.SetPointCloud(*pc);

									MyAssert(encoder.Encode(options, &buffer).ok(), 2001);

									int nSize = buffer.size();
									/*
									DecoderBuffer dec_buffer;
									dec_buffer.Init(buffer.data(), buffer.size());
									PointCloudKdTreeDecoder decoder;

									std::unique_ptr<PointCloud> out_pc(new PointCloud());
									DecoderOptions dec_options;
									MyAssert(decoder.Decode(dec_options, &dec_buffer, out_pc.get()).ok(), 2002);
									MyAssert(out_pc->num_points() == nPointsNew, 1003);
									*/

									BYTE * p_send = new BYTE[nSize+CHUNK_HEADER_LEN];


									memcpy(p_send, pp_pc, CHUNK_HEADER_LEN);
									memcpy(p_send+CHUNK_HEADER_LEN, buffer.data(), buffer.size());
									//memcpy(pp_pc+nCompressedSize, pp_pc, CHUNK_HEADER_LEN);
									//memcpy(pp_pc+nCompressedSize+CHUNK_HEADER_LEN, buffer.data(), buffer.size());
									seqnum = (int)(NDKGetTime_comm() - t1);
									if (dynamic_skip){
										SendMessage_Data_dynamic(chunkID, cellID, quality, cls, seqnum, p_send, nSize+CHUNK_HEADER_LEN, 0);
									} else{
										SendMessage_Data(chunkID, cellID, quality, cls, seqnum, p_send, nSize+CHUNK_HEADER_LEN, 0);

									}
									//printf("%d %d %d %d %d %d ", chunkID, cellID, quality, cls, seqnum, nSize+CHUNK_HEADER_LEN);
									//SendMessage_Data(chunkID, cellID, quality, cls, seqnum, pp_pc+nCompressedSize, nSize+CHUNK_HEADER_LEN, 0);
									nCompressedSize += nSize + CHUNK_HEADER_LEN;




								}
							}
						}


						//printf("end: %lu\n", NDKGetTime_comm());
						uint64_t t2 = NDKGetTime_comm();

						oss << "CPU compression takes:" << t2-t_compression_start << std::endl;

						logMessage = oss.str();

						oss.str("");
						oss.clear();

						writeToLogFile(logMessage);


						t_total = t2 - t1;
						//printf("Total preparing time: %lu\n", t_total);
						//printf("nCompressedSize: %d\n", nCompressedSize);


						oss << "CPU Total preparing time:" << t_total << std::endl;

						logMessage = oss.str();

						oss.str("");
						oss.clear();

						writeToLogFile(logMessage);


						oss << "CPU nCompressedSize:" << nCompressedSize << std::endl;

						logMessage = oss.str();

						oss.str("");
						oss.clear();

						writeToLogFile(logMessage);

						delete [] pDataX;
						delete [] pDataY;
						delete [] pDataZ;
						delete [] cData1;
						delete [] cData2;
						delete [] cData3;
						delete [] sData;

						// delete [] pDataX_outer;
						// delete [] pDataY_outer;
						// delete [] pDataZ_outer;
						// delete [] cData1_outer;
						// delete [] cData2_outer;
						// delete [] cData3_outer;
						// delete [] sData_outer;

						// delete [] pointBuf;
						// delete [] colorBuf;




					}

					p+=5;
				}

				//transmit in a single bundle
				if (n > 0) TransmitMessages(); //the flag will be updated inside this call
				break;
			}

		default:
			MyAssert(0, 3538);
	}
}


void VIDEO_COMM::UpdatePendingTiles_start(int nTiles, int seqnum, BYTE * pData /* a batch of tiles */) {
	switch (tileXMitMode) {
		case XMIT_QUEUE_UPDATE_REPLACE_USER:
		case XMIT_QUEUE_UPDATE_ONCE:
			{
				MESSAGE_QUEUE & q = sendMsgQueue;
				/*
				if (q.size > 0 && tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_USER) {
					//first clear the queue
					int h = q.head;

					if (sendingPri != -1) {
						MyAssert(q.size > 0, 3557);
						h++;
						if (h >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE)
							h -= SETTINGS::SEND_MESSAGE_QUEUE_SIZE;
					}
					int tail = h;

					while (h != q.tail) {
						MyAssert(q.data[h].id == MSG_VIDEO_DATA, 3546);
						BYTE * pData = q.data[h].pData;
						int chunkID = ReadWORD(pData + 5);
						int tileID = pData[7];
						ttb.UnmarkTile(chunkID, tileID);
						h++;
						if (h >= SETTINGS::SEND_MESSAGE_QUEUE_SIZE)
							h -= SETTINGS::SEND_MESSAGE_QUEUE_SIZE;
						q.size--;
					}

					MyAssert((sendingPri == -1 && q.size == 0) || q.size == 1, 3547);
					q.tail = tail;
				}
				*/

				//then pump in the data
				BYTE * p = pData;
				int n = 0;

				

				int x, y, z;
				int nTotalPointsNew = 0;
				uint64_t t_total = 0;
				int nCompressedSize = 0;

				int segments_x ,segments_y, segments_z;
				int minBox_x, minBox_y, minBox_z;
				int maxBox_x, maxBox_y, maxBox_z;
				int segmentation = 2;
				int step;
				if (segmentation == 1) step = 2000;
				else if (segmentation == 2) step = 1000;
				else if (segmentation == 4) step = 500;
				else if (segmentation == 8) step = 250;
				else if (segmentation == 10) step = 200;


				segments_x = segments_y = segments_z = segmentation;
				minBox_x = -1000;
				minBox_y = -50;
				minBox_z = -1050;

				maxBox_x = 1000;
				maxBox_y = 1950;
				maxBox_z = 950;

				segments_x = (maxBox_x - minBox_x) / step;
				segments_y = (maxBox_y - minBox_y) / step;
				segments_z = (maxBox_z - minBox_z) / step;


				//InfoMessage("nTiles = %d", nTiles);

				for (int i=0; i<nTiles; i++) {
					int chunkID = ReadWORD(p);
					int tileID = ReadWORD(p + 2);

					//InfoMessage("### Chunk=%d tile=%d", chunkID, (int)tileID);

					if (ttb.TileStatus(chunkID, tileID) == ttb.STATUS_NOTQUEUED) {
												printf("### Chunk=%d tile=%d\n", chunkID, (int)tileID);
						reqCounter++;
						n++;
						//BYTE cls = p[4];
						BYTE cls = 0;
						BYTE quality = p[4];

						p += 5;

						float gazedata;
						for (int gaze_idx = 0; gaze_idx < 3; gaze_idx++){
							for (int gaze_90fps_idx = 0; gaze_90fps_idx < 3; gaze_90fps_idx++){
								int temp_idx = 3 * (30 - 6 - 2 + gaze_idx) + gaze_90fps_idx;
								gazedata = ReadFloat(p);
								replayGazeCenterX[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeCenterY[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeCenterZ[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeDirX[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeDirY[temp_idx] = gazedata;
								p += 4;

								gazedata = ReadFloat(p);
								replayGazeDirZ[temp_idx] = gazedata;
								p += 4;
							}

						}
						float eyeX, eyeY, eyeZ, dirX, dirY, dirZ;
						Predict_Linear(30, &eyeX, &eyeY, &eyeZ, &dirX, &dirY, &dirZ);
												

						InfoMessage("%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf", eyeX, eyeY, eyeZ, dirX, dirY, dirZ);



						InfoMessage("REQ_BATCH#%d cls=%d chunkID=%d tileID=%d quality=%d seq=%d qlen=%d",
							reqCounter, (int)cls, (int)chunkID, (int)tileID, (int)quality, seqnum, q.size
						);

						Gaze gaze;
	
						//Gaze gaze;
						gaze.position.x = -2.0;
						gaze.position.y = 1.5;
						gaze.position.z = 0.0;
						gaze.direction.x = 1.0;
						gaze.direction.y = 0.0;
						gaze.direction.z = 0.0;

						int dataLen;
						//BYTE * pp_pc = VIDEO_DATA::GetChunk(chunkID, tileID, quality, &dataLen);
						BYTE * pp_pc = VIDEO_DATA::GetChunk(chunkID, tileID, 0, &dataLen);

						//compress point cloud retrieved from pp
						int nPoints = (dataLen-CHUNK_HEADER_LEN) / (sizeof(short)+sizeof(char)) / 3;
						printf("nPoints: %d\n", nPoints);

						// char * pointBuf = new char[nPoints * sizeof(short) * 3];
						// char * colorBuf = new char[nPoints * sizeof(char) * 3];

						// int pDataSize = sizeof(short) * nPoints * 3;
						// memcpy(pointBuf, pp_pc+CHUNK_HEADER_LEN, pDataSize);

						// int cDataSize = sizeof(char) * nPoints * 3;
						// memcpy(colorBuf, pp_pc+CHUNK_HEADER_LEN+pDataSize, cDataSize);



						printf("get_fovea: %lu\n", NDKGetTime_comm());
						uint64_t t1 = NDKGetTime_comm();

						//std::vector<int> selectedPoints_vector = get_fovea(pp_pc, dataLen, gaze, d_points, d_classification);
						std::vector<Point> selectedPoints_vector_inner;
						std::vector<Point> selectedPoints_vector_outer;

						get_fovea(pp_pc, dataLen, gaze, d_points, selectedPoints_vector_inner, selectedPoints_vector_outer, false, 1.0, true);
						printf("after get_fovea: %lu\n", NDKGetTime_comm());
						

						nPoints = selectedPoints_vector_inner.size();
						short * pDataX = new short[nPoints];
						short * pDataY = new short[nPoints];
						short * pDataZ = new short[nPoints];

						char * cData1 = new char[nPoints];
						char * cData2 = new char[nPoints];
						char * cData3 = new char[nPoints];

						float * sData = new float[nPoints];


						int index = 0;


						//for(int select_idx : selectedPoints_vector) {
						for(Point select_point : selectedPoints_vector_inner) {
							//pOffset = select_idx.id * sizeof(short) * 3;
							//cOffset = select_idx.id * sizeof(char) * 3;
							//short * pp = (short *)(pointBuf + pOffset);
							pDataX[index] = short(select_point.x * 1000);
							pDataY[index] = short(select_point.y * 1000);
							pDataZ[index] = short(select_point.z * 1000);

							//char * pc = (char *)(colorBuf + cOffset);
							cData1[index] = char(select_point.r);
							cData2[index] = char(select_point.g);
							cData3[index] = char(select_point.b);

							//sData[index] = 0.5f;
							sData[index] = select_point.point_size;
							index++;
						}


						printf("start segmentation and compression: %lu\n", NDKGetTime_comm());



						//int mod = 100;
						for (x = 0; x < segments_x; x++) {
							for (y = 0; y < segments_y; y++) {
								for (z = 0; z < segments_z; z++) {
									



									int cellID = z + y * segments_z + x * segments_z * segments_y;
									int nPointsNew = 0;

									int lowX = minBox_x + x * step;
									int lowY = minBox_y + y * step;
									int lowZ = minBox_z + z * step;
									int idx = 0;
									vector<int> selected;
									for (int index = 0; index < nPoints; index++) {
										if (pDataX[index] > lowX && pDataX[index] <= lowX + step && pDataY[index] > lowY &&
											pDataY[index] <= lowY + step && pDataZ[index] > lowZ && pDataZ[index] <= lowZ + step) {
											//if (rand() % 100 < mod) {

											selected.push_back(index);
											idx++;

										}
									}

									if (idx == 0){
										BYTE * p_send_0 = new BYTE[CHUNK_HEADER_LEN];
										memcpy(p_send_0, pp_pc, CHUNK_HEADER_LEN);
										seqnum = (int)(NDKGetTime_comm() - t1);

										SendMessage_Data(chunkID, cellID, quality, cls, seqnum, p_send_0, CHUNK_HEADER_LEN, 0);
										continue;
									}



									nPointsNew = idx;
									//printf("nPointsNew: %d\n", nPointsNew);
									PointCloudBuilder builder;
									builder.Start(nPointsNew);
									const int pos_att_id =
										builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
										//builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_INT32);
									const int color_att_id =
										builder.AddAttribute(GeometryAttribute::COLOR, 3, DT_INT8);
									const int size_att_id =
										builder.AddAttribute(GeometryAttribute::GENERIC, 1, DT_FLOAT32);

									for (int index = 0; index < nPointsNew; index++) {
										std::array<float, 3> point;
										//point[0] = (pDataX[selected[index]] - lowX) / 1000.0f;
										//point[1] = (pDataY[selected[index]] - lowY) / 1000.0f;
										//point[2] = (pDataZ[selected[index]] - lowZ) / 1000.0f;
										//point[0] = pDataX[selected[index]];
										//point[1] = pDataY[selected[index]];
										//point[2] = pDataZ[selected[index]];
										point[0] = pDataX[selected[index]] / 1000.0f;
										point[1] = pDataY[selected[index]] / 1000.0f;
										point[2] = pDataZ[selected[index]] / 1000.0f;
										//std::array<int, 3> point;
										//point[0] = pDataX[selected[index]];
										//point[1] = pDataY[selected[index]];
										//point[2] = pDataZ[selected[index]];
										builder.SetAttributeValueForPoint(pos_att_id, PointIndex(index), &(point)[0]);

										std::array<uint8_t, 3> color;
										color[0] = cData1[selected[index]];
										color[1] = cData2[selected[index]];
										color[2] = cData3[selected[index]];
										builder.SetAttributeValueForPoint(color_att_id, PointIndex(index), &(color)[0]);

										std::array<float, 1> size;
										size[0] = sData[selected[index]];
										//size[0] = 0.5;
										builder.SetAttributeValueForPoint(size_att_id, PointIndex(index), &(size)[0]);
									}

									std::unique_ptr<PointCloud> pc = builder.Finalize(false);
									MyAssert(pc != nullptr, 1001);
									MyAssert(pc->num_points() == nPointsNew, 1002);

									int compression_level = 0;
									EncoderBuffer buffer;
									PointCloudSequentialEncoder encoder;
									EncoderOptions options = EncoderOptions::CreateDefaultOptions();
									int quantization_bits = 10;
									options.SetGlobalInt("quantization_bits", quantization_bits);
									options.SetSpeed(10- compression_level, 10 - compression_level);
									encoder.SetPointCloud(*pc);

									MyAssert(encoder.Encode(options, &buffer).ok(), 2001);

									int nSize = buffer.size();
									/*
									DecoderBuffer dec_buffer;
									dec_buffer.Init(buffer.data(), buffer.size());
									PointCloudKdTreeDecoder decoder;

									std::unique_ptr<PointCloud> out_pc(new PointCloud());
									DecoderOptions dec_options;
									MyAssert(decoder.Decode(dec_options, &dec_buffer, out_pc.get()).ok(), 2002);
									MyAssert(out_pc->num_points() == nPointsNew, 1003);
									*/

									BYTE * p_send = new BYTE[nSize+CHUNK_HEADER_LEN];


									memcpy(p_send, pp_pc, CHUNK_HEADER_LEN);
									memcpy(p_send+CHUNK_HEADER_LEN, buffer.data(), buffer.size());
									//memcpy(pp_pc+nCompressedSize, pp_pc, CHUNK_HEADER_LEN);
									//memcpy(pp_pc+nCompressedSize+CHUNK_HEADER_LEN, buffer.data(), buffer.size());
									seqnum = (int)(NDKGetTime_comm() - t1);
									SendMessage_Data(chunkID, cellID, quality, cls, seqnum, p_send, nSize+CHUNK_HEADER_LEN, 0);
									//printf("%d %d %d %d %d %d ", chunkID, cellID, quality, cls, seqnum, nSize+CHUNK_HEADER_LEN);
									//SendMessage_Data(chunkID, cellID, quality, cls, seqnum, pp_pc+nCompressedSize, nSize+CHUNK_HEADER_LEN, 0);
									nCompressedSize += nSize + CHUNK_HEADER_LEN;




								}
							}
						}


						printf("end: %lu\n", NDKGetTime_comm());
						uint64_t t2 = NDKGetTime_comm();
						t_total = t2 - t1;
						printf("Total preparing time: %lu\n", t_total);
						printf("nCompressedSize: %d\n", nCompressedSize);

						delete [] pDataX;
						delete [] pDataY;
						delete [] pDataZ;
						delete [] cData1;
						delete [] cData2;
						delete [] cData3;
						delete [] sData;

						// delete [] pDataX_outer;
						// delete [] pDataY_outer;
						// delete [] pDataZ_outer;
						// delete [] cData1_outer;
						// delete [] cData2_outer;
						// delete [] cData3_outer;
						// delete [] sData_outer;

						// delete [] pointBuf;
						// delete [] colorBuf;




					}

					p+=5;
				}

				//transmit in a single bundle
				if (n > 0) TransmitMessages(); //the flag will be updated inside this call
				break;
			}

		default:
			MyAssert(0, 3538);
	}
}

void VIDEO_COMM::ReceiveMessage() {
	int fd = fds[0];
	int & nRecv = rcvdBytes;

	VIDEO_MESSAGE * pM = &rcvdMsg;

	while (1) {
		int nToRead;
		if (nRecv < 5)
			nToRead = 5;
		else
			nToRead = pM->msgLen;

		int n = read(fd, pM->pData + nRecv, nToRead - nRecv);
		if (n > 0) {
			nRecv += n;
			if (nRecv == 5) {

				/////////////////////////////////////////////////
				//handle wasted bytes due to LB
				//TODO: this is an inefficient solution
				int i;
				for (i=0; i<5; i++) {
					if (pM->pData[i] != 0xFF) break;
				}
				if (i>0) {
					if (i<5) memmove(pM->pData, pM->pData+i, 5-i);
					nRecv -= i;
					continue;
				}
				/////////////////////////////////////////////////

				pM->id = pM->pData[0];
				pM->msgLen = ReadInt(pM->pData + 1);
				MyAssert(pM->msgLen < SETTINGS::RECV_BUFFER_SIZE, 3409);
				//InfoMessage("*** msgLen = %d ***", pM->msgLen);

				MyAssert(pM->msgLen > 5, 3388);
			} else if (nRecv > 5 && nRecv == pM->msgLen) {
				ProcessMessage(pM);
				nRecv = 0;
			}
		} else if (n == 0 || (n < 0 && errno == ECONNRESET)) {
			//connection closed
			MyAssert(0, 3386);
			close(fd);
		} else if (n < 0 && (errno == EWOULDBLOCK)) {
			break;
		} else {
			ErrorMessage("Unexpected recv error %d: %s.", errno, strerror(errno));
			MyAssert(0, 3387);
		}
	}
}

void VIDEO_COMM::SendMessage_SelectVideo(const char * name) {
	static BYTE msgBuf[64];

	VIDEO_MESSAGE m;
	m.id = MSG_SELECT_VIDEO;

	int len = (int)strlen(name);
	MyAssert(len > 0 && len < 48, 3379);
	m.msgLen = len + 1 + 11;

	msgBuf[0] = m.id;
	WriteInt(msgBuf + 1, m.msgLen);
	WriteShort(msgBuf + 5, (short)(-1));	//bw
	WriteWORD(msgBuf + 7, (WORD)(100));		//bwDamp
	//WriteWORD(msgBuf + 9, (WORD)XMIT_QUEUE_UPDATE_REPLACE_KERNEL);
	WriteWORD(msgBuf + 9, (WORD)XMIT_QUEUE_UPDATE_ONCE);
	strcpy((char *) (msgBuf + 11), name);
	m.pData = msgBuf;

	sendMsgQueue.Enqueue(&m);
	TransmitMessages();
}

void VIDEO_COMM::SendMessage_RequestRandomChunk() {
	int chunkID = rand() % VIDEO_DATA::nChunks;
	int tileID = rand() % (VIDEO_DATA::nTilesX * VIDEO_DATA::nTilesY * VIDEO_DATA::nTilesZ);
	int quality = rand() % VIDEO_DATA::nQualities;

	static int seqnum = 0;

	SendMessage_RequestChunk(chunkID, tileID, quality, 0, seqnum++);
}

void VIDEO_COMM::SendMessage_RequestBatch(int seqnum, int len, BYTE * reqData) {
	VIDEO_MESSAGE m;
	m.id = MSG_BATCH_REQUESTS;
	m.msgLen = len;
	m.pData = reqData;

	WriteInt(reqData + 5, seqnum);

	sendMsgQueue.Enqueue(&m);
	TransmitMessages();
}

void VIDEO_COMM::SendMessage_RequestChunk(WORD chunkID, WORD tileID, BYTE quality, BYTE cls, int seqnum) {
	//InfoMessage("Sending request: c=%d t=%d q=%d c=%d seq=%d", chunkID, (int)tileID, (int)quality, (int)cls, seqnum);

	VIDEO_MESSAGE m;
	m.id = MSG_REQUEST_CHUNK;
	m.msgLen = CHUNK_HEADER_LEN;
	m.pData = vms.AllocateBlock(CHUNK_HEADER_LEN);

	m.pData[0] = m.id;
	WriteInt(m.pData + 1, m.msgLen);
	WriteWORD(m.pData + 5, chunkID);
	WriteWORD(m.pData + 7, tileID);
	//m.pData[7] = tileID;
	m.pData[9] = quality;
	//m.pData[9] = cls;
	WriteInt(m.pData + 10, seqnum);
	m.pData[14] = 0;	//TODO: flag

	sendMsgQueue.Enqueue(&m);
	TransmitMessages();
}

void VIDEO_COMM::SendMessage_VideoMetaData() {

	int encLen;
	BYTE * encData;
	VIDEO_DATA::GetEncodedData(encLen, encData);
	MyAssert(encLen > 0 && encData != NULL, 3714);

	VIDEO_MESSAGE m;
	m.id = MSG_VIDEO_METADATA;
	m.msgLen = ReadInt(encData + 1);
	m.pData = encData;

	InfoMessage("Metadata size = %d bytes", m.msgLen);
	sendMsgQueue.Enqueue(&m);
	TransmitMessages();
}

void VIDEO_COMM::SendMessage_Data(WORD chunkID, WORD tileID, BYTE quality, BYTE cls, int seqnum, BYTE * pData, int dataLen, int bTransmit) {
	VIDEO_MESSAGE m;
	m.id=  MSG_VIDEO_DATA;

	//IMPORTANT: the first 14 bytes of pData are reserved
	//the 14 bytes are included in dataLen
	pData[0] = m.id;
	WriteInt(pData + 1, dataLen);
	WriteWORD(pData + 5, chunkID);
	WriteWORD(pData + 7, tileID);
	//pData[7] = tileID;
	pData[9] = quality;

	MyAssert(ReadInt(pData+1) == dataLen, 3715);
	//pData[9] = cls;
	WriteInt(pData + 10, seqnum);
	pData[14] = 0;	//TODO: flag


	m.msgLen = dataLen;
	m.pData = pData;



	sendMsgQueue.Enqueue(&m);
	if (bTransmit) TransmitMessages();
}

void VIDEO_COMM::SendMessage_Data_dynamic(WORD chunkID, WORD tileID, BYTE quality, BYTE cls, int seqnum, BYTE * pData, int dataLen, int bTransmit) {
	VIDEO_MESSAGE m;
	m.id=  MSG_VIDEO_DATA;

	//IMPORTANT: the first 14 bytes of pData are reserved
	//the 14 bytes are included in dataLen
	pData[0] = m.id;
	WriteInt(pData + 1, dataLen);
	WriteWORD(pData + 5, chunkID);
	WriteWORD(pData + 7, tileID);
	//pData[7] = tileID;
	pData[9] = quality;

	MyAssert(ReadInt(pData+1) == dataLen, 3715);
	//pData[9] = cls;
	WriteInt(pData + 10, seqnum);
	
	pData[14] = 1;	//TODO: flag


	m.msgLen = dataLen;
	m.pData = pData;



	sendMsgQueue.Enqueue(&m);
	if (bTransmit) TransmitMessages();
}

int VIDEO_COMM::ConnectionSetup(const char * remoteIP) {
	fds[0] = -1;

	////////////////////////////// Local Proxy ///////////////////////////////
	if (mode == MODE_CLIENT) {
		MyAssert(remoteIP != NULL, 1719);
		fds[0] = socket(AF_INET, SOCK_STREAM, 0);
		//SetMaxSegSize(fd[i], MAGIC_MSS_VALUE);
		if (fds[0] < 0) return R_FAIL;
		SetSocketNoDelay_TCP(fds[0]);

		//TODO: set socket buffer
		//SetSocketBuffer(fd[i], PROXY_SETTINGS::pipeReadBufLocalProxy, PROXY_SETTINGS::pipeWriteBufLocalProxy); //This MUST be called before connect()!

		struct sockaddr_in serverAddr;
		memset(&serverAddr, 0, sizeof(serverAddr));
		serverAddr.sin_family = AF_INET;
		serverAddr.sin_port = htons((unsigned short)SETTINGS::SERVER_PORT);
		inet_pton(AF_INET, remoteIP, &serverAddr.sin_addr);

		if (connect(fds[0], (const struct sockaddr *) &serverAddr, sizeof(serverAddr)) != 0)
			return R_FAIL;

		//SetCongestionControl(fd[i], PROXY_SETTINGS::pipeProtocol[i].c_str());
		SetNonBlockIO(fds[0]);

		DWORD clientIP;
		WORD clientPort;
		GetClientIPPort(fds[0], clientIP, clientPort);
		InfoMessage("Pipe established. Local port=%d", (int)clientPort);


	} else { ////////////////////////////// server ///////////////////////////////
		MyAssert(remoteIP == NULL, 1718);

		int listenFD = socket(AF_INET, SOCK_STREAM, 0);
		if (listenFD < 0) return R_FAIL;

		struct sockaddr_in serverAddr;
		memset(&serverAddr, 0, sizeof(sockaddr_in));
		serverAddr.sin_family = AF_INET;
		serverAddr.sin_port = htons((unsigned short)SETTINGS::SERVER_PORT);
		serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

		int optval = 1;
		int r = setsockopt(listenFD, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
		MyAssert(r == 0, 1762);

		//TODO: set socket buffer
		//SetSocketBuffer(listenFD, PROXY_SETTINGS::pipeReadBufRemoteProxy, PROXY_SETTINGS::pipeWriteBufRemoteProxy); //This must be called for listenFD !

		if (bind(listenFD, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) != 0) return R_FAIL;
		if (listen(listenFD, 32) != 0) return R_FAIL;

		//wait for the connection
		struct sockaddr_in clientAddr;
		socklen_t clientAddrLen = sizeof(clientAddr);
		fds[0] = accept(listenFD, (struct sockaddr *)&clientAddr, &clientAddrLen);
		if (fds[0] == -1) return R_FAIL;

		//TODO: check endianness
		tcpTuple.serverIP = SETTINGS::SERVER_IP;
		tcpTuple.serverPort = (WORD)SETTINGS::SERVER_PORT;
		tcpTuple.clientIP = (DWORD)clientAddr.sin_addr.s_addr;
		tcpTuple.clientPort = (WORD)clientAddr.sin_port;

		//SetCongestionControl(fd[i], PROXY_SETTINGS::pipeProtocol[i].c_str());

		SetSocketNoDelay_TCP(fds[0]);
		SetNonBlockIO(fds[0]);
		int clientPort = (int)ntohs(clientAddr.sin_port);
		clientIPStr = ConvertDWORDToIP((DWORD)clientAddr.sin_addr.s_addr);

		InfoMessage("Pipe established. Client IP=%s Client port=%d", clientIPStr.c_str(), clientPort);


		close(listenFD);
	}

	memset(peers, 0, sizeof(peers));
	peers[0].fd = -1;
	peers[0].fd = fds[0];
	peers[0].events = POLLRDNORM;

	return R_SUCC;
}

void VIDEO_COMM::MainLoop() {
	while (1) {
		int nReady = poll(peers, 1, SETTINGS::POLL_TIMEOUT);
		MyAssert(nReady >= 0, 1699);

		int peerFD = peers[0].fd;
		MyAssert(peerFD >= 0, 3377);


		if (peers[0].revents & (POLLRDNORM | POLLERR | POLLHUP)) {
			ReceiveMessage();
		}

		if (peers[0].revents & POLLWRNORM) {
			//peers[0].events &= ~POLLWRNORM;
			if (tileXMitMode == XMIT_QUEUE_UPDATE_REPLACE_KERNEL && KERNEL_INTERFACE::IsLBEnabled()) {
				TransmitContainers();
			} else {
				TransmitMessages();
			}
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////

#define SPERKE_MAGIC 122

#define SPERKE_IOCTL_GLOBAL_INIT	_IOW(SPERKE_MAGIC, 0, unsigned char *)
#define SPERKE_IOCTL_VIDEO_INIT		_IOW(SPERKE_MAGIC, 1, unsigned char *)
#define SPERKE_IOCTL_NEW_BATCH		_IOW(SPERKE_MAGIC, 2, unsigned long)
#define SPERKE_IOCTL_START_LB			_IOW(SPERKE_MAGIC, 3, int)
#define SPERKE_IOCTL_STOP_LB			_IOW(SPERKE_MAGIC, 4, int)

int KERNEL_INTERFACE::fd = -1;
BYTE * KERNEL_INTERFACE::sperke_data_base = NULL;
CHUNK_INFO * KERNEL_INTERFACE::pMeta = NULL;
BYTE * KERNEL_INTERFACE::pBuf = NULL;
BYTE * KERNEL_INTERFACE::pBatch = NULL;
int * KERNEL_INTERFACE::pFastShare = NULL;
int KERNEL_INTERFACE::bLateBindingEnabled = 0;

#define PAGE_SHIFT      12
#define PAGE_SIZE       (1UL << PAGE_SHIFT)
#define PAGE_MASK       (~(PAGE_SIZE-1))
#define PAGE_ALIGN(addr)        (((addr)+PAGE_SIZE-1)&PAGE_MASK)

void KERNEL_INTERFACE::GlobalInit() {
	MyAssert(VIDEO_COMM::isKMLoaded && VIDEO_COMM::mode == MODE_SERVER, 3711);
	fd = open("/dev/sperke", O_RDWR);
	MyAssert(fd >= 0, 3699);

	unsigned char opt[20];
	WriteInt(opt,	 SETTINGS::SERVER_BUFFER_SIZE);
	WriteInt(opt+4,	 SETTINGS::MAX_CTQ);
	WriteInt(opt+8,	 SETTINGS::MAX_CT);
	WriteInt(opt+12, SETTINGS::MAX_BATCHES);
	WriteInt(opt+16, SETTINGS::MAX_BATCH_SIZE);
	int sperke_data_size = ioctl(fd, SPERKE_IOCTL_GLOBAL_INIT, opt);

	MyAssert(
	sperke_data_size == PAGE_ALIGN(
		sizeof(CHUNK_INFO) * SETTINGS::MAX_CTQ +
		SETTINGS::SERVER_BUFFER_SIZE +
		SETTINGS::MAX_BATCHES * SETTINGS::MAX_BATCH_SIZE +
		SETTINGS::MAX_CT + FAST_SHARE_SIZE
	), 3701);

	sperke_data_base = (unsigned char *)mmap(0, sperke_data_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
	MyAssert(sperke_data_base != MAP_FAILED && sperke_data_base != NULL, 2025);

	pFastShare =  (int*) sperke_data_base;
	BYTE * pMap =	(BYTE *)pFastShare + FAST_SHARE_SIZE;
	pMeta = 	(CHUNK_INFO *)(pMap + SETTINGS::MAX_CT);
	pBuf = 		(BYTE *)pMeta + sizeof(CHUNK_INFO) * SETTINGS::MAX_CTQ;
	pBatch =	pBuf + SETTINGS::SERVER_BUFFER_SIZE;
}

void KERNEL_INTERFACE::VideoInit() {
	MyAssert(fd >= 0, 3700);

	unsigned char opt[24];
	WriteInt(opt,	 VIDEO_DATA::nChunks);
	WriteInt(opt+4,	 VIDEO_DATA::nTilesX * VIDEO_DATA::nTilesY * VIDEO_DATA::nTilesZ);
	WriteInt(opt+8,  VIDEO_DATA::nQualities);

	TCP_TUPLE & tt = VIDEO_COMM::tcpTuple;
	MyAssert(tt.serverIP > 0, 3713);
	memcpy(opt+12, &tt.serverIP, 4);
	memcpy(opt+16, &tt.clientIP, 4);
	memcpy(opt+20, &tt.serverPort, 2);
	memcpy(opt+22, &tt.clientPort, 2);

	int ret;
	ret = ioctl(fd, SPERKE_IOCTL_VIDEO_INIT, opt);
	MyAssert(ret == 0, 2033);
}

int KERNEL_INTERFACE::OnNewBatch(int seqnum) {
	MyAssert(fd>=0 && bLateBindingEnabled, 3712);

	pFastShare[0] = VIDEO_COMM::userOutBytes + VIDEO_COMM::userQueuedBytes;
	pFastShare[1] = VIDEO_COMM::curBatch;
	pFastShare[2] = seqnum;
	return ioctl(fd, SPERKE_IOCTL_NEW_BATCH, 0);
}

void KERNEL_INTERFACE::EnableLateBinding() {
	MyAssert(fd>=0 && !bLateBindingEnabled, 3708);
	int ret = ioctl(fd, SPERKE_IOCTL_START_LB, 0);
	MyAssert(ret == 0, 3709);
	bLateBindingEnabled = 1;
}

void KERNEL_INTERFACE::DisableLateBinding() {

	if (fd < 0 || !bLateBindingEnabled) return;
	int ret = ioctl(fd, SPERKE_IOCTL_STOP_LB, 0);

	if (ret != 0) {
		ErrorMessage("DisableLateBinding failed");
	}

	bLateBindingEnabled = 0;
}

int KERNEL_INTERFACE::IsLBEnabled() {
	return bLateBindingEnabled;
}

void KERNEL_INTERFACE::DetectKernelModule() {
	int _fd = open("/dev/sperke", O_RDWR);
	if (_fd == -1) {
		VIDEO_COMM::isKMLoaded = 0;
		InfoMessage("Kernel Module NOT Detected");
	} else {
		close(fd);
		VIDEO_COMM::isKMLoaded = 1;
		InfoMessage("Kernel Module Detected");
	}
}

//////////////////////////////////////////////////////////////////////////////////////////

vector<int> TC_INTERFACE::bw;

int TC_INTERFACE::CheckFileForExec(const char * pathname) {
	int r = access(pathname, R_OK | W_OK | X_OK);
	if (r == 0) return 1; else return 0;
}

void TC_INTERFACE::CleanUpAndSystemCheck() {
	//check root
	if (geteuid() != 0) {
		InfoMessage("Need root permission");
		MyAssert(0, 3741);
	}

	char filename[256];
	sprintf(filename, "%s/emulate.sh", SETTINGS::BASE_DIRECTORY.c_str());
	if (!CheckFileForExec(filename)) {
		InfoMessage("Cannot find emulate.sh");
		MyAssert(0, 3739);
	}

	sprintf(filename, "%s/emulate_cleanup.sh", SETTINGS::BASE_DIRECTORY.c_str());
	if (!CheckFileForExec(filename)) {
		InfoMessage("Cannot find emulate_cleanup.sh");
		MyAssert(0, 3740);
	}

	int r = system(filename);
}

void TC_INTERFACE::LoadBWTrace(const char * filename, double dampFactor) {
	char fullpath[256];
	sprintf(fullpath, "%s/%s", SETTINGS::BASE_DIRECTORY.c_str(), filename);

	FILE * ifs = fopen(fullpath, "rb");
	MyAssert(ifs != NULL, 3742);
	int n;

	bw.clear();
	double f1, f2, f3;
	int i1, i2, i3;

	while (!feof(ifs)) {
		//n = fscanf(ifs, "%lf %d %lf %lf %d %d", &f1, &i1, &f2, &f3, &i2, &i3);
		n = fscanf(ifs, "%d", &i2);	// Fixed by Bo Han 06092019
		MyAssert(i2 >= 0, 3663);
		if (n != 1) break;
		dampFactor = 0.2;	//NanWu
		int kbps = int(i2 * dampFactor * 8 / 1000 + 0.5f);
		if (kbps < 10) kbps = 10;	//to satisfy tc
		printf("%d %d\n", i2, kbps);
		bw.push_back(kbps);	//bytes to kbits
	}

	fclose(ifs);
}

void TC_INTERFACE::InitReplay(const char * clientIP, int baseRTT) {
	char filename[256];
	sprintf(filename, "%s/emulate.sh %d %d %s %d",
		SETTINGS::BASE_DIRECTORY.c_str(),
		baseRTT, 5000, clientIP, SETTINGS::SERVER_PORT);	//set initial BW to 5000kbps
	int r = system(filename);
}

//essentially implement what is implemented in emulate_prof.sh
void * TC_INTERFACE::ReplayThread(void * arg) {
	int interval = (int)(long)arg;
	int n = bw.size();

	interval *= 1000;	//ms -> micro-sec
	const int EXE_TIME_COMPENSATION = 1;	//1ms
	interval -= EXE_TIME_COMPENSATION * 1000;

	char cmd[256];

	std::ostringstream oss;
	std::string logMessage;

	while (1) {
		for (int i=0; i<n; i++) {
			//original cmd: sudo tc qdisc change dev ifb1 handle 1: root tbf rate $a"kbit" burst 20k latency 50ms
			sprintf(cmd, "sudo tc qdisc change dev ifb1 handle 1: root tbf rate %d\"kbit\" burst 20k latency 1ms", bw[i]);		// Latency not related to network RTT (Bo Han 06092019).
			InfoMessage(cmd);

			oss << cmd << std::endl;

			logMessage = oss.str();

			oss.str("");
			oss.clear();

			writeToLogFile(logMessage);
			int r = system(cmd);
			usleep(interval);
		}
	}
}

void TC_INTERFACE::StartReplay(int interval) {
	static int bStarted = 0;
	MyAssert(!bStarted, 3743);
	bStarted = 1;

	#ifndef VS_SIMULATION
	pthread_t replayThread;
	int r = pthread_create(&replayThread, NULL, ReplayThread, (void *)(long)interval);
	MyAssert(r == 0, 1724);
	#endif
}

