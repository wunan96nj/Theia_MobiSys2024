#include "stdafx.h"
#include "videodata.h"
#include "videocomm.h"
#include "tools.h"
#include "settings.h"
#include <math.h>

int VIDEO_DATA::nChunks = 0;
int VIDEO_DATA::nTilesX = 0;
int VIDEO_DATA::nTilesY = 0;
int VIDEO_DATA::nTilesZ = 0;
int VIDEO_DATA::nQualities = 0;
int VIDEO_DATA::gopSize = 0;
int VIDEO_DATA::tileWidth = 0;
int VIDEO_DATA::tileHeight = 0;

int VIDEO_DATA::encodedMetaDataLen = 0;
BYTE * VIDEO_DATA::pEncodedMetaData = NULL;

BYTE * VIDEO_DATA::pBuf = NULL;
CHUNK_INFO * VIDEO_DATA::pMeta = NULL;

float * VIDEO_DATA::bitrateUtilities = NULL;

long VIDEO_DATA::bufSize = 0;

int * VIDEO_DATA::chunkSizes = NULL;
BYTE * VIDEO_DATA::pBatch = NULL;

void VIDEO_DATA::Init() {
	if (VIDEO_COMM::mode == MODE_SERVER) {

		if (VIDEO_COMM::isKMLoaded) {
			KERNEL_INTERFACE::GlobalInit();
		}

		pBuf = new BYTE[SETTINGS::SERVER_BUFFER_SIZE];
		MyAssert(pBuf != NULL, 3397);
		pMeta = new CHUNK_INFO[SETTINGS::MAX_CTQ];
		MyAssert(pMeta != NULL, 3398);
		pBatch = NULL;

		/*
		if (tileXMitMode == bKM) {
			KERNEL_INTERFACE::GlobalInit();
			pBuf = KERNEL_INTERFACE::pBuf;
			pMeta = KERNEL_INTERFACE::pMeta;
			pBatch = KERNEL_INTERFACE::pBatch;
			//pMap = KERNEL_INTERFACE::pMap;
		} else {
			pBuf = new BYTE[SETTINGS::SERVER_BUFFER_SIZE];
			MyAssert(pBuf != NULL, 3397);
			pMeta = new CHUNK_INFO[SETTINGS::MAX_CTQ];
			MyAssert(pMeta != NULL, 3398);
			pBatch = NULL;
		}
		*/

		pEncodedMetaData = new BYTE[SETTINGS::MAX_CTQ * 6 + 33 + 3 * MAX_QUALITIES * sizeof(float)];
		MyAssert(pEncodedMetaData != NULL, 3403);

	} else {
		chunkSizes = new int[SETTINGS::MAX_CTQ];
		MyAssert(chunkSizes != NULL, 3413);
		MyAssert(VIDEO_COMM::isKMLoaded == 0, 3825);
	}
}

void VIDEO_DATA::SwitchToKernelMemory() {
	MyAssert(VIDEO_COMM::isKMLoaded, 3826);
	pBuf = KERNEL_INTERFACE::pBuf;
	pMeta = KERNEL_INTERFACE::pMeta;
	pBatch = KERNEL_INTERFACE::pBatch;
	//pMap = KERNEL_INTERFACE::pMap;
}

void VIDEO_DATA::UnloadVideo() {
	bufSize = 0;
	encodedMetaDataLen = 0;
}


void VIDEO_DATA::DecodeMetaData(BYTE * pData, int metaLen) {
	MyAssert(VIDEO_COMM::mode == MODE_CLIENT, 3414);

	BYTE * p = pData;
	MyAssert(*p == MSG_VIDEO_METADATA, 3411);
	p++;

	MyAssert(ReadInt(p) == metaLen, 3412);
	p+=sizeof(int);

	VIDEO_DATA::nChunks = ReadInt(p); p += sizeof(int);
	VIDEO_DATA::nTilesX  = ReadInt(p); p += sizeof(int);
	VIDEO_DATA::nTilesZ  = ReadInt(p); p += sizeof(int);
	VIDEO_DATA::nQualities = ReadInt(p); p += sizeof(int);
	VIDEO_DATA::gopSize = ReadInt(p); p += sizeof(int);
	VIDEO_DATA::tileWidth = ReadInt(p); p += sizeof(int);
	VIDEO_DATA::tileHeight = ReadInt(p); p += sizeof(int);

	int nTiles = nTilesX * nTilesY * nTilesZ;
	MyAssert(nChunks * nTiles * nQualities <= SETTINGS::MAX_CTQ, 3422);

	int buLen = 3 * MAX_QUALITIES * sizeof(float);

	//dump utilities
	float * bu = (float *)p;
	for (int i=0; i<3; i++) {
		printf("Util Func %d: ", i);
		int k = i * MAX_QUALITIES;
		for (int j=0; j<nQualities; j++) {
			printf("%.2f ", bu[k]);
			k++;
		}
		printf("\n");
	}

	p += buLen;
	for (int i=0; i<nChunks; i++) {
		for (int j=0; j<nTiles; j++) {
			for (int k=0; k<nQualities; k++) {
				int len = ReadWORD(p);
				p += sizeof(WORD);

				if (len == 0) {
					len = ReadInt(p);
					p += sizeof(int);
				}

				chunkSizes[i*nTiles*nQualities + j*nQualities + k] = len;
			}
		}
	}
}

void VIDEO_DATA::EncodeMetaData() {
	MyAssert(encodedMetaDataLen == 0 && VIDEO_COMM::mode == MODE_SERVER, 3404);

	BYTE * p = pEncodedMetaData;
	*p = MSG_VIDEO_METADATA;	p++;
	p += sizeof(int);	//skip the total size for now;
	WriteInt(p, nChunks);	 p += sizeof(int);
	WriteInt(p, nTilesX);	 p += sizeof(int);
	WriteInt(p, nTilesZ);	 p += sizeof(int);
	WriteInt(p, nQualities); p += sizeof(int);
	WriteInt(p, gopSize);	 p += sizeof(int);
	WriteInt(p, tileWidth);	 p += sizeof(int);
	WriteInt(p, tileHeight); p += sizeof(int);
	//33 bytes so far

	int buLen = 3 * MAX_QUALITIES * sizeof(float);
	memcpy(p, bitrateUtilities, buLen);
	p += buLen;

	int nTiles = nTilesX * nTilesY * nTilesZ;

	for (int i=0; i<nChunks; i++) {
		for (int j=0; j<nTiles; j++) {
			for (int k=0; k<nQualities; k++) {
				int points = pMeta[i*nTiles*nQualities + j*nQualities + k].points;
				if (points == 0) continue;

				int frameID = pMeta[i*nTiles*nQualities + j*nQualities + k].frameID;
				if (!(frameID >= 0 && frameID <= 0xFFFF)) InfoMessage("%d %d %d %d %d %d", frameID, i, j, k, nTiles, nQualities);
				MyAssert(frameID >= 0 && frameID <= 0xFFFF, 3405);
				WriteWORD(p, (WORD)frameID);
				p+=2;

				int cellID = pMeta[i*nTiles*nQualities + j*nQualities + k].cellID;
				MyAssert(cellID >= 0 && cellID <= 0xFFFF, 3406);
				WriteWORD(p, (WORD)cellID);
				p+=2;

				int len = pMeta[i*nTiles*nQualities + j*nQualities + k].len;
				/*
				if (frameID == 299 && cellID == 26) {
					printf("%d %d %d %d\n", frameID, cellID, len, points);
					getchar();
				}
				*/
				MyAssert(len > 0, 3407);
				if (len <= 0xFFFF) {
					WriteWORD(p, (WORD)len);
					p+=2;
				} else {
					WriteWORD(p, 0);
					p+=2;
					WriteInt(p, len);
					p+=sizeof(int);
				}

				MyAssert(points > 0, 3408);
				if (points <= 0xFFFF) {
					WriteWORD(p, (WORD)points);
					p+=2;
				} else {
					WriteWORD(p, 0);
					p+=2;
					WriteInt(p, points);
					p+=sizeof(int);
				}
				*p = k; p++;
			}
			//getchar();
		}
	}

	encodedMetaDataLen = p - pEncodedMetaData;
	WriteInt(pEncodedMetaData + 1, encodedMetaDataLen);
}

char * VIDEO_DATA::SkipComment(FILE * ifs) {
	static char line[256];
	while (!feof(ifs)) {
		if (fgets(line, sizeof(line), ifs) == NULL) {
			MyAssert(0, 3429);
			return NULL;
		}

		Chomp(line);
		if (strlen(line) == 0) continue;
		if (strlen(line) >= 2 && line[0] == '/' && line[1] == '/') continue;


		//InfoMessage("***** Line = %s", line);

		return line;
	}

	MyAssert(0, 3430);
	return NULL;
}

void VIDEO_DATA::EncodeFrameSizes(BYTE * pData, vector<int> & sizes) {
	//no compression
	for (int i=0; i<gopSize; i++) {
		WriteInt(pData, sizes[i]);
		pData += sizeof(int);
	}

	/*
	//compression
	int n = 0;

	int largeNumLim = (bufLim - sizeof(WORD) * gopSize) / (sizeof(WORD) + sizeof(int));
	MyAssert(largeNumLim >= 0, 3444);

	int largeNum = 0;

	for (int i=0; i<gopSize; i++) {
		if (sizes[i] <= 0xFFFF) {
			WriteWORD(pData, sizes[i]);
			pData += sizeof(WORD);
		} else {
			MyAssert(++largeNum <= largeNumLim, 3445);
			WriteWORD(pData, 0);
			pData += sizeof(WORD);
			WriteInt(pData, sizes[i]);
			pData += sizeof(int);
		}
	}
	*/
}

void VIDEO_DATA::LoadVideo(const char * name) {
	InfoMessage("Loading video...");
	if (nChunks != 0) {
		UnloadVideo();
	}

	char filename[256];
	int r;
	char * _line;
	vector<string> crf;
	vector<string> v;
	vector<string> bitrate;

	//Step 1: read info.txt
	sprintf(filename, "%s/%s/info.txt", SETTINGS::BASE_DIRECTORY.c_str(), name);
	FILE * ifs = OpenFileForRead(filename);
	_line = SkipComment(ifs);
	r = sscanf(_line, "%d", &nTilesX); MyAssert(r == 1, 3428);

	if (strstr(name, "1x1x1") != NULL) nTilesY = 1;
	else if (strstr(name, "2x2x2") != NULL) nTilesY = 2;
	else if (strstr(name, "4x4x4") != NULL) nTilesY = 4;
	else if (strstr(name, "8x8x8") != NULL) nTilesY = 8;
	else if (strstr(name, "10x10x10") != NULL) nTilesY = 10;
	else {
		InfoMessage("No supported segmentation (%s)", name);
		return;
	}

	_line = SkipComment(ifs);
	r = sscanf(_line, "%d", &nTilesZ); MyAssert(r == 1, 3428);

	_line = SkipComment(ifs);
	r = sscanf(_line, "%d", &nQualities); MyAssert(r == 1, 3431);

	_line = SkipComment(ifs);
	Split(_line, (const char *) " ", crf);
	MyAssert(crf.size() == nQualities, 3432);

	_line = SkipComment(ifs);
	r = sscanf(_line, "%d", &gopSize); MyAssert(r == 1, 3431);

	_line = SkipComment(ifs);
	r = sscanf(_line, "%d", &tileWidth); MyAssert(r == 1, 3431);

	_line = SkipComment(ifs);
	r = sscanf(_line, "%d", &tileHeight); MyAssert(r == 1, 3431);

	_line = SkipComment(ifs);
	Split(_line, (const char *) " ", bitrate);
	MyAssert(bitrate.size() == nQualities, 3432);

	MyAssert(nTilesX>0 && nTilesZ>0 && tileWidth>0 && tileHeight>0, 3447);
	fclose(ifs);

	long tmpBufSize = 256*1024*1024;
	BYTE * pTmp = new BYTE[tmpBufSize];
	MyAssert(pTmp != NULL, 3434);
	int * pChunkLen;
	VIDEO_DATA::nChunks = 1;	//Bo Han 0530
	BYTE * pFrameSizes;
	char line[256];

	int nTiles = nTilesX * nTilesY * nTilesZ;
	int frameID = 0;
	int rFrameID = 0;

	bufSize = 0;
	//read each (tile, quality)
	for (int i = 0; i < nQualities; i++) {
        //dat file original point cloud
        sprintf(filename, "%s/%s/pc_%s.dat", SETTINGS::BASE_DIRECTORY.c_str(), name, crf[i].c_str());
		long filesize;
		r = GetFileSize(filename, filesize);
		MyAssert(r, 3433);

		ifs = OpenFileForRead(filename);

		//fclose(ifs);

        //video data file
        /*
		sprintf(filename, "%s/%s/video.%s", SETTINGS::BASE_DIRECTORY.c_str(), name, crf[i].c_str());
		long filesize;
		r = GetFileSize(filename, filesize);
		MyAssert(r, 3433);

		if (filesize >= tmpBufSize) {
			tmpBufSize = long(filesize * 1.1);
			//InfoMessage("%lu\n", tmpBufSize);
			delete [] pTmp;
			pTmp = new BYTE[tmpBufSize];
			MyAssert(pTmp != NULL, 3815);
		}

		ifs = OpenFileForRead(filename);
		r = fread(pTmp, filesize, 1, ifs);
		MyAssert(r == 1, 3435);
		fclose(ifs);
		*/

		//metafile
		/*
		sprintf(filename, "%s/%s/meta.%s", SETTINGS::BASE_DIRECTORY.c_str(), name, crf[i].c_str());
		ifs = OpenFileForRead(filename);
		int cellID = 0;
		int nPoints = 0;
		int nSize = 0;

        */
		frameID = 0;
		long offset = 0;
        int cellID = 0;
		int nPoints = 0;
		int nSize = 0;
		//read frame-by-frame
		//getchar();
		//nTiles = 1;
		while (!feof(ifs)) {
			for (int j = 0; j < nTiles; j++) {
                printf("frameID: %d\n", frameID);
				rFrameID = frameID;
				MyAssert(rFrameID == frameID, 5001);

				cellID = j;
				MyAssert(cellID == j, 5002);

				if (j == 0){
                    r = fread(&nPoints, sizeof(int), 1, ifs);
                    if (r != 1) break;
                    printf("nPoints: %d\n", nPoints);
                    int frameTime;
                    r = fread(&frameTime, sizeof(int), 1, ifs);
                    MyAssert(r == 1, 4006);

                    char * pointBuf = new char[nPoints * sizeof(short) * 3];
                    char * colorBuf = new char[nPoints * sizeof(char) * 3];

                    int pDataSize = sizeof(short) * nPoints * 3;
                    r = fread(pointBuf, pDataSize, 1, ifs);
                    MyAssert(r == 1, 4005);

                    int cDataSize = sizeof(char) * nPoints * 3;
                    r = fread(colorBuf, cDataSize, 1, ifs);
                    MyAssert(r == 1, 4008);

                    nSize = nPoints*sizeof(short)*3+nPoints*sizeof(char)*3;

                    int idx = frameID*nTiles*nQualities + j*nQualities + i;
                    pMeta[idx].offset = bufSize;
                    pMeta[idx].frameID = frameID;
                    pMeta[idx].cellID = cellID;
                    pMeta[idx].quality = i;
                    pMeta[idx].len = nSize + CHUNK_HEADER_LEN;
                    pMeta[idx].points = nPoints;

                    BYTE * pData = pBuf + bufSize;
                    pData[0] = MSG_VIDEO_DATA;
                    WriteInt(pData + 1, nSize + CHUNK_HEADER_LEN);
                    WriteWORD(pData + 5, (WORD)frameID);
                    WriteWORD(pData + 7, (WORD)cellID);
                    pData[9] = (BYTE)i;
                    WriteInt(pData + 10, -1);	//seqnum - this will not be used by the client
                    pData[14] = 0;	//bLast
                    bufSize += CHUNK_HEADER_LEN;

                    //memcpy(pBuf + bufSize , pTmp + offset + , nSize);
                    memcpy(pBuf + bufSize , pointBuf, pDataSize);
                    memcpy(pBuf + bufSize + pDataSize, colorBuf, cDataSize);

                    offset += nSize; //not needed, copy from point and color buf
                    bufSize += nSize; //the start of the frame from pBuf

                    //printf("%d %d %d %d %ld %ld %ld\n", frameID, cellID, nPoints, nSize, offset, bufSize, SETTINGS::SERVER_BUFFER_SIZE);
                    MyAssert(bufSize < SETTINGS::SERVER_BUFFER_SIZE, 3441);


				} else {
                    nSize = 0;
                    nPoints = 0;
                    int idx = frameID*nTiles*nQualities + j*nQualities + i;
                    pMeta[idx].offset = bufSize;
                    pMeta[idx].frameID = frameID;
                    pMeta[idx].cellID = cellID;
                    pMeta[idx].quality = i;
                    pMeta[idx].len = nSize + CHUNK_HEADER_LEN;
                    pMeta[idx].points = nPoints;

                    BYTE * pData = pBuf + bufSize;
                    pData[0] = MSG_VIDEO_DATA;
                    WriteInt(pData + 1, nSize + CHUNK_HEADER_LEN);
                    WriteWORD(pData + 5, (WORD)frameID);
                    WriteWORD(pData + 7, (WORD)cellID);
                    pData[9] = (BYTE)i;
                    WriteInt(pData + 10, -1);	//seqnum - this will not be used by the client
                    pData[14] = 0;	//bLast
                    bufSize += CHUNK_HEADER_LEN;

                    //memcpy(pBuf + bufSize , pTmp + offset + , nSize);


                    offset += nSize; //not needed, copy from point and color buf
                    bufSize += nSize; //the start of the frame from pBuf

                    //printf("%d %d %d %d %ld %ld %ld\n", frameID, cellID, nPoints, nSize, offset, bufSize, SETTINGS::SERVER_BUFFER_SIZE);
                    MyAssert(bufSize < SETTINGS::SERVER_BUFFER_SIZE, 3441);

				}

			}
			frameID++;

		}

		fclose(ifs);
	}


	nChunks = rFrameID;

	if (bitrateUtilities == NULL) {
		bitrateUtilities = new float[3 * MAX_QUALITIES];
	}

	//ComputeBitrateUtility(BITRATE_UTILITY_LINEAR, bitrateUtilities, bitrate); //Nan skip for now
	//ComputeBitrateUtility(BITRATE_UTILITY_LOG, bitrateUtilities + MAX_QUALITIES, bitrate); //Nan skip for now
	//ComputeBitrateUtility(BITRATE_UTILITY_HD, bitrateUtilities + MAX_QUALITIES * 2, bitrate); //Nan skip for now

	EncodeMetaData();
	//FillInFixedVideoDataHeaders();

	InfoMessage("Video %s loaded. nChunks=%d nTiles=%dx%dx%d nQualities=%d gopSize=%d frameSize=%dx%d totalSize=%.2lf MB",
		name, nChunks, nTilesX, nTilesY, nTilesZ, nQualities, gopSize,tileWidth,tileHeight,
		bufSize / 1e6
	);
}

void VIDEO_DATA::ComputeBitrateUtility(int bitrateUtility, float * bu, vector<string> & bitrate) {
	int nTiles = nTilesX * nTilesY * nTilesZ;
	for (int i=0; i<nQualities; i++) {
		bu[i] = 0.0f;
	}

	if (bitrateUtility == BITRATE_UTILITY_HD) {
		//sigcomm'17 settings
		// added more qualities Ding
		MyAssert(nQualities == 5, 3683);
		bu[0] = 1.0f / nTiles;
		bu[1] = 2.0f / nTiles;
		bu[2] = 3.0f / nTiles;
		bu[3] = 12.0f / nTiles;
		bu[4] = 15.0f / nTiles;
		//bu[5] = 18.0f / nTiles; // addedy randomly
		return;
	}

	/*
	for (int i=0; i<nChunks; i++) {
		for (int j=0; j<nTiles; j++) {
			for (int k=0; k<nQualities; k++) {
				int size = GetChunkSize(i, j, k);
				bu[k] += size;
			}
		}
	}

	//convert to mbps
	float minBitrateQ = 1e30f;
	for (int i=0; i<nQualities; i++) {
		bu[i] = bu[i] * 8.0f / (1e6f * nTiles * nChunks);
		if (bu[i] < minBitrateQ)
			minBitrateQ = bu[i];
	}
	*/

	float minBitrateQ = 1e30f;
	for (int i=0; i<nQualities; i++) {
		bu[i] = (float)atof(bitrate[i].c_str());
		if (bu[i] < minBitrateQ)
			minBitrateQ = bu[i];
	}

	//apply the utility function
	switch (bitrateUtility) {
		case BITRATE_UTILITY_LINEAR:
			{
				break;	//nothing to do
			}

		case BITRATE_UTILITY_LOG:
			{
				for (int i=0; i<nQualities; i++) {
					bu[i] = log(bu[i] / minBitrateQ) / nTiles;
				}
				break;
			}

		default:
			MyAssert(0, 3682);
	}
}

void VIDEO_DATA::FillInFixedVideoDataHeaders() {
	int nTiles = nTilesX * nTilesY * nTilesZ;
	for (int i=0; i<nChunks; i++) {
		for (int j=0; j<nTiles; j++) {
			for (int k=0; k<nQualities; k++) {
				int len;
				BYTE * pData = GetChunk(i, j, k, &len);
				pData[0] = MSG_VIDEO_DATA;
				WriteInt(pData + 1, len);
				WriteWORD(pData + 5, (WORD)i);
				pData[7] = (BYTE)j;
				pData[8] = (BYTE)k;
				pData[9] = 0xFF;		//cls - this will not be used by the client
				WriteInt(pData + 10, -1);	//seqnum - this will not be used by the client
				pData[14] = 0;	//bLast
			}
		}
	}
}

void VIDEO_DATA::LoadVideo_Emulation(CHUNK_INFO * pInfo, int chunkID, int tileID, int quality) {
	//reserve 14 bytes for message header

	pInfo->offset = bufSize;
	//pInfo->len = rand() % 8192 + 11;
	pInfo->len = rand() % 81920 + 11;

	MyAssert(bufSize + pInfo->len < SETTINGS::SERVER_BUFFER_SIZE, 3399);

	for (int i=0; i<pInfo->len; i++) {
		pBuf[bufSize + i] = rand() % 256;
	}

	memset(pBuf + bufSize, 0, CHUNK_HEADER_LEN);	//fill 0 for header;
	bufSize += pInfo->len;
}

void VIDEO_DATA::LoadVideo_Emulation(const char * name) {
	if (nChunks != 0) {
		UnloadVideo();
	}

	/*
	nChunks = 1000;
	nTiles = 20;
	nQualities = 5;
	*/

	nChunks = 100;
	nTilesX = 5;
	nTilesZ = 4;
	nQualities = 5;

	int nTiles = nTilesX * nTilesY * nTilesZ;
	//for now, generate them synthetically
	for (int i=0; i<nChunks; i++) {
		for (int j=0; j<nTiles; j++) {
			for (int k=0; k<nQualities; k++) {
				CHUNK_INFO * pInfo = &pMeta[i*nTiles*nQualities + j*nQualities + k];
				LoadVideo_Emulation(pInfo, i, j, k);
			}
		}
	}

	//encode meta data
	EncodeMetaData();

	InfoMessage("Video %s loaded. %d chunks, %d tiles, %d qualities, %.2lf MB",
		name, nChunks, nTiles, nQualities,
		bufSize / 1e6
	);
}

BYTE * VIDEO_DATA::GetChunk(int chunkID, int tileID, int quality, int * dataLen) {
	int nTiles = nTilesX * nTilesY * nTilesZ;

	if (!(chunkID>=0 && chunkID<nChunks &&
		tileID>=0 && tileID<nTiles &&
		quality>=0 && quality<nQualities))
		InfoMessage("%d %d %d %d %d %d \n", chunkID, nChunks, tileID, nTiles, quality, nQualities);

	// MyAssert(
	// 	chunkID>=0 && chunkID<nChunks &&
	// 	tileID>=0 && tileID<nTiles &&
	// 	quality>=0 && quality<nQualities, 3401
	// );

	MyAssert(
		chunkID>=0 && chunkID<nChunks &&
		tileID>=0 && tileID<nTiles &&
		quality>=0, 3401
	);

	CHUNK_INFO * pInfo = &pMeta[chunkID*nTiles*nQualities + tileID*nQualities + quality];
	*dataLen = pInfo->len;

	return pBuf + pInfo->offset;
}

void VIDEO_DATA::GetEncodedData(int & encLen, BYTE * & encData) {
	MyAssert(VIDEO_COMM::mode == MODE_SERVER, 3406);
	encLen = encodedMetaDataLen;
	encData = pEncodedMetaData;
}

int VIDEO_DATA::GetChunkSize(int chunkID, int tileID, int quality) {
	int idx = chunkID * nTilesX * nTilesY * nTilesZ * nQualities + tileID * nQualities + quality;
	return pMeta[idx].len;
}

int VIDEO_DATA::GetChunkPoints(int chunkID, int tileID, int quality) {
	int idx = chunkID * nTilesX * nTilesY * nTilesZ * nQualities + tileID * nQualities + quality;
	return pMeta[idx].points;
}

