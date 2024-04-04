// 360video_networking.cpp : Defines the entry point for the console application.
//

#include "theia_server.h"
#include "tools.h"
#include "videocomm.h"
#include "videodata.h"

void DisplayUsage(char * * argv) {
	printf("client mode:\n");
	printf("%s 1 [serverIP]\n", argv[0]);
	printf("Server mode:\n");
	printf("%s 2 \n", argv[0]);
	exit(-1);
}

void * TestingThread(void * arg) {
	MyAssert(VIDEO_COMM::mode == MODE_CLIENT, 3415);
	sleep(1);
	InfoMessage("Testing thread started");

	const char * reqTraceName = "batchedReq.dat";
	//const char * reqTraceName = "C:\\vm\\shared\\360emu\\batchedReq.dat";
	
	//load request file
	int reqTraceSize = 0;
	/*
	int r = GetFileSize(reqTraceName, reqTraceSize);
	MyAssert(r, 3693);
	BYTE * reqTrace = new BYTE[reqTraceSize];
	FILE * ifs = fopen(reqTraceName, "rb");
	MyAssert(ifs != NULL, 3692);	
	r = fread(reqTrace, reqTraceSize, 1, ifs);
	MyAssert(r == 1, 3694);
	fclose(ifs);
	*/

	//parse the requests
	/************************************************************************/
	/* #define MSG_BATCH_REQUESTS 4
	//ID (1 byte)	 0
	//len (4B) = XX  1
	//seqnumber (4B) 5
	//lists of tiles (5B*x) 9
	//{chunk_id 2B, tileID 1B, quality 1B, class 1B}                                                                     */
	/************************************************************************/
	/*
	vector<int> reqOffset;
	int pos = 0;
	while (pos < reqTraceSize) {
		MyAssert(reqTrace[pos] == MSG_BATCH_REQUESTS && pos + 9 < reqTraceSize, 3695);
		int len = ReadInt(reqTrace + pos + 1);
		MyAssert(len > 9 && pos + len <= reqTraceSize && (len - 9) % 5 == 0, 3696);
		reqOffset.push_back(pos);
		pos += len;
	}
	
	MyAssert(pos == reqTraceSize, 3697);
	*/

	//VIDEO_COMM::SendMessage_SelectVideo("mega.coaster.2x4");
	VIDEO_COMM::SendMessage_SelectVideo("elephant.4x6");
	
	//wait for the metadata response
	while (VIDEO_DATA::nChunks == 0) {
		pthread_yield();
	}
	
	/*
	int nReqs = reqOffset.size();
	int seqnum = 0;
	for (int i=0; i<nReqs; i+=500) { //seqnum starts at 0
		InfoMessage("Batched Request %d", i);
		int len = ReadInt(reqTrace + reqOffset[i] + 1);
		VIDEO_COMM::SendMessage_RequestBatch(seqnum, len, reqTrace + reqOffset[i]);
		seqnum++;
		usleep(100*1000);	//100ms
	}
	*/

	/*
	while (1) {		
		for (int i=0; i<100; i++) {
			//VIDEO_COMM::SendMessage_RequestChunk(i, 0, 0, 1);
			VIDEO_COMM::SendMessage_RequestRandomChunk();
		};
	}
	*/
	
	//delete [] reqTrace;

	return NULL;
}

void StartTestingThread() {
	pthread_t testing_thread;	
	int r = pthread_create(&testing_thread, NULL, TestingThread, NULL);
	MyAssert(r == 0, 1724);
}

int main(int argc, char * * argv) {
	MyAssert(sizeof(long) == 8, 3703);

	srand(time(NULL));
	InfoMessage("Version: %s", MY_VERSION);

	if (argc < 4) {
		DisplayUsage(argv);
	}

	VIDEO_COMM::mode = atoi(argv[1]);
	VIDEO_COMM::bw_id = atoi(argv[3]);
	if (atoi(argv[2]) == 4){
		VIDEO_COMM::skip = true;
		VIDEO_COMM::augmentation = true;
	}
	if (atoi(argv[2]) == 1){
		VIDEO_COMM::skip = false;
		VIDEO_COMM::augmentation = false;
	}
	if (atoi(argv[2]) == 2){
		VIDEO_COMM::skip = true;
		VIDEO_COMM::augmentation = false;
	}
	if (atoi(argv[2]) == 3){
		VIDEO_COMM::skip = false;
		VIDEO_COMM::augmentation = true;
	}
	
	if (VIDEO_COMM::mode == MODE_CLIENT) {
		SETTINGS::ReadRawSettingsFromFile("./theia.client.cfg");
	} else {
		SETTINGS::ReadRawSettingsFromFile("./theia.server.cfg");
	}
	SETTINGS::ApplySettings();
	InfoMessage("augmentation %d skip %d trace_id %d", VIDEO_COMM::augmentation, VIDEO_COMM::skip, VIDEO_COMM::bw_id);

	VIDEO_COMM::Init();	
	
	switch (VIDEO_COMM::mode) {
		case MODE_CLIENT:	//client mode
			{	
				if (argc != 3) DisplayUsage(argv);
				VIDEO_DATA::Init();				
				InfoMessage("Client mode: IP=%s", argv[2]);
				
				if (VIDEO_COMM::ConnectionSetup(argv[2]) != R_SUCC) {
					MyAssert(0, 3486);
				}
				StartTestingThread();
				VIDEO_COMM::MainLoop();
				break;
			}

		case MODE_SERVER:	//server mode
			{
				if (argc != 4) DisplayUsage(argv);

				InfoMessage("Server mode");
				TC_INTERFACE::CleanUpAndSystemCheck(); //Nan 1222
				//KERNEL_INTERFACE::DetectKernelModule();

				VIDEO_DATA::Init();		

				InfoMessage("Connection setup");
				if (VIDEO_COMM::ConnectionSetup(NULL) != R_SUCC) {
					MyAssert(0,3487);
				}
				InfoMessage("Start the main loop");
				VIDEO_COMM::MainLoop();
				break;
			}

		default:
			DisplayUsage(argv);
			break;
	}

	return 0;
}
