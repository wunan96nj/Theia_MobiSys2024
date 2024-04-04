#ifndef _SETTINGS_H_
#define _SETTINGS_H_

#include "vivo_server.h"

//Show debug message?
//#define DEBUG_MESSAGE

//Output levels
//#define DEBUG_LEVEL_VERBOSE
#define DEBUG_LEVEL_INFO
//#define DEBUG_LEVEL_WARNING

#define DEBUG_ENABLE_ASSERTION

struct SETTINGS {
private:
	SETTINGS();
	~SETTINGS();

public:
	static int SERVER_PORT; //6001
	static int POLL_TIMEOUT; //5000 //millisecond
	static int SEND_MESSAGE_QUEUE_SIZE; //500
	static int RECV_BUFFER_SIZE; //1000000

	static long SERVER_BUFFER_SIZE; //1024*1024*1024
	
	static int MAX_BATCHES;
	static int MAX_BATCH_SIZE;
	
	static string BASE_DIRECTORY;	//"/home/fengqian/360/data", "/home/xiaoq/vr360/video/DASH"
	
	static int MAX_CTQ;		//chunk * tile * quality
	static int MAX_CT;		//chunk * tile

	static unsigned int SERVER_IP;

	static int BW_BASE_RTT;	//in ms

public:
	static void ReadRawSettingsFromFile(const char * filename);
	static void ApplySettings();
	
protected:
	static map<string, string> settings;
	static int FindInt(const char * key);
	static long FindLong(const char * key);
	static double FindDouble(const char * key);
	static const char * FindString(const char * key);
	static void CheckSettings();

};

#endif
