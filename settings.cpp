#include "stdafx.h"

#include "settings.h"
#include "tools.h"
#include "videocomm.h"
#include <iostream>


int SETTINGS::SERVER_PORT = -1;
int SETTINGS::POLL_TIMEOUT = -1;
int SETTINGS::SEND_MESSAGE_QUEUE_SIZE = -1;
int SETTINGS::RECV_BUFFER_SIZE = -1;

long SETTINGS::SERVER_BUFFER_SIZE = -1;

int SETTINGS::MAX_CTQ = -1;
int SETTINGS::MAX_CT = -1;

int SETTINGS::MAX_BATCHES = -1;
int SETTINGS::MAX_BATCH_SIZE = -1;

DWORD SETTINGS::SERVER_IP = 0;

string SETTINGS::BASE_DIRECTORY;

//bandwidth replay (server side)
int SETTINGS::BW_BASE_RTT = 0;

map<string, string> SETTINGS::settings;

void SETTINGS::ApplySettings() {
	SERVER_PORT = FindInt("SERVER_PORT");
	POLL_TIMEOUT = FindInt("POLL_TIMEOUT");
	SEND_MESSAGE_QUEUE_SIZE = FindInt("SEND_MESSAGE_QUEUE_SIZE");
	RECV_BUFFER_SIZE = FindInt("RECV_BUFFER_SIZE");
	SERVER_BUFFER_SIZE = FindLong("SERVER_BUFFER_SIZE");
	BASE_DIRECTORY = FindString("BASE_DIRECTORY");

	MAX_BATCHES = FindInt("MAX_BATCHES");
	MAX_BATCH_SIZE = FindInt("MAX_BATCH_SIZE");
	
	MAX_CTQ	= FindInt("MAX_CTQ");
	MAX_CT = FindInt("MAX_CT");

	string serverIPAddr = FindString("SERVER_IP");
	SERVER_IP = ConvertIPToDWORD(serverIPAddr.c_str());
	
	BW_BASE_RTT = FindInt("BW_BASE_RTT");
	
	CheckSettings();	
}

void SETTINGS::CheckSettings() {	
	MyAssert(POLL_TIMEOUT > 0, 3479);
	MyAssert(SEND_MESSAGE_QUEUE_SIZE > 0, 3480);
	MyAssert(RECV_BUFFER_SIZE > 0, 3481);
	MyAssert(MAX_CTQ > 0 && MAX_CT > 0, 3482);
	MyAssert(SERVER_BUFFER_SIZE > 0, 3483);
	MyAssert(!BASE_DIRECTORY.empty(), 3485);

	if (VIDEO_COMM::mode == MODE_SERVER) {
		MyAssert(SERVER_PORT > 0 && SERVER_IP > 0, 3478);
	}
}

void SETTINGS::ReadRawSettingsFromFile(const char * filename) {
	MyAssert(settings.size() == 0, 1939);
	FILE * ifs = fopen(filename, "rb");
	MyAssert(ifs != NULL, 1940);
	char buf[2048];
	while (!feof(ifs)) {
		if (fgets(buf, sizeof(buf), ifs) == NULL) break;

		char * comment = strstr(buf, "#");
		if (comment != NULL) *comment = 0;

		char * key = ChompSpaceTwoSides(buf);
		if (key[0] == 0) continue;		
		char * value = strstr(key, "=");
		MyAssert(value != NULL, 1941);
		*value = 0;
		value++;

		char * ckey = ChompSpaceTwoSides(key);
		char * cvalue = ChompSpaceTwoSides(value);
		//MyAssert(strlen(ckey)>0 && strlen(cvalue)>0, 1942); //allow it to be empty
		MyAssert(settings.find(ckey) == settings.end(), 1943);

		settings[ckey] = cvalue; 
		
	}
	// print map 
	//for(std::map<string,string>::iterator it=settings.begin(); it!=settings.end();++it)
	//	std::cout << it->first << "=>" << it->second << "\n";

	fclose(ifs);
}


int SETTINGS::FindInt(const char * key) {
	if (settings.find(key) != settings.end()) {
		int r = atoi(settings[key].c_str());
		VerboseMessage("[setting] %s = %d", key, r);
		return r;
	} else {
		printf("%s\n", key);
		MyAssert(0, 1944);
		return 0;
	}
}

long SETTINGS::FindLong(const char * key) {
	if (settings.find(key) != settings.end()) {
		long r = atol(settings[key].c_str());
		VerboseMessage("[setting] %s = %ld", key, r);
		return r;
	} else {
		printf("%s\n", key);
		MyAssert(0, 1944);
		return 0;
	}
}

double SETTINGS::FindDouble(const char * key) {
	if (settings.find(key) != settings.end()) {
		double r = atof(settings[key].c_str());
		VerboseMessage("[setting] %s = %.3lf", key, r);
		return r;
	} else {
		MyAssert(0, 1945);
		return 0;
	}
}

const char * SETTINGS::FindString(const char * key) {
	if (settings.find(key) != settings.end()) {
		const char * r = settings[key].c_str();
		VerboseMessage("[setting] %s = %s", key, r);
		return r;
	} else {
		MyAssert(0, 1946);
		return 0;
	}
}

