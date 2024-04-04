#ifndef _TOOLS_H_
#define _TOOLS_H_

#include "theia_server.h"
#include "videocomm.h"

inline void MyAssert(int x, int assertID) {
#ifdef DEBUG_ENABLE_ASSERTION
	if (!x) {
		fprintf(stderr, "Assertion failure: %d\n", assertID);
		fprintf(stderr, "errno = %d (%s)\n", errno, strerror(errno));

		if (VIDEO_COMM::isKMLoaded) KERNEL_INTERFACE::DisableLateBinding();
		exit(-1);
	}
#endif
}

//void MyAssert(int x, int assertID);
void SetNonBlockIO(int fd);
void SetSocketBuffer(int fd, int readBufSize, int writeBufSize);
void SetMaxSegSize(int fd, int nBytes);
void SetSocketNoDelay_TCP(int fd);
void SetQuickACK(int fd);

DWORD GetCongestionWinSize(int fd);

const char * ConvertDWORDToIP(DWORD ip);
DWORD ConvertIPToDWORD(const char * ip);

static inline WORD ReverseWORD(WORD x) {
	return
		(x & 0xFF) << 8 |
		(x & 0xFF00) >> 8;
}

void GetClientIPPort(int fd, DWORD & ip, WORD & port);

const char * GetTimeString();

void DebugMessage(const char * format, ...);
void WarningMessage(const char * format, ...);
void InfoMessage(const char * format, ...);
void ErrorMessage(const char * format, ...);
void VerboseMessage(const char * format, ...);

char * Chomp(char * str);
char * ChompSpace(char * str);
char * ChompSpaceTwoSides(char * str);

int GetFileSize(const char * filename, long & size);

void Split(char * str, const char * seps, char * * s, int & n);
void Split(char * str, const char * seps, vector<string> & s);

void SetCongestionControl(int fd, const char * tcpVar);

/*
int FindStr(const char * str, const BYTE * pBuf, int n);
int FindRequest(int & rr, const BYTE * pBuf, int n);
int FindResponse(int & rr, const BYTE * pBuf, int n);
*/

#define TCPSOCKET_2_PIPEBUFFER 1
#define PIPEBUFFER_2_PIPESOCKET 2
#define PIPESOCKET_2_TCPBUFFER 3
#define TCPBUFFER_2_TCPSOCKET 4


double GetMillisecondTS();

inline DWORD Reverse(DWORD x) {
	return
		(x & 0xFF) << 24 |
		(x & 0xFF00) << 8 |
		(x & 0xFF0000) >> 8 |
		(x & 0xFF000000) >> 24;
}

inline WORD Reverse(WORD x) {
	return
		(x & 0xFF) << 8 |
		(x & 0xFF00) >> 8;
}

inline int Reverse(int x) {
	return (int)Reverse(DWORD(x));
}

unsigned long GetHighResTimestamp();

inline unsigned long GetLogicTime() {
	static unsigned long l = 100;
	return l++;
}

void WriteInt(BYTE * pData, int x);
void WriteWORD(BYTE * pData, WORD x);
void WriteShort(BYTE * pData, short x);
void WriteFloat(BYTE * pData, float x);
int ReadInt(BYTE * pData);
WORD ReadWORD(BYTE * pData);
short ReadShort(BYTE * pData);
float ReadFloat(BYTE * pData);

FILE * OpenFileForRead(const char * filename);
void writeToLogFile(const std::string& logMessage);
#endif
