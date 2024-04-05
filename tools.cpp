#include "stdafx.h"
#include "tools.h"

/*
inline void MyAssert(int x, int assertID) {
#ifdef DEBUG_ENABLE_ASSERTION
	if (!x) {
		fprintf(stderr, "Assertion failure: %d\n", assertID);
		fprintf(stderr, "errno = %d (%s)\n", errno, strerror(errno));
		exit(-1);
	}
#endif
}
*/

void SetNonBlockIO(int fd) {
	int val = fcntl(fd, F_GETFL, 0);
	if (fcntl(fd, F_SETFL, val | O_NONBLOCK) != 0) {
		MyAssert(0, 1616);
	}
}

//no need for this if local and remote proxies are deployed on two machines
void SetMaxSegSize(int fd, int nBytes) {
	int r = setsockopt(fd, IPPROTO_TCP, TCP_MAXSEG, &nBytes, sizeof(int));
	MyAssert(r == 0, 1751);
}

void SetSocketBuffer(int fd, int readBufSize, int writeBufSize) {
	int r1 = 0;
	int r2 = 0;
	if (readBufSize != 0) {	
		r1 = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &readBufSize, sizeof(int));	MyAssert(r1 == 0, 1786);
	}
		
	if (writeBufSize != 0) {
		r2 = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &writeBufSize, sizeof(int)); MyAssert(r2 == 0, 1787);
	}
	
	/*
	socklen_t s1, s2;
	s1 = s2 = 4;
	r1 = getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &readBufSize, &s1);
	r2 = getsockopt(fd, SOL_SOCKET, SO_SNDBUF, &writeBufSize, &s2);
	*/

	MyAssert(r1 == 0 && r2 == 0, 1769);

	//InfoMessage("*** %d %d readBufSize=%d writeBufSize=%d ***", (int)s1, (int)s2, readBufSize, writeBufSize);
}

void SetSocketNoDelay_TCP(int fd) {
	static int enable = 1;
	int r = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enable,sizeof(int));
	MyAssert(r == 0, 1768);
}

void SetQuickACK(int fd) {
	static int enable = 1;
	int r = setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &enable, sizeof(int));
	MyAssert(r == 0, 2040);
}


const char * ConvertDWORDToIP(DWORD ip) {
	static char ipstr[5][128];
	static int count = 0;

	int i = count++;
	if (count == 5) count = 0;
	sprintf(ipstr[i], "%d.%d.%d.%d",
		(ip & 0x000000FF),
		(ip & 0x0000FF00) >> 8,
		(ip & 0x00FF0000) >> 16,
		(ip & 0xFF000000) >> 24
		);
	return ipstr[i];
}

void GetClientIPPort(int fd, DWORD & ip, WORD & port) {
	struct sockaddr_in sin;
	socklen_t addrlen = sizeof(sin);
	if (getsockname(fd, (struct sockaddr *)&sin, &addrlen) == 0 &&
		sin.sin_family == AF_INET &&
		addrlen == sizeof(sin)) 
	{
		ip = sin.sin_addr.s_addr;
		port = ntohs(sin.sin_port);
	}
	else {
		ip = port = 0;
		MyAssert(0, 1697);
	}
}

const char * GetTimeString() {
	static char timeStr[32];

	time_t t = time(NULL);
	struct tm tm = *localtime(&t);

	sprintf(timeStr, "%02d:%02d:%02d", tm.tm_hour, tm.tm_min, tm.tm_sec);
	return timeStr;
}

void DebugMessage(const char * format, ...) {
#ifdef DEBUG_MESSAGE
	static char dest[2048];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	va_end(argptr);
	//::MessageBox(AfxGetMainWnd()->m_hWnd, dest, "Mobile WebPage Profiler", MB_OK | MB_ICONEXCLAMATION);
	fprintf(stderr, "[DEBUG] %s %s\n", GetTimeString(), dest);
#endif
}

void WarningMessage(const char * format, ...) {
#ifdef DEBUG_LEVEL_WARNING
	static char dest[2048];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	va_end(argptr);
	//::MessageBox(AfxGetMainWnd()->m_hWnd, dest, "Mobile WebPage Profiler", MB_OK | MB_ICONEXCLAMATION);
	fprintf(stderr, "[WARN] %s %s\n", GetTimeString(), dest);
#endif
}

void InfoMessage(const char * format, ...) {
#ifdef DEBUG_LEVEL_INFO
	static char dest[2048];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	va_end(argptr);
	//::MessageBox(AfxGetMainWnd()->m_hWnd, dest, "Mobile WebPage Profiler", MB_OK | MB_ICONINFORMATION);
	fprintf(stderr, "[INFO] %s %s\n", GetTimeString(), dest);
#endif
}

void ErrorMessage(const char * format, ...) {
	static char dest[2048];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	va_end(argptr);
	//::MessageBox(AfxGetMainWnd()->m_hWnd, dest, "Mobile WebPage Profiler", MB_OK | MB_ICONSTOP);
	fprintf(stderr, "[ERROR] %s %s\n", GetTimeString(), dest);
	fprintf(stderr, "errno = %d (%s)\n", errno, strerror(errno));
}

void VerboseMessage(const char * format, ...) {
#ifdef DEBUG_LEVEL_VERBOSE
	static char dest[2048];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	va_end(argptr);
	//::MessageBox(AfxGetMainWnd()->m_hWnd, dest, "Mobile WebPage Profiler", MB_OK | MB_ICONSTOP);
	fprintf(stderr, "[VERBOSE] %s %s\n", GetTimeString(), dest);
#endif
}

DWORD GetCongestionWinSize(int fd) {
#ifndef VS_SIMULATION
	struct tcp_info ti;
	int tcpInfoSize;
	int r = getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, (socklen_t *)&tcpInfoSize);
	MyAssert(r == 0, 1770);
	return ti.tcpi_snd_cwnd;
#else
	return 0;
#endif
}

char * Chomp(char * str) {
	int len = strlen(str);
	while (len>0 && str[len-1] < 32) len--;
	str[len] = 0;

	return str;
}

char * ChompSpace(char * str) {
	int len = strlen(str);
	while (len>0 && str[len-1] <= 32) len--;
	str[len] = 0;

	return str;
}

char * ChompSpaceTwoSides(char * str) {
	int len = strlen(str);
	while (len>0 && str[len-1] <= 32) len--;
	str[len] = 0;

	len = strlen(str);
	int i=0;
	while (i<len && str[i]<=32) i++;
	return str+i;
}

DWORD ConvertIPToDWORD(const char * ip) {
	static char ip0[128];
	strcpy(ip0, ip);

	const char * p[4];
	int pp = 0;
	p[0] = ip0;
	
	int i=0;

	while (1) {
		MyAssert(ip0[i] != 0, 1771);
		if (ip0[i] == '.') {
			ip0[i++] = 0;
			p[++pp] = ip0 + i;
			if (pp == 3) break;
		} else {
			i++;
		}
	}
	
	return 
		(DWORD(atoi(p[0]))		) | 
		(DWORD(atoi(p[1])) << 8	) | 
		(DWORD(atoi(p[2])) << 16)  | 
		(DWORD(atoi(p[3])) << 24);
}

int GetFileSize(const char * filename, long & size) {
	FILE * ifs = fopen(filename, "rb");
	if (ifs == NULL) return 0;

	fseek(ifs, 0, SEEK_END);
	size = (long)ftell(ifs);
	fclose(ifs);

	return 1;
}

void Split(char * str, const char * seps, vector<string> & s) {	
	s.clear();
	char * token;
	token = strtok(str, seps);
	while( token != NULL )
	{		
		s.push_back(token);		
		token = strtok( NULL, seps );
	}
}

void Split(char * str, const char * seps, char * * s, int & n) {	
	int n0 = n;
	n = 0;	
	char * token;
	token = strtok(str, seps);
	while( token != NULL )
	{		
		s[n++] = token;
		MyAssert(n<=n0, 1915);
		token = strtok( NULL, seps );
	}
}

void SetCongestionControl(int fd, const char * tcpVar) {
	if (!strcmp(tcpVar, "default") || !strcmp(tcpVar, "DEFAULT")) return;
	int r = setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, tcpVar, (int)strlen(tcpVar));
	MyAssert(r == 0, 1815);	
}

FILE * OpenFileForRead(const char * filename) {
	FILE * ifs = fopen(filename, "rb");
	
	if (ifs == NULL) {
		ErrorMessage("File not found: %s", filename);
		MyAssert(0, 3427);
		return NULL;
	}
	
	return ifs;
}

void writeToLogFile(const std::string& logMessage) {
    std::ofstream logFile("./Logs/log.txt", std::ios_base::app);

    if (logFile.is_open()) {
        logFile << logMessage << std::endl;
        logFile.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}

/*
int FindStr(const char * str, const BYTE * pBuf, int n) {
	int m = strlen(str);

	const BYTE * p = pBuf;
	const BYTE * pEnd = pBuf + n - m;
	int i;
	while (p<=pEnd) {
		i=0;
		const BYTE * pp = p;
		while (i<m && *pp == (BYTE)str[i]) {
			pp++;
			i++;
		}
		if (i==m) return p-pBuf;
		p++;
	}

	return -1;
}

int FindRequest(int & rr, const BYTE * pBuf, int n) {
	const char * sig = "/small.";
	static char buf[5];

	int k = FindStr(sig, pBuf, n);
	if (k != -1) {
		memcpy(buf, pBuf + k + strlen(sig), 3);
		if (buf[1] == '.') buf[1] = 0;
		else if (buf[2] == '.') buf[2] = 0; 
		else MyAssert(0, 1963);
		rr = atoi(buf);
		return 1;
	} else {
		return 0;
	}
}

int FindResponse(int & rr, const BYTE * pBuf, int n) {
	const char * sig = "SMALL.";
	static char buf[5];

	int k = FindStr(sig, pBuf, n);
	if (k != -1) {
		memcpy(buf, pBuf + k + strlen(sig), 3);
		if (buf[1] == '.') buf[1] = 0;
		else if (buf[2] == '.') buf[2] = 0; 
		else MyAssert(0, 1964);
		rr = atoi(buf);
		return 1;
	} else {
		return 0;
	}
}
*/


double GetMillisecondTS() {
#ifndef VS_SIMULATION
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6 ;
#else
	return 0.0f;
#endif
}

/*
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

*/

/*
extern unsigned long highResTimestampBase;

unsigned long GetHighResTimestamp() {
#ifndef VS_SIMULATION
	struct timeval tv;
	gettimeofday(&tv, NULL);
	unsigned long tNow = tv.tv_sec * 1000000 + tv.tv_usec;
	
	return tNow - highResTimestampBase;
#else
	return 0;
#endif
}

*/

void WriteInt(BYTE * pData, int x) {
	memcpy(pData, &x, sizeof(int));
}

void WriteShort(BYTE * pData, short x) {
	memcpy(pData, &x, sizeof(short));
}

void WriteWORD(BYTE * pData, WORD x) {
	memcpy(pData, &x, sizeof(WORD));
}

void WriteFloat(BYTE * pData, float x) {
	memcpy(pData, &x, sizeof(float));
}

int ReadInt(BYTE * pData) {
	return *((int *)pData);
}

WORD ReadWORD(BYTE * pData) {
	return *((WORD *)pData);
}

short ReadShort(BYTE * pData) {
	return *((short *)pData);
}

float ReadFloat(BYTE * pData) {
	return *((float *)pData);
}

