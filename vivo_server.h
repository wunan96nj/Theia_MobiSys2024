#ifndef _VIVO_SERVER_H_
#define _VIVO_SERVER_H_

#define MY_VERSION "ViVo_Server_05222019"

#define SOL_TCP 6

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>


#include <linux/socket.h>
#include <sys/socket.h>
#include <poll.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#include <arpa/inet.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <pthread.h>
#include <fcntl.h>
#include <linux/tcp.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <linux/fb.h>


#include <linux/sockios.h>	//for SIOCOUTQ

using namespace std;
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "settings.h"

#define R_SUCC 1
#define R_FAIL 0

#define MODE_CLIENT 1
#define MODE_SERVER 2


typedef unsigned char BYTE;
typedef unsigned int DWORD;
typedef unsigned short WORD;

typedef long long INT64;
typedef unsigned long long DWORD64;


#endif
