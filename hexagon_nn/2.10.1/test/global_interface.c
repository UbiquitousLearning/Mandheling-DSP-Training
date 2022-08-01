#include <stdio.h>
#include "remote.h"

// These are the includes and definitions needed to make fastRPC work.
// FastRPC is the communication channel that allows the ARM core
//   (running the main application) to send requests to the DSP core
//   (which does all the heavy math).
#define adspmsgd_start(_a, _b, _c)
#define adspmsgd_stop()
#include "rpcmem.h"
#include "AEEStdErr.h"
#include <sys/types.h>
int fastrpc_setup()
{
    int retVal=0;

    adspmsgd_start(0,RPCMEM_HEAP_DEFAULT,4096);
    rpcmem_init();
	return retVal;
}

void fastrpc_teardown()
{
        rpcmem_deinit();
        adspmsgd_stop();
}

int hexagon_nn_global_teardown()
{
	printf("global teardown!!\n");
	fastrpc_teardown();
	return 0;
}

int hexagon_nn_global_init()
{
    #pragma weak remote_session_control
    int ret = -1;
    if (remote_session_control) {
        // printf("***************** remote_session_control is TRUE ****************\n");
        struct remote_rpc_control_unsigned_module data;
        data.enable = 1;
        data.domain = CDSP_DOMAIN_ID;
        ret = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *) &data, sizeof(data));
        // printf("***************** remote_session_control returned %d ****************\n", ret);
    } else {
        return -1;
    }
	if (fastrpc_setup() != 0) return 1;
    
	return 0;
}