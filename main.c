#include "mnist_data.h"
#include "stdio.h"

#ifdef NO_HOSTEDIO
#define PRINTF(...)
#else
#define PRINTF(...) { printf(__VA_ARGS__); }
#endif

int cnn_Recogn(int);


int main(void) {
    int retVal = cnn_Recogn(0);
    PRINTF("Actual Value:    %d\n", 3);
    PRINTF("Predicted Value: %d\n", retVal);
    return 0;
}

    
