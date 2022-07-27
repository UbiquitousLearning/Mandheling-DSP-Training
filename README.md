### Mandheling
This repo is an open-source project for Mandheling: Mixed-Precision On-Device DNN  Training with DSP Offloading.
It contains two part: exection engine and the Hexagon DSP backend.

### execution-engine build
* For Android users.
```
    cd execution-engine
    cd project/android

    mkdir Train_64
    cd Train_64
    ../build_64.sh
```

### Hexagon DSP backend build
The Hexagon DSP backend is dependent on Hexagon SDK, so users should first install the SDK, and then replace the hexagon_nn in libs/hexagon_nn/
Follow the README in SDK to build the hexagon nn dynamic link library --- libhexagon_nn_skel.so


### Run the Mnist test
1. Push the execution engine to device
```
cd execution-engine/project/android/Train_64
../build_64.sh
echo 1 >> DSP.txt
echo 0 >> parallel.txt

adb push ../Train_64 /data/local/tmp/
```

2. Push the hexagon libraries
```
adb push  libhexagon_nn_skel.so /data/local/tmp/Train_64
adb push  libhexagon_interface_arm64.so /data/local/tmp/Train_64
```

3. Push Mnist dataset
```
adb push MNIST_data /data/local/tmp/
```

4. Run Mnist test
```
export LD_LIBRARY_PATH=/data/local/tmp/Train_64/
cd /data/local/tmp/Train_64/
cp tools/train/libMNNTrain.so .
./runTrainDemo.out NITIInt8Train ../MNIST_data/

```