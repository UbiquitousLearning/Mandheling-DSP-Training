#!/bin/bash

DEVDIR=/data/local/tmp/wrapperexample/

adb shell mkdir -p $DEVDIR
adb shell rm -rf $DEVDIR/*

BINDIR=$DEVDIR/bin
adb shell mkdir -p $BINDIR
adb push android_Release/wrapperexample $BINDIR/

LIBDIR=$DEVDIR/lib
adb shell mkdir -p $LIBDIR
adb push android_Release/libgnustl_shared.so $LIBDIR

DSPDIR=$DEVDIR/dsp
adb shell mkdir -p $DSPDIR
adb push $HEXNN_ROOT/libs/libhexagon_nn_skel.so $DSPDIR/
adb push $HEXNN_ROOT/libs/libhexagon_nn_skel_v65.so $DSPDIR/
adb push $HEXNN_ROOT/libs/libhexagon_nn_skel_v66.so $DSPDIR/

echo "---------------------------------------------"
echo "Running wrapperexample"
echo "---------------------------------------------"
adb shell "export LD_LIBRARY_PATH=$LIBDIR && export ADSP_LIBRARY_PATH=\"$DSPDIR/\" && $BINDIR/wrapperexample"
