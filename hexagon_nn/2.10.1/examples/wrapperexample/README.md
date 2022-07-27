# Wrapper Example App
A simple app to flush out the make process, call a couple hexnn APIs
using the prebuilt libraries from a Hexagon-NN release.

This app is meant to test the Smart Wrapper functionality.

## BUILDING

### Setup your hexagon SDK:
```
pushd <HEXAGON_SDK_ROOT>
source setup_sdk_env.source
popd
```
### Build the android app with Smart Wrapper
```
export HEXNN_ROOT=<PATH_TO_HEXNN_RELEASE>
export ANDROID_NDK_ROOT=<PATH_TO_NDK>
cd $HEXNN_ROOT/examples/wrapperexample/
make tree V=android_Release VERBOSE=1 SMART_WRAPPER=1
```

## RUNNING
```
./runwrapperexample.sh
```