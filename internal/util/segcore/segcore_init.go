package segcore

import (
	_ "github.com/milvus-io/milvus/internal/util/cgo"
)

/*
#cgo pkg-config: milvus_core
#cgo LDFLAGS: -ldl

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stdint.h>
#include "segcore/segcore_init_c.h"

static int32_t
SegcoreGoSageCuvsGpuMemoryPoolStats(int32_t device_id,
                                    uint64_t* reserved_current,
                                    uint64_t* reserved_high,
                                    uint64_t* used_current,
                                    uint64_t* used_high) {
    typedef int (*function_t)(int, uint64_t*, uint64_t*, uint64_t*, uint64_t*);
    function_t function = (function_t)dlsym(RTLD_DEFAULT, "sage_cuvs_cuda_async_pool_stats_v1");
    return function == NULL ? 1 : function(device_id, reserved_current, reserved_high, used_current, used_high);
}

static int32_t
SegcoreGoSageCuvsGpuMemoryPoolSetReleaseThreshold(int32_t device_id,
                                                  uint64_t release_threshold_bytes) {
    typedef int (*function_t)(int, uint64_t);
    function_t function = (function_t)dlsym(
        RTLD_DEFAULT, "sage_cuvs_cuda_async_pool_set_release_threshold_v1");
    return function == NULL ? 1 : function(device_id, release_threshold_bytes);
}

static int32_t
SegcoreGoSageCuvsGpuMemoryPoolResetHighWater(int32_t device_id) {
    typedef int (*function_t)(int);
    function_t function = (function_t)dlsym(
        RTLD_DEFAULT, "sage_cuvs_cuda_async_pool_reset_high_water_v1");
    return function == NULL ? 1 : function(device_id);
}

static int32_t
SegcoreGoSageCuvsGpuMemoryPoolTrim(int32_t device_id,
                                   uint64_t min_bytes_to_keep,
                                   int synchronize_device) {
    typedef int (*function_t)(int, uint64_t, int);
    function_t function = (function_t)dlsym(RTLD_DEFAULT, "sage_cuvs_cuda_async_pool_trim_v1");
    return function == NULL ? 1 : function(device_id, min_bytes_to_keep, synchronize_device);
}

*/
import "C"

// IndexEngineInfo contains all the information about the index engine.
type IndexEngineInfo struct {
	MinIndexVersion     int32
	CurrentIndexVersion int32
	MaxIndexVersion     int32
}

// GetIndexEngineInfo returns the minimal, current, and maximum version of the index engine.
func GetIndexEngineInfo() IndexEngineInfo {
	cMinimal, cCurrent, cMaximum := C.GetMinimalIndexVersion(), C.GetCurrentIndexVersion(), C.GetMaximumIndexVersion()
	return IndexEngineInfo{
		MinIndexVersion:     int32(cMinimal),
		CurrentIndexVersion: int32(cCurrent),
		MaxIndexVersion:     int32(cMaximum),
	}
}

func getSageCuvsGpuMemoryPoolStats(deviceID int32) (SageCuvsGpuMemoryPoolStats, int32) {
	var reservedCurrent, reservedHigh, usedCurrent, usedHigh C.uint64_t
	status := C.SegcoreGoSageCuvsGpuMemoryPoolStats(
		C.int32_t(deviceID),
		&reservedCurrent,
		&reservedHigh,
		&usedCurrent,
		&usedHigh,
	)
	return SageCuvsGpuMemoryPoolStats{
		ReservedCurrent: uint64(reservedCurrent),
		ReservedHigh:    uint64(reservedHigh),
		UsedCurrent:     uint64(usedCurrent),
		UsedHigh:        uint64(usedHigh),
	}, int32(status)
}

func setSageCuvsGpuMemoryPoolReleaseThreshold(deviceID int32, bytes uint64) int32 {
	return int32(C.SegcoreGoSageCuvsGpuMemoryPoolSetReleaseThreshold(C.int32_t(deviceID), C.uint64_t(bytes)))
}

func resetSageCuvsGpuMemoryPoolHighWater(deviceID int32) int32 {
	return int32(C.SegcoreGoSageCuvsGpuMemoryPoolResetHighWater(C.int32_t(deviceID)))
}

func trimSageCuvsGpuMemoryPool(deviceID int32, minBytesToKeep uint64, synchronize bool) int32 {
	synchronizeDevice := C.int(0)
	if synchronize {
		synchronizeDevice = C.int(1)
	}
	return int32(C.SegcoreGoSageCuvsGpuMemoryPoolTrim(
		C.int32_t(deviceID),
		C.uint64_t(minBytesToKeep),
		synchronizeDevice,
	))
}
