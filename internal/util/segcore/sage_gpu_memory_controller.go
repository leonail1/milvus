// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package segcore

import (
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const bytesPerMiB = uint64(1024 * 1024)

// SageCuvsGpuMemoryPoolStats covers transient cuVS allocations for one CUDA
// device.  The resident index dataset is intentionally accounted separately.
type SageCuvsGpuMemoryPoolStats struct {
	ReservedCurrent uint64
	ReservedHigh    uint64
	UsedCurrent     uint64
	UsedHigh        uint64
}

type sageGpuMemoryBackend interface {
	stats(deviceID int32) (SageCuvsGpuMemoryPoolStats, int32)
	setReleaseThreshold(deviceID int32, bytes uint64) int32
	resetHighWater(deviceID int32) int32
	trim(deviceID int32, minBytesToKeep uint64, synchronize bool) int32
}

type sageGpuMemoryControllerConfig struct {
	enabled                     bool
	servingReleaseThreshold     uint64
	maintenanceReleaseThreshold uint64
	trimTarget                  uint64
	trimHysteresis              uint64
	trimCooldown                time.Duration
	maxDevices                  int32
}

type sageGpuMemoryController struct {
	config      sageGpuMemoryControllerConfig
	backend     sageGpuMemoryBackend
	now         func() time.Time
	active      atomic.Int64
	maintenance atomic.Int64
	lastCheck   atomic.Int64
	mu          sync.Mutex
}

func newSageGpuMemoryController(
	config sageGpuMemoryControllerConfig,
	backend sageGpuMemoryBackend,
	now func() time.Time,
) *sageGpuMemoryController {
	return &sageGpuMemoryController{config: config, backend: backend, now: now}
}

func loadSageGpuMemoryControllerConfig() sageGpuMemoryControllerConfig {
	enabled, valid := readSageBoolEnv("SAGE_MILVUS_GPU_MEMORY_CONTROLLER_ENABLED", false)
	servingMiB, validServing := readSageUintEnv("SAGE_MILVUS_GPU_SERVING_RELEASE_THRESHOLD_MB", 64)
	maintenanceMiB, validMaintenance := readSageUintEnv("SAGE_MILVUS_GPU_MAINTENANCE_RELEASE_THRESHOLD_MB", 2048)
	trimTargetMiB, validTarget := readSageUintEnv("SAGE_MILVUS_GPU_TRIM_TARGET_MB", 64)
	trimHysteresisMiB, validHysteresis := readSageUintEnv("SAGE_MILVUS_GPU_TRIM_HYSTERESIS_MB", 128)
	cooldownMillis, validCooldown := readSageUintEnv("SAGE_MILVUS_GPU_TRIM_COOLDOWN_MS", 1000)
	maxDevices, validDevices := readSageUintEnv("SAGE_MILVUS_GPU_MAX_VISIBLE_DEVICES", 16)
	valid = valid && validServing && validMaintenance && validTarget && validHysteresis && validCooldown && validDevices
	valid = valid && maxDevices > 0 && maxDevices <= 64
	valid = valid && servingMiB <= ^uint64(0)/bytesPerMiB
	valid = valid && maintenanceMiB <= ^uint64(0)/bytesPerMiB
	valid = valid && trimTargetMiB <= ^uint64(0)/bytesPerMiB
	valid = valid && trimHysteresisMiB <= ^uint64(0)/bytesPerMiB
	valid = valid && trimTargetMiB <= ^uint64(0)-trimHysteresisMiB
	valid = valid && maintenanceMiB >= servingMiB
	valid = valid && cooldownMillis <= uint64((time.Duration(1<<63-1))/time.Millisecond)

	return sageGpuMemoryControllerConfig{
		enabled:                     enabled && valid,
		servingReleaseThreshold:     servingMiB * bytesPerMiB,
		maintenanceReleaseThreshold: maintenanceMiB * bytesPerMiB,
		trimTarget:                  trimTargetMiB * bytesPerMiB,
		trimHysteresis:              trimHysteresisMiB * bytesPerMiB,
		trimCooldown:                time.Duration(cooldownMillis) * time.Millisecond,
		maxDevices:                  int32(maxDevices),
	}
}

func readSageBoolEnv(name string, fallback bool) (bool, bool) {
	value, present := os.LookupEnv(name)
	if !present || value == "" {
		return fallback, true
	}
	parsed, err := strconv.ParseBool(value)
	return parsed, err == nil
}

func readSageUintEnv(name string, fallback uint64) (uint64, bool) {
	value, present := os.LookupEnv(name)
	if !present || value == "" {
		return fallback, true
	}
	parsed, err := strconv.ParseUint(value, 10, 64)
	return parsed, err == nil
}

func (controller *sageGpuMemoryController) beginServing() func() {
	if !controller.config.enabled {
		return func() {}
	}
	controller.active.Add(1)
	return func() {
		if controller.active.Add(-1) == 0 && controller.maintenance.Load() == 0 {
			controller.maybeReclaim(false)
		}
	}
}

func (controller *sageGpuMemoryController) beginMaintenance() func() {
	if !controller.config.enabled {
		return func() {}
	}
	if controller.maintenance.Add(1) == 1 {
		controller.mu.Lock()
		controller.forEachDevice(func(deviceID int32, _ SageCuvsGpuMemoryPoolStats) {
			controller.backend.setReleaseThreshold(deviceID, controller.config.maintenanceReleaseThreshold)
			controller.backend.resetHighWater(deviceID)
		})
		controller.mu.Unlock()
	}
	return func() {
		if controller.maintenance.Add(-1) != 0 {
			return
		}
		controller.mu.Lock()
		controller.forEachDevice(func(deviceID int32, _ SageCuvsGpuMemoryPoolStats) {
			controller.backend.setReleaseThreshold(deviceID, controller.config.servingReleaseThreshold)
		})
		controller.mu.Unlock()
		controller.maybeReclaim(true)
	}
}

func (controller *sageGpuMemoryController) maybeReclaim(force bool) {
	if !controller.config.enabled || controller.active.Load() != 0 || controller.maintenance.Load() != 0 {
		return
	}
	now := controller.now()
	if !force && now.Sub(time.Unix(0, controller.lastCheck.Load())) < controller.config.trimCooldown {
		return
	}

	controller.mu.Lock()
	defer controller.mu.Unlock()
	if controller.active.Load() != 0 || controller.maintenance.Load() != 0 {
		return
	}
	now = controller.now()
	if !force && now.Sub(time.Unix(0, controller.lastCheck.Load())) < controller.config.trimCooldown {
		return
	}
	controller.lastCheck.Store(now.UnixNano())
	controller.forEachDevice(func(deviceID int32, stats SageCuvsGpuMemoryPoolStats) {
		trimBoundary := controller.config.trimTarget + controller.config.trimHysteresis
		if stats.ReservedCurrent <= trimBoundary {
			return
		}
		minBytesToKeep := controller.config.trimTarget
		if stats.UsedCurrent > minBytesToKeep {
			minBytesToKeep = stats.UsedCurrent
		}
		controller.backend.trim(deviceID, minBytesToKeep, false)
	})
}

func (controller *sageGpuMemoryController) forEachDevice(
	action func(deviceID int32, stats SageCuvsGpuMemoryPoolStats),
) {
	for deviceID := int32(0); deviceID < controller.config.maxDevices; deviceID++ {
		stats, status := controller.backend.stats(deviceID)
		if status != 0 {
			break
		}
		action(deviceID, stats)
	}
}
