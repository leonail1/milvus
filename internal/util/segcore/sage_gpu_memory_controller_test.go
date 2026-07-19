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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type fakeSageGpuMemoryBackend struct {
	mu          sync.Mutex
	statsValue  SageCuvsGpuMemoryPoolStats
	thresholds  []uint64
	trimTargets []uint64
	resetCount  int
}

func (backend *fakeSageGpuMemoryBackend) stats(deviceID int32) (SageCuvsGpuMemoryPoolStats, int32) {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	if deviceID != 0 {
		return SageCuvsGpuMemoryPoolStats{}, 2
	}
	return backend.statsValue, 0
}

func (backend *fakeSageGpuMemoryBackend) setReleaseThreshold(_ int32, bytes uint64) int32 {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	backend.thresholds = append(backend.thresholds, bytes)
	return 0
}

func (backend *fakeSageGpuMemoryBackend) resetHighWater(_ int32) int32 {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	backend.resetCount++
	return 0
}

func (backend *fakeSageGpuMemoryBackend) trim(_ int32, minBytesToKeep uint64, _ bool) int32 {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	backend.trimTargets = append(backend.trimTargets, minBytesToKeep)
	return 0
}

func testSageGpuMemoryController(backend *fakeSageGpuMemoryBackend, now func() time.Time) *sageGpuMemoryController {
	return newSageGpuMemoryController(sageGpuMemoryControllerConfig{
		enabled:                     true,
		servingReleaseThreshold:     64 * bytesPerMiB,
		maintenanceReleaseThreshold: 2048 * bytesPerMiB,
		trimTarget:                  64 * bytesPerMiB,
		trimHysteresis:              128 * bytesPerMiB,
		trimCooldown:                time.Second,
		maxDevices:                  2,
	}, backend, now)
}

func TestSageGpuMemoryControllerWaitsForLastQuery(t *testing.T) {
	now := time.Unix(10, 0)
	backend := &fakeSageGpuMemoryBackend{statsValue: SageCuvsGpuMemoryPoolStats{
		ReservedCurrent: 512 * bytesPerMiB,
	}}
	controller := testSageGpuMemoryController(backend, func() time.Time { return now })

	endFirst := controller.beginServing()
	endSecond := controller.beginServing()
	endFirst()
	require.Empty(t, backend.trimTargets)
	endSecond()
	require.Equal(t, []uint64{64 * bytesPerMiB}, backend.trimTargets)
}

func TestSageGpuMemoryControllerUsesHysteresisAndCooldown(t *testing.T) {
	now := time.Unix(10, 0)
	backend := &fakeSageGpuMemoryBackend{statsValue: SageCuvsGpuMemoryPoolStats{
		ReservedCurrent: 192 * bytesPerMiB,
	}}
	controller := testSageGpuMemoryController(backend, func() time.Time { return now })

	controller.maybeReclaim(false)
	require.Empty(t, backend.trimTargets)
	backend.statsValue.ReservedCurrent = 512 * bytesPerMiB
	controller.maybeReclaim(false)
	require.Empty(t, backend.trimTargets)
	now = now.Add(time.Second)
	controller.maybeReclaim(false)
	require.Equal(t, []uint64{64 * bytesPerMiB}, backend.trimTargets)
}

func TestSageGpuMemoryControllerReferenceCountsMaintenance(t *testing.T) {
	now := time.Unix(10, 0)
	backend := &fakeSageGpuMemoryBackend{statsValue: SageCuvsGpuMemoryPoolStats{
		ReservedCurrent: 512 * bytesPerMiB,
	}}
	controller := testSageGpuMemoryController(backend, func() time.Time { return now })

	endFirst := controller.beginMaintenance()
	endSecond := controller.beginMaintenance()
	require.Equal(t, []uint64{2048 * bytesPerMiB}, backend.thresholds)
	require.Equal(t, 1, backend.resetCount)
	endFirst()
	require.Len(t, backend.thresholds, 1)
	endSecond()
	require.Equal(t, []uint64{2048 * bytesPerMiB, 64 * bytesPerMiB}, backend.thresholds)
	require.Equal(t, []uint64{64 * bytesPerMiB}, backend.trimTargets)
}
