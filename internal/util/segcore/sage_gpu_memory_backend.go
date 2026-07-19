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
	"fmt"
	"strconv"
	"time"

	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

type sageSegcoreGpuMemoryBackend struct{}

func (sageSegcoreGpuMemoryBackend) stats(deviceID int32) (SageCuvsGpuMemoryPoolStats, int32) {
	stats, status := getSageCuvsGpuMemoryPoolStats(deviceID)
	recordSageGpuPoolOperation(deviceID, "stats", status)
	if status == 0 {
		nodeID := fmt.Sprint(paramtable.GetNodeID())
		device := strconv.FormatInt(int64(deviceID), 10)
		metrics.SageCuvsGpuPoolBytes.WithLabelValues(nodeID, device, "reserved_current").Set(float64(stats.ReservedCurrent))
		metrics.SageCuvsGpuPoolBytes.WithLabelValues(nodeID, device, "reserved_high").Set(float64(stats.ReservedHigh))
		metrics.SageCuvsGpuPoolBytes.WithLabelValues(nodeID, device, "used_current").Set(float64(stats.UsedCurrent))
		metrics.SageCuvsGpuPoolBytes.WithLabelValues(nodeID, device, "used_high").Set(float64(stats.UsedHigh))
	}
	return stats, status
}

func (sageSegcoreGpuMemoryBackend) setReleaseThreshold(deviceID int32, bytes uint64) int32 {
	status := setSageCuvsGpuMemoryPoolReleaseThreshold(deviceID, bytes)
	recordSageGpuPoolOperation(deviceID, "set_release_threshold", status)
	return status
}

func (sageSegcoreGpuMemoryBackend) resetHighWater(deviceID int32) int32 {
	status := resetSageCuvsGpuMemoryPoolHighWater(deviceID)
	recordSageGpuPoolOperation(deviceID, "reset_high_water", status)
	return status
}

func (sageSegcoreGpuMemoryBackend) trim(deviceID int32, minBytesToKeep uint64, synchronize bool) int32 {
	status := trimSageCuvsGpuMemoryPool(deviceID, minBytesToKeep, synchronize)
	operation := "trim_async"
	if synchronize {
		operation = "trim_sync"
	}
	recordSageGpuPoolOperation(deviceID, operation, status)
	return status
}

func recordSageGpuPoolOperation(deviceID int32, operation string, status int32) {
	metrics.SageCuvsGpuPoolOperations.WithLabelValues(
		fmt.Sprint(paramtable.GetNodeID()),
		strconv.FormatInt(int64(deviceID), 10),
		operation,
		strconv.FormatInt(int64(status), 10),
	).Inc()
}

var defaultSageGpuMemoryController = newSageGpuMemoryController(
	loadSageGpuMemoryControllerConfig(),
	sageSegcoreGpuMemoryBackend{},
	time.Now,
)

// BeginSageGpuServingPhase accounts for a database query.  The returned
// function must be deferred.  Reclamation only runs after the last concurrent
// query exits, outside maintenance, and after a cooldown.
func BeginSageGpuServingPhase() func() {
	return defaultSageGpuMemoryController.beginServing()
}

// BeginSageGpuMaintenancePhase temporarily raises the allocator retention
// budget for index build/load/compaction.  Nested and concurrent phases are
// reference counted; the final exit restores the serving budget.
func BeginSageGpuMaintenancePhase() func() {
	return defaultSageGpuMemoryController.beginMaintenance()
}

// NotifySageGpuMemoryReclaimable records a reclaim debt after a database object
// is released.  It never trims while work is active; the last tracked owner
// repays the debt with a synchronized trim at the quiescence boundary.
func NotifySageGpuMemoryReclaimable() {
	defaultSageGpuMemoryController.maybeReclaim(true)
}
