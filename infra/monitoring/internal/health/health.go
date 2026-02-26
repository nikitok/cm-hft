package health

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"
)

// MetricsAgeProvider is a function that returns the timestamp (unix millis) of the last metric update.
type MetricsAgeProvider func() int64

// Handler serves the /health endpoint for the monitoring service.
type Handler struct {
	startTime          time.Time
	metricsAgeProvider MetricsAgeProvider
	logger             *slog.Logger
}

// NewHandler creates a new health Handler.
// metricsAgeProvider should return the unix-millis timestamp of the last metric update.
func NewHandler(metricsAgeProvider MetricsAgeProvider, logger *slog.Logger) *Handler {
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{
		startTime:          time.Now(),
		metricsAgeProvider: metricsAgeProvider,
		logger:             logger,
	}
}

// ServeHTTP handles HTTP requests for the health endpoint.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	lastUpdate := h.metricsAgeProvider()
	metricsAgeMs := now.UnixMilli() - lastUpdate
	if metricsAgeMs < 0 {
		metricsAgeMs = 0
	}

	uptimeSeconds := int64(now.Sub(h.startTime).Seconds())

	status := StatusOK
	if metricsAgeMs > 10_000 {
		status = StatusDegraded
	}

	resp := MonitoringHealth{
		Status:        status,
		MetricsAgeMs:  metricsAgeMs,
		UptimeSeconds: uptimeSeconds,
	}

	w.Header().Set("Content-Type", "application/json")
	if status != StatusOK {
		w.WriteHeader(http.StatusServiceUnavailable)
	} else {
		w.WriteHeader(http.StatusOK)
	}

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		h.logger.Error("failed to write health response", "error", err)
	}
}

// BuildMonitoringHealth constructs a MonitoringHealth snapshot without writing HTTP.
// Useful for aggregation in the periodic checker.
func (h *Handler) BuildMonitoringHealth() MonitoringHealth {
	now := time.Now()
	lastUpdate := h.metricsAgeProvider()
	metricsAgeMs := now.UnixMilli() - lastUpdate
	if metricsAgeMs < 0 {
		metricsAgeMs = 0
	}

	uptimeSeconds := int64(now.Sub(h.startTime).Seconds())

	status := StatusOK
	if metricsAgeMs > 10_000 {
		status = StatusDegraded
	}

	return MonitoringHealth{
		Status:        status,
		MetricsAgeMs:  metricsAgeMs,
		UptimeSeconds: uptimeSeconds,
	}
}
