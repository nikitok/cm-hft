// Package health defines health check types and handlers for the CM.HFT platform.
package health

// Status represents the health state of a service.
type Status string

const (
	StatusOK       Status = "ok"
	StatusDegraded Status = "degraded"
	StatusDown     Status = "down"
)

// MonitoringHealth is the health response for the monitoring service itself.
type MonitoringHealth struct {
	Status       Status `json:"status"`
	MetricsAgeMs int64  `json:"metrics_age_ms"`
	UptimeSeconds int64 `json:"uptime_seconds"`
}

// TradingCoreHealth is the health response expected from the Rust trading core.
type TradingCoreHealth struct {
	Status         Status            `json:"status"`
	TradingEnabled bool              `json:"trading_enabled"`
	Exchanges      map[string]string `json:"exchanges"`
	Position       map[string]float64 `json:"positions"`
	UptimeSeconds  int64             `json:"uptime_seconds"`
}

// ServiceHealth is a generic wrapper for any service's health check result.
type ServiceHealth struct {
	Name    string `json:"name"`
	Status  Status `json:"status"`
	Message string `json:"message,omitempty"`
	Latency int64  `json:"latency_ms"`
}

// AggregatedHealth represents the combined health of all monitored services.
type AggregatedHealth struct {
	Status   Status          `json:"status"`
	Services []ServiceHealth `json:"services"`
}

// Checker is the interface that health-checkable services must implement.
type Checker interface {
	// Name returns the name of the service being checked.
	Name() string

	// Check performs the health check and returns the service health result.
	// It should respect the context for timeout and cancellation.
	Check() ServiceHealth
}
