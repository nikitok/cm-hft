package health

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// PeriodicChecker polls registered health check targets at a configurable interval
// and maintains the aggregated health status.
type PeriodicChecker struct {
	interval time.Duration
	timeout  time.Duration
	targets  []Target
	client   *http.Client
	logger   *slog.Logger

	mu     sync.RWMutex
	latest AggregatedHealth
}

// Target represents a remote service to health-check via HTTP GET.
type Target struct {
	Name string
	URL  string
}

// PeriodicCheckerConfig holds configuration for the PeriodicChecker.
type PeriodicCheckerConfig struct {
	// Interval between health check rounds.
	Interval time.Duration

	// Timeout for individual HTTP health check requests.
	Timeout time.Duration
}

// DefaultPeriodicCheckerConfig returns sensible defaults.
func DefaultPeriodicCheckerConfig() PeriodicCheckerConfig {
	return PeriodicCheckerConfig{
		Interval: 10 * time.Second,
		Timeout:  5 * time.Second,
	}
}

// NewPeriodicChecker creates a PeriodicChecker that polls the given targets.
func NewPeriodicChecker(cfg PeriodicCheckerConfig, targets []Target, logger *slog.Logger) *PeriodicChecker {
	if logger == nil {
		logger = slog.Default()
	}
	return &PeriodicChecker{
		interval: cfg.Interval,
		timeout:  cfg.Timeout,
		targets:  targets,
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
		logger: logger,
		latest: AggregatedHealth{
			Status:   StatusDown,
			Services: nil,
		},
	}
}

// Run starts the periodic health check loop. It blocks until ctx is cancelled.
func (pc *PeriodicChecker) Run(ctx context.Context) error {
	// Run an initial check immediately.
	pc.checkAll(ctx)

	ticker := time.NewTicker(pc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			pc.checkAll(ctx)
		}
	}
}

// Latest returns the most recent aggregated health status.
func (pc *PeriodicChecker) Latest() AggregatedHealth {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.latest
}

// checkAll performs health checks on all targets concurrently and updates latest.
func (pc *PeriodicChecker) checkAll(ctx context.Context) {
	results := make([]ServiceHealth, len(pc.targets))
	var wg sync.WaitGroup

	for i, target := range pc.targets {
		wg.Add(1)
		go func(idx int, t Target) {
			defer wg.Done()
			results[idx] = pc.checkTarget(ctx, t)
		}(i, target)
	}

	wg.Wait()

	agg := AggregatedHealth{
		Status:   StatusOK,
		Services: results,
	}

	for _, svc := range results {
		if svc.Status == StatusDown {
			agg.Status = StatusDown
			break
		}
		if svc.Status == StatusDegraded {
			agg.Status = StatusDegraded
		}
	}

	pc.mu.Lock()
	pc.latest = agg
	pc.mu.Unlock()

	pc.logger.Info("health check round complete",
		"overall_status", string(agg.Status),
		"services_checked", len(results),
	)
}

// checkTarget performs a single HTTP health check against a target.
func (pc *PeriodicChecker) checkTarget(ctx context.Context, t Target) ServiceHealth {
	start := time.Now()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, t.URL, nil)
	if err != nil {
		return ServiceHealth{
			Name:    t.Name,
			Status:  StatusDown,
			Message: fmt.Sprintf("request creation failed: %v", err),
			Latency: time.Since(start).Milliseconds(),
		}
	}

	resp, err := pc.client.Do(req)
	if err != nil {
		return ServiceHealth{
			Name:    t.Name,
			Status:  StatusDown,
			Message: fmt.Sprintf("request failed: %v", err),
			Latency: time.Since(start).Milliseconds(),
		}
	}
	defer resp.Body.Close()

	latency := time.Since(start).Milliseconds()

	if resp.StatusCode >= 500 {
		return ServiceHealth{
			Name:    t.Name,
			Status:  StatusDown,
			Message: fmt.Sprintf("HTTP %d", resp.StatusCode),
			Latency: latency,
		}
	}

	// Try to parse the response body for a "status" field.
	var body struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		// Could not parse, but the HTTP status was ok.
		return ServiceHealth{
			Name:    t.Name,
			Status:  StatusOK,
			Message: fmt.Sprintf("HTTP %d (body not parseable)", resp.StatusCode),
			Latency: latency,
		}
	}

	status := StatusOK
	switch Status(body.Status) {
	case StatusDegraded:
		status = StatusDegraded
	case StatusDown:
		status = StatusDown
	}

	return ServiceHealth{
		Name:    t.Name,
		Status:  status,
		Latency: latency,
	}
}
