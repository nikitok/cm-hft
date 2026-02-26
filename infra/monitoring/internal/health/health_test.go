package health_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cm-hft/monitoring/internal/health"
)

func TestHealthHandler_OKResponse(t *testing.T) {
	// Provider returns "just now" timestamp.
	provider := func() int64 {
		return time.Now().UnixMilli()
	}

	h := health.NewHandler(provider, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}

	ct := rec.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("expected Content-Type application/json, got %s", ct)
	}

	var resp health.MonitoringHealth
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Status != health.StatusOK {
		t.Errorf("expected status ok, got %s", resp.Status)
	}

	if resp.UptimeSeconds < 0 {
		t.Errorf("uptime_seconds should be >= 0, got %d", resp.UptimeSeconds)
	}

	// MetricsAgeMs should be very small since provider returns "now".
	if resp.MetricsAgeMs > 1000 {
		t.Errorf("metrics_age_ms unexpectedly large: %d", resp.MetricsAgeMs)
	}
}

func TestHealthHandler_DegradedWhenMetricsStale(t *testing.T) {
	// Provider returns a timestamp from 15 seconds ago (> 10s threshold).
	provider := func() int64 {
		return time.Now().Add(-15 * time.Second).UnixMilli()
	}

	h := health.NewHandler(provider, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("expected status 503, got %d", rec.Code)
	}

	var resp health.MonitoringHealth
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Status != health.StatusDegraded {
		t.Errorf("expected status degraded, got %s", resp.Status)
	}

	if resp.MetricsAgeMs < 10000 {
		t.Errorf("metrics_age_ms should be > 10000, got %d", resp.MetricsAgeMs)
	}
}

func TestHealthHandler_ResponseFormat(t *testing.T) {
	provider := func() int64 {
		return time.Now().UnixMilli()
	}

	h := health.NewHandler(provider, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	// Verify all expected JSON fields are present.
	var raw map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&raw); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	requiredFields := []string{"status", "metrics_age_ms", "uptime_seconds"}
	for _, field := range requiredFields {
		if _, ok := raw[field]; !ok {
			t.Errorf("missing required field: %s", field)
		}
	}
}

func TestHealthHandler_UptimeIncreases(t *testing.T) {
	provider := func() int64 {
		return time.Now().UnixMilli()
	}

	h := health.NewHandler(provider, nil)

	// First request.
	req1 := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec1 := httptest.NewRecorder()
	h.ServeHTTP(rec1, req1)

	var resp1 health.MonitoringHealth
	json.NewDecoder(rec1.Body).Decode(&resp1)

	time.Sleep(1100 * time.Millisecond)

	// Second request.
	req2 := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, req2)

	var resp2 health.MonitoringHealth
	json.NewDecoder(rec2.Body).Decode(&resp2)

	if resp2.UptimeSeconds <= resp1.UptimeSeconds {
		t.Errorf("uptime should increase: first=%d, second=%d", resp1.UptimeSeconds, resp2.UptimeSeconds)
	}
}

func TestBuildMonitoringHealth(t *testing.T) {
	provider := func() int64 {
		return time.Now().UnixMilli()
	}

	h := health.NewHandler(provider, nil)
	mh := h.BuildMonitoringHealth()

	if mh.Status != health.StatusOK {
		t.Errorf("expected status ok, got %s", mh.Status)
	}

	if mh.UptimeSeconds < 0 {
		t.Errorf("uptime should be >= 0, got %d", mh.UptimeSeconds)
	}
}
