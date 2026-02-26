package health_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/cm-hft/monitoring/internal/health"
)

func TestPeriodicChecker_ChecksTargets(t *testing.T) {
	// Create a test HTTP server that responds with health status.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer server.Close()

	targets := []health.Target{
		{Name: "test-service", URL: server.URL},
	}

	cfg := health.PeriodicCheckerConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, targets, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	// Wait for at least one check to complete.
	time.Sleep(200 * time.Millisecond)

	latest := pc.Latest()
	if latest.Status != health.StatusOK {
		t.Errorf("expected overall status ok, got %s", latest.Status)
	}

	if len(latest.Services) != 1 {
		t.Fatalf("expected 1 service, got %d", len(latest.Services))
	}

	if latest.Services[0].Name != "test-service" {
		t.Errorf("expected service name 'test-service', got %s", latest.Services[0].Name)
	}

	if latest.Services[0].Status != health.StatusOK {
		t.Errorf("expected service status ok, got %s", latest.Services[0].Status)
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_DetectsDownService(t *testing.T) {
	// Create a server that returns 500.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	targets := []health.Target{
		{Name: "failing-service", URL: server.URL},
	}

	cfg := health.PeriodicCheckerConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, targets, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	time.Sleep(200 * time.Millisecond)

	latest := pc.Latest()
	if latest.Status != health.StatusDown {
		t.Errorf("expected overall status down, got %s", latest.Status)
	}

	if len(latest.Services) != 1 {
		t.Fatalf("expected 1 service, got %d", len(latest.Services))
	}

	if latest.Services[0].Status != health.StatusDown {
		t.Errorf("expected service status down, got %s", latest.Services[0].Status)
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_DetectsDegradedService(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "degraded"})
	}))
	defer server.Close()

	targets := []health.Target{
		{Name: "degraded-service", URL: server.URL},
	}

	cfg := health.PeriodicCheckerConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, targets, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	time.Sleep(200 * time.Millisecond)

	latest := pc.Latest()
	if latest.Status != health.StatusDegraded {
		t.Errorf("expected overall status degraded, got %s", latest.Status)
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_MultipleTargets_DownOverridesDegraded(t *testing.T) {
	okServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer okServer.Close()

	downServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer downServer.Close()

	targets := []health.Target{
		{Name: "ok-service", URL: okServer.URL},
		{Name: "down-service", URL: downServer.URL},
	}

	cfg := health.PeriodicCheckerConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, targets, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	time.Sleep(200 * time.Millisecond)

	latest := pc.Latest()
	if latest.Status != health.StatusDown {
		t.Errorf("expected overall status down (one service is down), got %s", latest.Status)
	}

	if len(latest.Services) != 2 {
		t.Fatalf("expected 2 services, got %d", len(latest.Services))
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_UnreachableTarget(t *testing.T) {
	// Use a URL that won't connect.
	targets := []health.Target{
		{Name: "unreachable", URL: "http://127.0.0.1:1"},
	}

	cfg := health.PeriodicCheckerConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  500 * time.Millisecond,
	}

	pc := health.NewPeriodicChecker(cfg, targets, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	time.Sleep(1 * time.Second)

	latest := pc.Latest()
	if latest.Status != health.StatusDown {
		t.Errorf("expected status down for unreachable target, got %s", latest.Status)
	}

	if len(latest.Services) != 1 {
		t.Fatalf("expected 1 service, got %d", len(latest.Services))
	}

	if latest.Services[0].Status != health.StatusDown {
		t.Errorf("expected service status down, got %s", latest.Services[0].Status)
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_NoTargets(t *testing.T) {
	cfg := health.PeriodicCheckerConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, nil, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	time.Sleep(150 * time.Millisecond)

	latest := pc.Latest()
	// With no targets, status should be OK (nothing is failing).
	if latest.Status != health.StatusOK {
		t.Errorf("expected status ok with no targets, got %s", latest.Status)
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_LatencyRecorded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(20 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer server.Close()

	targets := []health.Target{
		{Name: "slow-service", URL: server.URL},
	}

	cfg := health.PeriodicCheckerConfig{
		Interval: 100 * time.Millisecond,
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, targets, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = pc.Run(ctx)
	}()

	time.Sleep(250 * time.Millisecond)

	latest := pc.Latest()
	if len(latest.Services) != 1 {
		t.Fatalf("expected 1 service, got %d", len(latest.Services))
	}

	// Latency should be at least 20ms since the server sleeps 20ms.
	if latest.Services[0].Latency < 15 {
		t.Errorf("expected latency >= 15ms, got %d", latest.Services[0].Latency)
	}

	cancel()
	wg.Wait()
}

func TestPeriodicChecker_ContextCancellation(t *testing.T) {
	cfg := health.PeriodicCheckerConfig{
		Interval: 1 * time.Hour, // Long interval.
		Timeout:  2 * time.Second,
	}

	pc := health.NewPeriodicChecker(cfg, nil, nil)

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		done <- pc.Run(ctx)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("checker did not exit after context cancellation")
	}
}

func TestDefaultPeriodicCheckerConfig(t *testing.T) {
	cfg := health.DefaultPeriodicCheckerConfig()
	if cfg.Interval != 10*time.Second {
		t.Errorf("unexpected default interval: %v", cfg.Interval)
	}
	if cfg.Timeout != 5*time.Second {
		t.Errorf("unexpected default timeout: %v", cfg.Timeout)
	}
}
