package collector_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/cm-hft/monitoring/internal/collector"
	"github.com/cm-hft/monitoring/internal/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// testSocket creates a temporary Unix domain socket path short enough for macOS
// (which limits Unix socket paths to ~104 characters). It uses /tmp directly
// with a unique name per test and registers a cleanup to remove the socket.
func testSocket(t *testing.T) string {
	t.Helper()
	f, err := os.CreateTemp("/tmp", "cm_test_*.sock")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	path := f.Name()
	f.Close()
	os.Remove(path) // Remove the file so we can bind to the path.
	t.Cleanup(func() { os.Remove(path) })
	return path
}

// startListener starts a Unix domain socket listener and returns it.
func startListener(t *testing.T, path string) net.Listener {
	t.Helper()
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("failed to listen on %s: %v", path, err)
	}
	return ln
}

func newTestMetrics(t *testing.T) (*prometheus.Registry, *metrics.Metrics) {
	t.Helper()
	reg := prometheus.NewRegistry()
	m := metrics.New(reg)
	return reg, m
}

func TestCollector_ParsesMetricUpdates(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	// Start listener.
	ln := startListener(t, sockPath)
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start collector.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = coll.Run(ctx)
	}()

	// Accept connection and send metric updates.
	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept failed: %v", err)
	}
	defer conn.Close()

	updates := []collector.MetricUpdate{
		{Metric: "pnl", Value: 500.25},
		{Metric: "latency", Value: 1500, Labels: map[string]string{"type": "wire_to_internal"}},
		{Metric: "position", Value: 0.001, Labels: map[string]string{"symbol": "btcusdt", "side": "long"}},
		{Metric: "order", Value: 1, Labels: map[string]string{"side": "buy", "status": "filled"}},
		{Metric: "fill", Value: 1, Labels: map[string]string{"side": "buy"}},
		{Metric: "connection_status", Value: 1, Labels: map[string]string{"exchange": "binance"}},
		{Metric: "message_rate", Value: 1200, Labels: map[string]string{"exchange": "binance"}},
		{Metric: "spread_captured", Value: 3.5},
		{Metric: "circuit_breaker", Value: 1},
		{Metric: "rate_limit", Value: 0.8, Labels: map[string]string{"exchange": "bybit"}},
	}

	for _, u := range updates {
		data, _ := json.Marshal(u)
		_, err := fmt.Fprintf(conn, "%s\n", data)
		if err != nil {
			t.Fatalf("failed to write update: %v", err)
		}
	}

	// Give the collector time to process.
	time.Sleep(200 * time.Millisecond)

	// Verify metrics.
	val := testutil.ToFloat64(m.PnLTotal)
	if val != 500.25 {
		t.Errorf("expected PnL=500.25, got %f", val)
	}

	val = testutil.ToFloat64(m.PositionQuantity.WithLabelValues("btcusdt", "long"))
	if val != 0.001 {
		t.Errorf("expected position btcusdt/long=0.001, got %f", val)
	}

	val = testutil.ToFloat64(m.OrdersTotal.WithLabelValues("buy", "filled"))
	if val != 1 {
		t.Errorf("expected orders buy/filled=1, got %f", val)
	}

	val = testutil.ToFloat64(m.FillsTotal.WithLabelValues("buy"))
	if val != 1 {
		t.Errorf("expected fills buy=1, got %f", val)
	}

	val = testutil.ToFloat64(m.ExchangeConnectionStatus.WithLabelValues("binance"))
	if val != 1 {
		t.Errorf("expected binance connection=1, got %f", val)
	}

	val = testutil.ToFloat64(m.ExchangeMessageRate.WithLabelValues("binance"))
	if val != 1200 {
		t.Errorf("expected binance message_rate=1200, got %f", val)
	}

	val = testutil.ToFloat64(m.SpreadCapturedBps)
	if val != 3.5 {
		t.Errorf("expected spread=3.5, got %f", val)
	}

	val = testutil.ToFloat64(m.CircuitBreakerActive)
	if val != 1 {
		t.Errorf("expected circuit_breaker=1, got %f", val)
	}

	val = testutil.ToFloat64(m.RateLimitUsageRatio.WithLabelValues("bybit"))
	if val != 0.8 {
		t.Errorf("expected rate_limit bybit=0.8, got %f", val)
	}

	cancel()
	wg.Wait()
}

func TestCollector_ReconnectsOnDisconnect(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	ln := startListener(t, sockPath)
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = coll.Run(ctx)
	}()

	// First connection: send a metric, then disconnect.
	conn1, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept failed: %v", err)
	}
	u := collector.MetricUpdate{Metric: "pnl", Value: 100}
	data, _ := json.Marshal(u)
	fmt.Fprintf(conn1, "%s\n", data)
	time.Sleep(100 * time.Millisecond)
	conn1.Close() // Force disconnect.

	// Second connection: collector should reconnect.
	conn2, err := ln.Accept()
	if err != nil {
		t.Fatalf("second accept failed: %v", err)
	}
	defer conn2.Close()

	u2 := collector.MetricUpdate{Metric: "pnl", Value: 200}
	data2, _ := json.Marshal(u2)
	fmt.Fprintf(conn2, "%s\n", data2)
	time.Sleep(100 * time.Millisecond)

	val := testutil.ToFloat64(m.PnLTotal)
	if val != 200 {
		t.Errorf("expected PnL=200 after reconnect, got %f", val)
	}

	cancel()
	wg.Wait()
}

func TestCollector_HandlesInvalidJSON(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	ln := startListener(t, sockPath)
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = coll.Run(ctx)
	}()

	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept failed: %v", err)
	}
	defer conn.Close()

	// Send invalid JSON followed by a valid update.
	fmt.Fprintf(conn, "this is not json\n")
	fmt.Fprintf(conn, "{invalid json too}\n")

	valid := collector.MetricUpdate{Metric: "pnl", Value: 42}
	data, _ := json.Marshal(valid)
	fmt.Fprintf(conn, "%s\n", data)

	time.Sleep(200 * time.Millisecond)

	// The valid metric should still be processed.
	val := testutil.ToFloat64(m.PnLTotal)
	if val != 42 {
		t.Errorf("expected PnL=42, got %f", val)
	}

	cancel()
	wg.Wait()
}

func TestCollector_HandlesEmptyLines(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	ln := startListener(t, sockPath)
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = coll.Run(ctx)
	}()

	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept failed: %v", err)
	}
	defer conn.Close()

	// Send empty lines interspersed with valid data.
	fmt.Fprintf(conn, "\n\n")
	u := collector.MetricUpdate{Metric: "pnl", Value: 99}
	data, _ := json.Marshal(u)
	fmt.Fprintf(conn, "%s\n", data)
	fmt.Fprintf(conn, "\n")

	time.Sleep(200 * time.Millisecond)

	val := testutil.ToFloat64(m.PnLTotal)
	if val != 99 {
		t.Errorf("expected PnL=99, got %f", val)
	}

	cancel()
	wg.Wait()
}

func TestCollector_LastUpdateTime(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	before := time.Now().UnixMilli()
	initialTime := coll.LastUpdateTime()
	if initialTime < before-1000 {
		t.Errorf("initial LastUpdateTime too old: %d vs now %d", initialTime, before)
	}

	ln := startListener(t, sockPath)
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = coll.Run(ctx)
	}()

	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept failed: %v", err)
	}
	defer conn.Close()

	time.Sleep(50 * time.Millisecond) // Ensure some time passes.

	beforeUpdate := time.Now().UnixMilli()
	u := collector.MetricUpdate{Metric: "pnl", Value: 1}
	data, _ := json.Marshal(u)
	fmt.Fprintf(conn, "%s\n", data)

	time.Sleep(100 * time.Millisecond)

	afterUpdate := coll.LastUpdateTime()
	if afterUpdate < beforeUpdate {
		t.Errorf("LastUpdateTime not updated: got %d, expected >= %d", afterUpdate, beforeUpdate)
	}

	cancel()
	wg.Wait()
}

func TestCollector_UnknownMetricType(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	ln := startListener(t, sockPath)
	defer ln.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = coll.Run(ctx)
	}()

	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("accept failed: %v", err)
	}
	defer conn.Close()

	// Unknown metric should not crash; it should be logged and skipped.
	u := collector.MetricUpdate{Metric: "unknown_metric", Value: 999}
	data, _ := json.Marshal(u)
	fmt.Fprintf(conn, "%s\n", data)

	// Follow with a valid metric to prove processing continued.
	u2 := collector.MetricUpdate{Metric: "pnl", Value: 77}
	data2, _ := json.Marshal(u2)
	fmt.Fprintf(conn, "%s\n", data2)

	time.Sleep(200 * time.Millisecond)

	val := testutil.ToFloat64(m.PnLTotal)
	if val != 77 {
		t.Errorf("expected PnL=77, got %f", val)
	}

	cancel()
	wg.Wait()
}

func TestCollector_ContextCancellation(t *testing.T) {
	sockPath := testSocket(t)
	_, m := newTestMetrics(t)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	cfg := collector.Config{
		SocketPath:        sockPath,
		InitialBackoff:    10 * time.Millisecond,
		MaxBackoff:        50 * time.Millisecond,
		BackoffMultiplier: 1.5,
	}

	coll := collector.New(cfg, m, logger)

	// Don't start a listener -- the collector will fail to connect, but
	// should exit cleanly when context is cancelled during backoff.

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		done <- coll.Run(ctx)
	}()

	// Cancel after a short time.
	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("collector did not exit after context cancellation")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := collector.DefaultConfig()
	if cfg.SocketPath != "/tmp/cm_hft_metrics.sock" {
		t.Errorf("unexpected default socket path: %s", cfg.SocketPath)
	}
	if cfg.InitialBackoff != 100*time.Millisecond {
		t.Errorf("unexpected initial backoff: %v", cfg.InitialBackoff)
	}
	if cfg.MaxBackoff != 30*time.Second {
		t.Errorf("unexpected max backoff: %v", cfg.MaxBackoff)
	}
	if cfg.BackoffMultiplier != 2.0 {
		t.Errorf("unexpected backoff multiplier: %f", cfg.BackoffMultiplier)
	}
}
