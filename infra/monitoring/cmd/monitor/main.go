// Package main is the entry point for the CM.HFT monitoring service.
// It starts an HTTP server exposing Prometheus metrics on /metrics and
// a health check endpoint on /health, while reading metric updates from
// the Rust trading core via a Unix domain socket.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cm-hft/monitoring/internal/collector"
	"github.com/cm-hft/monitoring/internal/health"
	"github.com/cm-hft/monitoring/internal/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	// Configuration via flags with env-var fallback.
	listenAddr := flag.String("listen", envOrDefault("CM_MONITOR_LISTEN", ":9090"), "HTTP listen address")
	socketPath := flag.String("socket", envOrDefault("CM_MONITOR_SOCKET", "/tmp/cm_hft_metrics.sock"), "Unix domain socket path for metric updates")
	healthTargetsFlag := flag.String("health-targets", envOrDefault("CM_MONITOR_HEALTH_TARGETS", ""), "Comma-separated health check targets (name=url,...)")
	healthInterval := flag.Duration("health-interval", parseDurationOrDefault(envOrDefault("CM_MONITOR_HEALTH_INTERVAL", "10s"), 10*time.Second), "Health check poll interval")
	logLevel := flag.String("log-level", envOrDefault("CM_MONITOR_LOG_LEVEL", "info"), "Log level (debug, info, warn, error)")
	flag.Parse()

	// Set up structured logging.
	var level slog.Level
	switch strings.ToLower(*logLevel) {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	logger.Info("starting CM.HFT monitoring service",
		"listen", *listenAddr,
		"socket", *socketPath,
	)

	// Create Prometheus registry and metrics.
	registry := prometheus.NewRegistry()
	m := metrics.New(registry)

	// Create collector.
	collectorCfg := collector.DefaultConfig()
	collectorCfg.SocketPath = *socketPath
	coll := collector.New(collectorCfg, m, logger)

	// Create health handler.
	healthHandler := health.NewHandler(func() int64 {
		return coll.LastUpdateTime()
	}, logger)

	// Parse health targets for the periodic checker.
	var targets []health.Target
	if *healthTargetsFlag != "" {
		for _, entry := range strings.Split(*healthTargetsFlag, ",") {
			parts := strings.SplitN(strings.TrimSpace(entry), "=", 2)
			if len(parts) == 2 {
				targets = append(targets, health.Target{
					Name: strings.TrimSpace(parts[0]),
					URL:  strings.TrimSpace(parts[1]),
				})
			}
		}
	}

	checkerCfg := health.DefaultPeriodicCheckerConfig()
	checkerCfg.Interval = *healthInterval
	periodicChecker := health.NewPeriodicChecker(checkerCfg, targets, logger)

	// Set up HTTP mux.
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	}))
	mux.Handle("/health", healthHandler)
	mux.HandleFunc("/health/all", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		latest := periodicChecker.Latest()
		if err := json.NewEncoder(w).Encode(latest); err != nil {
			logger.Error("failed to write aggregated health response", "error", err)
		}
	})

	server := &http.Server{
		Addr:         *listenAddr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Context for graceful shutdown.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start collector in background.
	go func() {
		if err := coll.Run(ctx); err != nil && ctx.Err() == nil {
			logger.Error("collector stopped unexpectedly", "error", err)
		}
	}()

	// Start periodic health checker in background.
	go func() {
		if err := periodicChecker.Run(ctx); err != nil && ctx.Err() == nil {
			logger.Error("periodic health checker stopped unexpectedly", "error", err)
		}
	}()

	// Start HTTP server in background.
	go func() {
		logger.Info("HTTP server starting", "addr", *listenAddr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("HTTP server error", "error", err)
			cancel()
		}
	}()

	// Wait for termination signal.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		logger.Info("received signal, shutting down", "signal", sig)
	case <-ctx.Done():
	}

	cancel()

	// Graceful shutdown with timeout.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("HTTP server shutdown error", "error", err)
	}

	logger.Info("monitoring service stopped")
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseDurationOrDefault(s string, d time.Duration) time.Duration {
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return d
	}
	return parsed
}
