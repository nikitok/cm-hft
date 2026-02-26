// Package collector reads metric updates from a Unix domain socket and applies them
// to Prometheus metrics. It reconnects on disconnect with exponential backoff.
package collector

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cm-hft/monitoring/internal/metrics"
)

// MetricUpdate represents a single metric update received from the Rust trading core
// via the Unix domain socket. Each JSON line maps to one MetricUpdate.
type MetricUpdate struct {
	// Metric is the metric name, e.g. "latency", "pnl", "position", "order", "fill",
	// "connection_status", "message_rate", "spread_captured", "circuit_breaker", "rate_limit".
	Metric string `json:"metric"`

	// Value is the numeric value for this metric update.
	Value float64 `json:"value"`

	// Labels contains optional key-value label pairs (e.g. "symbol", "side", "exchange", "status", "type").
	Labels map[string]string `json:"labels,omitempty"`

	// Timestamp is the Unix timestamp in milliseconds when the metric was emitted.
	Timestamp int64 `json:"timestamp,omitempty"`
}

// Config holds configuration for the Collector.
type Config struct {
	// SocketPath is the path to the Unix domain socket.
	SocketPath string

	// InitialBackoff is the initial delay before reconnecting after a disconnect.
	InitialBackoff time.Duration

	// MaxBackoff is the maximum delay between reconnection attempts.
	MaxBackoff time.Duration

	// BackoffMultiplier scales the backoff after each failed attempt.
	BackoffMultiplier float64
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		SocketPath:        "/tmp/cm_hft_metrics.sock",
		InitialBackoff:    100 * time.Millisecond,
		MaxBackoff:        30 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// Collector reads JSON-line metric updates from a Unix domain socket
// and applies them to the Prometheus metrics.
type Collector struct {
	cfg     Config
	metrics *metrics.Metrics
	logger  *slog.Logger

	lastUpdateTime atomic.Int64 // unix millis of last update

	mu   sync.Mutex
	conn net.Conn
}

// New creates a new Collector with the given config and metrics.
func New(cfg Config, m *metrics.Metrics, logger *slog.Logger) *Collector {
	if logger == nil {
		logger = slog.Default()
	}
	c := &Collector{
		cfg:     cfg,
		metrics: m,
		logger:  logger,
	}
	c.lastUpdateTime.Store(time.Now().UnixMilli())
	return c
}

// LastUpdateTime returns the time of the last metric update as unix milliseconds.
func (c *Collector) LastUpdateTime() int64 {
	return c.lastUpdateTime.Load()
}

// Run starts the collector loop. It connects to the Unix domain socket, reads
// metric updates, and applies them. On disconnect it reconnects with exponential
// backoff. Run blocks until ctx is cancelled.
func (c *Collector) Run(ctx context.Context) error {
	backoff := c.cfg.InitialBackoff

	for {
		select {
		case <-ctx.Done():
			c.closeConn()
			return ctx.Err()
		default:
		}

		err := c.connectAndRead(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return err
			}
			c.logger.Warn("connection lost, will reconnect",
				"error", err,
				"backoff", backoff,
			)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}

		backoff = time.Duration(float64(backoff) * c.cfg.BackoffMultiplier)
		if backoff > c.cfg.MaxBackoff {
			backoff = c.cfg.MaxBackoff
		}
	}
}

// connectAndRead establishes a connection and reads lines until an error occurs.
func (c *Collector) connectAndRead(ctx context.Context) error {
	dialer := net.Dialer{}
	conn, err := dialer.DialContext(ctx, "unix", c.cfg.SocketPath)
	if err != nil {
		return fmt.Errorf("dial unix %s: %w", c.cfg.SocketPath, err)
	}

	c.mu.Lock()
	c.conn = conn
	c.mu.Unlock()

	c.logger.Info("connected to metrics socket", "path", c.cfg.SocketPath)

	// When context is cancelled, close the connection to unblock the scanner.
	go func() {
		<-ctx.Done()
		c.closeConn()
	}()

	defer c.closeConn()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var update MetricUpdate
		if err := json.Unmarshal(line, &update); err != nil {
			c.logger.Warn("failed to parse metric update",
				"error", err,
				"line", string(line),
			)
			continue
		}

		c.applyUpdate(&update)
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read socket: %w", err)
	}

	return io.EOF
}

// closeConn safely closes the current connection.
func (c *Collector) closeConn() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
}

// applyUpdate maps a MetricUpdate to the appropriate Prometheus metric.
func (c *Collector) applyUpdate(u *MetricUpdate) {
	c.lastUpdateTime.Store(time.Now().UnixMilli())

	switch u.Metric {
	case "latency":
		t := u.Labels["type"]
		if t == "" {
			t = "unknown"
		}
		c.metrics.LatencyNanoseconds.WithLabelValues(t).Observe(u.Value)

	case "pnl":
		c.metrics.PnLTotal.Set(u.Value)

	case "position":
		symbol := u.Labels["symbol"]
		side := u.Labels["side"]
		c.metrics.PositionQuantity.WithLabelValues(symbol, side).Set(u.Value)

	case "order":
		side := u.Labels["side"]
		status := u.Labels["status"]
		c.metrics.OrdersTotal.WithLabelValues(side, status).Add(u.Value)

	case "fill":
		side := u.Labels["side"]
		c.metrics.FillsTotal.WithLabelValues(side).Add(u.Value)

	case "connection_status":
		exchange := u.Labels["exchange"]
		c.metrics.ExchangeConnectionStatus.WithLabelValues(exchange).Set(u.Value)

	case "message_rate":
		exchange := u.Labels["exchange"]
		c.metrics.ExchangeMessageRate.WithLabelValues(exchange).Set(u.Value)

	case "spread_captured":
		c.metrics.SpreadCapturedBps.Set(u.Value)

	case "circuit_breaker":
		c.metrics.CircuitBreakerActive.Set(u.Value)

	case "rate_limit":
		exchange := u.Labels["exchange"]
		c.metrics.RateLimitUsageRatio.WithLabelValues(exchange).Set(u.Value)

	default:
		c.logger.Warn("unknown metric type", "metric", u.Metric)
	}
}
