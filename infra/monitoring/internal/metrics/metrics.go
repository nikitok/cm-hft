// Package metrics defines and registers Prometheus metrics for the CM.HFT trading platform.
package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

// Metrics holds all Prometheus metric collectors for the HFT platform.
type Metrics struct {
	LatencyNanoseconds       *prometheus.HistogramVec
	PnLTotal                 prometheus.Gauge
	PositionQuantity         *prometheus.GaugeVec
	OrdersTotal              *prometheus.CounterVec
	FillsTotal               *prometheus.CounterVec
	ExchangeConnectionStatus *prometheus.GaugeVec
	ExchangeMessageRate      *prometheus.GaugeVec
	SpreadCapturedBps        prometheus.Gauge
	CircuitBreakerActive     prometheus.Gauge
	RateLimitUsageRatio      *prometheus.GaugeVec
}

var (
	instance *Metrics
	once     sync.Once
)

// New creates a new Metrics instance and registers all collectors with the given registry.
// If registry is nil, prometheus.DefaultRegisterer is used.
func New(registry prometheus.Registerer) *Metrics {
	if registry == nil {
		registry = prometheus.DefaultRegisterer
	}

	m := &Metrics{
		LatencyNanoseconds: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "cm_latency_nanoseconds",
				Help:    "Wire-to-internal latency and order-to-ack latency in nanoseconds.",
				Buckets: []float64{100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000},
			},
			[]string{"type"},
		),
		PnLTotal: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "cm_pnl_total",
				Help: "Current profit and loss in quote currency.",
			},
		),
		PositionQuantity: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "cm_position_quantity",
				Help: "Current position quantity per symbol and side.",
			},
			[]string{"symbol", "side"},
		),
		OrdersTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cm_orders_total",
				Help: "Total number of orders by side and status.",
			},
			[]string{"side", "status"},
		),
		FillsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cm_fills_total",
				Help: "Total number of fills by side.",
			},
			[]string{"side"},
		),
		ExchangeConnectionStatus: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "cm_exchange_connection_status",
				Help: "Exchange connection status: 1=connected, 0=disconnected.",
			},
			[]string{"exchange"},
		),
		ExchangeMessageRate: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "cm_exchange_message_rate",
				Help: "Messages per second from each exchange.",
			},
			[]string{"exchange"},
		),
		SpreadCapturedBps: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "cm_spread_captured_bps",
				Help: "Spread captured in basis points.",
			},
		),
		CircuitBreakerActive: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "cm_circuit_breaker_active",
				Help: "Circuit breaker status: 1=triggered, 0=inactive.",
			},
		),
		RateLimitUsageRatio: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "cm_rate_limit_usage_ratio",
				Help: "Rate limit usage ratio per exchange (0.0-1.0).",
			},
			[]string{"exchange"},
		),
	}

	registry.MustRegister(
		m.LatencyNanoseconds,
		m.PnLTotal,
		m.PositionQuantity,
		m.OrdersTotal,
		m.FillsTotal,
		m.ExchangeConnectionStatus,
		m.ExchangeMessageRate,
		m.SpreadCapturedBps,
		m.CircuitBreakerActive,
		m.RateLimitUsageRatio,
	)

	return m
}

// Default returns the singleton Metrics instance registered with the default Prometheus registry.
func Default() *Metrics {
	once.Do(func() {
		instance = New(nil)
	})
	return instance
}
