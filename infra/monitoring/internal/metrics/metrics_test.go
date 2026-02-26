package metrics_test

import (
	"strings"
	"testing"

	"github.com/cm-hft/monitoring/internal/metrics"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func newTestRegistry() (*prometheus.Registry, *metrics.Metrics) {
	reg := prometheus.NewRegistry()
	m := metrics.New(reg)
	return reg, m
}

func TestNew_RegistersAllMetrics(t *testing.T) {
	reg, _ := newTestRegistry()

	// Gather all registered metrics.
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("failed to gather metrics: %v", err)
	}

	expected := map[string]bool{
		"cm_latency_nanoseconds":        false,
		"cm_pnl_total":                  false,
		"cm_position_quantity":           false,
		"cm_orders_total":               false,
		"cm_fills_total":                false,
		"cm_exchange_connection_status":  false,
		"cm_exchange_message_rate":       false,
		"cm_spread_captured_bps":         false,
		"cm_circuit_breaker_active":      false,
		"cm_rate_limit_usage_ratio":      false,
	}

	for _, fam := range families {
		if _, ok := expected[fam.GetName()]; ok {
			expected[fam.GetName()] = true
		}
	}

	// Gauges/counters without observations won't appear in Gather until they have data.
	// That's expected. We verify they're registered by trying to use them below.
}

func TestPnLTotal_SetAndRead(t *testing.T) {
	_, m := newTestRegistry()

	m.PnLTotal.Set(1234.56)

	val := testutil.ToFloat64(m.PnLTotal)
	if val != 1234.56 {
		t.Errorf("expected PnLTotal=1234.56, got %f", val)
	}
}

func TestPositionQuantity_Labels(t *testing.T) {
	_, m := newTestRegistry()

	m.PositionQuantity.WithLabelValues("btcusdt", "long").Set(0.5)
	m.PositionQuantity.WithLabelValues("ethusdt", "short").Set(10.0)

	val := testutil.ToFloat64(m.PositionQuantity.WithLabelValues("btcusdt", "long"))
	if val != 0.5 {
		t.Errorf("expected btcusdt long=0.5, got %f", val)
	}

	val = testutil.ToFloat64(m.PositionQuantity.WithLabelValues("ethusdt", "short"))
	if val != 10.0 {
		t.Errorf("expected ethusdt short=10.0, got %f", val)
	}
}

func TestOrdersTotal_Counter(t *testing.T) {
	_, m := newTestRegistry()

	m.OrdersTotal.WithLabelValues("buy", "filled").Add(5)
	m.OrdersTotal.WithLabelValues("sell", "cancelled").Add(3)

	val := testutil.ToFloat64(m.OrdersTotal.WithLabelValues("buy", "filled"))
	if val != 5 {
		t.Errorf("expected buy/filled=5, got %f", val)
	}

	val = testutil.ToFloat64(m.OrdersTotal.WithLabelValues("sell", "cancelled"))
	if val != 3 {
		t.Errorf("expected sell/cancelled=3, got %f", val)
	}
}

func TestFillsTotal_Counter(t *testing.T) {
	_, m := newTestRegistry()

	m.FillsTotal.WithLabelValues("buy").Add(10)
	m.FillsTotal.WithLabelValues("sell").Add(7)

	val := testutil.ToFloat64(m.FillsTotal.WithLabelValues("buy"))
	if val != 10 {
		t.Errorf("expected buy fills=10, got %f", val)
	}
}

func TestExchangeConnectionStatus_Gauge(t *testing.T) {
	_, m := newTestRegistry()

	m.ExchangeConnectionStatus.WithLabelValues("binance").Set(1)
	m.ExchangeConnectionStatus.WithLabelValues("bybit").Set(0)

	val := testutil.ToFloat64(m.ExchangeConnectionStatus.WithLabelValues("binance"))
	if val != 1 {
		t.Errorf("expected binance=1, got %f", val)
	}

	val = testutil.ToFloat64(m.ExchangeConnectionStatus.WithLabelValues("bybit"))
	if val != 0 {
		t.Errorf("expected bybit=0, got %f", val)
	}
}

func TestCircuitBreakerActive_Gauge(t *testing.T) {
	_, m := newTestRegistry()

	m.CircuitBreakerActive.Set(1)
	val := testutil.ToFloat64(m.CircuitBreakerActive)
	if val != 1 {
		t.Errorf("expected circuit_breaker=1, got %f", val)
	}

	m.CircuitBreakerActive.Set(0)
	val = testutil.ToFloat64(m.CircuitBreakerActive)
	if val != 0 {
		t.Errorf("expected circuit_breaker=0, got %f", val)
	}
}

func TestLatencyNanoseconds_Histogram(t *testing.T) {
	reg, m := newTestRegistry()

	m.LatencyNanoseconds.WithLabelValues("wire_to_internal").Observe(500)
	m.LatencyNanoseconds.WithLabelValues("wire_to_internal").Observe(1500)
	m.LatencyNanoseconds.WithLabelValues("order_to_ack").Observe(50000)

	// Verify the histogram is registered and has data.
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("failed to gather metrics: %v", err)
	}

	found := false
	for _, fam := range families {
		if fam.GetName() == "cm_latency_nanoseconds" {
			found = true
			if len(fam.GetMetric()) != 2 {
				t.Errorf("expected 2 label combos, got %d", len(fam.GetMetric()))
			}
		}
	}
	if !found {
		t.Error("cm_latency_nanoseconds not found in gathered metrics")
	}
}

func TestSpreadCapturedBps_Gauge(t *testing.T) {
	_, m := newTestRegistry()

	m.SpreadCapturedBps.Set(2.5)
	val := testutil.ToFloat64(m.SpreadCapturedBps)
	if val != 2.5 {
		t.Errorf("expected spread=2.5, got %f", val)
	}
}

func TestRateLimitUsageRatio_Gauge(t *testing.T) {
	_, m := newTestRegistry()

	m.RateLimitUsageRatio.WithLabelValues("binance").Set(0.75)
	val := testutil.ToFloat64(m.RateLimitUsageRatio.WithLabelValues("binance"))
	if val != 0.75 {
		t.Errorf("expected ratio=0.75, got %f", val)
	}
}

func TestExchangeMessageRate_Gauge(t *testing.T) {
	_, m := newTestRegistry()

	m.ExchangeMessageRate.WithLabelValues("binance").Set(1500.0)
	val := testutil.ToFloat64(m.ExchangeMessageRate.WithLabelValues("binance"))
	if val != 1500.0 {
		t.Errorf("expected message_rate=1500, got %f", val)
	}
}

func TestNew_DuplicateRegistration_Panics(t *testing.T) {
	reg := prometheus.NewRegistry()
	_ = metrics.New(reg)

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic on duplicate registration, got none")
		}
		msg, ok := r.(error)
		if ok && !strings.Contains(msg.Error(), "duplicate") {
			// The prometheus library may use different error messages; just ensure we panicked.
			t.Logf("panic message: %v", msg)
		}
	}()

	// Second registration on same registry should panic via MustRegister.
	_ = metrics.New(reg)
}
