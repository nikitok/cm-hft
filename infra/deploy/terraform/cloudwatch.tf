# =============================================================================
# CM.HFT — CloudWatch Logs & Metrics
# =============================================================================
# Centralized logging with 30-day retention.
# Metric filters create CloudWatch metrics from log patterns for alerting.
# =============================================================================

# ---------------------------------------------------------------------------
# Log Groups
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "trading_core" {
  name              = "/cm-hft/trading-core"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-trading-core-logs"
    Role = "trading"
  }
}

resource "aws_cloudwatch_log_group" "monitoring" {
  name              = "/cm-hft/monitoring"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-monitoring-logs"
    Role = "monitoring"
  }
}

# ---------------------------------------------------------------------------
# Metric Filters — surface errors as CloudWatch metrics for alarming
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "trading_errors" {
  name           = "${var.project_name}-trading-error-count"
  pattern        = "{ $.level = \"ERROR\" }"
  log_group_name = aws_cloudwatch_log_group.trading_core.name

  metric_transformation {
    name          = "TradingCoreErrorCount"
    namespace     = "CM-HFT"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "monitoring_errors" {
  name           = "${var.project_name}-monitoring-error-count"
  pattern        = "{ $.level = \"ERROR\" }"
  log_group_name = aws_cloudwatch_log_group.monitoring.name

  metric_transformation {
    name          = "MonitoringErrorCount"
    namespace     = "CM-HFT"
    value         = "1"
    default_value = "0"
  }
}

# ---------------------------------------------------------------------------
# CloudWatch Alarms — alert on elevated error rates
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "trading_errors_high" {
  alarm_name          = "${var.project_name}-trading-errors-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TradingCoreErrorCount"
  namespace           = "CM-HFT"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Trading core error rate exceeds threshold — potential issue with order execution or market data feed."
  treat_missing_data  = "notBreaching"

  tags = {
    Name = "${var.project_name}-trading-errors-alarm"
  }
}
