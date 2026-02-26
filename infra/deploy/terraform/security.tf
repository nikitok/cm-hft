# =============================================================================
# CM.HFT — Security Groups
# =============================================================================
# Principle of least privilege: only open ports that are strictly required.
# Trading core has minimal exposure; monitoring is VPC-internal only.
# =============================================================================

# ---------------------------------------------------------------------------
# Trading Core Security Group
# ---------------------------------------------------------------------------

resource "aws_security_group" "trading_core" {
  name        = "${var.project_name}-trading-core"
  description = "Security group for HFT trading core instance"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-trading-core-sg"
  }
}

# SSH access — restricted to specific CIDRs (bastion or VPN)
resource "aws_vpc_security_group_ingress_rule" "trading_ssh" {
  count = length(var.ssh_allowed_cidrs) > 0 ? length(var.ssh_allowed_cidrs) : 0

  security_group_id = aws_security_group.trading_core.id
  description       = "SSH from allowed CIDR"
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
  cidr_ipv4         = var.ssh_allowed_cidrs[count.index]
}

# Kill switch HTTP endpoint — accessible only from within VPC
resource "aws_vpc_security_group_ingress_rule" "trading_kill_switch" {
  security_group_id = aws_security_group.trading_core.id
  description       = "Kill switch HTTP endpoint (VPC only)"
  from_port         = 8080
  to_port           = 8080
  ip_protocol       = "tcp"
  cidr_ipv4         = aws_vpc.main.cidr_block
}

# Outbound: HTTPS to exchanges and APIs
resource "aws_vpc_security_group_egress_rule" "trading_https" {
  security_group_id = aws_security_group.trading_core.id
  description       = "HTTPS to exchanges and external APIs"
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
  cidr_ipv4         = "0.0.0.0/0"
}

# Outbound: all traffic within VPC (for inter-service communication)
resource "aws_vpc_security_group_egress_rule" "trading_vpc_all" {
  security_group_id = aws_security_group.trading_core.id
  description       = "All traffic within VPC"
  ip_protocol       = "-1"
  cidr_ipv4         = aws_vpc.main.cidr_block
}

# ---------------------------------------------------------------------------
# Monitoring Security Group
# ---------------------------------------------------------------------------

resource "aws_security_group" "monitoring" {
  name        = "${var.project_name}-monitoring"
  description = "Security group for monitoring stack (Prometheus, Grafana, Loki)"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-monitoring-sg"
  }
}

# Prometheus — VPC only
resource "aws_vpc_security_group_ingress_rule" "monitoring_prometheus" {
  security_group_id = aws_security_group.monitoring.id
  description       = "Prometheus (VPC only)"
  from_port         = 9090
  to_port           = 9090
  ip_protocol       = "tcp"
  cidr_ipv4         = aws_vpc.main.cidr_block
}

# Grafana — VPC only (expose via ALB or SSH tunnel for external access)
resource "aws_vpc_security_group_ingress_rule" "monitoring_grafana" {
  security_group_id = aws_security_group.monitoring.id
  description       = "Grafana (VPC only)"
  from_port         = 3000
  to_port           = 3000
  ip_protocol       = "tcp"
  cidr_ipv4         = aws_vpc.main.cidr_block
}

# Outbound: all traffic within VPC
resource "aws_vpc_security_group_egress_rule" "monitoring_vpc_all" {
  security_group_id = aws_security_group.monitoring.id
  description       = "All traffic within VPC"
  ip_protocol       = "-1"
  cidr_ipv4         = aws_vpc.main.cidr_block
}
