# =============================================================================
# CM.HFT — EC2 Instances
# =============================================================================
# Trading core runs on c5n.xlarge for enhanced networking (ENA).
# Placed in private subnet — no direct internet access, egress via NAT.
# Uses Amazon Linux 2023 for kernel tuning support and long-term updates.
# =============================================================================

# ---------------------------------------------------------------------------
# AMI Data Source — Latest Amazon Linux 2023
# ---------------------------------------------------------------------------

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ---------------------------------------------------------------------------
# Trading Core Instance
# ---------------------------------------------------------------------------

resource "aws_instance" "trading_core" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.private[0].id
  vpc_security_group_ids = [aws_security_group.trading_core.id]
  iam_instance_profile   = aws_iam_instance_profile.trading_core.name

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 100
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "${var.project_name}-trading-core-root"
    }
  }

  # Install Docker and pull trading image on first boot.
  # In production, consider baking a custom AMI with Packer instead.
  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -euxo pipefail

    # Install Docker
    dnf update -y
    dnf install -y docker
    systemctl enable docker
    systemctl start docker

    # Install Docker Compose plugin
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

    # Add ec2-user to docker group
    usermod -aG docker ec2-user

    # Install CloudWatch agent for log shipping
    dnf install -y amazon-cloudwatch-agent

    # Kernel tuning for low-latency networking
    cat >> /etc/sysctl.d/99-hft.conf <<SYSCTL
    net.core.rmem_max=16777216
    net.core.wmem_max=16777216
    net.ipv4.tcp_rmem=4096 87380 16777216
    net.ipv4.tcp_wmem=4096 87380 16777216
    net.core.netdev_max_backlog=5000
    net.ipv4.tcp_window_scaling=1
    net.ipv4.tcp_timestamps=0
    net.ipv4.tcp_sack=1
    net.ipv4.tcp_low_latency=1
    SYSCTL
    sysctl --system
  EOF
  )

  tags = {
    Name = "${var.project_name}-trading-core"
    Role = "trading"
  }
}

# ---------------------------------------------------------------------------
# Monitoring Instance (smaller, in same private subnet)
# ---------------------------------------------------------------------------

resource "aws_instance" "monitoring" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.private[1].id
  vpc_security_group_ids = [aws_security_group.monitoring.id]
  iam_instance_profile   = aws_iam_instance_profile.trading_core.name

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 50
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "${var.project_name}-monitoring-root"
    }
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -euxo pipefail
    dnf update -y
    dnf install -y docker
    systemctl enable docker
    systemctl start docker
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    usermod -aG docker ec2-user
    dnf install -y amazon-cloudwatch-agent
  EOF
  )

  tags = {
    Name = "${var.project_name}-monitoring"
    Role = "monitoring"
  }
}
