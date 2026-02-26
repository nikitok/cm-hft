# =============================================================================
# CM.HFT — Terraform Outputs
# =============================================================================

output "trading_core_private_ip" {
  description = "Private IP address of the trading core instance"
  value       = aws_instance.trading_core.private_ip
}

output "monitoring_private_ip" {
  description = "Private IP address of the monitoring instance"
  value       = aws_instance.monitoring.private_ip
}

output "s3_bucket_name" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.data.id
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}
