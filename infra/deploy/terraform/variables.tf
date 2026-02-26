# =============================================================================
# CM.HFT — Terraform Variables
# =============================================================================

variable "aws_region" {
  description = "AWS region for all resources. Tokyo chosen for proximity to Asian exchanges."
  type        = string
  default     = "ap-northeast-1"
}

variable "environment" {
  description = "Deployment environment (production, staging, development)."
  type        = string
  default     = "production"

  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "instance_type" {
  description = "EC2 instance type for trading core. c5n.xlarge provides enhanced networking (up to 25 Gbps) critical for low-latency trading."
  type        = string
  default     = "c5n.xlarge"
}

variable "ssh_allowed_cidrs" {
  description = "List of CIDR blocks allowed SSH access to bastion/trading instances. Keep this as narrow as possible."
  type        = list(string)
  default     = []
}

variable "project_name" {
  description = "Project name used for resource naming and tagging."
  type        = string
  default     = "cm-hft"
}
