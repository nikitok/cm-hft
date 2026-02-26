# =============================================================================
# CM.HFT — Terraform Main Configuration
# =============================================================================
# Infrastructure for a low-latency HFT trading platform on AWS.
# Uses ap-northeast-1 (Tokyo) for proximity to Asian exchanges.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment and configure for remote state storage.
  # Requires an existing S3 bucket and DynamoDB table for locking.
  #
  # backend "s3" {
  #   bucket         = "cm-hft-terraform-state"
  #   key            = "production/terraform.tfstate"
  #   region         = "ap-northeast-1"
  #   dynamodb_table = "cm-hft-terraform-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
