# =============================================================================
# CM.HFT — S3 Storage
# =============================================================================
# Data bucket for trading logs, backtest results, and market data snapshots.
# Versioning enabled for data integrity; lifecycle rules manage storage costs.
# =============================================================================

resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${var.environment}"

  tags = {
    Name = "${var.project_name}-data-${var.environment}"
  }
}

# ---------------------------------------------------------------------------
# Versioning — protect against accidental overwrites/deletes
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ---------------------------------------------------------------------------
# Server-Side Encryption — AES256 (SSE-S3)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# ---------------------------------------------------------------------------
# Lifecycle Rules — tiered storage to reduce costs
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-and-expire"
    status = "Enabled"

    # Apply to all objects in the bucket
    filter {}

    # Move to Infrequent Access after 90 days (cheaper for rarely accessed data)
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    # Delete after 365 days (adjust based on regulatory requirements)
    expiration {
      days = 365
    }

    # Clean up incomplete multipart uploads after 7 days
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# ---------------------------------------------------------------------------
# Block All Public Access — trading data must never be publicly accessible
# ---------------------------------------------------------------------------

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
