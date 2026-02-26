# =============================================================================
# CM.HFT — IAM Roles & Policies
# =============================================================================
# Least-privilege IAM for trading core EC2 instances.
# Grants only:
#   - S3 access to the project data bucket
#   - CloudWatch Logs for log shipping
#   - SSM Session Manager (eliminates need for SSH key management)
# =============================================================================

# ---------------------------------------------------------------------------
# EC2 Assume Role Policy
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------------
# Trading Core IAM Role
# ---------------------------------------------------------------------------

resource "aws_iam_role" "trading_core" {
  name               = "${var.project_name}-trading-core-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = {
    Name = "${var.project_name}-trading-core-role"
  }
}

# ---------------------------------------------------------------------------
# S3 Access Policy — restricted to project bucket only
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "s3_access" {
  statement {
    sid    = "S3BucketAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject",
    ]
    resources = [
      aws_s3_bucket.data.arn,
      "${aws_s3_bucket.data.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name   = "${var.project_name}-s3-access"
  role   = aws_iam_role.trading_core.id
  policy = data.aws_iam_policy_document.s3_access.json
}

# ---------------------------------------------------------------------------
# CloudWatch Logs Policy — restricted to project log groups
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "cloudwatch_logs" {
  statement {
    sid    = "CloudWatchLogsAccess"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/cm-hft/*",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/cm-hft/*:*",
    ]
  }
}

resource "aws_iam_role_policy" "cloudwatch_logs" {
  name   = "${var.project_name}-cloudwatch-logs"
  role   = aws_iam_role.trading_core.id
  policy = data.aws_iam_policy_document.cloudwatch_logs.json
}

# ---------------------------------------------------------------------------
# SSM Session Manager — for secure shell access without SSH keys
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.trading_core.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ---------------------------------------------------------------------------
# Instance Profile
# ---------------------------------------------------------------------------

resource "aws_iam_instance_profile" "trading_core" {
  name = "${var.project_name}-trading-core-profile"
  role = aws_iam_role.trading_core.name
}
