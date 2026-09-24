data "aws_vpc" "default" {
  default = true
}

data "aws_caller_identity" "current" {}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_subnet" "default" {
  for_each = toset(data.aws_subnets.default.ids)
  id       = each.value
}

locals {
  # Use every default public subnet supported by EKS. In us-east-1,
  # use1-az3 is explicitly unsupported by EKS.
  public_subnet_ids = sort([
    for subnet in data.aws_subnet.default : subnet.id
    if subnet.map_public_ip_on_launch && subnet.availability_zone_id != "use1-az3"
  ])
  tags = {
    Project   = var.cluster_name
    ManagedBy = "terraform"
  }
}

# EKS and its always-on administrative node.
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name                                     = var.cluster_name
  kubernetes_version                       = var.kubernetes_version
  endpoint_public_access                   = true
  endpoint_private_access                  = true
  endpoint_public_access_cidrs             = var.cluster_public_access_cidrs
  enable_cluster_creator_admin_permissions = true
  authentication_mode                      = "API_AND_CONFIG_MAP"
  encryption_config                        = null
  create_kms_key                           = false
  cloudwatch_log_group_retention_in_days   = 1

  vpc_id                   = data.aws_vpc.default.id
  subnet_ids               = local.public_subnet_ids
  control_plane_subnet_ids = local.public_subnet_ids
  node_security_group_tags = { "karpenter.sh/discovery" = var.cluster_name }

  addons = {
    coredns                = {}
    kube-proxy             = {}
    vpc-cni                = { before_compute = true }
    eks-pod-identity-agent = { before_compute = true }
  }

  eks_managed_node_groups = {
    admin = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = [var.admin_instance_type]
      subnet_ids     = local.public_subnet_ids
      min_size       = 1
      max_size       = 1
      desired_size   = 1
      labels = {
        "node-role.formatec/admin" = "true"
      }
    }
  }

  tags = local.tags
}

# Karpenter discovers existing public subnets and the EKS node security group
# by this cluster-specific tag. The EKS module owns its security-group tags.
resource "aws_ec2_tag" "karpenter_subnet_discovery" {
  for_each    = toset(local.public_subnet_ids)
  resource_id = each.value
  key         = "karpenter.sh/discovery"
  value       = var.cluster_name
}

# Data and container registries. They are not tied to the lifetime of a GPU node.
resource "aws_s3_bucket" "frutas" {
  bucket        = "${var.cluster_name}-${data.aws_caller_identity.current.account_id}"
  force_destroy = true # Lab cleanup removes all objects; export anything worth keeping first.
  tags          = local.tags
}

resource "aws_s3_bucket_public_access_block" "frutas" {
  bucket                  = aws_s3_bucket.frutas.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "trainer" {
  name         = "${var.cluster_name}-trainer"
  force_delete = true
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

resource "aws_ecr_repository" "inference" {
  name         = "${var.cluster_name}-inference"
  force_delete = true
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

# The version-pinned upstream template owns Karpenter's node role, six controller
# policies, SQS interruption queue and EventBridge rules. Terraform owns the stack.
data "http" "karpenter_template" {
  url = "https://raw.githubusercontent.com/aws/karpenter-provider-aws/v1.12.1/website/content/en/preview/getting-started/getting-started-with-karpenter/cloudformation.yaml"
}

resource "aws_cloudformation_stack" "karpenter" {
  name          = "Karpenter-${var.cluster_name}"
  template_body = data.http.karpenter_template.response_body
  capabilities  = ["CAPABILITY_NAMED_IAM"]
  parameters    = { ClusterName = module.eks.cluster_name }
  tags          = local.tags
}

data "aws_iam_policy_document" "pod_identity_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "karpenter_controller" {
  name               = "${var.cluster_name}-karpenter"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_trust.json
  tags               = local.tags
}

locals {
  karpenter_policy_names = toset([
    "NodeLifecycle", "IAMIntegration", "EKSIntegration", "Interruption", "ResourceDiscovery", "ZonalShift"
  ])
}

resource "aws_iam_role_policy_attachment" "karpenter_controller" {
  for_each   = local.karpenter_policy_names
  role       = aws_iam_role.karpenter_controller.name
  policy_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/KarpenterController${each.key}Policy-${var.cluster_name}"
  depends_on = [aws_cloudformation_stack.karpenter]
}

resource "aws_eks_access_entry" "karpenter_node" {
  cluster_name  = module.eks.cluster_name
  principal_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/KarpenterNodeRole-${var.cluster_name}"
  type          = "EC2_LINUX"
  depends_on    = [aws_cloudformation_stack.karpenter]
}

resource "aws_eks_pod_identity_association" "karpenter" {
  cluster_name    = module.eks.cluster_name
  namespace       = "kube-system"
  service_account = "karpenter"
  role_arn        = aws_iam_role.karpenter_controller.arn
  depends_on      = [aws_iam_role_policy_attachment.karpenter_controller]
}

data "aws_iam_policy_document" "frutas_s3" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.frutas.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.frutas.arn}/*"]
  }
}

resource "aws_iam_role" "frutas_s3" {
  name               = "${var.cluster_name}-s3"
  assume_role_policy = data.aws_iam_policy_document.pod_identity_trust.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "frutas_s3" {
  name   = "frutas-artifacts"
  role   = aws_iam_role.frutas_s3.id
  policy = data.aws_iam_policy_document.frutas_s3.json
}

resource "aws_eks_pod_identity_association" "frutas_s3" {
  cluster_name    = module.eks.cluster_name
  namespace       = "default"
  service_account = "frutas-s3"
  role_arn        = aws_iam_role.frutas_s3.arn
  depends_on      = [aws_iam_role_policy.frutas_s3]
}
