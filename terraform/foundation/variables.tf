variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "formatec-frutas-pro"
}

variable "admin_instance_type" {
  type    = string
  default = "t3a.large"
}

variable "kubernetes_version" {
  type    = string
  default = "1.34"
}

variable "cluster_public_access_cidrs" {
  description = "Restringir al IP público propio /32 para la demo."
  type        = list(string)
  default     = ["127.0.0.1/32"]
}
