output "cluster_name" { value = module.eks.cluster_name }
output "vpc_id" { value = data.aws_vpc.default.id }
output "aws_region" { value = var.aws_region }
output "s3_bucket" { value = aws_s3_bucket.frutas.bucket }
output "trainer_repository_url" { value = aws_ecr_repository.trainer.repository_url }
output "inference_repository_url" { value = aws_ecr_repository.inference.repository_url }
