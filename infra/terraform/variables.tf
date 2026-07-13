variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for the resources (Cloud Run, Scheduler, bucket). Defaults to us-central1, which is Always Free eligible for GCS -- other regions incur small storage costs."
  type        = string
  default     = "us-central1"
}

variable "bq_location" {
  description = "Location of the BigQuery dataset. Defaults to the US multi-region, which is Always Free eligible for the 10GB storage tier."
  type        = string
  default     = "US"
}

variable "bq_dataset" {
  description = "BigQuery dataset name"
  type        = string
  default     = "fin_data"
}

variable "raw_bucket_name" {
  description = "Name of the GCS bucket used as the bronze/landing zone"
  type        = string
}

variable "image" {
  description = "Docker image for the pipelines (Artifact Registry), e.g. southamerica-east1-docker.pkg.dev/PROJECT/fin-data/pipelines:latest"
  type        = string
}

variable "stocks_schedule" {
  description = "Cron for the stocks pipeline (default: 10pm every weekday, after B3 closes)"
  type        = string
  default     = "0 22 * * 1-5"
}

variable "crypto_schedule" {
  description = "Cron for the crypto pipeline (default: hourly)"
  type        = string
  default     = "0 * * * *"
}

variable "brapi_token" {
  description = "brapi.dev API token (optional on the free tier). Stored in Secret Manager."
  type        = string
  default     = ""
  sensitive   = true
}
