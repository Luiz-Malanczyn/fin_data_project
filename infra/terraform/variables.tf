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

variable "gemini_api_key" {
  description = "Google Gemini API key (free tier, AI Studio -- must come from a project with no billing account attached, or free-tier requests get billing-gated). Stored in Secret Manager. The news job is skipped entirely if empty."
  type        = string
  default     = ""
  sensitive   = true
}

variable "news_schedule" {
  description = "Cron for the daily news-sentiment backfill job. Default: once a day, off-peak."
  type        = string
  default     = "30 6 * * *"
}

variable "recommendations_schedule" {
  description = "Cron for the daily recommendation refresh. Default: 30 minutes after the default stocks_schedule (22:00 BRT), so it always runs against a fresh close."
  type        = string
  default     = "30 22 * * 1-5"
}
