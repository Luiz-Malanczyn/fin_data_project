locals {
  common_env = [
    { name = "GCP_PROJECT", value = var.project_id },
    { name = "BQ_DATASET", value = var.bq_dataset },
    { name = "RAW_BUCKET", value = var.raw_bucket_name },
  ]
}

resource "google_cloud_run_v2_job" "stocks_pipeline" {
  name     = "fin-data-stocks"
  location = var.region

  template {
    template {
      service_account = google_service_account.pipelines_runtime.email
      max_retries      = 1

      containers {
        image = var.image
        args  = ["stock"]

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.value.name
            value = env.value.value
          }
        }

        dynamic "env" {
          for_each = var.brapi_token != "" ? [1] : []
          content {
            name = "BRAPI_TOKEN"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.brapi_token[0].secret_id
                version = "latest"
              }
            }
          }
        }
      }
    }
  }
}

resource "google_cloud_run_v2_job" "crypto_pipeline" {
  name     = "fin-data-crypto"
  location = var.region

  template {
    template {
      service_account = google_service_account.pipelines_runtime.email
      max_retries      = 1

      containers {
        image = var.image
        args  = ["crypto"]

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.value.name
            value = env.value.value
          }
        }
      }
    }
  }
}
