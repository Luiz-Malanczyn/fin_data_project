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

# Daily news-sentiment backfill (see src/ml/news_data.py) -- distinct from
# the price pipelines above: it's rate-limited by Gemini's free tier
# (~500 requests/day) rather than by how much history there is to catch
# up on, so it self-paces via a BigQuery-backed checkpoint and just picks
# up where yesterday's run left off. Only created when a Gemini key is
# configured, since the job can't do anything without one.
resource "google_cloud_run_v2_job" "news_pipeline" {
  count    = var.gemini_api_key != "" ? 1 : 0
  name     = "fin-data-news"
  location = var.region

  depends_on = [google_secret_manager_secret_iam_member.pipelines_gemini_api_key_accessor]

  template {
    template {
      service_account = google_service_account.pipelines_runtime.email
      max_retries      = 1
      timeout          = "3600s"

      containers {
        image = var.image
        args  = ["news"]

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.value.name
            value = env.value.value
          }
        }

        env {
          name = "GEMINI_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.gemini_api_key[0].secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }
}
