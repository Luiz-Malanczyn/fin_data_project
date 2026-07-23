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

# Daily recommendation refresh (see src/ml/selection.py): re-runs today's
# live prediction for whichever (asset, horizon) combos have a persisted,
# backtest-proven track record, using the models already baked into the
# image (see Dockerfile's `COPY models ./models` and .gcloudignore, which
# deliberately does NOT exclude the local models/ directory the way
# .gitignore does -- Cloud Run Jobs are stateless containers, so the
# trained model files have to travel with the image or the job has
# nothing to predict with). No retraining happens here; it needs
# fin-data-stocks to have already run so today's close price is in
# BigQuery, hence the later schedule.
resource "google_cloud_run_v2_job" "recommendations_pipeline" {
  name     = "fin-data-recommendations"
  location = var.region

  template {
    template {
      service_account = google_service_account.pipelines_runtime.email
      max_retries      = 1
      timeout          = "1800s"

      containers {
        image = var.image
        args  = ["recommendations"]

        # Cloud Run Jobs default to 512Mi, which OOM-killed this job in
        # practice (signal 9). 1Gi still wasn't enough -- deserialized
        # tree-ensemble models (RandomForest/GradientBoosting/XGBoost)
        # take up meaningfully more memory than their ~63MB on-disk size,
        # and nothing in the prediction loop frees a combo's live feature
        # frame (full price history joined against macro/fundamentals/
        # news data) before moving to the next of 11 sequential
        # predictions, so usage climbs across the run rather than peaking
        # once. 2Gi is a generous, cheap margin for a job that runs a few
        # minutes once a day.
        resources {
          limits = {
            cpu    = "1000m"
            memory = "4Gi"
          }
        }

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
