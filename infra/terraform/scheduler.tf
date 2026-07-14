resource "google_cloud_scheduler_job" "stocks_schedule" {
  name      = "fin-data-stocks-schedule"
  region    = var.region
  schedule  = var.stocks_schedule
  time_zone = "America/Sao_Paulo"

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.stocks_pipeline.name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }
}

resource "google_cloud_scheduler_job" "crypto_schedule" {
  name      = "fin-data-crypto-schedule"
  region    = var.region
  schedule  = var.crypto_schedule
  time_zone = "America/Sao_Paulo"

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.crypto_pipeline.name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }
}

resource "google_cloud_scheduler_job" "news_schedule" {
  count     = var.gemini_api_key != "" ? 1 : 0
  name      = "fin-data-news-schedule"
  region    = var.region
  schedule  = var.news_schedule
  time_zone = "America/Sao_Paulo"

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.news_pipeline[0].name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }
}
