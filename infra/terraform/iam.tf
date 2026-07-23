# Service account used by the Cloud Run Jobs to run the pipelines.
resource "google_service_account" "pipelines_runtime" {
  account_id   = "fin-data-pipelines"
  display_name = "Fin Data - pipelines runtime"
}

resource "google_project_iam_member" "pipelines_bq_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.pipelines_runtime.email}"
}

resource "google_project_iam_member" "pipelines_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.pipelines_runtime.email}"
}

resource "google_storage_bucket_iam_member" "pipelines_raw_bucket_writer" {
  bucket = google_storage_bucket.raw_bronze.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipelines_runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "pipelines_brapi_token_accessor" {
  count     = var.brapi_token != "" ? 1 : 0
  secret_id = google_secret_manager_secret.brapi_token[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipelines_runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "pipelines_gemini_api_key_accessor" {
  count     = var.gemini_api_key != "" ? 1 : 0
  secret_id = google_secret_manager_secret.gemini_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipelines_runtime.email}"
}

# Service account used by Cloud Scheduler to trigger the Cloud Run Jobs.
resource "google_service_account" "scheduler_invoker" {
  account_id   = "fin-data-scheduler"
  display_name = "Fin Data - scheduler invoker"
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_can_run_stocks" {
  project  = google_cloud_run_v2_job.stocks_pipeline.project
  location = google_cloud_run_v2_job.stocks_pipeline.location
  name     = google_cloud_run_v2_job.stocks_pipeline.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_can_run_crypto" {
  project  = google_cloud_run_v2_job.crypto_pipeline.project
  location = google_cloud_run_v2_job.crypto_pipeline.location
  name     = google_cloud_run_v2_job.crypto_pipeline.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_can_run_news" {
  count    = var.gemini_api_key != "" ? 1 : 0
  project  = google_cloud_run_v2_job.news_pipeline[0].project
  location = google_cloud_run_v2_job.news_pipeline[0].location
  name     = google_cloud_run_v2_job.news_pipeline[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_can_run_recommendations" {
  project  = google_cloud_run_v2_job.recommendations_pipeline.project
  location = google_cloud_run_v2_job.recommendations_pipeline.location
  name     = google_cloud_run_v2_job.recommendations_pipeline.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}
