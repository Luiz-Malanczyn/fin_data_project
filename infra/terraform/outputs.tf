output "bq_dataset" {
  value = google_bigquery_dataset.fin_data.dataset_id
}

output "raw_bucket" {
  value = google_storage_bucket.raw_bronze.name
}

output "artifact_registry_repo" {
  value = google_artifact_registry_repository.pipelines.name
}

output "pipelines_runtime_service_account" {
  value = google_service_account.pipelines_runtime.email
}
