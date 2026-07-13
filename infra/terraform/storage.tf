# Bronze layer: raw payload of every API call, for reprocessing and
# traceability (each BigQuery row points back to a raw_uri here).
resource "google_storage_bucket" "raw_bronze" {
  name     = var.raw_bucket_name
  location = var.region

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}
