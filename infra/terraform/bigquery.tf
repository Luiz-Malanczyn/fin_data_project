resource "google_bigquery_dataset" "fin_data" {
  dataset_id = var.bq_dataset
  location   = var.bq_location
}

# Single history table, shared by every investment type.
# Partitioned by date and clustered by (type, id) -- the "indexes" used
# for filtering/joining in BigQuery.
resource "google_bigquery_table" "investment_history" {
  dataset_id = google_bigquery_dataset.fin_data.dataset_id
  table_id   = "investment_history"

  time_partitioning {
    type  = "DAY"
    field = "event_date"
  }

  clustering = ["investment_type", "investment_id"]

  schema = jsonencode([
    { name = "event_date", type = "DATE", mode = "REQUIRED" },
    { name = "investment_type", type = "STRING", mode = "REQUIRED" },
    { name = "investment_id", type = "STRING", mode = "REQUIRED" },
    { name = "source", type = "STRING", mode = "REQUIRED" },
    { name = "currency", type = "STRING", mode = "NULLABLE" },
    { name = "open", type = "FLOAT64", mode = "NULLABLE" },
    { name = "high", type = "FLOAT64", mode = "NULLABLE" },
    { name = "low", type = "FLOAT64", mode = "NULLABLE" },
    { name = "close", type = "FLOAT64", mode = "NULLABLE" },
    { name = "volume", type = "FLOAT64", mode = "NULLABLE" },
    { name = "extra", type = "JSON", mode = "NULLABLE" },
    { name = "ingestion_ts", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "raw_uri", type = "STRING", mode = "NULLABLE" },
  ])
}

# Dimension table for tracked assets. Starts as a mirror of watchlist.yaml;
# can become the source of truth later on (adding an asset = an INSERT).
resource "google_bigquery_table" "dim_investment" {
  dataset_id = google_bigquery_dataset.fin_data.dataset_id
  table_id   = "dim_investment"

  schema = jsonencode([
    { name = "investment_id", type = "STRING", mode = "REQUIRED" },
    { name = "investment_type", type = "STRING", mode = "REQUIRED" },
    { name = "name", type = "STRING", mode = "NULLABLE" },
    { name = "source", type = "STRING", mode = "NULLABLE" },
    { name = "active", type = "BOOL", mode = "NULLABLE" },
    { name = "added_at", type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "metadata", type = "JSON", mode = "NULLABLE" },
  ])
}
