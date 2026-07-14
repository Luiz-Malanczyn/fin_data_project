# brapi.dev works without a token on the free tier, so the secret (and the
# env var wiring in cloud_run.tf / the accessor binding in iam.tf) is only
# created when a real token is provided via the brapi_token variable.
resource "google_secret_manager_secret" "brapi_token" {
  count     = var.brapi_token != "" ? 1 : 0
  secret_id = "brapi-token"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "brapi_token" {
  count       = var.brapi_token != "" ? 1 : 0
  secret      = google_secret_manager_secret.brapi_token[0].id
  secret_data = var.brapi_token
}

# The news job (src/ml/news_data.py) has no operation without a Gemini key,
# unlike brapi -- so both the secret and the job itself are only created
# when gemini_api_key is provided.
resource "google_secret_manager_secret" "gemini_api_key" {
  count     = var.gemini_api_key != "" ? 1 : 0
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "gemini_api_key" {
  count       = var.gemini_api_key != "" ? 1 : 0
  secret      = google_secret_manager_secret.gemini_api_key[0].id
  secret_data = var.gemini_api_key
}
