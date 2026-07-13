resource "google_artifact_registry_repository" "pipelines" {
  location      = var.region
  repository_id = "fin-data"
  format        = "DOCKER"
  description   = "Docker images for the extraction pipelines"
}
