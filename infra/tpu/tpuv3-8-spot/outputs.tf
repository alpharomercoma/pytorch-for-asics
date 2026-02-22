# =============================================================================
# Output Values
# =============================================================================

# -----------------------------------------------------------------------------
# TPU Outputs
# -----------------------------------------------------------------------------

output "tpu_id" {
  description = "The unique identifier of the TPU VM"
  value       = google_tpu_v2_vm.tpu.id
}

output "tpu_name" {
  description = "The name of the TPU VM"
  value       = google_tpu_v2_vm.tpu.name
}

output "tpu_zone" {
  description = "The zone where the TPU VM is located"
  value       = google_tpu_v2_vm.tpu.zone
}

output "tpu_state" {
  description = "The current state of the TPU VM"
  value       = google_tpu_v2_vm.tpu.state
}

output "tpu_health" {
  description = "The health status of the TPU VM"
  value       = google_tpu_v2_vm.tpu.health
}

output "tpu_network_endpoints" {
  description = "The network endpoints where TPU workers can be accessed"
  value       = google_tpu_v2_vm.tpu.network_endpoints
}

output "tpu_accelerator_type" {
  description = "The TPU accelerator type"
  value       = var.tpu_accelerator_type
}

output "tpu_runtime_version" {
  description = "The TPU runtime version"
  value       = var.tpu_runtime_version
}

output "tpu_api_version" {
  description = "The API version that created this TPU Node"
  value       = google_tpu_v2_vm.tpu.api_version
}

# -----------------------------------------------------------------------------
# Disk Outputs
# -----------------------------------------------------------------------------

output "disk_id" {
  description = "The unique identifier of the persistent disk"
  value       = google_compute_disk.tpu_disk.id
}

output "disk_name" {
  description = "The name of the persistent disk"
  value       = google_compute_disk.tpu_disk.name
}

output "disk_self_link" {
  description = "The self link of the persistent disk"
  value       = google_compute_disk.tpu_disk.self_link
}

output "disk_zone" {
  description = "The zone where the disk is located"
  value       = google_compute_disk.tpu_disk.zone
}

output "disk_size_gb" {
  description = "The size of the disk in GB"
  value       = google_compute_disk.tpu_disk.size
}

output "disk_type" {
  description = "The type of the persistent disk"
  value       = google_compute_disk.tpu_disk.type
}

# -----------------------------------------------------------------------------
# Connection Information
# -----------------------------------------------------------------------------

output "disk_attached_to_tpu" {
  description = "Whether the disk is attached to the TPU VM"
  value       = var.attach_disk_to_tpu
}

output "connection_info" {
  description = "Summary of the TPU and disk connection"
  value = {
    tpu_name        = google_tpu_v2_vm.tpu.name
    tpu_zone        = google_tpu_v2_vm.tpu.zone
    disk_name       = google_compute_disk.tpu_disk.name
    disk_zone       = google_compute_disk.tpu_disk.zone
    disk_attached   = var.attach_disk_to_tpu
    attachment_mode = var.attach_disk_to_tpu ? var.disk_attachment_mode : "N/A"
  }
}

# -----------------------------------------------------------------------------
# Available TPU Configurations (for reference)
# -----------------------------------------------------------------------------

output "available_runtime_versions" {
  description = "Available TPU runtime versions in the specified zone"
  value       = data.google_tpu_v2_runtime_versions.available.versions
}

output "available_accelerator_types" {
  description = "Available TPU accelerator types in the specified zone"
  value       = data.google_tpu_v2_accelerator_types.available.types
}
