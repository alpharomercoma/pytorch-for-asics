# =============================================================================
# Google Cloud TPU v3-8 with Persistent Disk - Main Configuration
# =============================================================================
#
# This Terraform configuration creates:
# - A TPU v3-8 VM instance as a spot VM
# - A balanced persistent disk
# - Connects the disk to the TPU VM
#
# IMPORTANT: The disk must be in the same zone as the TPU for attachment.
# =============================================================================

# -----------------------------------------------------------------------------
# Local Values
# -----------------------------------------------------------------------------

locals {
  # Common tags/labels for all resources
  common_labels = merge(var.tpu_labels, {
    project = var.project_id
  })

  # Ensure disk zone matches TPU zone for attachment
  # TPU data_disks can only attach disks in the same zone
  effective_disk_zone = var.attach_disk_to_tpu ? var.tpu_zone : var.disk_zone
}

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

# Get available TPU runtime versions for validation
data "google_tpu_v2_runtime_versions" "available" {
  provider = google-beta
  zone     = var.tpu_zone
}

# Get available TPU accelerator types for validation
data "google_tpu_v2_accelerator_types" "available" {
  provider = google-beta
  zone     = var.tpu_zone
}

# -----------------------------------------------------------------------------
# Persistent Disk Resource
# -----------------------------------------------------------------------------

resource "google_compute_disk" "tpu_disk" {
  name = var.disk_name
  type = var.disk_type
  zone = local.effective_disk_zone
  size = var.disk_size_gb

  labels = merge(local.common_labels, var.disk_labels)

  # Physical block size for the disk
  physical_block_size_bytes = 4096

  # Lifecycle configuration
  lifecycle {
    # Prevent accidental deletion of the disk
    prevent_destroy = false

    # Create new disk before destroying old one during replacement
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# TPU v2 VM Resource (v3-8 Accelerator)
# -----------------------------------------------------------------------------

resource "google_tpu_v2_vm" "tpu" {
  provider = google-beta

  name            = var.tpu_name
  zone            = var.tpu_zone
  runtime_version = var.tpu_runtime_version
  description     = "TPU v3-8 Spot VM for ML workloads - managed by Terraform"

  # TPU Accelerator Configuration
  # Using accelerator_type for simple v3-8 configuration
  accelerator_type = var.tpu_accelerator_type

  # Scheduling Configuration - Spot VM
  scheduling_config {
    preemptible = var.tpu_preemptible
    spot        = var.tpu_spot
  }

  # Network Configuration
  network_config {
    network             = var.network
    subnetwork          = var.subnetwork
    enable_external_ips = var.enable_external_ips
    can_ip_forward      = false
  }

  # Data Disk Attachment
  # Only attach if enabled and disk is in the same zone
  dynamic "data_disks" {
    for_each = var.attach_disk_to_tpu ? [1] : []
    content {
      source_disk = google_compute_disk.tpu_disk.id
      mode        = var.disk_attachment_mode
    }
  }

  # Shielded Instance Configuration (security best practice)
  shielded_instance_config {
    enable_secure_boot = true
  }

  # Labels for resource organization
  labels = local.common_labels

  # Metadata for the TPU VM
  metadata = {
    managed_by = "terraform"
  }

  # Tags for network firewall rules
  tags = ["tpu-vm", "ml-workload"]

  # Explicit dependency on disk
  depends_on = [google_compute_disk.tpu_disk]

  # Lifecycle configuration
  lifecycle {
    # Ignore changes to labels that may be added by GCP
    ignore_changes = [
      labels["goog-dataproc-cluster-name"],
      labels["goog-dataproc-cluster-uuid"],
    ]
  }
}
