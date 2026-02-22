# TPU v3-8 Spot VM

TPU v3-8 Spot VM with persistent disk for cost-effective ML training.

## Specifications

| Property | Value |
|----------|-------|
| Accelerator | TPU v3-8 (4 chips, 8 TensorCores) |
| Pricing | Spot/Preemptible |
| Runtime | tpu-ubuntu2204-base |
| Zone | europe-west4-a (Netherlands) |
| Storage | 50 GB pd-balanced |
| Shielded VM | Secure boot enabled |

> **Note:** TPU v3-8 is available in limited zones. `europe-west4-a` is one of the supported zones.

## Architecture

```
Zone: europe-west4-a
├── TPU v3-8 VM (Spot)
│   ├── Ubuntu 22.04 base
│   └── Secure Boot enabled
└── Persistent Disk (50 GB pd-balanced)
```

## Deploy

```bash
cd tpuv3-8-spot
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project_id

terraform init
terraform validate
terraform plan
terraform apply
```

## Inputs

| Variable | Description | Default |
|----------|-------------|---------|
| `project_id` | GCP project ID | (required) |
| `tpu_zone` | TPU zone | `europe-west4-a` |
| `tpu_accelerator_type` | Accelerator type | `v3-8` |
| `tpu_spot` | Use spot pricing | `true` |
| `disk_size_gb` | Disk size (GB) | `50` |
| `attach_disk_to_tpu` | Attach disk to TPU | `true` |
| `enable_external_ips` | Assign public IPs to TPU workers | `false` |

> **Important:** The persistent disk must be in the same zone as the TPU VM.

## Cost Estimate

| Resource | On-Demand | Spot (est.) |
|----------|-----------|-------------|
| TPU v3-8 (4 chips) | $8.80/hr | ~$0.88–3.52/hr |
| 50 GB pd-balanced | $6.00/mo | $6.00/mo |

Spot prices are dynamic. Check the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) for current estimates.

## Cleanup

```bash
terraform destroy
```
