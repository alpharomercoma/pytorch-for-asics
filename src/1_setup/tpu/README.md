# Google Cloud TPU Templates

Google Cloud TPU templates using Terraform (HCL).

## Templates

### [tpuv3-8-spot](./tpuv3-8-spot)

TPU v3-8 Spot VM with persistent disk for cost-effective ML training.

| Property | Value |
|----------|-------|
| Accelerator | TPU v3-8 (Spot) |
| Zone | europe-west4-a (Netherlands) |
| Storage | 50 GB pd-balanced |
| Cost | ~$0.88–3.52/hr (spot) vs $8.80/hr (on-demand) |
