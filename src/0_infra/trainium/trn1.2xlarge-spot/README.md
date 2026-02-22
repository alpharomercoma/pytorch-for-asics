# Trainium Spot Instance

Cost-optimized Trainium1 spot instance for ML workloads with auto-shutdown.

## Specifications

| Property | Value |
|----------|-------|
| Instance Type | trn1.2xlarge (Spot) |
| Accelerator | 1× Trainium chip (16 NeuronCores) |
| OS | Ubuntu 24.04 LTS (x86_64) |
| Region | us-east-1 (N. Virginia) |
| CPU | 2 cores × 1 thread (reduced from 8 vCPUs) |
| Memory | 32 GiB |
| Storage | 50 GiB GP3 (encrypted) |
| Spot Behavior | Persistent request, STOP on interruption |

> **Note:** Trainium instances require a service quota increase. By default, AWS accounts have 0 vCPUs for Trainium. Request quota for "Running On-Demand Trn Instances" in the [Service Quotas Console](https://console.aws.amazon.com/servicequotas/).

## Auto-Shutdown

### SSH/Activity Detection (Systemd Timer)

Checks every 5 minutes with a 10-minute boot grace period. Detects:
- Active SSH sessions
- SSM Session Manager connections
- Screen/tmux sessions
- Neuron compiler and runtime processes
- ML training/inference scripts
- CPU usage > 10%
- Memory usage > 80%

Shuts down after 2 consecutive idle checks (10 minutes).

> No CloudWatch CPU alarm — during training, compute is offloaded to NeuronCores while host CPU idles, which would cause false positives.

## Architecture

```
VPC (single AZ)
└── trn1.2xlarge (Spot)
    ├── Neuron SDK pre-installed
    ├── 50 GiB GP3
    └── Systemd Timer (activity detection, 10 min) → shutdown -h now
```

## Deploy

```bash
cd trainium-spot
pnpm install
pnpm run build
npx cdk bootstrap   # first time only
npx cdk deploy -c sshAllowedCidr="$(curl -s https://checkip.amazonaws.com)/32"
```

Default SSH ingress is `127.0.0.1/32`. Set `sshAllowedCidr` explicitly at deploy time to allow remote SSH.

## Connect

```bash
export AWS_REGION=us-east-1
export KEY_PAIR_ID=$(aws ec2 describe-key-pairs --region $AWS_REGION \
  --query "KeyPairs[?KeyName=='TrainiumSpotStack-keypair'].[KeyPairId]" --output text)

aws ssm get-parameter --name /ec2/keypair/$KEY_PAIR_ID --region $AWS_REGION \
  --with-decryption --query Parameter.Value --output text \
  > ~/.ssh/trainium-spot.pem && chmod 400 ~/.ssh/trainium-spot.pem

export IP=$(aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:Name,Values=TrainiumSpotStack-trainium" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

ssh -i ~/.ssh/trainium-spot.pem ubuntu@$IP
```

## Restart After Shutdown

```bash
export AWS_REGION=us-east-1
export INSTANCE_ID=$(aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:Name,Values=TrainiumSpotStack-trainium" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

aws ec2 start-instances --instance-ids $INSTANCE_ID --region $AWS_REGION
```

## Cost Estimate

| Resource | On-Demand | Spot (est.) |
|----------|-----------|-------------|
| trn1.2xlarge | $1.34/hr | ~$0.40/hr |
| 50 GiB GP3 | $4.00/mo | $4.00/mo |

## Cleanup

```bash
npx cdk destroy
```
