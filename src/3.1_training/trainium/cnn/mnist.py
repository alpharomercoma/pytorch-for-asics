#!/usr/bin/env python3
"""
TorchNeuron MNIST Training Example

This script trains a CNN on MNIST dataset, optimized for AWS Trainium using TorchNeuron.

Key TorchNeuron features:
- Automatic device detection via torch_xla
- Automatic graph compilation and optimization via XLA compiler
- Explicit synchronization with torch_xla.sync() for XLA devices

Note: torch.compile is NOT used for XLA/Neuron devices because torch_xla provides
its own graph compilation system. The XLA compiler automatically optimizes and
compiles computation graphs for Trainium devices.

Usage:
    # Standard training on Trainium
    python main.py --epochs 3

    # CPU-only (for testing)
    python main.py --no-accel --epochs 1 --dry-run
"""

import argparse
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

try:
    import torch_neuronx
    import torch_xla
    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False


class Net(nn.Module):
    """CNN architecture for MNIST classification."""
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_device_type(device) -> str:
    """Extract device type string from device object."""
    if device is None:
        return 'cpu'
    device_str = str(device)
    # Handle formats like 'neuron:0', 'cuda:0', 'cpu'
    return device_str.split(':')[0]


def train(args, model, device, train_loader, optimizer, epoch, use_mixed_precision=False):
    """Train model for one epoch."""
    model.train()
    device_type = get_device_type(device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Use autocast for mixed precision training on accelerators
        # TorchNeuron uses XLA backend
        if use_mixed_precision and device_type in ('xla', 'cuda'):
            # Note: autocast may not be supported for XLA in all versions
            with torch.autocast(device_type='cpu', enabled=False):  # Disabled for XLA compatibility
                output = model(data)
                loss = F.nll_loss(output, target)
        else:
            output = model(data)
            loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()

        # XLA requires explicit step marking to trigger execution
        if device_type == 'xla' and NEURON_AVAILABLE:
            torch_xla.sync()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, use_mixed_precision=False):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    device_type = get_device_type(device)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Use autocast for mixed precision inference
            if use_mixed_precision and device_type in ('xla', 'cuda'):
                # Note: autocast may not be supported for XLA in all versions
                with torch.autocast(device_type='cpu', enabled=False):  # Disabled for XLA compatibility
                    output = model(data)
            else:
                output = model(data)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # XLA requires explicit step marking to trigger execution
    if device_type == 'xla' and NEURON_AVAILABLE:
        torch_xla.sync()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def synchronize_device(device):
    """Synchronize device to ensure all operations are complete.

    For XLA/Neuron devices, we need to explicitly mark step boundaries.
    """
    device_type = get_device_type(device)

    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'xla':
        # XLA requires explicit synchronization
        if NEURON_AVAILABLE:
            torch_xla.sync()


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='TorchNeuron MNIST Example - Train CNN on AWS Trainium',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator (use CPU)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--save-model', action='store_true',
                        help='save the trained model')
    
    # TorchNeuron-specific arguments
    parser.add_argument('--compile', action='store_true',
                        help='enable torch.compile optimization (Note: not used for XLA/Neuron - XLA handles compilation automatically)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='enable mixed precision training with autocast')
    
    args, _ = parser.parse_known_args()

    # Device selection
    # TorchNeuron uses XLA backend for Trainium devices
    use_accel = not args.no_accel and NEURON_AVAILABLE

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch_xla.device()
        device_type = 'xla'  # TorchNeuron uses XLA backend
        print(f"Using Neuron device: {device} (type: {device_type})")
    else:
        device = torch.device("cpu")
        device_type = 'cpu'
        print("Using CPU")

    # DataLoader configuration
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if use_accel:
        accel_kwargs = {
            'num_workers': 1,
            'persistent_workers': True,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    # MNIST dataset with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Create model and move to device
    model = Net().to(device)
    
    # Apply torch.compile optimization
    # Note: XLA/Neuron devices use torch_xla's automatic graph compilation
    # and do NOT require torch.compile (which uses PyTorch's inductor backend)
    if args.compile and use_accel:
        if device_type == 'xla':
            print("Note: XLA/Neuron devices use automatic graph compilation via torch_xla.")
            print("torch.compile is not needed and may cause errors. Running without torch.compile.")
            # XLA compilation is handled automatically by torch_neuronx/torch_xla
            # Do NOT use torch.compile(model) here - it doesn't support XLA backend
        elif device_type == 'cuda':
            print("Applying torch.compile with default backend")
            model = torch.compile(model)
        else:
            print(f"torch.compile not supported for device type: {device_type}")
    elif args.compile and not use_accel:
        print("Warning: --compile flag ignored when using CPU")

    # Optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Mixed precision setting
    use_mixed_precision = args.mixed_precision and use_accel
    if use_mixed_precision:
        print(f"Mixed precision training enabled for device type: {device_type}")

    # Training loop with timing
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, use_mixed_precision)
        test(model, device, test_loader, use_mixed_precision)
        scheduler.step()

    # Synchronize device before measuring end time
    synchronize_device(device)

    # Print total training time
    total_time = time.time() - start_time
    formatted = str(timedelta(seconds=int(total_time)))
    print(f"Total time: {formatted} ({total_time:.3f} seconds)")

    # Save model if requested
    if args.save_model:
        model_path = "mnist_cnn.pt"
        # Handle compiled model (extract underlying module if needed)
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()