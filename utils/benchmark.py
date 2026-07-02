from dataset.dataset import dataloaders
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import torch

IMAGES = './car-segmentation/images'
MASKS = './car-segmentation/masks'
CLASSES = 5
CONFIGS = [(4, 0), (4, 2), (8, 2), (8, 4)]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.down1 = DoubleConv(3, 16)
        self.down2 = DoubleConv(16, 32)
        self.bottleneck = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.conv1 = DoubleConv(64, 32)
        self.conv2 = DoubleConv(32, 16)

        self.out = nn.Conv2d(16, n, 1)
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        b = self.bottleneck(self.pool(d2))

        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)
        return self.out(u2)

def train(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += images.size(0)
    return total_loss / len(loader), total_samples

def benchmark(img_dir, mask_dir, configs, classes):
    results = []

    for batch, workers in configs:
        print(f"\n[WARNING] (Batch, Workers): ({batch}, {workers}).")
        train_loader, _, _ = dataloaders(img_dir, mask_dir, batch=batch, workers=workers)
        model = UNet(classes).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
    
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.time()
        loss, samples = train(train_loader, model, optimizer, criterion)
        
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        throughput = samples / elapsed
        peak_memory = 0

        if DEVICE.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\n[WARNING] Time: {elapsed:.2f} sec, Throughput: {throughput:.2f} img/sec, Loss: {loss:.2f}, GPU mem: {peak_memory:.2f} MB.")
        results.append({'batch': batch, 'workers': workers, 'time': elapsed, 'throughput': throughput, 'loss': loss, 'gpu_memory': peak_memory})
    return results

def get_plot(results):
    labels = [f'B{r["batch"]}-W{r["workers"]}' for r in results]
    x = np.arange(len(labels))

    # Time
    plt.figure(figsize=(12, 5))
    times = [r['time'] for r in results]
    bars = plt.bar(x, times)
    plt.xticks(x, labels)
    plt.ylabel('Seconds')
    plt.title('Training Time Benchmark')
    
    for bar, val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}s', ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Throughput
    plt.figure(figsize=(12, 5))
    throughput = [r['throughput'] for r in results]
    plt.plot(x, throughput, marker='o', linewidth=3)
    plt.xticks(x, labels)
    plt.ylabel('Images / sec')
    plt.title('Training Throughput')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    #GPU memory
    if DEVICE.type == 'cuda':
        plt.figure(figsize=(12, 5))
        mem = [r['gpu_memory'] for r in results]
        plt.bar(x, mem)
        plt.xticks(x, labels)
        plt.ylabel('MB')
        plt.title('Peak GPU Memory')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    results = benchmark(IMAGES, MASKS, CONFIGS, CLASSES)
    print('\nResults')
    print('-' * 40)
    for r in results:
        print(r)
    get_plot(results)
