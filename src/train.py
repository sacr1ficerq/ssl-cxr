import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

from .model import SimCLR, VICReg, SSLModelWithProbe

device = 'cuda'


class LARS(optim.Optimizer):
    def __init__(self, params, lr=1.0, momentum=0.9, weight_decay=1e-6, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)

    # this is written by LLM because I dont have time
    # to figure out how this optimizer works and writing it myself
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)

                if param_norm != 0 and grad_norm != 0:
                    local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm)
                else:
                    local_lr = 1.0

                actual_lr = lr * local_lr

                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(p.grad.data)
                    p.data.add_(buf, alpha=-actual_lr)
                else:
                    p.data.add_(p.grad.data, alpha=-actual_lr)

        return loss


class LinearWarmupCosineAnnealing:
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
                  for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                  for base_lr in self.base_lrs]

        for param_group, learning_rate in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = learning_rate


def get_ssl_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])


class SSLDataset:
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


def pretrain_ssl(model, train_loader, epochs=100, lr=0.3):
    model = model.to(device)
    optimizer = LARS(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = LinearWarmupCosineAnnealing(optimizer, warmup_epochs=10, max_epochs=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for view1, view2 in train_loader:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step(epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model


def linear_probe(encoder, train_loader, test_loader, epochs=100):
    classifier = nn.Linear(512, 10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        classifier.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                features = encoder(imgs)
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    features = encoder(imgs)
                    logits = classifier(features)
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")


def pretrain_ssl_with_online_probe(model, train_loader, val_loader, num_classes,
                                   epochs=50, lr=0.3, device='cuda'):
    ssl_model = model.to(device)
    probe_model = SSLModelWithProbe(ssl_model, num_classes).to(device)

    ssl_params = list(ssl_model.encoder.parameters()) + list(ssl_model.projection_head.parameters())
    optimizer = LARS(ssl_params, lr=lr, momentum=0.9, weight_decay=1e-6)
    probe_optimizer = optim.Adam(probe_model.linear_probe.parameters(), lr=1e-3)

    scheduler = LinearWarmupCosineAnnealing(optimizer, warmup_epochs=10, max_epochs=epochs)
    cls_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        ssl_model.train()
        probe_model.linear_probe.train()

        total_ssl_loss = 0
        total_cls_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            probe_optimizer.zero_grad()

            ssl_loss = ssl_model(view1, view2)
            ssl_loss.backward()
            optimizer.step()

            with torch.no_grad():
                h1 = ssl_model.encoder(view1)

            logits = probe_model.linear_probe(h1.detach())
            cls_loss = cls_criterion(logits, labels)
            cls_loss.backward()
            probe_optimizer.step()

            total_ssl_loss += ssl_loss.item()
            total_cls_loss += cls_loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step(epoch)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | SSL Loss: {total_ssl_loss/len(train_loader):.4f} | "
              f"Probe Loss: {total_cls_loss/len(train_loader):.4f} | Probe Acc: {train_acc:.2f}%")

        if (epoch + 1) % 5 == 0:
            val_acc = evaluate_probe(ssl_model.encoder, probe_model.linear_probe, val_loader, device)
            print(f"Val Accuracy: {val_acc:.2f}%")

    return ssl_model

def evaluate_probe(encoder, probe, loader, device):
    encoder.eval()
    probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                images = batch['view1'].to(device)
                labels = batch['labels'].to(device)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

            features = encoder(images)
            logits = probe(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def offline_linear_probe(encoder, train_loader, test_loader, num_classes, epochs=100, device='cuda'):
    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                imgs = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = encoder(imgs)

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            acc = evaluate_classifier(encoder, classifier, test_loader, device)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    final_acc = evaluate_classifier(encoder, classifier, test_loader, device)
    return final_acc


def evaluate_classifier(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                imgs = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

            features = encoder(imgs)
            logits = classifier(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == '__main__':
    base_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    ssl_dataset = SSLDataset(base_dataset, get_ssl_transforms())
    ssl_loader = DataLoader(ssl_dataset, batch_size=256, shuffle=True, num_workers=4)

    model = SimCLR()
    model = pretrain_ssl(model, ssl_loader, epochs=100)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    linear_probe(model.encoder, train_loader, test_loader, epochs=100)
