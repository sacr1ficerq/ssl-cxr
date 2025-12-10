import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from .model import SimCLR, VICReg, SSLModelWithProbe
from .dataset import SSLDataset


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


device = 'cuda'


def compute_class_weights(train_loader, num_classes):
    class_counts = torch.zeros(num_classes)

    for batch in train_loader:
        # handle both SSL (view1, view2, labels) and SFT (images, labels) formats
        if len(batch) == 3:
            labels = batch[2]
        else:
            labels = batch[1]

        for label in labels:
            class_counts[label] += 1

    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    return class_weights


def evaluate_probe_with_auc(encoder, probe, loader, device):
    encoder.eval()
    probe.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # handle both formats
            if len(batch) == 3:
                images, labels = batch[0], batch[2]
            else:
                images, labels = batch[0], batch[1]

            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            logits = probe(features)
            probs = torch.softmax(logits, dim=1)[:, 1]  #P(positive class)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    roc_auc = roc_auc_score(all_labels, all_probs)
    return roc_auc * 100


def evaluate_probe(encoder, probe, loader, device):
    encoder.eval()
    probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, labels = batch[0], batch[2]
            else:
                images, labels = batch[0], batch[1]

            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            logits = probe(features)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def evaluate_classifier(encoder, classifier, loader, device):
    """Evaluate classifier with ROC-AUC metric."""
    encoder.eval()
    classifier.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            features = encoder(imgs)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    roc_auc = roc_auc_score(all_labels, all_probs)
    return roc_auc * 100


def pretrain_ssl(model,
                 train_loader,
                 val_loader=None,
                 epochs=100,
                 lr=0.3,
                 log_dir='runs/ssl_pretraining',
                 device='cuda'):
    writer = SummaryWriter(log_dir=log_dir)

    model = model.to(device)
    optimizer = LARS(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        warmup_epochs=10,
        max_epochs=epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (view1, view2, label) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/SSL_batch', loss.item(), global_step)

        scheduler.step(epoch)

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/SSL_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_loss = evaluate_ssl_loss(model, val_loader, device)
            writer.add_scalar('Loss/SSL_val', val_loss, epoch)
            print(f"  Validation Loss: {val_loss:.4f}")

    writer.close()
    return model


def evaluate_ssl_loss(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for view1, view2, _ in loader:
            view1, view2 = view1.to(device), view2.to(device)
            loss = model(view1, view2)
            total_loss += loss.item()

    return total_loss / len(loader)



def pretrain_ssl_with_online_probe(model,
                                   train_loader,
                                   val_loader,
                                   num_classes,
                                   epochs=50,
                                   lr=0.3,
                                   device='cuda',
                                   log_dir='runs/ssl_training'):
    writer = SummaryWriter(log_dir=log_dir)

    ssl_model = model.to(device)
    probe_model = SSLModelWithProbe(ssl_model, num_classes).to(device)

    ssl_params = list(ssl_model.encoder.parameters()) + list(ssl_model.projection_head.parameters())
    optimizer = LARS(ssl_params, lr=lr, momentum=0.9, weight_decay=1e-6)
    probe_optimizer = optim.Adam(probe_model.linear_probe.parameters(), lr=1e-3)

    scheduler = LinearWarmupCosineAnnealing(optimizer, warmup_epochs=10, max_epochs=epochs)

    class_weights = compute_class_weights(train_loader, num_classes).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Class weights: {class_weights.cpu().numpy()}")

    for epoch in range(epochs):
        ssl_model.train()
        probe_model.linear_probe.train()
        total_ssl_loss = 0
        total_cls_loss = 0
        correct = 0
        total = 0

        for batch_idx, (view1, view2, labels) in enumerate(train_loader):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

            optimizer.zero_grad()
            ssl_loss = ssl_model(view1, view2)
            ssl_loss.backward()
            optimizer.step()

            probe_optimizer.zero_grad()
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

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/SSL_batch', ssl_loss.item(), global_step)
            writer.add_scalar('Loss/Probe_batch', cls_loss.item(), global_step)

        scheduler.step(epoch)

        avg_ssl_loss = total_ssl_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        train_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/SSL_epoch', avg_ssl_loss, epoch)
        writer.add_scalar('Loss/Probe_epoch', avg_cls_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        print(f"Epoch {epoch+1}/{epochs} | SSL Loss: {avg_ssl_loss:.4f} | "
              f"Probe Loss: {avg_cls_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {current_lr:.6f}")

        if (epoch + 1) % 5 == 0:
            val_auc = evaluate_probe_with_auc(ssl_model.encoder, probe_model.linear_probe, val_loader, device)
            val_acc = evaluate_probe(ssl_model.encoder, probe_model.linear_probe, val_loader, device)

            writer.add_scalar('ROC-AUC/Validation', val_auc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)

            print(f"  -> Val ROC-AUC: {val_auc:.2f}% | Val Acc: {val_acc:.2f}%")

    writer.close()
    return ssl_model


def offline_linear_probe(encoder,
                        train_loader,
                        test_loader,
                        num_classes,
                        epochs=100,
                        device='cuda',
                        log_dir='runs/linear_probe'):
    writer = SummaryWriter(log_dir=log_dir)

    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    class_weights = compute_class_weights(train_loader, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Class weights: {class_weights.cpu().numpy()}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = encoder(imgs)

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_batch', loss.item(), global_step)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        writer.add_scalar('Loss/Train_epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        if (epoch + 1) % 10 == 0:
            test_auc = evaluate_classifier(encoder, classifier, test_loader, device)
            test_acc = evaluate_probe(encoder, classifier, test_loader, device)

            writer.add_scalar('ROC-AUC/Test', test_auc, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test ROC-AUC: {test_auc:.2f}% | Test Acc: {test_acc:.2f}%")

    final_auc = evaluate_classifier(encoder, classifier, test_loader, device)
    final_acc = evaluate_probe(encoder, classifier, test_loader, device)

    writer.add_hparams(
        {'num_classes': num_classes, 'epochs': epochs},
        {'final_auc': final_auc, 'final_acc': final_acc}
    )

    writer.close()
    print(f"\nFinal Test ROC-AUC: {final_auc:.2f}% | Final Test Acc: {final_acc:.2f}%")

    return final_auc


def linear_probe(encoder,
                train_loader,
                test_loader,
                num_classes=10,
                epochs=100,
                device='cuda',
                log_dir='runs/linear_probe_simple'):
    """
    Simple linear probe (legacy function, updated with ROC-AUC).
    """
    writer = SummaryWriter(log_dir=log_dir)

    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    class_weights = compute_class_weights(train_loader, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                features = encoder(imgs)

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        if (epoch + 1) % 10 == 0:
            auc = evaluate_classifier(encoder, classifier, test_loader, device)
            acc = evaluate_probe(encoder, classifier, test_loader, device)

            writer.add_scalar('ROC-AUC/Test', auc, epoch)
            writer.add_scalar('Accuracy/Test', acc, epoch)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Test ROC-AUC: {auc:.2f}% | Test Acc: {acc:.2f}%")

    writer.close()
