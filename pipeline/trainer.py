import torch
import torch.optim as optim
import torch.nn as nn

from .utils import save_model_parameters

class EMNISTTrainer():
    def __init__(self, 
                 model, 
                 train_loader,
                 val_loader,
                 test_loader,
                 classes,
                 train_config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes

        self.train_config = train_config
        if self.train_config is not None:
            self.epochs = self.train_config['epochs']
            self.learning_rate = self.train_config['learning_rate']
            self.patience = self.train_config['patience']
        else:
            self.epochs = 10
            self.learning_rate = 0.001
            self.patience = 5

        self.device = next(model.parameters()).device

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        self.best_model_state = None

    def _train_epoch(self, optimizer, criterion):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total

        return epoch_loss, epoch_accuracy

    def _validate_epoch(self, criterion):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in (self.val_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = 100. * correct / total

        return epoch_loss, epoch_accuracy

    def train_pipeline(self):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        early_stopping_counter = 0

        for epoch in range(self.epochs):
            print(f'Epoch [{epoch+1:02d}/{self.epochs}]')

            train_loss, train_accuracy = self._train_epoch(
                optimizer, criterion
            )

            val_loss, val_accuracy = self._validate_epoch(
                criterion
            )

            scheduler.step(val_accuracy)

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Early Stopping check
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0

                # Save best model
                save_model_parameters(
                    self.model,
                    self.train_losses,
                    self.train_accuracies,
                    self.val_losses,
                    self.val_accuracies,
                    f'best_emnist_model_base.pth'
                )
            else:
                early_stopping_counter += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(f'LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'Best Val Acc: {self.best_val_accuracy:.2f}%, Early Stopping: {early_stopping_counter}/{self.patience}')
            print('-' * 60)

            if early_stopping_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

