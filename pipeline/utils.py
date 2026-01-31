import torch

def save_model_parameters(
  model, train_losses, train_accuracies, val_losses, val_accuracies, filepath
):
  """
  Save model with metadata
  """
  torch.save({
    'model_state_dict': model.state_dict(),
    'training_history': {
      'train_losses': train_losses,
      'train_accuracies': train_accuracies,
      'val_losses': val_losses,
      'val_accuracies': val_accuracies,
    }
  }, filepath)

  print(f"Model saved to {filepath}")