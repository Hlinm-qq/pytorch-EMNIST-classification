import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def load_model(filepath, model):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    training_history = checkpoint['training_history']
    train_losses = training_history['train_losses']
    train_accuracies = training_history['train_accuracies']
    val_losses = training_history['val_losses']
    val_accuracies = training_history['val_accuracies']

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluation(trainer, 
               test_loader,
               classes):
    """
    Perform model evaluation
    """
    model = trainer.model
    device = trainer.device
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=[str(c) for c in classes]))
    
    return all_predictions, all_targets, all_probabilities

def plot_training_history(train_losses, 
                          train_accuracies, 
                          val_losses, 
                          val_accuracies, 
                          save_path='emnist_training_history.png'
                          ):
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    axes = axes.flatten()

    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title("Train & Val Losses")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accuracies, label='Training Accuracy')
    axes[1].plot(val_accuracies, label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title("Train & Val Accuracies")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"training history saved successfully to: {save_path}")

    # Close the figure to free memory
    plt.close(fig)

def get_confusion_matrix_subset(full_mapping, desired_labels_string=''):
 
    subset_indices = []
    subset_labels = []
    
    # Convert the desired labels string into a set for fast lookup
    desired_set = set(desired_labels_string)
    
    # Iterate through the full mapping
    for index, label in enumerate(full_mapping):
        if desired_set:
            if label in desired_set:
                subset_indices.append(index)
                subset_labels.append(label)
        else:
            subset_indices.append(index)
            subset_labels.append(label)
            
    subset_indices.sort()
    subset_labels.sort()
    return subset_indices, subset_labels

def predict_one(model, img_tensor, device, classes):

    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_label = classes[predicted_idx.item()]
        confidence_score = confidence.item()
        
    return predicted_label, confidence_score, probabilities.squeeze().cpu().numpy()

def inference(model, test_loader, device, classes, 
              save_path='emnist_inference.png',
              n_samples=5):
    test_dataset = test_loader.dataset

    fig_size_x = min(max(8, n_samples+8), 40)
    fig_size_y = min(max(4, int(n_samples/2)+5), 20)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(fig_size_x, fig_size_y))

    for i in range(n_samples):
        idx = np.random.randint(len(test_dataset))
        image, true_label = test_dataset[idx]
        true_label = classes[true_label]

        # inference
        predicted_label, confidence, all_probs = predict_one(model, image, device, classes)

        axes[0, i].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}", fontsize=20)
        axes[0, i].axis('off')

        axes[1, i].bar(range(len(all_probs)), all_probs)
        axes[1, i].set_title("Label Probs")
        axes[1, i].set_xlabel("Label")
        axes[1, i].set_ylabel("Probability")
        axes[1, i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"inference saved successfully to: {save_path}")

    # Close the figure to free memory
    plt.close(fig)