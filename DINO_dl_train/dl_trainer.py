import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from transformers import AutoModel
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer_classifier import TransformerClassifier
# import accuracy calc class in torch
from torchmetrics import Accuracy, Precision


# Train classifier with improvements
def train(train_loader, test_loader, args):
    # Define the model
    model = TransformerClassifier(input_dim=args.input_dim, num_classes=args.num_classes)
    model.to(args.device)

    class_weights = torch.tensor([1.0, 0.1])  # 每类权重 (示例)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion.to(args.device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    precision_calc = Precision(task="binary", average='macro')
    best_precision = 0.0
    best_model = None

    non_hybrid_weight = 1
    hybrid_weight = 1
    class_weights = {0: non_hybrid_weight, 1: hybrid_weight}

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_preds_all = []
        train_labels_all = []
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_features, batch_labels = batch_features.to(args.device), batch_labels.to(args.device)
            outputs = model(batch_features)
            # print("batch_features.shape:", batch_features.shape)
            # print("outputs.shape:", outputs.shape)
            # print("batch_labels.shape:", batch_labels.shape)
            # print("outputs device:", outputs.device)
            # print("batch_labels device:", batch_labels.device)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Save predictions and labels for metrics calculation
            train_preds_all.append(outputs.cpu().detach())
            train_labels_all.append(batch_labels.cpu().detach())
        
        # Calculate precision
        train_preds_all = torch.cat(train_preds_all)
        print("train_preds_all.shape(before):", train_preds_all.shape)
        train_preds_all = torch.argmax(train_preds_all, dim=-1)
        train_labels_all = torch.cat(train_labels_all)
        train_accuracy, train_precision, train_recall, train_f1 = calc_metrics(train_preds_all, train_labels_all)
        print("train_preds_all.shape(after):", train_preds_all.shape)
        print("train_labels_all.shape:", train_labels_all.shape)
        
        # Evaluate the model
        val_preds, val_labels = validate(model, test_loader, args)
        print(val_preds)
        print(val_labels)
        
        val_accuracy, val_precision, val_recall, val_f1 = calc_metrics(val_preds, val_labels)
        
        if val_precision > best_precision:
            best_precision = val_precision
            best_model = model
            torch.save(best_model, args.clf_save_dir / f"trained_{args.cls_model_name}_classifier_epoch_{epoch}.pth")
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"\tTrain: Acc - {train_accuracy:.4f}, Precision - {train_precision:.4f}, Recall - {train_recall:.4f}, F1 - {train_f1:.4f}")
        print(f"\tVal: Acc - {val_accuracy:.4f}, Precision - {val_precision:.4f}, Recall - {val_recall:.4f}, F1 - {val_f1:.4f}")
    return best_model


# Function to evaluate the model
def validate(model, test_loader, args):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for x_test, label_test in test_loader:
            x_test, label_test = x_test.to(args.device), label_test.to(args.device)  # Move data to device
            outputs = model(x_test)  # Forward pass
            predictions = torch.argmax(outputs, dim=-1)  # Get predicted class indices
            all_predictions.append(predictions.cpu().detach())  # Move predictions to CPU and store
            all_labels.append(label_test.cpu().detach())  # Move labels to CPU and store

    # Concatenate all predictions and labels
    preds = torch.cat(all_predictions).cpu().detach()
    y_val = torch.cat(all_labels).cpu().detach()
    print("preds_val.shape(before):", preds.shape)
    print("y_val.shape:", y_val.shape)
    # Turn the binary labels into a numpy array (B,2) to (B,)
    preds = torch.argmax(preds, dim=-1)
    print("preds_val.shape(after):", preds.shape)
    return preds, y_val

def calc_metrics(preds, y_val):
    # Compute metrics
    TP = ((preds == 1) & (y_val == 1)).sum()
    TN = ((preds == 0) & (y_val == 0)).sum()
    FP = ((preds == 1) & (y_val == 0)).sum()
    FN = ((preds == 0) & (y_val == 1)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, f1
    

# def evaluate(model, test_loader):
#     # 测试模型
#     model.eval()
#     test_features = torch.randn(10, 1, input_dim)  # 随机生成10个样本
#     logits = model(test_features)
#     predictions = torch.argmax(logits, dim=-1)
#     print("Predictions:", predictions)

#     # if classifier_config == "svm":
#     #     clf = make_pipeline(StandardScaler(), SVC(gamma='scale', C=1, class_weight='balanced', probability=True))
#     # elif classifier_config == "sgd":
#     #     base_clf = SGDClassifier(
#     #         loss="log_loss",
#     #         alpha=0.001,
#     #         penalty="l2",
#     #         eta0=0.001,
#     #         n_iter_no_change=100,
#     #         learning_rate='adaptive',
#     #         max_iter=1000,
#     #         class_weight='balanced'
#     #     )
#     #     clf = CalibratedClassifierCV(base_clf)  # Calibrate SGD to get probability estimates
#     # elif classifier_config == "knn":
#     #     clf = KNeighborsClassifier(n_neighbors=5)  # Increased k value for better generalization
#     # elif classifier_config == "gaussian":
#     #     clf = GaussianProcessClassifier(random_state=0)
#     # elif classifier_config == "xgb":
#     #     clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=hybrid_weight/non_hybrid_weight)
#     # else:
#     #     raise ValueError("Invalid classifier_config")

#     # # Train the classifier on the training set
#     # clf.fit(X_train, y_train)

#     # # Evaluate on the validation set
#     # preds = clf.predict(X_val)
    
#     correct = preds == y_val

#     hybrid_correct = correct[y_val == 1].sum()
#     non_hybrid_correct = correct[y_val == 0].sum()

#     acc = clf.score(X_val, y_val)
#     h_acc = hybrid_correct / (y_val == 1).sum() if (y_val == 1).sum() > 0 else 0 # precision
#     nh_acc = non_hybrid_correct / (y_val == 0).sum() if (y_val == 0).sum() > 0 else 0 # recall

#     return clf, acc, h_acc, nh_acc

# Get prediction scores (probability estimates)
def get_scores(clf, X):
    return clf.predict_proba(X)[:, 1]