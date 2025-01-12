import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from transformer_classifier import TransformerClassifier, MLPClassifier
from copy import deepcopy
from focal_loss import FocalLoss
from evaluation import evaluate

# Train classifier with improvements
def train(train_loader, test_loader, args):
    # Define the model
    if args.cls_model_name == "transformer":
        model = TransformerClassifier(input_dim=args.input_dim, num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    elif args.cls_model_name == "mlp":
        model = MLPClassifier(input_dim=args.input_dim, num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    else:
        raise ValueError("Invalid cls_model_name")
    
    model.to(args.device)

    class_weights = torch.tensor([0.0457,1.0])  # 每类权重 (示例)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    criterion.to(args.device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # precision_calc = Precision(task="binary", average='macro')
    best_f1 = 0.0
    best_model = deepcopy(model)

    # non_hybrid_weight = 1
    # hybrid_weight = 1
    # class_weights = {0: non_hybrid_weight, 1: hybrid_weight}

    # 训练循环
    # num_epochs = 8
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        train_preds_all = []
        train_labels_all = []
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_features, batch_labels = batch_features.to(args.device), batch_labels.to(args.device)
            # print("batch_features.shape", batch_features.shape)
            outputs = model(batch_features)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Save predictions and labels for metrics calculation
            train_preds_all.append(outputs.cpu().detach())
            train_labels_all.append(batch_labels.cpu().detach())
        
        # Calculate precision
        train_preds_all = torch.cat(train_preds_all)
        train_preds_all = torch.argmax(train_preds_all, dim=-1)
        train_labels_all = torch.cat(train_labels_all)
        train_accuracy, train_precision, train_recall, train_f1 = calc_metrics(train_preds_all, train_labels_all)
        
        # Evaluate the model
        val_preds, val_labels = validate(model, test_loader, args)
        # val_preds = torch.argmax(val_preds, dim=-1)
        
        val_accuracy, val_precision, val_recall, val_f1 = calc_metrics(torch.argmax(val_preds, dim=-1), val_labels)
        val_preds_for_hybrid = torch.softmax(val_preds, dim=-1)[:,1]
        val_hybrid_recall,val_hybrid_precision,val_hybrid_f1,val_hybrid_roc_auc,val_hybrid_acc = evaluate(val_preds_for_hybrid.numpy(), val_labels.numpy(), reversed=False)
        
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"\tTrain: Acc - {train_accuracy:.4f}, Precision - {train_precision:.4f}, Recall - {train_recall:.4f}, F1 - {train_f1:.4f}")
        print(f"\tVal: Acc - {val_accuracy:.4f}, Precision - {val_precision:.4f}, Recall - {val_recall:.4f}, F1 - {val_f1:.4f}")
        print(f"\tVal Hybrid: Recall - {val_hybrid_recall:.4f}, Precision - {val_hybrid_precision:.4f}, F1 - {val_hybrid_f1:.4f}, ROC AUC - {val_hybrid_roc_auc:.4f}, Acc - {val_hybrid_acc:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = deepcopy(model)
            torch.save(best_model, args.clf_save_dir / f"trained_classifier_epoch_{epoch+1}.pth")
            print(f"Best model updated! Saved model at epoch {epoch+1} with val f1 {val_f1:.4f}")
        
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
            # predictions = torch.argmax(outputs, dim=-1)  # Get predicted class indices
            all_predictions.append(outputs.cpu().detach())  # Move predictions to CPU and store
            all_labels.append(label_test.cpu().detach())  # Move labels to CPU and store

    # Concatenate all predictions and labels
    preds = torch.cat(all_predictions).cpu().detach()
    y_val = torch.cat(all_labels).cpu().detach()
    # print("preds_val.shape(before):", preds.shape)
    # print("y_val.shape:", y_val.shape)
    # Turn the binary labels into a numpy array (B,2) to (B,)
    # preds = torch.argmax(preds, dim=-1)
    # print("preds_val.shape(after):", preds.shape)
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