import os
import csv
import argparse
from pathlib import Path
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ButterflyDataset, ClassifierDataset
from data_utils import data_transforms, load_data
from evaluation import evaluate, print_evaluation
from model_utils import get_feats_and_meta, get_dino_model
from dl_trainer import train, validate, calc_metrics
import yaml

# Configuration         
def parse_args():
    parser = argparse.ArgumentParser(description="Butterfly Anomaly Training Configuration")

    parser.add_argument("--root_data_dir", type=Path, default=Path("/tsukimi/datasets/butterfly_anomaly"), help="Root directory for dataset")
    parser.add_argument("--data_file", type=Path, help="Path to the data CSV file")
    parser.add_argument("--img_dir", type=Path, help="Directory for images")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--input_dim", type=int, default=1536, help="Input dimension size")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer dimension size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cls_model_name", type=str, choices=["mlp", "transformer"], default="transformer", help="Classifier model name")
    parser.add_argument("--clf_save_dir", type=Path, help="Directory to save trained classifiers")

    args = parser.parse_args()

    # Dynamically set derived arguments
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if args.data_file is None:
        args.data_file = args.root_data_dir / "butterfly_anomaly_train.csv"
    if args.img_dir is None:
        args.img_dir = args.root_data_dir / "train_downsized_flat"
    if args.clf_save_dir is None:
        args.clf_save_dir = args.root_data_dir / f"trained_clfs_{args.cls_model_name}_{time_str}"

    return args

args = parse_args()
os.makedirs(args.clf_save_dir, exist_ok=True)
# Save the configuration into a yaml file
config_filename = args.clf_save_dir / "config.yaml"
with open(config_filename, 'w') as config_file:
    yaml.dump(vars(args), config_file, default_flow_style=False)

# Make sure the experiment is reproducible
torch.manual_seed(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(233)


def setup_data_and_model():
    # Load Data
    train_data, test_data = load_data(args.data_file, args.img_dir)

    # Model setup
    model = get_dino_model()
    return model.to(args.device), train_data, test_data

def prepare_data_loaders(train_data, test_data):
    train_sig_dset = ButterflyDataset(train_data, args.img_dir, transforms=data_transforms())
    tr_sig_dloader = DataLoader(train_sig_dset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_dset = ButterflyDataset(test_data, args.img_dir, transforms=data_transforms())
    test_dl = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    return tr_sig_dloader, test_dl

def prepare_classifier_data_loaders(tr_features, tr_labels, test_features, test_labels):
    tr_cls_dataset = ClassifierDataset(tr_features, tr_labels)
    test_cls_dataset = ClassifierDataset(test_features, test_labels)
    tr_cls_dloader = DataLoader(tr_cls_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_cls_dloader = DataLoader(test_cls_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return tr_cls_dloader, test_cls_dloader
    
def extract_features(tr_sig_dloader, test_dl, model):
    tr_features, tr_labels = get_feats_and_meta(tr_sig_dloader, model, args.device)
    test_features, test_labels = get_feats_and_meta(test_dl, model, args.device)
    return tr_features, tr_labels, test_features, test_labels


def train_and_evaluate(tr_cls_dloader, val_cls_dloader, test_features, test_labels):
    # configs = ["svm","sgd","knn","gaussian","xgb"]
    csv_output = []
    score_output = []

    model = train(tr_cls_dloader, val_cls_dloader, args)
    preds, labels = validate(model, val_cls_dloader, args)
    accuracy, precision, recall, f1 = calc_metrics(torch.argmax(preds, dim=-1), labels)

    # Save model to the specified path
    model_filename = args.clf_save_dir / f"best_classifier.pth"
    torch.save(model, model_filename)
    
    print(f"Saved {args.cls_model_name} classifier to {model_filename}")
    print(f"{args.cls_model_name}: Acc - {accuracy:.4f}, Hacc - {precision:.4f}, NHacc - {recall:.4f}, F1 - {f1:.4f}")
    
    # Get scores for the test dataset
    preds = torch.softmax(preds, dim=-1)[:,1]
    eval_scores = evaluate(preds.numpy(), labels.numpy(), reversed=False)
    print_evaluation(*eval_scores)
    csv_output.append([f"DiNO Features + {args.cls_model_name}"] + list(eval_scores))
    
    # Save individual scores for analysis
    for idx, pred in enumerate(preds):
        score_output.append([idx, pred, test_labels[idx]])
            
    return csv_output, score_output


def main():
    model, train_data, test_data = setup_data_and_model()
    tr_sig_dloader, test_dl = prepare_data_loaders(train_data, test_data)
    tr_features, tr_labels, test_features, test_labels = extract_features(tr_sig_dloader, test_dl, model)
    tr_cls_dloader, test_cls_dloader = prepare_classifier_data_loaders(tr_features, tr_labels, test_features, test_labels)
    csv_output, score_output = train_and_evaluate(tr_cls_dloader, test_cls_dloader, test_features, test_labels)
    
    # Save evaluation results
    csv_filename = args.clf_save_dir / "classifier_evaluation_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Configuration", "AUC", "Precision", "Recall", "F1-score"])
        writer.writerows(csv_output)
    
    # Save individual scores
    scores_filename = args.clf_save_dir / "classifier_scores.csv"
    with open(scores_filename, mode='w', newline='') as score_file:
        score_writer = csv.writer(score_file)
        score_writer.writerow(["Index", "Score", "True Label"])
        score_writer.writerows(score_output)
    
if __name__ == "__main__":
    main()