import os
import csv
from pathlib import Path

import torch
import pickle 
from torch.utils.data import DataLoader

from dataset import ButterflyDataset, ClassifierDataset
from data_utils import data_transforms, load_data
from evaluation import evaluate, print_evaluation
from model_utils import get_feats_and_meta, get_dino_model
from classifier import train, get_scores
from types import SimpleNamespace

# Configuration         
args = SimpleNamespace()
args.root_data_dir = Path("/tsukimi/datasets/butterfly_anomaly")
args.data_file = args.root_data_dir / "butterfly_anomaly_train.csv"
args.img_dir = args.root_data_dir / "train_downsized_flat"
args.device = "cuda:0"
args.batch_size = 4
args.input_dim = 1536
args.num_classes = 2
args.cls_model_name = "transformer"
args.clf_save_dir = args.root_data_dir / f"trained_clfs_{args.cls_model_name}"
os.makedirs(args.clf_save_dir, exist_ok=True)


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


def train_and_evaluate(tr_cls_dloader, test_cls_dloader, test_features, test_labels):
    configs = ["svm","sgd","knn","gaussian","xgb"]
    csv_output = []
    score_output = []

    clf, acc, h_acc, nh_acc = train(tr_cls_dloader, test_cls_dloader, args)

    # Save model to the specified path
    model_filename = args.clf_save_dir / f"trained_{args.cls_model_name}_classifier.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)
    print(f"Saved {args.cls_model_name} classifier to {model_filename}")
    print(f"{args.cls_model_name}: Acc - {acc:.4f}, Hacc - {h_acc:.4f}, NHacc - {nh_acc:.4f}")
    
    # Get scores for the test dataset
    scores = get_scores(clf, test_features)
    eval_scores = evaluate(scores, test_labels, reversed=False)
    print_evaluation(*eval_scores)
    csv_output.append([f"DiNO Features + {args.cls_model_name}"] + list(eval_scores))
    
    # Save individual scores for analysis
    for idx, score in enumerate(scores):
        score_output.append([idx, score, test_labels[idx]])
            
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