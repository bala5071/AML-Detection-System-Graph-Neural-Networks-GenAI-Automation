import torch
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from dataPreprocessing import Dataset
from model import AMLGCN
from SAR.generateSAR import generate_sars

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    dataset = Dataset(root='Data')
    data = dataset[0]

    split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.01)
    data = split(data)
    data = data.to(device)

    print("Initializing model architecture...")
    model = AMLGCN(
        in_channels=data.num_features,
        hidden_channels=64,
        out_channels=1
    ).to(device)

    print("Loading best model weights...")
    try:
        checkpoint = torch.load('best_aml_gcn_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from Epoch {checkpoint['epoch']} (Val AUC: {checkpoint['val_auc']:.4f})")
    except FileNotFoundError:
        print("Error: 'best_aml_gcn_model.pt' not found. Run train.py first.")
        return

    print("\nStarting Evaluation...")
    model.eval()
    
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    
    with torch.no_grad():

        logits = model(data.x, data.edge_index, data.edge_weight)

        probs = torch.sigmoid(logits).squeeze()
        
        test_probs = probs[test_idx]
        test_y = data.y[test_idx]

        test_pred_binary = (test_probs > 0.5).int()

        y_true = test_y.cpu().numpy()
        y_scores = test_probs.cpu().numpy()
        y_pred = test_pred_binary.cpu().numpy()

        auc = roc_auc_score(y_true, y_scores)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print("-" * 40)
        print(f"Test AUC:       {auc:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test F1 Score:  {f1:.4f}")
        print("-" * 40)
        print("Confusion Matrix:")
        print(cm)
        print("-" * 40)

        print("\nTriggering SAR Generation for High-Risk Accounts...")
        generate_sars(data, probs, node_subset=test_idx)


if __name__ == "__main__":
    evaluate()