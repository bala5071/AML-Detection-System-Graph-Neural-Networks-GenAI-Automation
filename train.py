import torch
from model import AMLGCN as GCN
from dataPreprocessing import Dataset
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, precision_score, roc_auc_score, recall_score
from SAR.generateSAR import generate_sars

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Dataset(root='Data')
data = dataset[0]

num_pos = int(data.y.sum())
num_neg = data.y.size(0) - num_pos

raw_pos_weight = num_neg / num_pos
pos_weight = torch.tensor([min(raw_pos_weight, 20.0)]).to(device)



epochs = 100

model = GCN(
    in_channels=data.num_features,
    hidden_channels=64,
    out_channels=1
).to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

split = T.RandomNodeSplit(
    split='train_rest',
    num_val=0.1,
    num_test=0.1
)
data = split(data)
data = data.to(device)


train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

print(f"\nTrain nodes: {train_idx.size(0)}")
print(f"Val nodes: {val_idx.size(0)}")
print(f"Test nodes: {test_idx.size(0)}")

best_val_auc = 0
best_epoch = 0

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(train_idx.size(0), device=device)
    train_idx_shuffled = train_idx[perm]
    
    total_loss = 0
    batch_size = 256
    num_batches = 0

    for i in range(0, train_idx_shuffled.size(0), batch_size):
        batch = train_idx_shuffled[i:i+batch_size]
        
        optimizer.zero_grad()

        pred = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(pred[batch], data.y[batch].unsqueeze(1).float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches

    if epoch % 5 == 0 or epoch == epochs - 1:
        model.eval()
        
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_weight)
            print("NaNs in model output:", torch.isnan(pred).sum().item())

            val_pred = pred[val_idx]
            val_y = data.y[val_idx]
            
            val_scores = torch.sigmoid(val_pred).squeeze()
            val_pred_binary = (val_scores > 0.5).int()

            print("NaNs in val_scores:", torch.isnan(val_scores).sum().item())
            print("Infs in val_scores:", torch.isinf(val_scores).sum().item())

            
            val_auc = roc_auc_score(val_y.cpu().numpy(), val_scores.cpu().numpy())
            k = int(0.05 * len(val_scores))  # top 5%
            topk_idx = torch.topk(val_scores, k).indices

            val_recall = val_y[topk_idx].sum().item() / val_y.sum().item()
            val_precision = val_y[topk_idx].float().mean().item()

            train_pred = pred[train_idx]
            train_y = data.y[train_idx]
            train_scores = torch.sigmoid(train_pred).squeeze()
            train_auc = roc_auc_score(train_y.cpu().numpy(), train_scores.cpu().numpy())
            
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train AUC: {train_auc:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Recall: {val_recall:.4f} | "
                f"Val Precision: {val_precision:.4f} | "
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                }, 'best_aml_gcn_model.pt')

print("-" * 80)
print(f"\nTraining completed!")
print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

print("\nEvaluating on test set...")
checkpoint = torch.load('best_aml_gcn_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index, data.edge_weight)
    
    test_pred = pred[test_idx]
    test_y = data.y[test_idx]
    
    test_scores = torch.sigmoid(test_pred).squeeze()
    test_pred_binary = (test_scores > 0.5).int()
    
    test_auc = roc_auc_score(test_y.cpu().numpy(), test_scores.cpu().numpy())
    test_recall = recall_score(test_y.cpu().numpy(), test_pred_binary.cpu().numpy())
    test_precision = precision_score(test_y.cpu().numpy(), test_pred_binary.cpu().numpy(), zero_division=0)
    test_f1 = f1_score(test_y.cpu().numpy(), test_pred_binary.cpu().numpy())
    generate_sars(data, test_scores)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test F1: {test_f1:.4f}")

print("\nModel saved as 'best_aml_gcn_model.pt'")
