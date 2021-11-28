import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.data.get_data import get_loaders
from src.models.cnn import SmallCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = get_loaders(batch_size=256, download=True)
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load("runs/best_model.pt", map_location=device))
    model.eval()
    
    ys = []
    ps = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.squeeze().long()
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu()
            ys.extend(y.tolist())
            ps.extend(pred.tolist())
            
    print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps))

if __name__ == "__main__":
    main()