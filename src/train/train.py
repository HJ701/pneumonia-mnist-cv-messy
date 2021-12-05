import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.data.get_data import get_loaders
from src.models.cnn import SmallCNN
from src.utils.stuff import seed_everything, ensure_dir

def run_one_epoch(model, loader, opt=None, device="cpu"):
    model.train() if opt else model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    all_y = []
    all_p = []

    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device) # medmnist labels are shape (N,1)
        if opt:
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            if opt:
                loss.backward()
                opt.step()
            losses.append(loss.item())
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_p.extend(list(pred))
            all_y.extend(list(y.detach().cpu().numpy()))
    
    acc = accuracy_score(all_y, all_p)
    return sum(losses)/len(losses), acc

def main():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = get_loaders(batch_size=128, download=True)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    ensure_dir("runs")
    best = 0.0
    
    for epoch in range(1, 9): # lol arbitrary
        tr_loss, tr_acc = run_one_epoch(model, train_loader, opt=opt, device=device)
        va_loss, va_acc = run_one_epoch(model, val_loader, opt=None, device=device)
        print("epoch", epoch, "train", tr_loss, tr_acc, "val", va_loss, va_acc)
        
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), "runs/best_model.pt")
    
    with open("runs/notes.txt", "w", encoding="utf-8") as f:
        f.write(f"best_val_acc={best}\n")
    
    # sloppy plot: just fake history by re-running prints? (ugh)
    # TODO: actually store history properly
    plt.figure()
    plt.title("Training done (no history yet)")
    plt.plot([0, 1], [0, best])
    plt.savefig("runs/plot.png")
    plt.close()

if __name__ == "__main__":
    main()