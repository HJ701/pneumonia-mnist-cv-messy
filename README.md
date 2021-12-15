# 🩺 PneumoniaMNIST: Computer Vision Exploration

![Project Banner](assets/banner.png)

## 👋 About the Project

This repository is a deep learning project developed with the motivation of "I'm not a doctor, but I can definitely try some AI" over a weekend. It performs pneumonia detection (binary classification) using low-resolution X-ray images from the MedMNIST dataset.

Don't let the "messy" in the repository name fool you; the code might be a bit scattered, but the results are serious! 🚀

## 🎯 What Does It Do?

- **Data**: Automatically downloads the PneumoniaMNIST dataset from MedMNIST
- **Model**: Uses a boutique and fast Convolutional Neural Network (CNN) named SmallCNN
- **Training**: Runs with logic for early stopping (maybe in the future) and saving the best model
- **Result**: Pneumonia or Normal? (Targeting 85%+ accuracy)

## 🛠️ Installation

Prepare your virtual environment and fasten your seatbelt.

```bash
# Clone the repository
git clone https://github.com/username/pneumonia-mnist-cv-messy.git
cd pneumonia-mnist-cv-messy

# Create a virtual environment (optional but recommended to keep your PC clean)
python -m venv .venv

# For Windows:
.venv\Scripts\activate

# For Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -e ".[dev]"
```

## 🚀 Usage

Whisper these magic words into your terminal to train the model and see the results:

### 1. Train the Model

```bash
python -m src.train.train
```

This command will download the data, train the model, and save the best model (`best_model.pt`) to the `runs/` folder. Grab your coffee; it won't take long. ☕

### 2. Evaluate Performance

```bash
python -m src.train.eval
```

This prints a report showing how smart (or confused) the model is on the test data.

## 🧠 Model Architecture

Instead of complex ResNets or Transformers, we adopted the "Simple is Beautiful" philosophy for this task:

| Layer | Detail |
|-------|--------|
| Input | 1×28×28 (Grayscale X-ray) |
| Conv1 | 16 filters, 3×3 kernel |
| Conv2 | 32 filters, 3×3 kernel |
| FC | Linear layers for classification |
| Output | 2 Classes (Normal / Pneumonia) |

## 📊 Sample Output

After training, you'll see cool tables like this in your terminal:

```
              precision    recall  f1-score   support

           0       0.92      0.88      0.90       234
           1       0.89      0.93      0.91       390

    accuracy                           0.91       624
   macro avg       0.91      0.90      0.90       624
weighted avg       0.91      0.91      0.91       624
```

**Note**: Results may vary slightly due to randomness. Don't panic.

## 📝 TODO

This project is a living organism. Here are the future plans (or things postponed due to laziness):

- [ ] Add better data augmentation techniques
- [ ] Move the logging system from `print()` to WandB or TensorBoard
- [ ] Clean up the code a bit (refactoring)
- [ ] Add a UI (Gradio/Streamlit) so our doctor friends can play with it too

## 🤝 Contributing

Pull requests are welcome! If you can make the code less "messy", you'll be our hero.

## 📄 License

© 2021 This project is open source and for educational purposes only. Please do not use it for actual medical diagnosis; trust AI, but go to a doctor. 🩺