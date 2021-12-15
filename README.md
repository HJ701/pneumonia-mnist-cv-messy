🫁 PneumoniaMNIST: Computer Vision Exploration

👋 Hello World! (About the Project)

This repo is a deep learning project coded with the motivation of "I'm not a doctor, but I can definitely try some AI" over a weekend. It performs Pneumonia Detection (Binary Classification) using low-resolution X-Ray images from the MedMNIST dataset.

Don't let the messy in the repo name fool you; the code might be a bit scattered, but the results are serious! 🚀

🎯 What Does It Do?

Data: Automatically downloads the PneumoniaMNIST dataset from MedMNIST.

Model: Uses a boutique and fast Convolutional Neural Network (CNN) named SmallCNN.

Training: Runs with logic for early stopping (maybe in the future) and saving the best model.

Result: Pneumonia or Normal? (Targeting 85%+ accuracy).

🛠️ Installation

Prepare your virtual environments and fasten your seatbelts.

# Clone the repo
git clone [https://github.com/username/pneumonia-mnist-cv-messy.git](https://github.com/username/pneumonia-mnist-cv-messy.git)
cd pneumonia-mnist-cv-messy

# Create a virtual environment (Optional but recommended, keep your PC clean)
python -m venv .venv
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -e ".[dev]"


🚀 Usage

Whisper these magic words into your terminal to train the model and see the results:

1. Train the Model

python -m src.train.train


This command will download the data, train the model, and save the best model (best_model.pt) to the runs/ folder. Grab your coffee, it won't take long. ☕

2. Evaluate Performance

python -m src.train.eval


Prints a report showing how smart (or confused) the model is on the test data.

🧠 Model Architecture (The Brain)

Instead of complex ResNets or Transformers, we adopted the Simple is Beautiful philosophy for this task:

Layer

Detail

Input

1x28x28 (Grayscale X-Ray)

Conv1

16 filters, 3x3 kernel

Conv2

32 filters, 3x3 kernel

FC

Linear layers for classification

Output

2 Classes (Normal / Pneumonia)

📊 Sample Output (Results)

After training, you'll see cool tables like this in your terminal:

              precision    recall  f1-score   support

           0       0.92      0.88      0.90       234
           1       0.89      0.93      0.91       390

    accuracy                           0.91       624
   macro avg       0.91      0.90      0.90       624
weighted avg       0.91      0.91      0.91       624


(Note: Results may vary slightly due to randomness, don't panic.)

📝 TODO (The "Messy" Part)

This project is a living organism. Here are the future plans (or things postponed due to laziness):

[ ] Add better Data Augmentation techniques.

[ ] Move the logging system from print() to WandB or Tensorboard.

[ ] Clean up the code a bit (Refactoring).

[ ] Add a UI (Gradio/Streamlit) so our doctor friends can play with it too.

🤝 Contributing

Pull requests are welcome! If you can make the code less "messy", you'll be our hero.

2021 © This project is open source and for educational purposes only. Please do not use it for actual medical diagnosis; trust AI, but go to a doctor. 🩺