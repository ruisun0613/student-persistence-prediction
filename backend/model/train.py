from backend.data.data_loader import load_students_data
from backend.data.preprocessing import clean_data, preprocessor, TARGET_COLUMN
from backend.model.network import PersistenceNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
from pathlib import Path
import joblib
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Fix random seeds (important for reproducibility)
RANDOM_SEED = 42

torch.manual_seed(42)
np.random.seed(42)

# Save model address
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents = True, exist_ok = True)
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"

def prepare_data():
    # Load data
    raw_df = load_students_data()
    # Clean data
    df = clean_data(raw_df)


    # Split the pre-train data
    X = df.drop(columns = TARGET_COLUMN)
    """
    Original:
    Persistence = 1 -> persisted
    Persistence = 0 -> did not persist

    Model target:
    AtRisk = 1 -> did not persist
    AtRisk = 0 -> persisted
    """
    y = 1 - df[TARGET_COLUMN]
    y.name = "AtRisk"

    # print("At-risk distribution:")
    # print(y.value_counts())

    # Train/Validation split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size = 0.3, random_state = RANDOM_SEED, stratify = y
    )

    # Validation/Test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,test_size = 0.5, random_state = RANDOM_SEED, stratify = y_temp
    )

    # Calculate class weight using training data only
    positive_count = (y_train == 1).sum()
    negative_count = (y_train == 0).sum()

    pos_weight = negative_count / positive_count
    # print(f"Positive (AtRisk=1): {positive_count}")
    # print(f"Negative (AtRisk=0): {negative_count}")
    # print(f"Positive weight: {pos_weight:.4f}")

    # test
    # print("Train:", X_train.shape, y_train.shape)
    # print("Validation:", X_val.shape, y_val.shape)
    # print("Test:", X_test.shape, y_test.shape)

    # fit preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # print("Processed Train:", X_train_processed.shape)
    # print("Processed Validation:", X_val_processed.shape)
    # print("Processed Test:", X_test_processed.shape)
    # print(type(X_train_processed))
    
    def to_tensor(x):
        return torch.tensor(x, dtype = torch.float32)

    X_train_tensor = to_tensor(X_train_processed)
    X_val_tensor = to_tensor(X_val_processed)
    X_test_tensor = to_tensor(X_test_processed)

    y_train_tensor = to_tensor(y_train.values).view(-1, 1)
    y_val_tensor = to_tensor(y_val.values).view(-1, 1)
    y_test_tensor = to_tensor(y_test.values).view(-1, 1)

    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True
    )

    # for batch_X, batch_y in train_loader:
    #     print(batch_X.shape)
    #     print(batch_y.shape)

    input_dim = X_train_tensor.shape[1]

    return {
        "train_loader": train_loader,
        "X_val": X_val_tensor,
        "y_val": y_val_tensor,
        "X_test": X_test_tensor,
        "y_test": y_test_tensor,
        "input_dim": input_dim,
        "pos_weight": pos_weight,
    }

# Train model
def train_model():
    data = prepare_data()

    model = PersistenceNN(input_dim = data["input_dim"])
    # print(model)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight = torch.tensor([data["pos_weight"]], dtype = torch.float32)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.001,
        weight_decay = 1e-5
    )
    # Training parameters
    best_loss = float("inf")
    EPOCHS = 50
    patience = 0
    patience_limit = 8
    best_model_state = None

    for epoch in range(EPOCHS):
        # ========= Training =========
        model.train()
        train_losses = []

        for batch_X, batch_y in data["train_loader"]:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ========= Validation =========
        model.eval()

        with torch.no_grad():
            val_logits = model(data["X_val"])
            val_loss = loss_fn(val_logits, data["y_val"])
            val_loss_value = val_loss.item()        # Pytorch Tensor -> Python float
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss = {np.mean(train_losses):.4f} | "
            f"Val Loss = {val_loss_value:.4f}"
        )

        # Early stopping
        if val_loss_value < best_loss - 1e-4:
            best_loss = val_loss_value
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping triggered!")
                break
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(data["X_test"])
        test_prob = torch.sigmoid(test_logits)
        test_pred = (test_prob >= 0.5).float()
    accuracy = (test_pred.eq(data["y_test"]).sum() / len(data["y_test"])).item()

    print(f"\nTrue Test Accuracy = {accuracy:.4f}")

    # confusion matrix
    y_true = data["y_test"].numpy().flatten()
    y_pred = test_pred.numpy().flatten()

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # classification report
    cr = classification_report(y_true, y_pred)
    print(cr)

    # Metrics
    metrics = {
        "accuracy" : accuracy,
        "at_risk_precision" : precision_score(y_true, y_pred),
        "at_risk_recall" : recall_score(y_true, y_pred),
        "at_risk_f1" : f1_score(y_true, y_pred),
        "confusion_matrix" : cm.tolist(),
        "pos_weight" : float(data["pos_weight"])
    }

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(metrics, METRICS_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
if __name__ == '__main__':
    train_model()