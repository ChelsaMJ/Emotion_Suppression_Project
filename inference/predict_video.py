from preprocessing.build_feature_sequence import extract_full_features
import torch
from models.lstm_model import SuppressionLSTM


def predict(video_path):

    features = extract_full_features(video_path)

    features = torch.tensor(features).unsqueeze(0).float()

    model = SuppressionLSTM(features.shape[2])
    model.load_state_dict(torch.load("suppression_model.pth"))
    model.eval()

    with torch.no_grad():
        score = model(features).item()

    return score