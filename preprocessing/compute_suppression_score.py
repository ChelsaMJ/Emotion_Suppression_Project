import numpy as np

def compute_suppression_score(sequence):

    # mean AU intensity
    mean_intensity = np.mean(sequence)

    # variability (temporal)
    variability = np.std(sequence)

    # duration factor
    duration = len(sequence)
    duration_norm = min(duration / 200, 1.0)

    # suppression formula
    suppression = (
        (1 - mean_intensity) * 0.5 +
        variability * 0.3 +
        (1 - duration_norm) * 0.2
    )

    suppression = np.clip(suppression, 0, 1)

    return float(suppression)