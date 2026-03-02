import numpy as np

<<<<<<< HEAD
def compute_score(window):
    window = np.nan_to_num(window)

    variance = np.var(window)
    peak = np.max(window)
    drop = np.mean(window[-1])
    peak_drop = peak - drop
    micro_spikes = np.sum(np.abs(np.diff(window, axis=0)) > 0.4)

    score = (peak_drop * 0.5) + (micro_spikes * 0.3) - (variance * 0.2)

    if np.isnan(score) or np.isinf(score):
        score = 0.0

    return score
=======
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
>>>>>>> 0f2154a8dac9fc6f08a028a2909743dd3e0515e4
