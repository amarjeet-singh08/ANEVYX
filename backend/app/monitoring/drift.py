import numpy as np

def calculate_wasserstein_1d(u_values, v_values):
    """
    Computes 1D Wasserstein distance.
    Equivalent to the area between the empirical CDFs of the two distributions.
    """
    if len(u_values) == 0 or len(v_values) == 0:
        return 0.0
    
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    
    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')
    
    deltas = np.diff(all_values)
    
    u_cdf = np.searchsorted(u_values[u_sorter], all_values[:-1], side='right') / len(u_values)
    v_cdf = np.searchsorted(v_values[v_sorter], all_values[:-1], side='right') / len(v_values)
    
    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    Compares two 1D distributions to detect drift.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    breakpoints = np.linspace(min(expected), max(expected), buckets + 1)
    
    # Calculate frequencies
    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero and log of zero
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

def detect_drift(baseline_scores: np.ndarray, recent_scores: np.ndarray) -> dict:
    """
    Calculate drift using PSI and 1D Wasserstein distance.
    Returns status: Normal, Watch, Drift Detected.
    """
    psi = calculate_psi(baseline_scores, recent_scores)
    w_dist = calculate_wasserstein_1d(baseline_scores, recent_scores)
    
    if psi > 0.2:
        status = "Drift Detected"
    elif psi > 0.1:
        status = "Watch"
    else:
        status = "Normal"
        
    return {
        "status": status,
        "psi": psi,
        "wasserstein_distance": float(w_dist)
    }

