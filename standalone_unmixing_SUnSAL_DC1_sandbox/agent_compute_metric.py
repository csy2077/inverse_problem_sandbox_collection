

# --- Extracted Dependencies ---

log = logging.getLogger(__name__)

def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
    """Return individual and global metric."""
    d = {}
    d["Overall"] = round(float(metric(X_hat, X_gt)), 4)
    if detail:
        for ii, label in enumerate(labels):
            if on_endmembers:
                x_gt, x_hat = X_gt[:, ii][:, None], X_hat[:, ii][:, None]
                d[label] = round(float(metric(x_hat, x_gt)), 4)
            else:
                d[label] = round(float(metric(X_hat[ii], X_gt[ii])), 4)
    log.info(f"{metric} => {d}")
    return d
