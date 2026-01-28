def extract_node_facts(node_id, data, scores):
    mask_out = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask_out]

    edge_weights = data.edge_weight[mask_out]

    facts = {
        "entity_id": int(node_id),
        "risk_score": float(scores[node_id]),
        "num_transactions": int(mask_out.sum().item()),
        "unique_counterparties": int(neighbors.unique().numel()),
        "avg_neighbor_risk": float(scores[neighbors].mean()) if neighbors.numel() > 0 else 0.0,
        "max_transaction_weight": float(edge_weights.max()) if edge_weights.numel() > 0 else 0.0,
        "mean_transaction_weight": float(edge_weights.mean()) if edge_weights.numel() > 0 else 0.0,
    }

    return facts
