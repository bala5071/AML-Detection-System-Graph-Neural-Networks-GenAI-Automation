def build_case_summary(facts):
    return f"""
Account {facts['entity_id']} was flagged by the transaction monitoring system with a risk score of {facts['risk_score']:.3f}.

The account conducted {facts['num_transactions']} transactions involving {facts['unique_counterparties']} unique counterparties.

Connected entities exhibited an average risk score of {facts['avg_neighbor_risk']:.3f}.

Transaction activity shows elevated values with a maximum normalized transaction weight of {facts['max_transaction_weight']:.3f}.
"""
