import os
import torch
from Features.featureExtraction import extract_node_facts
from Features.case import build_case_summary
from SAR.llm import build_sar_chain

def generate_sars(
    data,
    scores,
    node_subset=None,
    threshold=0.995,
):

    TOP_N = 5
    chain = build_sar_chain()

    if node_subset is not None:
        scores_subset = scores[node_subset]
        cutoff = torch.quantile(scores_subset, threshold)
        flagged_nodes = node_subset[scores_subset >= cutoff]
    else:
        cutoff = torch.quantile(scores, threshold)
        flagged_nodes = torch.where(scores >= cutoff)[0]

    sar_results = []

    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/SAR_Report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SUSPICIOUS ACTIVITY REPORT (SAR)\n")
        f.write("=" * 70 + "\n\n")

        for i, node_id in enumerate(flagged_nodes.tolist(), start=1):
            facts = extract_node_facts(node_id, data, scores)
            summary = build_case_summary(facts)

            sar_text = chain.invoke({"case_summary": summary})

            sar_results.append({
                "entity_id": facts["entity_id"],
                "risk_score": round(float(facts["risk_score"]), 4),
                "sar": sar_text
            })

            f.write(f"SAR CASE #{i}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Subject Entity ID: {facts['entity_id']}\n")
            f.write(f"Model Risk Score: {round(float(facts['risk_score']), 4)}\n\n")
            f.write("Narrative Summary:\n")
            f.write(sar_text.strip() + "\n\n")
            f.write("Model Attribution:\n")
            f.write(
                "The subject entity was identified through a graph-based "
                "risk detection model leveraging transactional relationships, "
                "network exposure, and anomalous behavior patterns.\n\n"
            )
            f.write("=" * 70 + "\n\n")

    print(f"SAR report generated at: {report_path}")
    return sar_results
