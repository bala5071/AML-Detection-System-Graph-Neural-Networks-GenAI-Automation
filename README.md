# AML Detection System: Graph Neural Networks & GenAI Automation

## Description

This project is about Anti Money Laundering (AML) detection system using Graph Neural Networks (GNNs) combined with Generative AI for explainability and regulatory reporting. Transactional data is modeled as a graph, where accounts are nodes and transactions are edges. A GNN learns risk patterns across the network to identify suspicious entities. High risk predictions are then automatically translated into human readable Suspicious Activity Reports (SARs) using Large Language Models (LLMs), bridging the gap between black box models and business or regulatory stakeholders.

This system is designed to reflect real world financial crime detection workflows used in banks and regulatory institutions.

## Features

- Graph based AML detection using Graph Convolutional Networks (GCN)
- Node level risk scoring for bank accounts
- Handles highly imbalanced financial crime datasets
- Robust evaluation with AUC, Precision, Recall, F1 Score, and confusion matrix
- Explainable AI pipeline that converts model outputs into natural language SARs
- Automated SAR generation using LLMs via LangChain
- Bank readable SAR reports stored as structured artifacts
- Modular and extensible architecture for research and production use

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- Scikit learn
- Pandas and NumPy
- Graph Neural Networks (GCN)
- LangChain
- Large Language Models (OpenAI or Gemini)
- CUDA (optional, for GPU acceleration)

## Installation

Clone the repository:

```bash
git clone https://github.com/bala5071/AML-Detection-System-Graph-Neural-Networks-GenAI-Automation.git
cd AML-Detection-System-Graph-Neural-Networks-GenAI-Automation
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux or macOS
venv\Scripts\activate      # Windows
```

Install all the required dependencies:

```bash
pip install -r requirements.txt
```
