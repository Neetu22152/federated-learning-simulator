# Federated Learning Simulator

A from-scratch simulation of Federated Averaging (FedAvg) with non-IID data, 
Byzantine attack/defense, and an interactive dashboard.

## Progress
- [x] Phase 1: Centralized MNIST baseline (99.1% accuracy)
- [x] Phase 2: FedAvg simulator — 10 clients IID (99.0% accuracy)
- [x] Phase 3: Non-IID data + Byzantine attack + median defense
- [x] Phase 4: Streamlit dashboard with live simulation
- [x] Phase 5: Research report (LaTeX)


# Federated Learning Simulator

A from-scratch implementation of **Federated Averaging (FedAvg)** with non-IID data heterogeneity, Byzantine fault simulation, and median-based defense — built to explore privacy-preserving distributed machine learning.

> Built as a research portfolio project targeting Master's programs in Software Engineering.

---

##  Motivation

Centralizing sensitive data for ML training raises serious privacy concerns — especially under regulations like the EU's **GDPR**. Federated Learning addresses this by training models across decentralized clients without raw data ever leaving the source.

This simulator explores:
- How well FedAvg converges under ideal (IID) conditions
- How **data heterogeneity** (non-IID) degrades convergence
- How a single **Byzantine attacker** can collapse the global model
- How **coordinate-wise median aggregation** defends against such attacks

---

## Key Results

| Experiment | Final Accuracy |
|---|---|
| Centralized baseline | 99.1% |
| FL — IID, no attack | 99.0% |
| FL — Non-IID (α=0.5), no attack | 98.8% |
| FL — Non-IID + Byzantine attack (no defense) | 9.8% |
| FL — Non-IID + Byzantine attack + Median defense | 98.84% |

**Finding:** A single malicious client using FedAvg (mean aggregation) reduces global accuracy to random-guess level (9.8%). Coordinate-wise median aggregation recovers performance to within 0.26% of the centralized baseline.

---

## Architecture
Server

├── Holds global model

├── Distributes weights each round

├── Aggregates client updates (FedAvg or Median)

└── Evaluates global accuracy


Clients (×10, simulated)

├── Each holds isolated local data partition

├── Trains locally for N epochs

├── Returns only weight updates (never raw data)

└── Byzantine clients return poisoned random weights

---

## Interactive Dashboard

The Streamlit dashboard lets you experiment with all parameters live:

- Number of clients, communication rounds, local epochs
- Non-IID skew level (Dirichlet α)
- Byzantine client fraction
- Aggregation defense (FedAvg vs Median)

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

federated-learning-simulator/

├── app.py                          # Streamlit dashboard

├── notebooks/

│   ├── phase1_mnist.ipynb          # Centralized baseline

│   ├── phase2_fedavg.ipynb         # FedAvg simulator

│   └── phase3_noniid_attack.ipynb  # Non-IID + attack + defense

├── models/

│   ├── centralized_baseline.png

│   ├── fedavg_iid.png

│   └── phase3_comparison.png

├── requirements.txt

└── README.md

---

## Algorithms

**FedAvg (McMahan et al., 2017)**
Each round: select clients → local SGD → average weights → update global model.

**Non-IID partitioning**
Dirichlet distribution (α) controls class imbalance per client. Lower α = more skewed.

**Byzantine attack**
Malicious clients replace honest weight updates with high-variance random noise (σ=10).

**Coordinate-wise Median defense**
Replace mean aggregation with element-wise median across client weights — outlier weights from Byzantine clients are naturally suppressed.

---

## References

- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- Blanchard et al. (2017) — [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://arxiv.org/abs/1703.02757)
- Li et al. (2020) — [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- Kairouz et al. (2021) — [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)

---

## Future Work

- Implement FedProx for better non-IID convergence
- Add differential privacy (Gaussian noise mechanism)
- Extend to CIFAR-10 for more complex tasks
- Communication efficiency: gradient compression


## Live Demo
 **[federated-learning-simulator.streamlit.app](https://federated-learning-simulator.streamlit.app)**

Adjust parameters and watch the global model learn in real time.