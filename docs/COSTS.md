# Cost Tracking

Total budget: **$100**

Every compute expenditure is logged here with receipts/timestamps.

---

## Summary

| Category | Budget | Spent | Remaining |
|---|---|---|---|
| Data storage | $0 | $0 | $0 |
| Tokenizer training | $0 | $0 | $0 |
| Hyperparameter search | $0 | $0 | $0 |
| Architecture ablations | $10 | $0 | $10 |
| Final training | $80 | $0 | $80 |
| Evaluation | $5 | $0 | $5 |
| Buffer | $5 | $0 | $5 |
| **Total** | **$100** | **$0** | **$100** |

---

## Detailed Log

| Date | Item | Provider | GPU | Hours | Cost | Cumulative | Notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | $0 | Project started |

---

## Cost Optimization Strategies

1. **Free-tier first:** Use Google Colab (free T4) for all ablations and debugging.
2. **Spot instances:** Use spot/preemptible instances for training runs (50-70% cheaper).
3. **Efficient code:** Minimize wasted compute through gradient accumulation, mixed precision.
4. **Checkpoint frequently:** Avoid losing progress to preemption.
5. **Right-size GPU:** Don't rent H100 for work that fits on a T4.

---

*Updated after every compute expenditure.*
