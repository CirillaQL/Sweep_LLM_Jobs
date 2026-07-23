# Minimum-SLO-violation overload fallback validation

This CPU-only Slurm job validates the scheduler rule:

> When no safe operating point exists, select the candidate with the lowest
> modeled overload/SLO-violation risk.

It reuses the four boundary workloads from Jobs 250005 and 250009 and evaluates
both fixed and automatic placement, for eight decisions in total. Every
decision must:

- return `OVERLOAD_FALLBACK`;
- retain `recommended.is_safe=false`;
- report `num_safe=0`;
- have no returned alternative with a lower overload-violation score.

The job requests no GPU TRES and does not start vLLM or issue any GPU command.
It tests selection semantics on the cluster without executing an unsafe
operating point.
