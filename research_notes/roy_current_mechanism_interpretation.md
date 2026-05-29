# Current Mechanism Interpretation After ODE-PDE Comparison

## Main conclusion

The best current mechanism diagnosis is `reaction_dominated_homogeneous_multistability`.

Current evidence supports reaction-dominated homogeneous multistability embedded in a PDE. Spatial patterning itself is not currently supported as the mechanism for basin selection in the tested parameterization.

## Evidence

- Representative ODE-PDE classification agreement was 3/3.
- ODE-PDE basin labels agreed for 90 percent of the q0-w0 grid.
- The 14 ODE-PDE disagreements all involved transient labels.
- Direct persistent/extinct disagreements were zero.
- Final spatial CV values were very small.
- Targeted perturbation tests produced zero outcome changes.

## What changed

Earlier language about spatial PDE bistability must be qualified. The PDE preserves basin-dependent outcomes, but the representative fields remain nearly homogeneous and the matched ODE reproduces most basin labels.

## What is supported

- The well-mixed eco-evolutionary reaction system supports indirect evolutionary rescue in the tested parameterization.
- The spatial PDE preserves persistent, extinct, and transient basin-dependent outcomes.
- Representative PDE solutions remain close to spatially homogeneous.
- ODE and PDE basin labels agree for most q0-w0 grid points.
- Small targeted perturbations do not change representative outcome classes.

## What is not supported

- Spatial-pattern-mediated rescue is not supported by the current evidence.
- The current results do not justify claiming that spatial structure generates the bistability.
- The conclusion should not be generalized across diffusion settings, trade-off forms, or broader parameter regions.

## Immediate next question

Why does the homogeneous eco-evolutionary reaction system have multiple basins, and under what trade-off conditions does indirect evolutionary rescue become basin-dependent?

## Consequence for manuscript language

Manuscript language should distinguish spatially extended dynamics from spatial-pattern-mediated dynamics. Allowed language is that the spatial PDE preserves basin-dependent outcomes and that the current mechanism diagnosis is reaction-dominated homogeneous multistability. The current results should not state that spatial structure causes bistability, that spatial PDE rescue is spatial-pattern-mediated, or that spatial structure generally suppresses or amplifies rescue.

## Remaining caveats

- A minority of ODE-PDE basin labels disagree.
- The disagreement audit identifies these cases as boundary or horizon sensitive in the current outputs.
- Transient-heavy regions still require caution if they become central to a manuscript claim.

## Next work

The next work is to analyze the homogeneous eco-evolutionary mechanism, then test robustness of that basin structure across trade-off parameters and refine the analytical equilibrium/stability interpretation. It should not restart broad PDE scanning before the current mechanism narrative is corrected.
