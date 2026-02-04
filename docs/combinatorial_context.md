# combinatorial_context.md

## 1. Goal Definition
**Objective:** Algorithmically rank a set of candidate robot trajectories ($T_1...T_N$) for a 6-DoF manipulator to identify the "Global Best" path for execution.

**Context:**
* **Input:** A toolpath consisting of discrete waypoints (Position + Quaternion) and a **Desired Cartesian Speed** (mm/s) for each segment.
* **System:** A 6-DoF Robotic Manipulator with defined joint limits (Position, Velocity).
* **Constraint:** The ranking must be hierarchical, strictly prioritizing physical feasibility and safety before optimizing for motion quality.

## 2. The Solution: "Waterfall" Hierarchical Ranking
We utilize a **Lexicographical Ranking System** (Tuple Sort) that filters trajectories through four distinct levels of strictness. A trajectory is only evaluated on "Level $N+1$" if it ties with others at "Level $N$".

### The Hierarchy
1.  **Level 1: Feasibility Gate (The "Must Haves")**
    * *Type:* Binary Hard Constraint (Pass/Fail).
    * *Logic:* Is the path physically executable without collision, teleportation, or motor over-speeding?
2.  **Level 2: Safety Tier (The "Risk Limit")**
    * *Type:* Discretized Hard Optimization.
    * *Logic:* Minimizes the risk of Singularity (Kinematic Lock-up). We "bin" this score to group "equally safe" paths together, preventing negligible safety differences from overshadowing smoothness.
3.  **Level 3: Smoothness Cost (The "Efficiency" Factor)**
    * *Type:* Continuous Soft Optimization.
    * *Logic:* Minimizes the "Normalized Kinetic Energy" required to execute the path. Penalizes jerkiness and high-speed joint movements.
4.  **Level 4: Dexterity Score (The "Quality" Bonus)**
    * *Type:* Continuous Soft Optimization.
    * *Logic:* Maximizes the robot's Manipulability (Control Authority/Agility).

---

## 3. Core Physics Engine (Kinodynamic Updates)
To support the ranking, we calculate physics based on **Desired Speed**, not just geometry.

### A. Time Step Derivation
Unlike geometric analysis (where $\Delta t = 1$), we derive $\Delta t$ from the toolpath requirements.
$$\Delta t_i = \frac{||P_{i+1} - P_i||}{\max(v_{desired, i}, \epsilon)}$$
* *Correction:* Requires input sanitization to filter duplicate points ($dist \approx 0$) to prevent DivisionByZero.

### B. Joint Velocity Calculation
velocities are calculated using the shortest angular path to handle wrapping (e.g., $359^\circ \to 1^\circ$).
$$\dot{q}_i = \frac{\text{shortest\_angular\_dist}(q_{i+1}, q_i)}{\Delta t_i}$$

### C. Time-Weighted Averaging (Critical Fix)
Because $\Delta t$ varies (slow segments vs. fast segments), simple means `mean()` are statistically biased. All "Average" scores (Smoothness, Dexterity) must be **Time-Weighted**:
$$\text{Score}_{avg} = \frac{\sum (\text{Score}_i \cdot \Delta t_i)}{\sum \Delta t_i}$$

---

## 4. The Metric Specifications

### Level 1: Feasibility Metrics
* **Reachability:** $100\%$ of waypoints must have valid IK solutions.
* **C0 Continuity:** $\max(|\Delta q|) < \text{Threshold}$ (e.g., $30^\circ$).
* **C1 Feasibility:** $\max(\text{Velocity Ratio}) \le 1.0$.
    * *Ratio:* $|\dot{q}| / \dot{q}_{limit}$.

### Level 2: Safety Metrics
* **Condition Number ($\kappa$):** $\sigma_{max} / \sigma_{min}$ (from SVD of Jacobian).
* **Safety Score:** $\max(\kappa)$ over the entire trajectory.
* **Safety Tier (Binned):**
    $$\text{Tier} = \lceil \frac{\max(\kappa)}{\text{bin\_size}} \rceil$$
    * *Note:* `bin_size` (e.g., 10.0) is configurable to control sort sensitivity.

### Level 3: Smoothness Metrics
* **Energy Intensity ($E_i$):** Sum of squared normalized velocities per step.
    $$E_i = \sum_{j=1}^{DoF} \left( \frac{\dot{q}_{j}}{\dot{q}_{limit,j}} \right)^2$$
* **Smoothness Cost:** Time-Weighted Average of $E$.

### Level 4: Dexterity Metrics
* **Manipulability ($w_i$):** $\sqrt{\det(J_i J_i^T)}$ (Yoshikawa Index).
* **Dexterity Score:** Time-Weighted Average of $w$.

---

## 5. Implementation Logic (Python Sort Key)

The sorting algorithm uses a tuple key. Python sorts tuples element-by-element. We sort in **Descending Order** (`reverse=True`), so "Better" values must be mathematically "Larger".

```python
def get_sort_key(traj):
    # 1. Feasibility (Boolean -> Int)
    # Valid = 1, Invalid = 0
    # Primary Sort Key: Valid trajectories always win.
    valid_score = int(traj.is_globally_valid) 

    # 2. Safety Tier (Int)
    # Lower Tier is Better (1 is best).
    # Negate it: -1 > -5.
    # If Invalid, force to -Infinity to ensure it ranks last.
    if valid_score == 0:
        safety_score = -float('inf')
    else:
        safety_score = -traj.safety_tier 

    # 3. Smoothness Cost (Float)
    # Lower Energy is Better.
    # Negate it: -0.1 > -0.9.
    smoothness_score = -traj.norm_energy_score

    # 4. Dexterity Score (Float)
    # Higher Manipulability is Better.
    # Positive value.
    dexterity_score = traj.avg_manipulability

    return (valid_score, safety_score, smoothness_score, dexterity_score)

# Execution
sorted_trajectories = sorted(trajectories, key=get_sort_key, reverse=True)