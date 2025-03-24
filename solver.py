from ortools.linear_solver import pywraplp
import numpy as np

def get_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def assign_workers_optimally(
    workers, 
    sites, 
    skill_weight=0.5, 
    distance_weight=0.5, 
    compatibility_weight=0.5,
    site_priorities=None,
    worker_compatibility=None
):
    """
    Assign workers to construction sites optimally using MILP.
    
    workers: list of (worker_id, {skill: proficiency}, (lat, lon))
    sites: list of (site_id, {required_skills}, (lat, lon))
    skill_weight: weight for skill proficiency in optimization (0-1)
    distance_weight: weight for distance in optimization (0-1)
    compatibility_weight: weight for worker compatibility in optimization (0-1)
    site_priorities: dict {site_id: priority_value} where higher values indicate higher priority
    worker_compatibility: dict {(worker_id1, worker_id2): compatibility_score} where higher values
                         indicate better compatibility between workers

    Returns: dict {site_id: list of assigned worker_ids}
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver not available.")

    # Set default values if not provided
    if site_priorities is None:
        site_priorities = {site[0]: 1 for site in sites}
    
    if worker_compatibility is None:
        # Default: all workers are neutral with each other (0)
        worker_compatibility = {
            (w1[0], w2[0]): 0 
            for w1 in workers 
            for w2 in workers 
            if w1[0] != w2[0]
        }
    
    # Create symmetrical compatibility matrix if entries are missing
    for (w1, w2), score in list(worker_compatibility.items()):
        if (w2, w1) not in worker_compatibility:
            worker_compatibility[(w2, w1)] = score
    
    # Normalize priorities to be between 0 and 1
    max_priority = max(site_priorities.values())
    normalized_priorities = {
        site_id: priority / max_priority for site_id, priority in site_priorities.items()
    }
    
    # Normalize compatibility scores to be between 0 and 1
    compat_values = [v for v in worker_compatibility.values()]
    max_compat = max(compat_values) if compat_values else 1
    min_compat = min(compat_values) if compat_values else 0
    compat_range = max_compat - min_compat if max_compat != min_compat else 1
    
    normalized_compatibility = {
        (w1, w2): (score - min_compat) / compat_range 
        for (w1, w2), score in worker_compatibility.items()
    }
    
    # Decision variables: x[w, s, skill] = 1 if worker w is assigned to site s for a specific skill
    x = {}
    
    # Slack variables for unfilled positions
    unfilled = {}
    
    # Variables for worker pairs at the same site
    same_site_pairs = {}
    
    # Find max distance for normalization
    max_distance = 0
    for w_id, skills, w_loc in workers:
        for s_id, required_skills, s_loc in sites:
            max_distance = max(max_distance, get_distance(w_loc, s_loc))
    
    # Create variables and objective
    objective = solver.Objective()
    
    # Create worker assignment variables and add to objective
    for w_id, skills, w_loc in workers:
        for s_id, required_skills, s_loc in sites:
            for skill in required_skills:
                if skill in skills:  # Worker has the required skill
                    x[w_id, s_id, skill] = solver.BoolVar(f'x[{w_id},{s_id},{skill}]')
                    
                    # Calculate normalized costs
                    distance = get_distance(w_loc, s_loc)
                    normalized_distance = distance / max_distance if max_distance > 0 else 0
                    proficiency = skills[skill]
                    
                    # Combined cost (minimize distance and maximize proficiency)
                    # Apply site priority as a multiplier to make high-priority sites more attractive
                    site_priority_factor = 1 / (normalized_priorities[s_id] + 0.1)  # +0.1 to avoid division by zero
                    cost = site_priority_factor * (distance_weight * normalized_distance + skill_weight * (1 - proficiency))
                    objective.SetCoefficient(x[w_id, s_id, skill], cost)
    
    # Create unfilled position variables
    for s_id, required_skills, _ in sites:
        for skill in required_skills:
            unfilled[s_id, skill] = solver.BoolVar(f'unfilled[{s_id},{skill}]')
            
            # Set a high penalty for unfilled positions, inversely proportional to priority
            # Higher priority sites have higher penalties for being unfilled
            unfilled_penalty = 100 * (2 - normalized_priorities[s_id])
            objective.SetCoefficient(unfilled[s_id, skill], unfilled_penalty)
    
    # Create variables for worker pairs at the same site
    worker_ids = [w[0] for w in workers]
    for i, w1_id in enumerate(worker_ids):
        for w2_id in worker_ids[i+1:]:  # Only consider each pair once
            for s_id, _, _ in sites:
                same_site_pairs[w1_id, w2_id, s_id] = solver.BoolVar(f'same_site[{w1_id},{w2_id},{s_id}]')
                
                # Get compatibility score (higher is better)
                if (w1_id, w2_id) in normalized_compatibility:
                    compat_score = normalized_compatibility[w1_id, w2_id]
                    
                    # We use 1-compat_score because we're minimizing
                    # A negative coefficient for good compatibility (to encourage them working together)
                    objective.SetCoefficient(
                        same_site_pairs[w1_id, w2_id, s_id], 
                        -compatibility_weight * compat_score
                    )
    
    objective.SetMinimization()

    # Constraint: Each worker is assigned to at most one site-skill combination
    for w_id, _, _ in workers:
        solver.Add(sum(x[w_id, s_id, skill] 
                       for s_id, required_skills, _ in sites 
                       for skill in required_skills 
                       if (w_id, s_id, skill) in x) <= 1)

    # Modified constraint: Each site gets exactly one worker per required skill OR has that position unfilled
    for s_id, required_skills, _ in sites:
        for skill in required_skills:
            solver.Add(
                sum(
                    x[w_id, s_id, skill]
                    for w_id, worker_skills, _ in workers
                    if (w_id, s_id, skill) in x
                ) + unfilled[s_id, skill] == 1
            )
    
    # Constraints for worker pairs at same site
    # If both workers w1 and w2 are assigned to site s, then same_site[w1,w2,s] = 1
    for (w1_id, w2_id, s_id), var in same_site_pairs.items():
        # Get all the possible skill assignments for w1 at site s
        w1_at_s_vars = [
            x[w1_id, s_id, skill] 
            for skill in next(s[1] for s in sites if s[0] == s_id)
            if (w1_id, s_id, skill) in x
        ]
        
        # Get all the possible skill assignments for w2 at site s
        w2_at_s_vars = [
            x[w2_id, s_id, skill] 
            for skill in next(s[1] for s in sites if s[0] == s_id)
            if (w2_id, s_id, skill) in x
        ]
        
        # If both workers have possible assignments at this site
        if w1_at_s_vars and w2_at_s_vars:
            # If either worker is not assigned to site s, then same_site must be 0
            solver.Add(var <= sum(w1_at_s_vars))
            solver.Add(var <= sum(w2_at_s_vars))
            
            # If both workers are assigned to site s, then same_site must be 1
            solver.Add(var >= sum(w1_at_s_vars) + sum(w2_at_s_vars) - 1)

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        assignments = {s_id: [] for s_id, _, _ in sites}
        for (w_id, s_id, _), var in x.items():
            if var.solution_value() == 1:
                assignments[s_id].append(w_id)
                
        # Report unfilled positions
        unfilled_positions = []
        for (s_id, skill), var in unfilled.items():
            if var.solution_value() == 1:
                unfilled_positions.append((s_id, skill))
        
        if unfilled_positions:
            print(f"Warning: {len(unfilled_positions)} positions unfilled: {unfilled_positions}")
                
        return assignments
    else:
        raise Exception("No optimal solution found.")

# Example usage:
workers = [
    (1, {"electrician": 0.9, "plumber": 0.3}, (45.0, 7.6)),
    (2, {"plumber": 0.8, "worker": 0.5}, (44.9, 7.5)),
    (3, {"electrician": 0.7, "worker": 0.6}, (45.1, 7.7)),
    (4, {"electrician": 1.0}, (34.1, 1.7)),
    (5, {"worker": 0.9, "plumber": 0.2}, (30.1, 1.7)),
    (6, {"plumber": 0.95, "electrician": 0.4}, (45.05, 7.55))
]
sites = [
    (101, {"electrician", "plumber"}, (45.05, 7.55)),
    (102, {"electrician", "worker"}, (45.2, 7.8)),
]

# Define site priorities (higher value = higher priority)
site_priorities = {
    101: 3,  # High priority
    102: 1   # Lower priority
}

# Define worker compatibility (higher is better, negative for incompatibility)
worker_compatibility = {
    (1, 2): 0.8,   # Workers 1 and 2 work very well together
    (1, 3): -0.5,  # Workers 1 and 3 don't work well together
    (2, 3): 0.6,   # Workers 2 and 3 work well together
    (4, 5): 0.7,   # Workers 4 and 5 work well together
    (5, 6): -0.3,  # Workers 5 and 6 don't work well together
    # Other pairs default to neutral (0)
}

# Basic assignment
assignments = assign_workers_optimally(workers, sites)
print("Basic assignment:", assignments)

# With compatibility preferences
assignments_compatible = assign_workers_optimally(
    workers, 
    sites, 
    compatibility_weight=0.7,
    worker_compatibility=worker_compatibility
)
print("With compatibility preferences:", assignments_compatible)

# Combined optimization with all factors
assignments_combined = assign_workers_optimally(
    workers, 
    sites, 
    skill_weight=0.3,
    distance_weight=0.3, 
    compatibility_weight=0.4,
    site_priorities=site_priorities,
    worker_compatibility=worker_compatibility
)
print("Combined optimization:", assignments_combined)