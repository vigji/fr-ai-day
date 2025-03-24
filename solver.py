def get_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


from ortools.linear_solver import pywraplp

def assign_workers_optimally(workers, sites):
    """
    Assign workers to construction sites optimally using MILP.
    
    workers: list of (worker_id, skill, (lat, lon))
    sites: list of (site_id, {required_skills}, (lat, lon))

    Returns: dict {site_id: assigned worker_id}
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver not available.")

    # Decision variables: x[w, s] = 1 if worker w is assigned to site s
    x = {}
    for w_id, skill, w_loc in workers:
        for s_id, s_skills, s_loc in sites:
            if skill in s_skills:  # Worker can only be assigned if skill matches
                x[w_id, s_id] = solver.BoolVar(f'x[{w_id},{s_id}]')

    # Objective: Minimize total distance
    objective = solver.Objective()
    for (w_id, s_id), var in x.items():
        w_loc = next(w[2] for w in workers if w[0] == w_id)
        s_loc = next(s[2] for s in sites if s[0] == s_id)
        distance = get_distance(w_loc, s_loc)
        objective.SetCoefficient(var, distance)
    objective.SetMinimization()

    # Constraint: Each worker is assigned to at most one site
    for w_id, _, _ in workers:
        solver.Add(sum(x[w_id, s_id] for s_id in {s[0] for s in sites if (w_id, s[0]) in x}) <= 1)

    # Modified constraint: Each site gets exactly one worker per required skill
    for s_id, s_skills, _ in sites:
        for skill in s_skills:
            solver.Add(
                sum(
                    x[w_id, s_id]
                    for w_id, w_skill, _ in workers
                    if (w_id, s_id) in x and w_skill == skill
                )
                == 1
            )

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        assignments = {s_id: None for s_id, _, _ in sites}
        for (w_id, s_id), var in x.items():
            if var.solution_value() == 1:
                assignments[s_id] = w_id
        return assignments
    else:
        raise Exception("No optimal solution found.")

# Example usage:
workers = [
    (1, "electrician", (45.0, 7.6)),
    (2, "plumber", (44.9, 7.5)),
    (3, "electrician", (45.1, 7.7)),
    (4, "electrician", (34.1, 1.7)),
    (5, "worker", (30.1, 1.7)),
]
sites = [
    (101, {"electrician", "plumber"}, (45.05, 7.55)),
    (102, {"electrician"}, (45.2, 7.8)),
]

assignments = assign_workers_optimally(workers, sites)
print(assignments)