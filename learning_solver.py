from ortools.sat.python import cp_model

num_nurses = 4
num_shifts = 3
num_days = 3
all_nurses = range(num_nurses)
all_shifts = range(num_shifts)
all_days = range(num_days)

shifts = {}

model = cp_model.CpModel()
for n in all_nurses:
    for d in all_days:
        for s in all_shifts:
            shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}")

model.Add(sum(shifts[(n, d, s)] for s in all_shifts) == 1 for n in all_nurses for d in all_days)
