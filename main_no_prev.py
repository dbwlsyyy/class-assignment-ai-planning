
import pandas as pd
from ortools.sat.python import cp_model
import math
from collections import defaultdict, Counter

CSV_PATH = "data.csv"
NUM_CLASSROOMS = 6
CLASS_SIZES = [34,34,33,33,33,33]
TIME_LIMIT_SECONDS = 120

def yes(val):
    if pd.isna(val): return False
    if isinstance(val, str):
        return val.strip().lower() in ("yes","y","true","1","예")
    return bool(val)

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df["orig_id"] = df["id"].astype(int)
id_to_index = {int(r["orig_id"]): idx for idx, r in df.iterrows()}
student_ids = list(df["orig_id"])

df["Leadership_bool"] = df.get("Leadership", pd.Series([None]*len(df))).apply(yes)
df["Piano_bool"] = df.get("Piano", pd.Series([None]*len(df))).apply(yes)
df["비등교_bool"] = df.get("비등교", pd.Series([None]*len(df))).apply(yes)
df["운동선호_bool"] = df.get("운동선호", pd.Series([None]*len(df))).apply(yes)
df["sex_norm"] = df["sex"].astype(str).str.lower().str.strip()
df["club_norm"] = df["클럽"].astype(str).str.strip()

def parse_rel(x):
    if pd.isna(x): return None
    try:
        return int(x)
    except:
        try: return int(float(x))
        except: return None

df["good_rel_id"] = df.get("좋은관계", pd.Series([None]*len(df))).apply(parse_rel)
df["bad_rel_id"] = df.get("나쁜관계", pd.Series([None]*len(df))).apply(parse_rel)

leaders = [int(r["orig_id"]) for _,r in df.iterrows() if r["Leadership_bool"]]
pianos = [int(r["orig_id"]) for _,r in df.iterrows() if r["Piano_bool"]]
nonatts = [int(r["orig_id"]) for _,r in df.iterrows() if r["비등교_bool"]]
athletic = [int(r["orig_id"]) for _,r in df.iterrows() if r["운동선호_bool"]]
males = [int(r["orig_id"]) for _,r in df.iterrows() if r["sex_norm"]=="boy"]

dislike_pairs = []
for _,r in df.iterrows():
    a = int(r["orig_id"])
    b = r["bad_rel_id"]
    if b is not None and b in id_to_index and b != a:
        dislike_pairs.append((a,int(b)))

caretaking_pairs = []
for _,r in df.iterrows():
    a = int(r["orig_id"])
    if r["비등교_bool"]:
        b = r["good_rel_id"]
        if b is not None and b in id_to_index and b != a:
            caretaking_pairs.append((a,int(b)))

club_to_ids = defaultdict(list)
for _,r in df.iterrows():
    club_to_ids[r["club_norm"]].append(int(r["orig_id"]))

scores = {int(r["orig_id"]): int(r["score"]) for _,r in df.iterrows()}
total_score_all = sum(scores.values())

model = cp_model.CpModel()
student_var = {}
assigned = {}
for sid in student_ids:
    student_var[sid] = model.NewIntVar(0, NUM_CLASSROOMS-1, f"st_{sid}_class")
    assigned[sid] = {}
    for c in range(NUM_CLASSROOMS):
        b = model.NewBoolVar(f"st_{sid}_in_c{c}")
        assigned[sid][c] = b
        model.Add(student_var[sid] == c).OnlyEnforceIf(b)
        model.Add(student_var[sid] != c).OnlyEnforceIf(b.Not())

for sid in student_ids:
    model.Add(sum(assigned[sid][c] for c in range(NUM_CLASSROOMS)) == 1)
for c, size in enumerate(CLASS_SIZES):
    model.Add(sum(assigned[sid][c] for sid in student_ids) == size)

for a,b in dislike_pairs:
    if a in student_ids and b in student_ids:
        model.Add(student_var[a] != student_var[b])
for a,b in caretaking_pairs:
    if a in student_ids and b in student_ids:
        model.Add(student_var[a] == student_var[b])

for c in range(NUM_CLASSROOMS):
    model.Add(sum(assigned[sid][c] for sid in leaders) >= 1)

if pianos:
    low = len(pianos)//NUM_CLASSROOMS
    high = math.ceil(len(pianos)/NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in pianos) >= low)
        model.Add(sum(assigned[sid][c] for sid in pianos) <= high)

if nonatts:
    low = len(nonatts)//NUM_CLASSROOMS
    high = math.ceil(len(nonatts)/NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in nonatts) >= low)
        model.Add(sum(assigned[sid][c] for sid in nonatts) <= high)

total_males = len(males)
for c,size in enumerate(CLASS_SIZES):
    exp = total_males * size / len(student_ids)
    model.Add(sum(assigned[sid][c] for sid in males) >= math.floor(exp))
    model.Add(sum(assigned[sid][c] for sid in males) <= math.ceil(exp))

if athletic:
    low = len(athletic)//NUM_CLASSROOMS
    high = math.ceil(len(athletic)/NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in athletic) >= low)
        model.Add(sum(assigned[sid][c] for sid in athletic) <= high)

for club,members in club_to_ids.items():
    k = len(members)
    low = k // NUM_CLASSROOMS
    high = math.ceil(k / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in members) >= low)
        model.Add(sum(assigned[sid][c] for sid in members) <= high)

class_total = {}
for c,size in enumerate(CLASS_SIZES):
    v = model.NewIntVar(0, total_score_all, f"class{c}_total")
    class_total[c] = v
    model.Add(v == sum(assigned[sid][c] * scores[sid] for sid in student_ids))
class_avg_scaled = {}
for c,size in enumerate(CLASS_SIZES):
    a = model.NewIntVar(0, total_score_all*100, f"class{c}_avg100")
    model.Add(a * size == class_total[c] * 100)
    class_avg_scaled[c] = a
min_avg = model.NewIntVar(0, total_score_all*100, "min_avg")
max_avg = model.NewIntVar(0, total_score_all*100, "max_avg")
model.AddMinEquality(min_avg, list(class_avg_scaled.values()))
model.AddMaxEquality(max_avg, list(class_avg_scaled.values()))
model.Minimize(max_avg - min_avg)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
solver.parameters.num_workers = 8

print("Solving (no prev constraint)...")
status = solver.Solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Solution found:", solver.StatusName(status))
    assignments = {sid: solver.Value(student_var[sid]) + 1 for sid in student_ids}
    out_df = df.copy()
    out_df["assigned_class"] = out_df["orig_id"].map(assignments)
    out_df.to_csv("result_no_prev.csv", index=False, encoding="utf-8-sig")
    for c in range(NUM_CLASSROOMS):
        members = out_df[out_df["assigned_class"]==c+1]
        print(f"Class {c+1}: size {len(members)}, avg score {members['score'].mean():.2f}")
    print("result_no_prev.csv 저장 완료")
else:
    print("No solution. Status:", solver.StatusName(status))
