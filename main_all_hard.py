
import pandas as pd
from ortools.sat.python import cp_model
import math
from collections import defaultdict, Counter

CSV_PATH = "data.csv"
NUM_CLASSROOMS = 6
CLASS_SIZES = [34, 34, 33, 33, 33, 33]
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
        try:
            return int(float(x))
        except:
            return None

df["good_rel_id"] = df.get("좋은관계", pd.Series([None]*len(df))).apply(parse_rel)
df["bad_rel_id"] = df.get("나쁜관계", pd.Series([None]*len(df))).apply(parse_rel)

leaders = [int(r["orig_id"]) for _,r in df.iterrows() if r["Leadership_bool"]]
pianos = [int(r["orig_id"]) for _,r in df.iterrows() if r["Piano_bool"]]
nonatts = [int(r["orig_id"]) for _,r in df.iterrows() if r["비등교_bool"]]
athletic = [int(r["orig_id"]) for _,r in df.iterrows() if r["운동선호_bool"]]
males = [int(r["orig_id"]) for _,r in df.iterrows() if r["sex_norm"]=="boy"]
females = [int(r["orig_id"]) for _,r in df.iterrows() if r["sex_norm"]=="girl"]

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

prev_groups = df.groupby("24년 학급")["orig_id"].apply(list).to_dict()
prev_pairs = []
for grp, members in prev_groups.items():
    for i in range(len(members)):
        for j in range(i+1,len(members)):
            prev_pairs.append((int(members[i]), int(members[j])))

club_to_ids = defaultdict(list)
for _,r in df.iterrows():
    club_to_ids[r["club_norm"]].append(int(r["orig_id"]))

scores = {int(r["orig_id"]): int(r["score"]) for _,r in df.iterrows()}
total_score_all = sum(scores.values())

# ----- 사전 불가능성 검사 -----
diagnostics = []
max_prev_group = max((len(v) for v in prev_groups.values()), default=0)
if max_prev_group > NUM_CLASSROOMS:
    diagnostics.append(f"전년도 그룹 중 크기가 {max_prev_group}로, 반 수({NUM_CLASSROOMS}) 초과 -> strict prev-class forbidding 불가능.")

if len(leaders) < NUM_CLASSROOMS:
    diagnostics.append(f"리더 수({len(leaders)}) < 반 수({NUM_CLASSROOMS}) -> 각 반 최소 1명 불가능")

if len(pianos) < NUM_CLASSROOMS:
    diagnostics.append(f"피아노 가능한 학생 수({len(pianos)}) < 반 수({NUM_CLASSROOMS}) → 균등 분배 불가능")

for sid in nonatts:
    row = df.loc[df["orig_id"] == sid].iloc[0]
    if pd.isna(row["good_rel_id"]):
        diagnostics.append(f"비등교 학생(id={sid})에게 챙겨주는 친구 지정이 없음 → 제약 충돌 가능성")

if len(males) == 0 or len(females) == 0:
    diagnostics.append("남녀 한쪽 성별이 전혀 없음 → 성비 균등 불가능")

if len(athletic) < NUM_CLASSROOMS:
    diagnostics.append(f"운동 능력 학생 수({len(athletic)}) < 반 수({NUM_CLASSROOMS}) → 균등 분배 불가능")

for club, members in club_to_ids.items():
    if len(members) < NUM_CLASSROOMS and len(members) > 0:
        diagnostics.append(f"클럽 '{club}' 멤버 수({len(members)}) < 반 수({NUM_CLASSROOMS}) → 완전 균등 분배 불가능")

max_score = max(scores.values())
min_score = min(scores.values())
if max_score - min_score > 90:
    diagnostics.append("성적 분포 편차가 매우 큼 → 클래스 간 평균 점수 균등 배분이 어려울 수 있음")

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
    low = len(pianos) // NUM_CLASSROOMS
    high = math.ceil(len(pianos) / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in pianos) >= low)
        model.Add(sum(assigned[sid][c] for sid in pianos) <= high)

if nonatts:
    low = len(nonatts) // NUM_CLASSROOMS
    high = math.ceil(len(nonatts) / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in nonatts) >= low)
        model.Add(sum(assigned[sid][c] for sid in nonatts) <= high)

total_males = len(males)
for c, size in enumerate(CLASS_SIZES):
    exp_males = total_males * size / len(student_ids)
    low = math.floor(exp_males)
    high = math.ceil(exp_males)
    model.Add(sum(assigned[sid][c] for sid in males) >= low)
    model.Add(sum(assigned[sid][c] for sid in males) <= high)

if athletic:
    low = len(athletic) // NUM_CLASSROOMS
    high = math.ceil(len(athletic) / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in athletic) >= low)
        model.Add(sum(assigned[sid][c] for sid in athletic) <= high)

for a,b in prev_pairs:
    if a in student_ids and b in student_ids:
        model.Add(student_var[a] != student_var[b])

for club, members in club_to_ids.items():
    k = len(members)
    low = k // NUM_CLASSROOMS
    high = math.ceil(k / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in members) >= low)
        model.Add(sum(assigned[sid][c] for sid in members) <= high)

class_total_score = {}
for c, size in enumerate(CLASS_SIZES):
    v = model.NewIntVar(0, total_score_all, f"class{c}_total_score")
    class_total_score[c] = v
    model.Add(v == sum(assigned[sid][c] * scores[sid] for sid in student_ids))

class_avg_scaled = {}
for c,size in enumerate(CLASS_SIZES):
    a = model.NewIntVar(0, total_score_all * 100, f"class{c}_avg100")
    model.Add(a * size == class_total_score[c] * 100)
    class_avg_scaled[c] = a

min_avg = model.NewIntVar(0, total_score_all*100, "min_avg")
max_avg = model.NewIntVar(0, total_score_all*100, "max_avg")
model.AddMinEquality(min_avg, list(class_avg_scaled.values()))
model.AddMaxEquality(max_avg, list(class_avg_scaled.values()))
model.Minimize(max_avg - min_avg)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
solver.parameters.num_workers = 8

print("Solving (all hard constraints)...")
status = solver.Solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Solution found:", solver.StatusName(status))
    assignments = []
    for sid in student_ids:
        cls = solver.Value(student_var[sid]) + 1
        assignments.append({"id": sid, "assigned_class": cls})
    out_df = pd.DataFrame(assignments)
    merged = df.merge(out_df, left_on="orig_id", right_on="id", how="left")
    merged.to_csv("result_all_hard.csv", index=False, encoding="utf-8-sig")
    print("result_all_hard.csv 저장 완료")
else:
    print("No solution (INFEASIBLE or TIME_LIMIT). Status:", solver.StatusName(status))
    diag = {"diagnostic": diagnostics, "status": solver.StatusName(status)}
    diag_df = pd.DataFrame({"note": diagnostics})
    diag_df.to_csv("diagnostic_all_hard.csv", index=False, encoding="utf-8-sig")
    print("diagnostic_all_hard.csv 저장 완료 (문제 원인 요약)")
