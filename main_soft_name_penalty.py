import pandas as pd
from ortools.sat.python import cp_model
import math
from collections import defaultdict, Counter

CSV_PATH = "data.csv"
NUM_CLASSROOMS = 6
CLASS_SIZES = [34, 34, 33, 33, 33, 33]
RESULT_CSV = "result_soft_name_penalty.csv"
TIME_LIMIT_SECONDS = 180
WEIGHT_SAME_FIRSTNAME = 2000

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df['orig_id'] = df['id'].astype(int)
student_ids = list(df['orig_id'])
student_idx_by_id = {int(r['orig_id']): i for i, r in df.iterrows()}

def yes(val):
    if pd.isna(val): return False
    if isinstance(val, str):
        return val.strip().lower() in ('yes','y','true','1','예')
    return bool(val)

df['Leadership_bool'] = df.get('Leadership', pd.Series([None]*len(df))).apply(yes)
df['Piano_bool'] = df.get('Piano', pd.Series([None]*len(df))).apply(yes)
df['비등교_bool'] = df.get('비등교', pd.Series([None]*len(df))).apply(yes)
df['운동선호_bool'] = df.get('운동선호', pd.Series([None]*len(df))).apply(yes)
df['sex_norm'] = df['sex'].astype(str).str.lower().str.strip()
df['club_norm'] = df['클럽'].astype(str).str.strip()

def parse_relation(x):
    if pd.isna(x): return None
    try:
        return int(x)
    except:
        try:
            return int(float(x))
        except:
            return None

df['good_rel_id'] = df.get('좋은관계', pd.Series([None]*len(df))).apply(parse_relation)
df['bad_rel_id'] = df.get('나쁜관계', pd.Series([None]*len(df))).apply(parse_relation)

def get_firstname(fullname):
    if pd.isna(fullname): return ""
    parts = str(fullname).strip().split()
    return parts[0] if parts else ""

df['firstname'] = df.get('name', pd.Series(['']*len(df))).apply(get_firstname)

scores = {int(r['orig_id']): int(r['score']) for _,r in df.iterrows()}

dislike_pairs = []
for _,r in df.iterrows():
    a = int(r['orig_id']); b = r['bad_rel_id']
    if b is not None and b in student_idx_by_id and b != a:
        dislike_pairs.append((a,int(b)))

caretaking_pairs = []
for _,r in df.iterrows():
    a = int(r['orig_id'])
    if r['비등교_bool']:
        b = r['good_rel_id']
        if b is not None and b in student_idx_by_id and b != a:
            caretaking_pairs.append((a,int(b)))

prev_class_groups = df.groupby('24년 학급')['orig_id'].apply(list).to_dict()
prev_classmate_pairs = []
for grp,members in prev_class_groups.items():
    if len(members)>1:
        for i in range(len(members)):
            for j in range(i+1,len(members)):
                prev_classmate_pairs.append((int(members[i]), int(members[j])))

club_to_ids = defaultdict(list)
for _,r in df.iterrows():
    club_to_ids[r['club_norm']].append(int(r['orig_id']))

piano_ids = [int(r['orig_id']) for _,r in df.iterrows() if r['Piano_bool']]
nonatt_ids = [int(r['orig_id']) for _,r in df.iterrows() if r['비등교_bool']]
male_ids = [int(r['orig_id']) for _,r in df.iterrows() if r['sex_norm']=='boy']
female_ids = [int(r['orig_id']) for _,r in df.iterrows() if r['sex_norm']=='girl']
athletic_ids = [int(r['orig_id']) for _,r in df.iterrows() if r['운동선호_bool']]
leader_ids = [int(r['orig_id']) for _,r in df.iterrows() if r['Leadership_bool']]

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

for c,size in enumerate(CLASS_SIZES):
    model.Add(sum(assigned[sid][c] for sid in student_ids) == size)

for a,b in dislike_pairs:
    if a in student_ids and b in student_ids:
        model.Add(student_var[a] != student_var[b])

for a,b in caretaking_pairs:
    if a in student_ids and b in student_ids:
        model.Add(student_var[a] == student_var[b])

for c in range(NUM_CLASSROOMS):
    model.Add(sum(assigned[sid][c] for sid in leader_ids) >= 1)

num_piano = len(piano_ids)
if num_piano > 0:
    low = num_piano // NUM_CLASSROOMS
    high = math.ceil(num_piano / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in piano_ids) >= low)
        model.Add(sum(assigned[sid][c] for sid in piano_ids) <= high)

num_nonatt = len(nonatt_ids)
if num_nonatt > 0:
    low = num_nonatt // NUM_CLASSROOMS
    high = math.ceil(num_nonatt / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in nonatt_ids) >= low)
        model.Add(sum(assigned[sid][c] for sid in nonatt_ids) <= high)

total_males = len(male_ids)
for c,size in enumerate(CLASS_SIZES):
    exp_males = total_males * size / len(student_ids)
    low = math.floor(exp_males); high = math.ceil(exp_males)
    model.Add(sum(assigned[sid][c] for sid in male_ids) >= low)
    model.Add(sum(assigned[sid][c] for sid in male_ids) <= high)

num_ath = len(athletic_ids)
if num_ath > 0:
    low = num_ath // NUM_CLASSROOMS
    high = math.ceil(num_ath / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in athletic_ids) >= low)
        model.Add(sum(assigned[sid][c] for sid in athletic_ids) <= high)

prev_penalties = []
for a,b in prev_classmate_pairs:
    if a in student_ids and b in student_ids:
        viol = model.NewBoolVar(f"prev_same_{a}_{b}")
        zlist = []
        for c in range(NUM_CLASSROOMS):
            z = model.NewBoolVar(f"z_prev_{a}_{b}_c{c}")
            zlist.append(z)
            model.Add(z <= assigned[a][c])
            model.Add(z <= assigned[b][c])
            model.Add(assigned[a][c] + assigned[b][c] - z <= 1)
        model.Add(sum(zlist) == viol)
        prev_penalties.append(viol)

for club,members in club_to_ids.items():
    k = len(members)
    low = k // NUM_CLASSROOMS
    high = math.ceil(k / NUM_CLASSROOMS)
    for c in range(NUM_CLASSROOMS):
        model.Add(sum(assigned[sid][c] for sid in members) >= low)
        model.Add(sum(assigned[sid][c] for sid in members) <= high)


firstname_to_ids = defaultdict(list)
for _, r in df.iterrows():
    sid = int(r['orig_id'])
    fn = r['firstname']
    if fn and str(fn).strip() != "":
        firstname_to_ids[fn].append(sid)

name_penalties = []
for fn, members in firstname_to_ids.items():
    if len(members) <= 1:
        continue
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            a = members[i]; b = members[j]
            if a in student_ids and b in student_ids:
                same_pair = model.NewBoolVar(f"same_first_{fn}_{a}_{b}")
                zlist = []
                for c in range(NUM_CLASSROOMS):
                    z = model.NewBoolVar(f"z_name_{fn}_{a}_{b}_c{c}")
                    zlist.append(z)
                    model.Add(z <= assigned[a][c])
                    model.Add(z <= assigned[b][c])
                    model.Add(assigned[a][c] + assigned[b][c] - z <= 1)
                model.Add(sum(zlist) == same_pair)
                name_penalties.append(same_pair)

total_score_all = sum(scores.values())
class_total_score = {}
for c,size in enumerate(CLASS_SIZES):
    v = model.NewIntVar(0, total_score_all, f"class{c}_total_score")
    class_total_score[c] = v
    model.Add(v == sum(assigned[sid][c] * scores[sid] for sid in student_ids))

class_avg_scaled = {}
for c,size in enumerate(CLASS_SIZES):
    a = model.NewIntVar(0, total_score_all*100, f"class{c}_avg100")
    class_avg_scaled[c] = a
    model.Add(a * size == class_total_score[c] * 100)

min_avg = model.NewIntVar(0, total_score_all*100, "min_avg")
max_avg = model.NewIntVar(0, total_score_all*100, "max_avg")
model.AddMinEquality(min_avg, list(class_avg_scaled.values()))
model.AddMaxEquality(max_avg, list(class_avg_scaled.values()))

obj = (max_avg - min_avg)
if prev_penalties:
    obj = obj + 1000 * sum(prev_penalties)
if name_penalties:
    obj = obj + WEIGHT_SAME_FIRSTNAME * sum(name_penalties)

model.Minimize(obj)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
solver.parameters.num_workers = 8
print("Solving...")

status = solver.Solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Status:", solver.StatusName(status))
    assignments = {c: [] for c in range(NUM_CLASSROOMS)}
    for sid in student_ids:
        c = solver.Value(student_var[sid])
        assignments[c].append(sid)

    rows = []
    for c in range(NUM_CLASSROOMS):
        members = assignments[c]
        totals = sum(scores[s] for s in members)
        avg = totals / len(members)
        row = {
            "반": c+1,
            "인원": len(members),
            "평균 점수": round(avg,2),
            "총점": totals,
            "리더 수": sum(1 for s in members if s in leader_ids),
            "피아노": sum(1 for s in members if s in piano_ids),
            "비등교": sum(1 for s in members if s in nonatt_ids),
            "운동": sum(1 for s in members if s in athletic_ids),
            "남": sum(1 for s in members if s in male_ids),
            "여": sum(1 for s in members if s in female_ids),
            "클럽 학생 수": sum(1 for s in members if df.loc[student_idx_by_id[s],'club_norm'] not in ("", "nan", "None")),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"결과 CSV 저장 완료 → {RESULT_CSV}")

    total_pairs = len(name_penalties)
    violated = sum(solver.Value(v) for v in name_penalties) if name_penalties else 0
    print("같은 firstname 쌍 총 개수:", total_pairs)
    print("같은 반에 배정된 쌍(위반):", violated)
    if total_pairs > 0:
        print(f"→ 분리 성공률: {(1 - violated/total_pairs)*100:.2f}%")

else:
    print("No solution; status:", solver.StatusName(status))

