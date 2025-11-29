import pandas as pd
import numpy as np

### load data ###
df = pd.read_excel("data/baseline_data_11_26_25.xlsx")

# filter to Pierce County
df = df[df['census_block_group'].astype(str).str.startswith('53053')].copy()
pierce_zips = df[df['census_block_group'].astype(str).str.startswith('53053')]['address_zip'].dropna().unique()
df = df[df['address_zip'].isin(pierce_zips)].copy()

# target variables
target_columns = [
    'defendant_appearance',
    'hearing_held',
    'defendant_hearing_attendance',
    'defendant_rep_merged',
    'writ_final',
    'dismissal_final',
    'old_final',
    'court_displacement',
]

# drop rows with missing values
valid_idx = df[target_columns].notna().all(axis=1)
df = df[valid_idx].copy()

print(f"analyzing {len(df)} cases\n")

### check assumption: defendant_rep_merged requires appearance OR hearing attendance ###

print("="*70)
print("ASSUMPTION 1: defendant_rep_merged requires appearance OR attendance")
print("="*70)

# cases where rep_merged = 1
rep_cases = df[df['defendant_rep_merged'] == 1]
print(f"\ncases with defendant_rep_merged=1: {len(rep_cases)}")

# of those, how many have appearance OR attendance?
has_appearance_or_attendance = rep_cases[
    (rep_cases['defendant_appearance'] == 1) | 
    (rep_cases['defendant_hearing_attendance'] == 1)
]
print(f"  with appearance OR attendance: {len(has_appearance_or_attendance)} ({100*len(has_appearance_or_attendance)/len(rep_cases):.1f}%)")

# violations (rep_merged but no appearance and no attendance)
violations = rep_cases[
    (rep_cases['defendant_appearance'] == 0) & 
    (rep_cases['defendant_hearing_attendance'] == 0)
]
print(f"  rep but no appearance/attendance: {len(violations)} ({100*len(violations)/len(rep_cases):.1f}%)")

### causal flow analysis ###

# stage 1: early dynamics

print(f"\ndefendant_appearance: {df['defendant_appearance'].sum()}/{len(df)} ({100*df['defendant_appearance'].mean():.1f}%)")

# of those with appearance, how many had hearing?
appeared = df[df['defendant_appearance'] == 1]
print(f"  → of those, hearing_held: {appeared['hearing_held'].sum()}/{len(appeared)} ({100*appeared['hearing_held'].mean():.1f}%)")

# of those with hearing, how many attended?
hearings = df[df['hearing_held'] == 1]
print(f"\ndefendant_hearing_attendance: {hearings['defendant_hearing_attendance'].sum()}/{len(hearings)} ({100*hearings['defendant_hearing_attendance'].mean():.1f}%)")

# stage 2: representation
print(f"\n\nSTAGE 2: Legal Representation")
print("-" * 70)
print(f"defendant_rep_merged: {df['defendant_rep_merged'].sum()}/{len(df)} ({100*df['defendant_rep_merged'].mean():.1f}%)")

# does representation correlate with appearance/attendance?
rep = df[df['defendant_rep_merged'] == 1]
print(f"  of those with rep:")
print(f"    appearance: {rep['defendant_appearance'].sum()}/{len(rep)} ({100*rep['defendant_appearance'].mean():.1f}%)")
print(f"    hearing attendance: {rep['defendant_hearing_attendance'].sum()}/{len(rep)} ({100*rep['defendant_hearing_attendance'].mean():.1f}%)")

# stage 3: case outcomes
print(f"\n\nSTAGE 3: Case Outcomes")
print("-" * 70)

outcomes = ['writ_final', 'dismissal_final', 'old_final', 'court_displacement']
for outcome in outcomes:
    count = df[outcome].sum()
    pct = 100 * df[outcome].mean()
    print(f"{outcome:30s}: {count:4d}/{len(df)} ({pct:5.1f}%)")

# check if cases have multiple outcomes
writ_cases = df[df['writ_final'] == 1]
dismissal_cases = df[df['dismissal_final'] == 1]
old_cases = df[df['old_final'] == 1]

print(f"writ_final AND dismissal_final: {((writ_cases['dismissal_final'] == 1).sum())} cases")
print(f"writ_final AND old_final: {((writ_cases['old_final'] == 1).sum())} cases")
print(f"dismissal_final AND old_final: {((dismissal_cases['old_final'] == 1).sum())} cases")

# can displacement happen with any outcome?
print(f"writ_final cases with displacement: {writ_cases['court_displacement'].sum()}/{len(writ_cases)} ({100*writ_cases['court_displacement'].mean():.1f}%)")
print(f"dismissal_final cases with displacement: {dismissal_cases['court_displacement'].sum()}/{len(dismissal_cases)} ({100*dismissal_cases['court_displacement'].mean():.1f}%)")
print(f"old_final cases with displacement: {old_cases['court_displacement'].sum()}/{len(old_cases)} ({100*old_cases['court_displacement'].mean():.1f}%)")

no_outcome = df[(df['writ_final'] == 0) & (df['dismissal_final'] == 0) & (df['old_final'] == 0)]
print(f"cases with none of [writ/dismissal/old]: {len(no_outcome)} ({100*len(no_outcome)/len(df):.1f}%)")
if len(no_outcome) > 0:
    print(f"  of those, displacement: {no_outcome['court_displacement'].sum()}/{len(no_outcome)} ({100*no_outcome['court_displacement'].mean():.1f}%)")

### correlation matrix ###

corr_matrix = df[target_columns].corr()
print(corr_matrix.round(3))

### representation impact analysis ###

outcomes_to_check = ['writ_final', 'dismissal_final', 'old_final', 'court_displacement']

with_rep = df[df['defendant_rep_merged'] == 1]
without_rep = df[df['defendant_rep_merged'] == 0]

print(f"\nwith vs without representation:")
print(f"with representation (n={len(with_rep)}):")
for outcome in outcomes_to_check:
    pct = 100 * with_rep[outcome].mean()
    print(f"  {outcome:30s}: {pct:5.1f}%")

print(f"\nwithout representation (n={len(without_rep)}):")
for outcome in outcomes_to_check:
    pct = 100 * without_rep[outcome].mean()
    print(f"  {outcome:30s}: {pct:5.1f}%")

print(f"\nwith rep - without rep):")
for outcome in outcomes_to_check:
    with_rep_rate = with_rep[outcome].mean()
    without_rep_rate = without_rep[outcome].mean()
    diff = 100 * (with_rep_rate - without_rep_rate)
    print(f"  {outcome:30s}: {diff:+6.1f} percentage points")

# conditional on appearance and attendance
appeared_and_attended = df[
    (df['defendant_appearance'] == 1) & 
    (df['defendant_hearing_attendance'] == 1)
]

print(f"\n\namong cases with appearance AND hearing attendance (n={len(appeared_and_attended)}):")

with_rep_sub = appeared_and_attended[appeared_and_attended['defendant_rep_merged'] == 1]
without_rep_sub = appeared_and_attended[appeared_and_attended['defendant_rep_merged'] == 0]

print(f"\nwith rep vs without rep:")
for outcome in outcomes_to_check:
    with_rep_rate = with_rep_sub[outcome].mean()
    without_rep_rate = without_rep_sub[outcome].mean()
    diff = 100 * (with_rep_rate - without_rep_rate)
    print(f"  {outcome:30s}: {diff:+6.1f} percentage points")