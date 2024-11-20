import pandas as pd
import numpy as np

# Load the election data
election = pd.read_csv("ElectionResults.csv")

# Convert rank columns to integer
for col in ['R', 'D', 'L', 'G', 'Adithi', 'Srinjoy', 'Tracy', 'Hunter', 'Archith']:
    election[col] = election[col].str.extract(r'(\d+)').astype(float).astype('Int64')

# Define isSenior and MF columns based on wing information
senior_wings = ["C100", "A100", "D100A", "D100B", "D200A", "D200B", "D300A", "D300B"]
junior_wings = ["A200", "C200A", "C200B", "C300A", "C300B", "B200", "A300", "B300"]
male_wings = ["C100", "D100A", "D100B", "A300", "B300", "C300A", "C300B", "D300A", "D300B"]
female_wings = ["A100", "C200A", "C200B", "B200", "A200", "D200A", "D200B"]

# Assign values for isSenior and MF
election['isSenior'] = np.where(election['Wing'].isin(senior_wings), True,
                        np.where(election['Wing'].isin(junior_wings), False,
                                 np.random.choice([True, False], len(election))))

election['MF'] = np.where(election['Wing'].isin(female_wings), 'F',
                 np.where(election['Wing'].isin(male_wings), 'M',
                          np.random.choice(['M', 'F'], len(election))))

# Summarize counts of seniors, juniors, guys, and girls
summary_counts = {
    "num_seniors": election['isSenior'].sum(),
    "num_juniors": len(election) - election['isSenior'].sum(),
    "num_guys": (election['MF'] == 'M').sum(),
    "num_girls": (election['MF'] == 'F').sum()
}
num_seniors = summary_counts['num_seniors']
num_juniors = summary_counts['num_juniors']
num_guys = summary_counts['num_guys']
num_girls = summary_counts['num_girls']

detailed_counts = election.groupby(['isSenior', 'MF']).size().unstack(fill_value=0)

# Rename columns for clarity
detailed_counts = detailed_counts.rename(columns={True: 'Senior', False: 'Junior'}, index={'M': 'Guys', 'F': 'Girls'})

# Flatten the result to create named columns for each group
# Ensure expected labels exist, defaulting to 0 if missing
counts_dict = {
    "Senior_Guys": detailed_counts.loc['Guys', 'Senior'] if ('Guys' in detailed_counts.index and 'Senior' in detailed_counts.columns) else 0,
    "Senior_Girls": detailed_counts.loc['Girls', 'Senior'] if ('Girls' in detailed_counts.index and 'Senior' in detailed_counts.columns) else 0,
    "Junior_Guys": detailed_counts.loc['Guys', 'Junior'] if ('Guys' in detailed_counts.index and 'Junior' in detailed_counts.columns) else 0,
    "Junior_Girls": detailed_counts.loc['Girls', 'Junior'] if ('Girls' in detailed_counts.index and 'Junior' in detailed_counts.columns) else 0
}

# Convert to a DataFrame for easier viewing (optional)
counts_df = pd.DataFrame([counts_dict])

print(counts_df)
# Separate issues into long format
issues_long = election.assign(Issues=election['Issues'].str.split(',')).explode('Issues')
issues_long['Issues'] = issues_long['Issues'].str.strip()

# Count issues by Senior vs Junior
issue_counts_class = issues_long.groupby('Issues').apply(
    lambda x: pd.Series({
        "Senior_Frequency": round(100 * x['isSenior'].sum() / num_seniors, 1),
        "Junior_Frequency": round(100 * (~x['isSenior']).sum() / num_juniors, 1)
    })
).reset_index()

# Count issues by Gender (MF)
issue_counts_gender = issues_long.groupby('Issues').apply(
    lambda x: pd.Series({
        "Guys_Frequency": round(100 * (x['MF'] == 'M').sum() / num_guys, 1),
        "Girls_Frequency": round(100 * (x['MF'] == 'F').sum() / num_girls, 1)
    })
).reset_index()

# Reshape party rankings to long format and calculate average rank for Senior vs Junior and Guys vs Girls
ranking_long = election.melt(
    id_vars=['Wing', 'isSenior', 'MF'], 
    value_vars=['R', 'D', 'L', 'G'], 
    var_name='party', 
    value_name='rank'
).dropna()

# Average rank for Senior vs. Junior
senior_junior_rankings = ranking_long.groupby(['isSenior', 'party']).agg(avg_rank=('rank', 'mean')).reset_index()
senior_junior_rankings = senior_junior_rankings.sort_values(['isSenior', 'avg_rank'])

# Average rank for Guys vs. Girls
gender_rankings = ranking_long.groupby(['MF', 'party']).agg(avg_rank=('rank', 'mean')).reset_index()
gender_rankings = gender_rankings.sort_values(['MF', 'avg_rank'])

# FPTP analysis for President and Senator
fptp_long = election.melt(
    id_vars=['isSenior', 'MF'], 
    value_vars=['President', 'Senator'], 
    var_name='position', 
    value_name='choice'
)

# Senior vs Junior FPTP choices
senior_junior_fptp = fptp_long.groupby(['position', 'isSenior', 'choice']).size().reset_index(name='vote_count')
senior_junior_fptp = senior_junior_fptp.sort_values(['position', 'isSenior', 'vote_count'], ascending=[True, True, False])

# Guys vs Girls FPTP choices
gender_fptp = fptp_long.groupby(['position', 'MF', 'choice']).size().reset_index(name='vote_count')
gender_fptp = gender_fptp.sort_values(['position', 'MF', 'vote_count'], ascending=[True, True, False])

# Reformat FPTP choices to wide format
presidential_fptp_class = senior_junior_fptp[senior_junior_fptp['position'] == 'President'].pivot_table(
    index='choice', columns='isSenior', values='vote_count', fill_value=0
).rename(columns={True: 'Senior', False: 'Junior'}).reset_index()

presidential_fptp_gender = gender_fptp[gender_fptp['position'] == 'President'].pivot_table(
    index='choice', columns='MF', values='vote_count', fill_value=0
).rename(columns={'M': 'Guys', 'F': 'Girls'}).reset_index()

senate_fptp_class = senior_junior_fptp[senior_junior_fptp['position'] == 'Senator'].pivot_table(
    index='choice', columns='isSenior', values='vote_count', fill_value=0
).rename(columns={True: 'Senior', False: 'Junior'}).reset_index()

senate_fptp_gender = gender_fptp[gender_fptp['position'] == 'Senator'].pivot_table(
    index='choice', columns='MF', values='vote_count', fill_value=0
).rename(columns={'M': 'Guys', 'F': 'Girls'}).reset_index()

# Display results
results = {
    "Senior vs Junior Average Party Rankings": senior_junior_rankings,
    "Guys vs Girls Average Party Rankings": gender_rankings,
    "Senior vs Junior President Choices": presidential_fptp_class,
    "Senior vs Junior Senate Choices": senate_fptp_class,
    "Guys vs Girls President Choices": presidential_fptp_gender,
    "Senior vs Junior Senate Choices": senate_fptp_gender
}

for key, value in results.items():
    print(f"\n{key}:\n", value)
