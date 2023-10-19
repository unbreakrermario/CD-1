import pandas as pd

d1 = {'Name': ['Pankaj', 'Meghna', 'Lisa'], 'Country': ['India', 'India', 'USA'], 'Role': ['CEO', 'CTO', 'CTO']}
df1 = pd.DataFrame(d1)
print('DataFrame 1:\n', df1)
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Pankaj', 'Anupam', 'Amit']})
print('DataFrame 2:\n', df2)

# INNER JOIN
df_merged = df1.merge(df2)
print('Result:\n', df_merged)
# LEFT
print('Result Left Join:\n', df1.merge(df2, how='left'))
# RIGHT
print('Result Right Join:\n', df1.merge(df2, how='right'))
# OUTER JOIN
print('Result Outer Join:\n', df1.merge(df2, how='outer'))

# USING COLUMNS TO MERGE
d1 = {'Name': ['Pankaj', 'Meghna', 'Lisa'], 'ID': [1, 2, 3], 'Country': ['India', 'India', 'USA'],
      'Role': ['CEO', 'CTO', 'CTO']}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Pankaj', 'Anupam', 'Amit']})

print('Result ID column:\n', df1.merge(df2, on='ID'))
print('Result Name column:\n', df1.merge(df2, on='Name'))
