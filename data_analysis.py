from data_extraction import load_data

def analyze_data():
  train_data = load_data()

  print(train_data.head())
  print(train_data.describe())
  print(train_data.info())

  print(f"\nData Features: {train_data.columns}") # Lists data features

  print("\nEmpty Values: ")
  print(train_data.isnull().sum()) # Number of null values per feature

  print("\nDuplicated Values: ")
  for column in train_data.columns:
    print(column, train_data[column].duplicated().sum()) # Number of duplicated values per feature

  print("\nUnique Values: ")
  for column in train_data.columns:
    print(column, train_data[column].nunique()) # Number of unique values per feature

  return train_data

analyze_data()
