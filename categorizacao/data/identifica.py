import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "resumos.csv")

df = pd.read_csv(csv_path)

categorias_unicas = df['category'].unique()

print("Categorias encontradas:")
for categoria in categorias_unicas:
    print("-", categoria)