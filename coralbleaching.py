import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
file_path = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
df = pd.read_csv(file_path)

df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace(r'\W+', '_', regex=True)
)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Temperature_Mean'] = pd.to_numeric(df['Temperature_Mean'], errors='coerce')
df['Percent_Bleaching'] = pd.to_numeric(df['Percent_Bleaching'], errors='coerce')
df['Depth_m'] = pd.to_numeric(df['Depth_m'], errors='coerce')

df_filtered = df[
    (df['Depth_m'] >= 0) &
    (df['Depth_m'] <= 5) &
    (df['Date'].dt.year.isin([2019, 2020])) &
    (df['Country_Name'].isin(['Indonesia', 'Philippines']))
].copy()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_filtered,
    x='Temperature_Mean',
    y='Percent_Bleaching',
    hue='Country_Name',
    s=100
)
plt.title('Bleaching at 0–3m (2019–2020)')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('% Bleaching')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.lineplot(
    data=df_filtered,
    x='Date',
    y='Percent_Bleaching',
    hue='Country_Name',
    marker='o'
)
plt.title('Bleaching Over Time (0–3m)')
plt.xlabel('Survey Date')
plt.ylabel('% Bleaching')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

sns.lmplot(
    data=df_filtered,
    x='Temperature_Mean',
    y='Percent_Bleaching',
    hue='Country_Name',
    aspect=1.5,
    height=6,
    ci=None,
    scatter_kws={'alpha':0.5, 's':80},
    line_kws={'linewidth':2.5}
)
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('% Bleaching')
plt.title('Bleaching Trend vs Sea Temperature (0–5m, 2019–2020)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_filtered,
    x='Temperature_Mean',
    y='Percent_Bleaching',
    hue='Country_Name',
    s=100,
    palette='Set2'
)
plt.title('Coral Bleaching (0–5m, 2019–2020)')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('% Bleaching')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

df_filtered['Month'] = df_filtered['Date'].dt.to_period('M')
monthly_avg = df_filtered.groupby(['Month','Country_Name'])['Percent_Bleaching'].mean().reset_index()
monthly_avg['Month'] = monthly_avg['Month'].dt.to_timestamp()
month_range = pd.date_range(start='2019-01-01', end='2019-12-31', freq='MS')

plt.figure(figsize=(14,6))
sns.lineplot(
    data=monthly_avg,
    x='Month',
    y='Percent_Bleaching',
    hue='Country_Name',
    marker='o'
)
plt.xticks(month_range, [d.strftime('%b %Y') for d in month_range], rotation=45)
plt.title('Monthly Average Bleaching (0–5m, 2019)')
plt.xlabel('Month')
plt.ylabel('% Bleaching (Mean)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(
    data=df_filtered,
    x='Country_Name',
    y='Temperature_Mean',
    palette='pastel'
)
plt.title('Distribution of Mean Sea Temperature (0–5m, 2019–2020)')
plt.xlabel('Country')
plt.ylabel('Temperature (°C)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_filtered,
    x='Longitude_Degrees',
    y='Latitude_Degrees',
    hue='Percent_Bleaching',
    size='Percent_Bleaching',
    sizes=(20,200),
    palette='coolwarm',
    alpha=0.7,
    edgecolor='gray'
)
plt.title('Bleaching by Location (0–5m, 2019–2020)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
