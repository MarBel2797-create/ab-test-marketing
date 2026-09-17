import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('marketing_AB.csv')
df.columns = df.columns.str.replace(' ', '_')

print("Размер данных:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nСколько пользователей в каждой группе:")
print(df['test_group'].value_counts())

print("\nОбщая конверсия по группам:")
print(df.groupby('test_group')['converted'].mean())

from statsmodels.stats.proportion import proportions_ztest
plt.figure(figsize=(8, 5))
sns.barplot(x='test_group', y='converted', data=df, errorbar=None)
plt.title('Конверсия в покупку по группам')
plt.ylabel('Доля купивших')
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(x='most_ads_day', y='converted', hue='test_group', data=df, errorbar=None)
plt.title('Конверсия по дням недели')
plt.xticks(rotation=45)
plt.show()

conversions = df.groupby('test_group')['converted'].sum()
nobs = df.groupby('test_group')['converted'].count()
z_stat, p_value = proportions_ztest(conversions, nobs)

print(f"Z-статистика: {z_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Результат статистически значим! Реклама работает.")
else:
    print("❌ Результат незначим. Разница может быть случайной.")

conv_ad = df[df['test_group'] == 'ad']['converted'].mean()
conv_psa = df[df['test_group'] == 'psa']['converted'].mean()

absolute_lift = conv_ad - conv_psa
relative_lift = (conv_ad / conv_psa - 1) * 100

print(f"Конверсия в группе 'ad': {conv_ad:.2%}")
print(f"Конверсия в группе 'psa': {conv_psa:.2%}")
print(f"Абсолютный прирост: {absolute_lift:.2%} п.п.")
print(f"Относительный прирост: {relative_lift:.1f}%")
