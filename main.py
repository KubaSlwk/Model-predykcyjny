"""
Jakub Kucharzewski
Projekt - Prognozowanie zużycia energii przy pomocy Decision Tree, Random Forest oraz XGBoost
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor




# Wczytanie danych z CSV
energy_data = pd.read_csv('Energy_consumption_dataset.csv')

# Sprawdzenie typow danych
energy_data.info()

print(energy_data.head())

pd.set_option('future.no_silent_downcasting', True)

# Konwersja zmiennych binarnych na numeryczne
binary_columns = {'Holiday': {"Yes": 1, "No": 0}, 'HVACUsage': {"On": 1, "Off": 0}, 'LightingUsage': {"On": 1, "Off": 0}}
for col, mapping in binary_columns.items():
    energy_data[col] = energy_data[col].replace(mapping).astype(int)

# Oznaczenie dni tygodnia liczbami
energy_data['DayOfWeek'] = energy_data['DayOfWeek'].replace({"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,  "Saturday": 5, "Sunday": 6,}).astype(int)

#Srednie zuzycie energii w miesiacu
month_avg_consumption = energy_data.groupby('Month')['EnergyConsumption'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(month_avg_consumption['Month'], month_avg_consumption['EnergyConsumption'], marker='o', linestyle='-', color='darkblue')
plt.title('Średnie zużycie energii w miesiącu', fontsize=20)
plt.xlabel('Miesiąc', fontsize=16)
plt.ylabel('Średnie zużycie energii', fontsize=16)
plt.xticks(range(0, 13))
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.show()

#Srednie zuzycie energii w godzine dnia
hour_avg_consumption = energy_data.groupby('Hour')['EnergyConsumption'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(hour_avg_consumption['Hour'], hour_avg_consumption['EnergyConsumption'], marker='o', linestyle='-', color='darkgreen')
plt.title('Średnie zużycie energii w godzinach dnia', fontsize=20)
plt.xlabel('Godziny', fontsize=16)
plt.ylabel('Średnie zużycie energii', fontsize=16)
plt.xticks(range(0, 25))
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.show()


#Srednie zuzycie energii w dniu tygodnia
day_avg_consumption = energy_data.groupby('DayOfWeek')['EnergyConsumption'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(day_avg_consumption['DayOfWeek'], day_avg_consumption['EnergyConsumption'], marker='o', linestyle='-', color='darkred')
plt.title('Średnie zużycie energii w dniach tygodnia', fontsize=20)
plt.xlabel('Dni tygodnia', fontsize=16)
plt.ylabel('Średnie zużycie energiin', fontsize=16)
plt.xticks(range(0, 8))
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.show()


energy_data = energy_data.drop(columns='DayOfWeek')
print(energy_data.columns)

#Miesiące przekształca na porę roku do której dany miesiąc należy. 1 - zima, 2 - wiosna itd.
def pora_roku(month):
    if 1 <= month < 3 or month == 12:
        return '1'
    elif 3 <= month < 7:
        return '2'
    elif 7 <= month < 10:
        return '3'
    else:
        return '4'

#Godziny przekształca na pory dnia, do ktorej dana godzina należy. 1 - rano, 2 - popołudnie itd.
def pora_dnia(hour):
    if 4 <= hour < 12:
        return '1'
    elif 12 <= hour < 16:
        return '2'
    elif 16 <= hour < 21:
        return '3'
    else:
        return '4'

#Przekształacam zmienne przy pomocy powyższych funkcji.
energy_data['TimeOfDay'] = energy_data['Hour'].apply(pora_dnia)
energy_data['Season'] = energy_data['Month'].apply(pora_roku)
energy_data = energy_data.drop('Hour', axis=1)
energy_data = energy_data.drop('Month', axis=1)
print(energy_data.columns)
print(energy_data[['TimeOfDay', 'Season']])

#One-Hot encoding
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(energy_data[['TimeOfDay']]))
OH_cols1 = pd.DataFrame(OH_encoder.fit_transform(energy_data[['Season']]))
OH_cols.columns = [f"TimeOfDay_{i}" for i in range(OH_cols.shape[1])]
OH_cols1.columns = [f"Season_{i}" for i in range(OH_cols.shape[1])]
OH_cols.index = energy_data.index
OH_cols1.index = energy_data.index

#Złączenie powstałych tabel oraz usunięcie tych pierwotnych.
energy_data_oh = pd.concat([energy_data, OH_cols, OH_cols1], axis=1)
energy_data_oh = energy_data_oh.drop('TimeOfDay', axis=1)
energy_data_oh = energy_data_oh.drop('Season', axis=1)


# Definicja cech i zmiennej docelowej
features = [col for col in energy_data_oh.columns if col != 'EnergyConsumption']
X = energy_data_oh[features]
y = energy_data_oh['EnergyConsumption']

# Podział na zestawy treningowy i testowy
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)


# Funkcja do oceny modeli
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return mae, r2, mse
# One-hot encoding dla 'DayOfWeek'

# Funkcja do testowania różnych modeli i parametrów
def test_models(model_class, param_name, param_values, X_train, X_test, y_train, y_test, model_name="Model"):
    print(f"\n{model_name}:")
    for param in param_values:
        model = model_class(**{param_name: param}, random_state=0) if model_class != XGBRegressor else model_class(**{param_name: param, 'learning_rate': 0.1, 'n_jobs': 4})
        mae, r2, mse = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{param}: MAE = {mae:.4f}, MSE = {mse:.4f}, R2 = {r2:.4f}")

# Decison Tree
leafs = [10, 20, 30, 50, 100, 150]
test_models(DecisionTreeRegressor, "max_leaf_nodes", leafs, train_X, test_X, train_y, test_y, "Decision Tree")


# Random Forest
estimators = [100, 200, 250, 325]
test_models(RandomForestRegressor, "n_estimators", estimators, train_X, test_X, train_y, test_y, "Random Forest")


# XGBoost
estimators = [10, 20, 30, 40, 45]
test_models(XGBRegressor, "n_estimators", estimators, train_X, test_X, train_y, test_y, "XGBoost")


#Najlepszy model
best_model = RandomForestRegressor(max_leaf_nodes=325, random_state=0)

best_model.fit(train_X, train_y)
predcitions = best_model.predict(test_X)
mae = mean_absolute_error(test_y, predcitions)
mse = mean_squared_error(test_y, predcitions)
r2 = r2_score(test_y, predcitions)
print(f"MAE: {mae}, MSE: {mse}, R Squared: {r2}")

# Tworzenie DataFrame do porównania
comparison_df = pd.DataFrame({'Actual': test_y, 'Predicted': predcitions})

# Wyświetlanie pierwszych 50 wierszy
print(comparison_df.head())

limited_test_y = test_y[:50]
limited_predictions = predcitions[:50]

plt.figure(figsize=(10, 6))
plt.plot(limited_test_y.values, label='Actual', marker='o', color='blue')
plt.plot(limited_predictions, label='Predicted', marker='o', color='lightgreen')
plt.xlabel('Observation Index')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()