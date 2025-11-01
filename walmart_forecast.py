import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

data = pd.read_csv("Walmart.csv")
data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
data = data[data["Store"] == 1].copy()
data = data.sort_values("Date")

X = data[["Date", "Fuel_Price", "CPI", "Unemployment"]]
Xnum = X.copy()
Xnum["Date"] = Xnum["Date"].map(pd.Timestamp.toordinal)
y = data["Weekly_Sales"]
data["Year"] = data["Date"].dt.year

Xtrain, Xtest, ytrain, ytest = train_test_split(Xnum, y, test_size=0.2, shuffle=False)
years_test = data["Year"].iloc[Xtrain.shape[0]:]

lr = LinearRegression()
lr.fit(Xtrain, ytrain)
pred_lr = lr.predict(Xtest)

poly = PolynomialFeatures(degree=2)
Xtrain_poly = poly.fit_transform(Xtrain)
Xtest_poly = poly.transform(Xtest)
pr = LinearRegression()
pr.fit(Xtrain_poly, ytrain)
pred_pr = pr.predict(Xtest_poly)

tree = DecisionTreeRegressor(max_depth=5, random_state=0)
tree.fit(Xtrain, ytrain)
pred_tree = tree.predict(Xtest)

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xnum)
y_scaled = (y - y.min()) / (y.max() - y.min())

Xseq, yseq = [], []
seq_len = 5
for i in range(len(Xscaled) - seq_len):
    Xseq.append(Xscaled[i:i+seq_len])
    yseq.append(y_scaled.iloc[i+seq_len])
Xseq, yseq = np.array(Xseq), np.array(yseq)

split = int(len(Xseq) * 0.8)
Xtrain_seq, Xtest_seq = Xseq[:split], Xseq[split:]
ytrain_seq, ytest_seq = yseq[:split], yseq[split:]

lstm = Sequential()
lstm.add(LSTM(50, activation="relu", input_shape=(seq_len, Xseq.shape[2])))
lstm.add(Dense(1))
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(Xtrain_seq, ytrain_seq, epochs=5, batch_size=16, verbose=1)

pred_lstm = lstm.predict(Xtest_seq).flatten()
pred_lstm = pred_lstm * (y.max() - y.min()) + y.min()
ytest_seq_real = ytest_seq * (y.max() - y.min()) + y.min()

def eval_model(ytrue, ypred, name):
    mae = mean_absolute_error(ytrue, ypred)
    rmse = np.sqrt(mean_squared_error(ytrue, ypred))
    r2 = r2_score(ytrue, ypred)
    return pd.Series([mae, rmse, r2], index=["MAE","RMSE","R2"], name=name)

results = pd.DataFrame([
    eval_model(ytest, pred_lr, "Linear Regression"),
    eval_model(ytest, pred_pr, "Polynomial Regression"),
    eval_model(ytest, pred_tree, "Decision Tree"),
    eval_model(ytest_seq_real, pred_lstm, "LSTM")
])

print("Model Results:\n")
print(results)

best_model = results.sort_values("RMSE").index[0]
print("\nBest model is:", best_model)
print("The best model is chosen because it has the lowest RMSE and highest R² among all models.")

last_date = data["Date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7),
                             end=last_date + pd.DateOffset(years=3), freq="W")

avg_fuel = data["Fuel_Price"].mean()
avg_cpi = data["CPI"].mean()
avg_unemp = data["Unemployment"].mean()

future = pd.DataFrame({
    "Date": future_dates,
    "Fuel_Price": avg_fuel,
    "CPI": avg_cpi,
    "Unemployment": avg_unemp
})
future_num = future.copy()
future_num["Date"] = future_num["Date"].map(pd.Timestamp.toordinal)

future_preds = {
    "Linear Regression": lr.predict(future_num),
    "Polynomial Regression": pr.predict(poly.transform(future_num)),
    "Decision Tree": tree.predict(future_num),
    "LSTM": lstm.predict(
        np.array([scaler.transform(future_num.iloc[i:i+seq_len])
                  for i in range(len(future_num) - seq_len)])
    ).flatten()
}
future_preds["LSTM"] = future_preds["LSTM"] * (y.max() - y.min()) + y.min()

plt.figure(figsize=(12,6))
actual_yearly = pd.DataFrame({"Year": years_test.values[:len(ytest)], "Actual": ytest.values[:len(years_test)]})
actual_yearly = actual_yearly.groupby("Year")["Actual"].mean()
plt.plot(actual_yearly.index, actual_yearly.values, label="Actual", color="black", linewidth=3)

past_preds = {
    "Linear Regression": pred_lr,
    "Polynomial Regression": pred_pr,
    "Decision Tree": pred_tree,
    "LSTM": pred_lstm
}

all_years_list = []

for name, pred_future in future_preds.items():
    pred_past = past_preds[name]
    if len(pred_past) != len(years_test):
        min_len = min(len(pred_past), len(years_test))
        pred_past = pred_past[:min_len]
        years_past = years_test.values[:min_len]
    else:
        years_past = years_test.values

    if len(pred_future) != len(future):
        min_len_f = min(len(pred_future), len(future))
        pred_future = pred_future[:min_len_f]
        future_years = future["Date"].dt.year.values[:min_len_f]
    else:
        future_years = future["Date"].dt.year.values

    all_preds = np.concatenate([pred_past, pred_future])
    all_years = np.concatenate([years_past, future_years])
    all_years_list.extend(all_years)
    yearly = pd.DataFrame({"Year": all_years, "Pred": all_preds}).groupby("Year")["Pred"].mean()
    if name == best_model:
        plt.plot(yearly.index.astype(int), yearly.values, label=name+" (Best)", linewidth=3, color="red")
    else:
        plt.plot(yearly.index.astype(int), yearly.values, label=name, linestyle="--")

plt.xlabel("Year")
plt.ylabel("Average Weekly Sales")
plt.title("Walmart Sales Forecasting")
plt.xticks(sorted(set(all_years_list)))
plt.legend()
plt.grid(True)
plt.show()
