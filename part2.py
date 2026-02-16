import numpy as np #numbers
import pandas as pd # data
import matplotlib.pyplot as plt #plots

from sklearn.model_selection import train_test_split #data split
from sklearn.preprocessing import StandardScaler #standardizer
from sklearn.linear_model import SGDRegressor #linear regression w/ gradient descent
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

#loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")


#print("Shape:", df.shape)
#print("\nMissing values:\n", df.isnull().sum())
#print("\nDuplicate rows:", df.duplicated().sum())

#removing duplicates only since there are no missing values
df = df.drop_duplicates()

#defining features and targets

X = df.drop("quality", axis=1)
y = df["quality"]

#standardizes features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Splits between training and testing data
#80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#print("# of training samples:", X_train.shape[0])
#print("# of Test samples:", X_test.shape[0])

# tuning parameters
learning_rates = [0.0001, 0.001, 0.01, 0.1]
iterations = [500, 1000, 2000]

results = []

best_test_mse = float("inf")
best_params = None

with open("part2_log.txt", "w") as f:

    for lr in learning_rates:
        for it in iterations:

            model = SGDRegressor(
                learning_rate='constant',
                eta0=lr,
                max_iter=it,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)#makes predictions
            y_test_pred = model.predict(X_test)
            
            #compute errors
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            evs = explained_variance_score(y_test, y_test_pred)

            results.append([lr, it, train_mse, test_mse])

            # writing each trial
            f.write(
                f"LearningRate={lr}, Iterations={it}, "
                f"Train MSE={train_mse:.4f}, "
                f"Test MSE={test_mse:.4f}, "
                f"R2={r2:.4f}, "
                f"EVS={evs:.4f}\n"
            )

            # checks if model is the best
            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_params = (lr, it)

    # Write best results at end 
    f.write("\nBest Parameters:\n")
    f.write(f"Learning Rate: {best_params[0]}\n")
    f.write(f"Iterations: {best_params[1]}\n")
    f.write(f"Best Test MSE: {best_test_mse:.4f}\n")


#building final model

#extracts best learning rate
#best_params = (best_learning_rate, best_iterations)
best_lr = best_params[0]
best_iter = best_params[1]

final_model = SGDRegressor(
    learning_rate='constant',
    eta0=best_lr,
    max_iter=best_iter,
    random_state=42
)

final_model.fit(X_train, y_train) #trains final model

#make predictions
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

#calculate training MSE
train_mse = mean_squared_error(y_train, y_train_pred)
#calculate test MSE
test_mse = mean_squared_error(y_test, y_test_pred)

# 4. Actual vs Predicted plot
plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.savefig("actual_vs_predicted.png")
#plt.show()

# 5. Feature coefficients plot
coefficients = pd.Series(final_model.coef_, index=X.columns)
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients")
plt.savefig("feature_coefficients.png")
#plt.show()



#calculate R^2 Score
r2 = r2_score(y_test, y_test_pred)

#calculate Explained Variance
evs = explained_variance_score(y_test, y_test_pred)

#Print results
#print("Train MSE:", train_mse)
#print("Test MSE:", test_mse)
#print("R2 Score:", r2)
#print("Explained Variance:", evs)
