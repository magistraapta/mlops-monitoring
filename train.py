import mlflow

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=1)

model.fit(X_train, y_train)


