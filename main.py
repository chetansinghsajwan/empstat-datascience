# %%
# Loading Datasets

from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset(name: str, print_info: bool = False) -> Optional[pd.DataFrame]:
    try:
        print(f"loading {name}...")

        path = f"inputs/{name}.csv"
        df = pd.read_csv(path)
        count = len(df)

        print(f"loading {name} done, count: {count}")

        if print_info:
            print(df.info())

        return df
    except Exception as ex:
        print(f"loading {name} failed, error: {ex}")
        return None


users = load_dataset("users")
subjects = load_dataset("subjects")
trainings = load_dataset("trainings")
assessments = load_dataset("assessments")

# merging datasets

df = (
    assessments.merge(
        users, left_on="user_id", right_on="id", how="inner", suffixes=("", "_user")
    )
    .merge(
        trainings,
        left_on="training_id",
        right_on="id",
        how="inner",
        suffixes=("", "_training"),
    )
    .merge(
        subjects,
        left_on="subject_id",
        right_on="id",
        how="inner",
        suffixes=("", "_subject"),
    )
)

# droping redundant columns after merging
df = df.drop(
    columns=[
        # user id
        "id",
        "id_subject",
        "id_training",
    ]
)

# renaming columns with resource prefix
df = df.rename(
    columns={
        "email": "user_email",
        "first_name": "user_first_name",
        "middle_name": "user_middle_name",
        "last_name": "user_last_name",
        "role": "user_role",
        "created_at": "user_created_at",
        "updated_at": "user_updated_at",
        # subject columns
        "name_subject": "subject_name",
        "created_at_subject": "subject_created_at",
        "updated_at_subject": "subject_updated_at",
        "min_marks": "subject_min_marks",
        "max_marks": "subject_max_marks",
        # training columns
        "id_training": "training_id",
        "name": "training_name",
        "mode": "training_mode",
        "total_time": "subject_total_time",
        "started_at": "training_started_at",
        "ended_at": "training_ended_at",
        "created_at_training": "training_created_at",
        "updated_at_training": "training_updated_at",
        "name_subject": "subject_name",
        # assessment columns
        "internet_allowed": "assessment_internet_allowed",
        "marks": "assessment_marks",
    }
)

# set correct data type

# string types
df["user_id"] = df["user_id"].astype("string")
df["training_id"] = df["training_id"].astype("string")
df["user_email"] = df["user_email"].astype("string")
df["user_first_name"] = df["user_first_name"].astype("string")
df["user_middle_name"] = df["user_middle_name"].astype("string")
df["user_last_name"] = df["user_last_name"].astype("string")
df["user_role"] = df["user_role"].astype("string")
df["training_name"] = df["training_name"].astype("string")
df["training_mode"] = df["training_mode"].astype("string")
df["subject_id"] = df["subject_id"].astype("string")
df["subject_name"] = df["subject_name"].astype("string")

# int types
df["assessment_marks"] = (
    pd.to_numeric(df["assessment_marks"], errors="coerce").fillna(0).astype(int)
)
df["subject_min_marks"] = (
    pd.to_numeric(df["subject_min_marks"], errors="coerce").fillna(0).astype(int)
)
df["subject_max_marks"] = (
    pd.to_numeric(df["subject_max_marks"], errors="coerce").fillna(0).astype(int)
)
df["subject_total_time"] = (
    pd.to_numeric(df["subject_total_time"], errors="coerce").fillna(0).astype(int)
)

# datetime types
df["user_created_at"] = pd.to_datetime(df["user_created_at"], errors="coerce")
df["user_updated_at"] = pd.to_datetime(df["user_updated_at"], errors="coerce")
df["subject_created_at"] = pd.to_datetime(df["subject_created_at"], errors="coerce")
df["subject_updated_at"] = pd.to_datetime(df["subject_updated_at"], errors="coerce")
df["training_started_at"] = pd.to_datetime(df["training_started_at"], errors="coerce")
df["training_ended_at"] = pd.to_datetime(df["training_ended_at"], errors="coerce")
df["training_created_at"] = pd.to_datetime(df["training_created_at"], errors="coerce")
df["training_updated_at"] = pd.to_datetime(df["training_updated_at"], errors="coerce")

# boolean types
df["assessment_internet_allowed"] = df["assessment_internet_allowed"].astype(bool)
df["promoted"] = df["promoted"].astype(bool)


print("merged dataset:")
print(df.info())

# %%
# Correlation mapping

corr_fields = [
    "assessment_marks",
    "promoted",
    "user_role",
    "training_mode",
    "subject_min_marks",
    "subject_max_marks",
    "subject_total_time",
]

corr_df = df[corr_fields]
corr_df = pd.get_dummies(corr_df)
corr_mat = corr_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()

# %%
# Feature Engineering

# Total unique trainings attended
df["total_trainings_attended"] = df.groupby("user_id")["training_id"].transform(
    "nunique"
)

# Range of marks achievable
df["subject_marks_range"] = df["subject_max_marks"] - df["subject_min_marks"]

# Time it took to complete the training
df["training_time_range"] = (
    pd.to_datetime(df["training_ended_at"]) - pd.to_datetime(df["training_started_at"])
).dt.days

# How old the training is
df["training_age"] = (
    pd.to_datetime("today") - pd.to_datetime(df["training_created_at"])
).dt.days

# Average score across assessments
df["average_score"] = df.groupby("user_id")["assessment_marks"].transform("mean")

# Completion rate for the most recent training (if applicable)
df["completion_rate"] = df["assessment_marks"] / df["subject_max_marks"]

# Create features for retention
df["user_service_age"] = (pd.to_datetime("today") - df["user_created_at"]).dt.days

print("dataset after feature engineering:")
print(df.info())

# %%
# Feature Selection

selected_columns = [
    "user_role",
    "user_service_age",
    "subject_total_time",
    "subject_marks_range",
    "training_time_range",
    "training_age",
    "training_mode",
    "total_trainings_attended",
    "completion_rate",
    "assessment_internet_allowed",
    "promoted",
]

df = df.filter(selected_columns)

print("dataset after feature selection:")
print(df.info())

# %%
# Correlation mapping after feature engineering

corr_df = pd.get_dummies(df)
corr_mat = corr_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()

# %%
# Model Generation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Using one-hot encoding for categorical values
df = pd.get_dummies(df)

# Splitting dataset into training and test datasets
x = df.drop(columns="promoted")
y = df["promoted"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=22
)

# Training Model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(x_train, y_train)

# %%
# Evaluating the model

y_pred = model.predict(x_test)

class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("classification report:")
print(class_report)

# Displaying confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
