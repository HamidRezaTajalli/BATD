from collections import Counter
import numpy as np
from models.XGBoost import XGBoostModel
from dataset.Eye_Movement import EyeMovement



clean_model = XGBoostModel()
poisoned_model = XGBoostModel()
poisoned_model.load_model("/home/htajalli/prjs0962/repos/BATD/saved_models/poisoned/xgboost_eye_movement_2_1.0_0.1_0.1_0.05_poisoned_model.pth")
dataset = EyeMovement()

trainset, testset = dataset.get_normal_datasets()

X_train, y_train = dataset._get_dataset_data(trainset)
X_test, y_test = dataset._get_dataset_data(testset)

clean_model.fit(X_train, y_train)

poisoned_model_preds = poisoned_model.predict(X_test)
clean_model_preds = clean_model.predict(X_test)

poisoned_model_accuracy = (poisoned_model_preds == y_test).mean()
clean_model_accuracy = (clean_model_preds == y_test).mean()

print(f"Poisoned Model Accuracy: {poisoned_model_accuracy * 100:.2f}%")
print(f"Clean Model Accuracy: {clean_model_accuracy * 100:.2f}%")


poisoned_model_preds = poisoned_model.predict(X_train)
clean_model_preds = clean_model.predict(X_train)

poisoned_model_accuracy = (poisoned_model_preds == y_train).mean()
clean_model_accuracy = (clean_model_preds == y_train).mean()

print(f"Poisoned Model Accuracy: {poisoned_model_accuracy * 100:.2f}%")
print(f"Clean Model Accuracy: {clean_model_accuracy * 100:.2f}%")


# Identify indices where the poisoned model is correct and the clean model is not
correct_poisoned_wrong_clean = np.where((poisoned_model_preds == y_train) & (clean_model_preds != y_train))[0]

# Extract the class labels for these indices
class_labels = y_train[correct_poisoned_wrong_clean]

# print how many of each class are in the class_labels
# Step 3: Count the number of samples for each class
class_counts = Counter(class_labels)
print("Class labels of samples correctly classified by poisoned model but not by clean model:")
print(class_counts)







