from catboost import CatBoostClassifier

from dataset.CovType import CovType
from sklearn.metrics import accuracy_score

from models.FTT import FTTModel


data_obj = CovType()


model = FTTModel(data_obj)
model_original = model.model_original
model_converted = model.model_converted


for name, param in model_original.named_parameters():
    print(name, param.shape)


print("--------------------------------")


for name, param in model_converted.named_parameters():
    print(name, param.shape)




# model = CatBoostClassifier(
#     iterations=10000,              # More iterations for complex data
#     learning_rate=0.01,           # Slower learning rate for better convergence
#     depth=10,                     # Deeper trees for capturing complex patterns
#     l2_leaf_reg=3,                # L2 regularization to prevent overfitting
#     border_count=128,             # More precise splits
#     loss_function='MultiClass',
#     eval_metric='Accuracy',
#     task_type='GPU',              # Change to 'GPU' if available
#     bootstrap_type='Bayesian',    # Advanced bootstrapping
#     bagging_temperature=1,
#     verbose=100,
#     early_stopping_rounds=50      # Stop if no improvement for 50 rounds
# )

# trainset, testset = data_obj.get_normal_datasets()

# X_train, y_train = data_obj._get_dataset_data(trainset)
# X_test, y_test = data_obj._get_dataset_data(testset)

# model.fit(X_train, y_train)

# # Make predictions
# predictions = model.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)

# print(f"CatBoost Model Accuracy: {accuracy * 100:.2f}%")


