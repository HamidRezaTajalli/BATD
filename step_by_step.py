
import torch
import logging
import time # Import time module for Unix timestamp
from attack import Attack
from dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py
from models.FTT import FTTModel
from models.Tabnet import TabNetModel
from dataset.BM import BankMarketing
from dataset.ACI import ACI
from dataset.HIGGS import HIGGS
from dataset.Diabetes import Diabetes
from dataset.Eye_Movement import EyeMovement
from dataset.KDD99 import KDD99
from dataset.CreditCard import CreditCard





# ============================================
# Logging Configuration
# ============================================

# Set up the logging configuration to display informational messages.
# This helps in tracking the progress and debugging if necessary.
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the logging message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)





# ============================================
# Main Execution Block
# ============================================


def main():
    """
    The main function orchestrates the attack setup by executing the initial model training steps,
    retraining on the poisoned dataset, and evaluating the backdoor's effectiveness.
    """

    logging.info("=== Starting Initial Model Training ===")

    # Step 0: Initialize device, target label, and other parameters needed for the attack.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_label = 1
    mu = 0.2
    beta = 0.1
    lambd = 0.1
    epsilon = 0.02
    
    # Step 1: Initialize the dataset object which can handle, convert and revert the dataset.
    data_obj = CreditCard()

    # Step 2: Initialize the model. If needed (optional), the model can be loaded from a saved model. Then the model is not needed to be trained again.
    # model = FTTModel(data_obj=data_obj)

    model = TabNetModel(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax'
        )

    model.to(device)

    # # load the trained model
    # model.load_model("")
    # model.to(device)


    # Step 3: Initialize the Attack class with model, data, and target label
    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=mu, beta=beta, lambd=lambd)

    # Step 4: Train the model on the clean training dataset
    
    attack.train(attack.converted_dataset)
    logging.info("=== Initial Model Training Completed ===")

    logging.info("=== Testing the model on the clean testing dataset ===")

    # Step 5: Test the model on the clean testing dataset
    attack.test(attack.converted_dataset[1])

    logging.info("=== Testing Completed ===")

    # Get current Unix timestamp
    unix_timestamp = int(time.time())


    # Save the model with Unix timestamp in the filename
    attack.model.save_model(f"./saved_models/clean/{attack.model.model_name}_{data_obj.dataset_name}_{unix_timestamp}")
    

    # Step 6: Select non-target samples from the training dataset
    D_non_target = attack.select_non_target_samples()

    # Step 7: Confidence-based sample ranking to create D_picked
    D_picked = attack.confidence_based_sample_ranking()


    # Step 8: Define the backdoor trigger
    attack.define_trigger()

    # Step 9: optimize the trigger
    attack.compute_mode() # compute the mode vector (if needed)

    attack.optimize_trigger()

    # attack.load_trigger() # load the already optimized trigger (if needed)


    print(attack.delta)

    print(attack.mode_vector)


    # Step 10: Construct the poisoned dataset

    poisoned_trainset, poisoned_train_samples = attack.construct_poisoned_dataset(attack.converted_dataset[0], epsilon=epsilon)
    poisoned_testset, poisoned_test_samples = attack.construct_poisoned_dataset(attack.converted_dataset[1], epsilon=1)
    attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)


    # Save the poisoned dataset
    attack.save_poisoned_dataset()
    # load the poisoned dataset (uncomment if needed)
    # attack.load_poisoned_dataset()


    # Step 11: Revert the poisoned dataset to the original categorical features
    FTT = True if attack.model.model_name == "FTTransformer" else False
    if data_obj.cat_cols:
        reverted_poisoned_trainset = attack.data_obj.Revert(attack.poisoned_dataset[0], FTT=FTT)
        reverted_poisoned_testset = attack.data_obj.Revert(attack.poisoned_dataset[1], FTT=FTT)
    else:
        reverted_poisoned_trainset = attack.poisoned_dataset[0]
        reverted_poisoned_testset = attack.poisoned_dataset[1]

    reverted_poisoned_dataset = (reverted_poisoned_trainset, reverted_poisoned_testset)

    # get the clean train and test datasets
    if FTT and data_obj.cat_cols:
        clean_dataset = attack.data_obj.get_normal_datasets_FTT()
    else:
        clean_dataset = attack.data_obj.get_normal_datasets()
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]

    logging.info("=== Training the model on the poisoned training dataset ===")
    # Step 12: Train the model on the poisoned training dataset
    converted = False if FTT and data_obj.cat_cols else True
    attack.train(reverted_poisoned_dataset, converted=converted)
    logging.info("=== Poisoned Training Completed ===")

    logging.info("=== Testing the model on the poisoned testing dataset and clean testing dataset ===")
    # Step 13: Test the model on the poisoned testing dataset and clean testing dataset
    attack.test(reverted_poisoned_testset, converted=converted)
    attack.test(clean_testset, converted=converted)
    logging.info("=== Testing Completed ===")

    # Save the poisoned model with Unix timestamp in the filename
    attack.model.save_model(f"./saved_models/poisoned/{attack.model.model_name}_{data_obj.dataset_name}_{unix_timestamp}")



# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()




# TODO:
# - comment in detial all the steps in  the main function. the attakc should be done in step by step manner exactly described in the paper.
# - If loaded path is given, the model is not needed to be trained again.