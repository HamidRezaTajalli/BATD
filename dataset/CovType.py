from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.datasets import fetch_covtype
import torch
from torch.utils.data import DataLoader, TensorDataset


class CovType:
    """
    This class is used to load the CoverType dataset and convert it into a format suitable for training a model.
    """
    
    def __init__(self, test_size=0.2, random_state=42, batch_size=64):

        # Define the categorical and numerical columns
        self.cat_cols = [
            "Wilderness_Area_0", "Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3",
            "Soil_Type_0", "Soil_Type_1", "Soil_Type_2", "Soil_Type_3", "Soil_Type_4",
            "Soil_Type_5", "Soil_Type_6", "Soil_Type_7", "Soil_Type_8", "Soil_Type_9",
            "Soil_Type_10", "Soil_Type_11", "Soil_Type_12", "Soil_Type_13", "Soil_Type_14",
            "Soil_Type_15", "Soil_Type_16", "Soil_Type_17", "Soil_Type_18", "Soil_Type_19",
            "Soil_Type_20", "Soil_Type_21", "Soil_Type_22", "Soil_Type_23", "Soil_Type_24",
            "Soil_Type_25", "Soil_Type_26", "Soil_Type_27", "Soil_Type_28", "Soil_Type_29",
            "Soil_Type_30", "Soil_Type_31", "Soil_Type_32", "Soil_Type_33", "Soil_Type_34",
            "Soil_Type_35", "Soil_Type_36", "Soil_Type_37", "Soil_Type_38", "Soil_Type_39"
        ]
        
        self.num_cols = [
            "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"
        ]

        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size


        # Load the CoverType dataset from sklearn as a pandas DataFrame
        data = fetch_covtype(as_frame=True)  # Load data as pandas DataFrame
        X = data['data']  # Features
        y = data['target']  # Target (Forest cover type classification)
        
        # Store original dataset for reference
        self.X_original = X.copy()
        self.y = y.copy()
        
        # Convert categorical columns using OrdinalEncoder
        # This transforms categorical string labels into integer encodings
        ordinal_encoder = OrdinalEncoder()
        self.X_encoded = X.copy()
        self.X_encoded.loc[:, self.cat_cols] = ordinal_encoder.fit_transform(X[self.cat_cols])
        
        # Apply StandardScaler to numerical features to standardize them
        scaler = StandardScaler()
        self.X_encoded[self.num_cols] = scaler.fit_transform(self.X_encoded[self.num_cols])
        
        # Initialize a dictionary to store the primary mappings for each categorical feature
        self.primary_mappings = {col: {} for col in self.cat_cols}

        # Initialize a dictionary to store the adaptive Delta r for each categorical feature
        self.delta_r_values = {col: 0.0 for col in self.cat_cols}

        # Initialize a dictionary to store the hierarchical mappings for each categorical feature
        self.hierarchical_mappings = {col: {} for col in self.cat_cols}
        
        # Initialize a dictionary to store the lookup tables for reverse mapping
        self.lookup_tables = {col: {} for col in self.cat_cols}
    

    def get_normal_datasets(self, test_size=None, random_state=None, batch_size=None):

        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
        
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        
        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        return train_dataset, test_dataset


    
    def compute_primary_frequency_mapping(self):
        """
        Computes the frequency of each category in the categorical features and assigns
        a unique numerical representation r_jl based on the provided formula:
        
        r_jl = (c_max_j - c_jl) / (c_max_j - 1)
        
        where:
            - c_jl is the count of category v_jl in feature j
            - c_max_j is the maximum count among all categories in feature j
        """
        # Iterate over each categorical feature
        for col in self.cat_cols:
            # Get the column data
            col_data = self.X_encoded[col]
            
            # Calculate the frequency (count) of each unique category in the feature
            freq_counts = col_data.value_counts().sort_index()  # Sort by category index for consistency
            
            # Identify the maximum frequency in the current feature
            c_max_j = freq_counts.max()
            
            # Handle the edge case where c_max_j == 1 to avoid division by zero
            if c_max_j == 1:
                # If all categories have the same frequency, assign r_jl = 1 for all
                r_jl = {category: 1.0 for category in freq_counts.index}
                self.primary_mappings[col] = r_jl
                print(f"Primary Mapping for feature '{col}': All categories have r_jl = 1.0\n")
                continue  # Move to the next feature
            
            # Initialize a dictionary to store r_jl for each category in the feature
            r_jl = {}
            
            # Compute r_jl for each category using the provided formula
            for category, count in freq_counts.items():
                r_value = (c_max_j - count) / (c_max_j - 1)
                r_jl[category] = round(r_value, 4)  # Rounded to 4 decimal places for precision
            
            # Store the mapping in the primary_mappings dictionary
            self.primary_mappings[col] = r_jl
            
            # Optional: Print the mapping for verification
            print(f"Primary Mapping for feature '{col}':")
            for category, r_value in r_jl.items():
                print(f"  Category {int(category)}: r_jl = {r_value}")
            print("\n")  # Add a newline for better readability

    def get_primary_mappings(self):
        """
        Returns the primary frequency-based mappings for all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is another dictionary mapping category to r_jl.
        """
        return self.primary_mappings
    


    def compute_adaptive_delta_r(self):
        """
        Determines the adaptive Delta r for each categorical feature based on the smallest
        decimal precision in the primary mapping. This ensures that Delta r is one order
        of magnitude smaller than the smallest decimal precision in Delta r_min.
        """
        # Iterate over each categorical feature
        for col in self.cat_cols:
            r_jl_mapping = self.primary_mappings[col]
            
            # If all r_jl are 1.0, skip Delta r computation as mapping cannot be refined
            if all(r == 1.0 for r in r_jl_mapping.values()):
                print(f"Feature '{col}' has all categories with r_jl = 1.0. Skipping Delta r computation.\n")
                continue  # Move to the next feature
            
            # Extract unique r_jl values and sort them in ascending order
            unique_r_jl = sorted(set(r_jl_mapping.values()))
            
            # Compute Delta r_min: the smallest difference between consecutive r_jl values
            delta_r_min = min(
                unique_r_jl[i+1] - unique_r_jl[i] for i in range(len(unique_r_jl) -1)
            )
            
            # Determine the first non-zero decimal place in Delta r_min
            # Convert Delta r_min to string to identify decimal places
            delta_r_min_str = f"{delta_r_min:.10f}"  # Format to 10 decimal places
            # Remove leading '0.' to focus on decimal digits
            decimal_part = delta_r_min_str.split('.')[1]
            
            # Initialize p to None
            p = None
            
            # Iterate through the decimal digits to find the first non-zero digit
            for idx, digit in enumerate(decimal_part, start=1):
                if digit != '0':
                    p = idx
                    break
            
            # If p is not found (delta_r_min is 0), set a default small p
            if p is None:
                p = 0  # Arbitrary large p to set Delta r very small
                print(f"Delta r_min for feature '{col}' is 0. Setting p={p} and Delta r accordingly.")
            
            # Set Delta r = 10^{-(p + 1)}
            delta_r = 10 ** (-(p + 1))
            
            # Store Delta r in the delta_r_values dictionary
            self.delta_r_values[col] = delta_r
            
            print(f"For feature '{col}':")
            print(f"  Delta r_min = {delta_r_min}")
            print(f"  Determined p = {p}")
            print(f"  Set Delta r = {delta_r}\n")
    
    def get_adaptive_delta_r(self):
        """
        Returns the adaptive Delta r values for all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is the corresponding Delta r.
        """
        return self.delta_r_values
        


    def compute_hierarchical_mapping(self):
        """
        Identifies tied categories within each categorical feature and assigns unique
        r'_jl values by adding incremental multiples of Delta r based on secondary ordering.
        
        Steps:
            1. Detect tied categories (categories with identical r_jl).
            2. Apply secondary ordering (e.g., alphabetical order) to tied categories.
            3. Assign unique r'_jl by adding (k-1)*Delta r to r_jl.
            4. Ensure that r'_jl <= 1. If not, adjust Delta r or redistribute offsets.
        """
        # Iterate over each categorical feature
        for col in self.cat_cols:
            primary_mapping = self.primary_mappings[col]
            delta_r = self.delta_r_values[col]
            
            # Skip features where all r_jl = 1.0
            if delta_r == 0.0:
                # Assign r'_jl = r_jl for all categories
                self.hierarchical_mappings[col] = primary_mapping.copy()
                # Build lookup table
                for category, r_value in primary_mapping.items():
                    self.lookup_tables[col][r_value] = category
                print(f"Hierarchical Mapping for feature '{col}': All categories have r'_jl = {primary_mapping[list(primary_mapping.keys())[0]]}\n")
                continue  # Move to the next feature
            
            # Invert the primary mapping to find categories with the same r_jl
            inverted_mapping = {}
            for category, r_value in primary_mapping.items():
                inverted_mapping.setdefault(r_value, []).append(category)
            
            # Initialize hierarchical mapping for the current feature
            hierarchical_mapping = {}
            
            # Iterate over each unique r_jl value
            for r_value, categories in inverted_mapping.items():
                if len(categories) == 1:
                    # No tie, assign r'_jl = r_jl
                    category = categories[0]
                    hierarchical_mapping[category] = r_value
                else:
                    # Tie detected, need to resolve
                    tied_categories = categories.copy()
                    
                    # Apply secondary ordering: sort categories numerically (since categories are encoded as integers)
                    # If original categories are strings, sort alphabetically. Assuming encoded as integers here.
                    tied_categories_sorted = sorted(tied_categories)
                    
                    # Assign unique r'_jl by adding (k-1)*Delta r
                    for idx, category in enumerate(tied_categories_sorted, start=1):
                        r_prime = r_value + (idx - 1) * delta_r

                        # # Ensure r'_jl <=1
                        # if r_prime > 1.0:
                        #     # Adjust Delta r or redistribute offsets
                        #     # Option 1: Reduce Delta r dynamically (not implemented here for simplicity)
                        #     # Option 2: Redistribute offsets evenly within the available range
                            
                        #     # Calculate available range
                        #     available_range = 1.0 - r_value
                        #     # Number of tied categories
                        #     k = len(tied_categories_sorted)
                        #     # New Delta r to fit within available range
                        #     if k > 1:
                        #         adjusted_delta_r = available_range / k
                        #     else:
                        #         adjusted_delta_r = 0.0  # Only one category, no adjustment needed
                            
                        #     # Recompute r'_jl with adjusted Delta r
                        #     r_prime = r_value + (idx - 1) * adjusted_delta_r
                            
                        #     # Update Delta r for future assignments (optional)
                        #     # self.delta_r_values[col] = adjusted_delta_r
                            
                        #     print(f"Adjusted Delta r for feature '{col}' due to overflow:")
                        #     print(f"  Original Delta r = {delta_r}")
                        #     print(f"  Adjusted Delta r = {adjusted_delta_r}")
                            
                        # Round r'_jl to 4 decimal places for consistency
                        r_prime = round(r_prime, 4)
                        
                        # Assign r'_jl to the category
                        hierarchical_mapping[category] = r_prime
                        
            # Store the hierarchical mapping
            self.hierarchical_mappings[col] = hierarchical_mapping
            
            # Build the lookup table for reverse mapping
            for category, r_prime in hierarchical_mapping.items():
                self.lookup_tables[col][r_prime] = category
            
            # Optional: Print the hierarchical mapping for verification
            print(f"Hierarchical Mapping for feature '{col}':")
            for category, r_prime in hierarchical_mapping.items():
                print(f"  Category {int(category)}: r'_jl = {r_prime}")
            print("\n")  # Add a newline for better readability
    
    def get_hierarchical_mappings(self):
        """
        Returns the hierarchical mappings for all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is another dictionary mapping category to r'_jl.
        """
        return self.hierarchical_mappings
    
    def get_lookup_tables(self):
        """
        Returns the lookup tables for reverse mapping of all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is another dictionary mapping r'_jl to category.
        """
        return self.lookup_tables
    
    def Revert(self, converted_dataset):
        """
        Reverts the converted numerical values back to their original categorical values.
        
        Args:
            converted_dataset (pd.DataFrame): The dataset with converted numerical categorical features.
        
        Returns:
            pd.DataFrame: The dataset with original categorical features restored.
        """
        # Create a copy to avoid modifying the original dataset
        reverted_dataset = converted_dataset.copy()
        
        # Iterate over each categorical feature
        for col in self.cat_cols:
            # Iterate over each row in the dataset
            for idx, r_prime in reverted_dataset[col].items():
                # Round r_prime to 4 decimal places to match the lookup table
                r_prime_rounded = round(r_prime, 4)
                
                # Retrieve the original category using the lookup table
                category = self.lookup_tables[col].get(r_prime_rounded, None)
                
                if category is not None:
                    reverted_dataset.at[idx, col] = category
                else:
                    # Handle cases where r_prime is not found in the lookup table
                    # This could be due to rounding errors or invalid values
                    # Assign a default category or handle as needed
                    reverted_dataset.at[idx, col] = 'Unknown'
        
        return reverted_dataset
    
    def Conv(self):
        """
        Applies the hierarchical mapping to convert categorical features in the dataset.
        
        Args:
            D_original (pd.DataFrame): The original dataset with categorical features.
        
        Returns:
            pd.DataFrame: The converted dataset with unique numerical representations for categorical features.
        """
        # Create a copy to avoid modifying the original dataset
        self.converted_X_encoded = self.X_encoded.copy()

        # Step 4a: Compute the primary frequency-based mapping
        self.compute_primary_frequency_mapping()
    
        # Step 4b: Compute the adaptive Delta r for each categorical feature
        self.compute_adaptive_delta_r()
        
        # Step 4c: Identify and resolve ties by assigning unique r'_jl values
        self.compute_hierarchical_mapping()
        
        # Iterate over each categorical feature
        for col in self.cat_cols:
            hierarchical_mapping = self.hierarchical_mappings[col]
            # Map each category to its r'_jl value
            self.converted_X_encoded[col] = self.converted_X_encoded[col].map(hierarchical_mapping)

        # Store the lookup table and hierarchical mapping for future using a Path()
        
        
        return self.converted_X_encoded
    

    def get_converted_dataset(self, test_size=None, random_state=None, batch_size=None):
        """
        Returns the converted dataset with unique numerical representations for categorical features.
        
        """


        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        self.converted_X_encoded = self.Conv()

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.converted_X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
        
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
        
        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        return train_dataset, test_dataset
    

    def save_mappings(self, directory='mappings/covtype'):
        """
        Saves the hierarchical mappings and lookup tables to pickle files within the specified directory.
        
        Args:
            directory (str or Path, optional): The directory where mapping files will be saved.
                                               Defaults to 'mappings/covtype'.
        """
        # Convert directory to Path object
        path = Path(directory)
        
        # Create the directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Define file paths for hierarchical mappings and lookup tables
        hierarchical_mappings_path = path / 'hierarchical_mappings.pkl'
        lookup_tables_path = path / 'lookup_tables.pkl'
        
        # Serialize the hierarchical mappings to pickle and save
        with open(hierarchical_mappings_path, 'wb') as f:
            pickle.dump(self.hierarchical_mappings, f)
        
        # Serialize the lookup tables to pickle and save
        with open(lookup_tables_path, 'wb') as f:
            pickle.dump(self.lookup_tables, f)
        
        print(f"Hierarchical mappings and lookup tables have been saved to '{path.resolve()}'.")

    def load_mappings(self, directory='mappings/covtype'):
        """
        Loads the hierarchical mappings and lookup tables from pickle files within the specified directory.
        
        Args:
            directory (str or Path, optional): The directory from where mapping files will be loaded.
                                               Defaults to 'mappings/covtype'.
        
        Raises:
            FileNotFoundError: If the mapping files are not found in the specified directory.
        """
        # Convert directory to Path object
        path = Path(directory)
        
        # Define file paths for hierarchical mappings and lookup tables
        hierarchical_mappings_path = path / 'hierarchical_mappings.pkl'
        lookup_tables_path = path / 'lookup_tables.pkl'
        
        # Check if both files exist
        if not hierarchical_mappings_path.exists() or not lookup_tables_path.exists():
            raise FileNotFoundError(f"Mapping files not found in directory '{path.resolve()}'. Please save mappings first.")
        
        # Load hierarchical mappings from pickle
        with open(hierarchical_mappings_path, 'rb') as f:
            self.hierarchical_mappings = pickle.load(f)
        
        # Load lookup tables from pickle
        with open(lookup_tables_path, 'rb') as f:
            self.lookup_tables = pickle.load(f)
        
        print(f"Hierarchical mappings and lookup tables have been loaded from '{path.resolve()}'.")



    def _get_dataset_data(self, dataset):
        """
        Extracts features and labels from the TensorDataset and converts them into appropriate numpy arrays.

        Parameters:
        dataset (TensorDataset): PyTorch TensorDataset object containing the data.

        Returns:
        X (numpy array): Features from the TensorDataset.
        y (numpy array): Labels from the TensorDataset.
        """
        X_tensor, y_tensor = dataset.tensors

        X = X_tensor.cpu().numpy()
        y = y_tensor.cpu().numpy()


        return X, y
