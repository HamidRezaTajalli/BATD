from dataset.CovType import CovType



# Example usage
if __name__ == "__main__":
    # Initialize the converter
    converter = CovType()
    
    # # Step 4a: Compute the primary frequency-based mapping
    # converter.compute_primary_frequency_mapping()
    
    # # Step 4b: Compute the adaptive Delta r for each categorical feature
    # converter.compute_adaptive_delta_r()
    
    # # Step 4c: Identify and resolve ties by assigning unique r'_jl values
    # converter.compute_hierarchical_mapping()
    
    # # Retrieve and display the hierarchical mappings
    # hierarchical_mappings = converter.get_hierarchical_mappings()
    # print("Hierarchical Mappings for Categorical Features:")
    # for feature, mapping in hierarchical_mappings.items():
    #     print(f"  Feature '{feature}':")
    #     for category, r_prime in mapping.items():
    #         print(f"    Category {int(category)}: r'_jl = {r_prime}")
    #     print("\n")
    
    # # Retrieve and display the lookup tables
    # lookup_tables = converter.get_lookup_tables()
    # print("Lookup Tables for Reverse Mapping:")
    # for feature, table in lookup_tables.items():
    #     print(f"  Feature '{feature}':")
    #     for r_prime, category in table.items():
    #         print(f"    r'_jl = {r_prime}: Category {int(category)}")
    #     print("\n")
    

    x_converted = converter.Conv()
    # Select 50 random rows (same for both original and converted data)
    random_rows = converter.X_encoded.sample(n=50).index

    # Print the 50 random rows of the Wilderness_Area_1 and Wilderness_Area_3 columns from the original encoded data
    print("Original Encoded Wilderness_Area_1 and Wilderness_Area_3:")
    print(converter.X_encoded.loc[random_rows, ['Wilderness_Area_1', 'Wilderness_Area_3']])
    print("\n")

    # Print the same 50 random rows of the Wilderness_Area_1 and Wilderness_Area_3 columns from the converted data
    print("Converted Wilderness_Area_1 and Wilderness_Area_3:")
    print(x_converted.loc[random_rows, ['Wilderness_Area_1', 'Wilderness_Area_3']])

    converter.save_mappings()