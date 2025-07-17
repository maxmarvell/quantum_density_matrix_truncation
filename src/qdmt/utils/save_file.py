import os

def safe_save_file(filepath):
    if os.path.exists(filepath):
        while True:
            # Prompt the user for their choice
            choice = input(f"⚠️  Warning: '{filepath}' already exists.\n"
                           "Type 'o' to overwrite or 'n' to choose a new file location: ").lower()
            
            if choice == 'o':
                # If 'overwrite', return the original filepath
                print("File will be overwritten.")
                return filepath
            
            elif choice == 'n':
                # If 'new location', prompt for the new path
                new_filepath = input("Enter the new file path: ")
                # You could add more validation here for the new_filepath if needed
                return new_filepath
            
            else:
                # Handle invalid input
                print("❌ Invalid choice. Please try again.")

    else:
        # If the file does not exist, no action is needed
        print("✅ File path is clear. Proceeding to save.")
        return filepath