from app.agent_setup import create_and_save_indexes
import os

# Define the directories
source_directory = "./data"  # Where your PDFs are located
output_directory = "./storage"  # Where you want to save the prebuilt indexes

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create and save the indexes
    create_and_save_indexes(source_directory, output_directory)

    # Confirmation message
    print("Indexes have been created and saved successfully.")