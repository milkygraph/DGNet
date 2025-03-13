import os
import sys

def find_files_with_zero(directory):
    """Searches for '0' in all files ending with '_outputs.txt' in the given directory."""
    if not os.path.exists(directory):
        print("Error: The specified directory does not exist.")
        sys.exit(1)

    matching_files = []

    for file in os.listdir(directory):
        if file.endswith("_outputs.txt"):
            file_path = os.path.join(directory, file)

            # Read the file and check if '0' is present
            with open(file_path, "r") as f:
                if "0" in f.read():
                    matching_files.append(file)

    # Print results
    if matching_files:
        print("Files containing '0':")
        for file in matching_files:
            print(file)
    else:
        print("No files contain '0'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    find_files_with_zero(directory_path)
