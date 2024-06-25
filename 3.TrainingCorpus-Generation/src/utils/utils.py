import os
from typing import List

def select_file_from_directory(directory_path: str, file_type: str) -> List[str]:
    """
    Prints the files in a directory and prompts the user to select one or more of them.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the files.
    file_type : str
        Type of files to select (e.g., "stopwords" or "equivalences").

    Returns
    -------
    List[str]
        List of selected file paths.
    """
    files = os.listdir(directory_path)
    # Print the list of available stopwords/equivalences
    print(f"-- -- Available {file_type} lists:")
    for index, file_name in enumerate(files):
        print(f"{index + 1}. {file_name}")

    # Prompt the user for input
    selected_files = []
    while True:
        selection = input("Enter the numbers of the files you want to select (comma-separated): ")
        selections = selection.split(",")
        try:
            for s in selections:
                file_index = int(s.strip())
                if 1 <= file_index <= len(files):
                    selected_files.append(os.path.join(directory_path, files[file_index - 1]))
                else:
                    print(f"-- -- Invalid selection: {file_index}. Ignoring...")
            break
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers.")

    return selected_files