import os

def find_files_with_extension(folder_path:str, extension:str) -> list:
    """
    Searches for all files in the specified folder that are of the specified file extension.
    
    folder_path: Path to folder where should be searched.
    extension: extension a files name should end with
    
    returns: List of file paths of found files.
    """
    if os.path.exists(folder_path):
        files = [os.path.join(folder_path, file_path) for file_path in os.listdir(folder_path) if file_path.endswith(extension)]
        return files
    else:
        return []
