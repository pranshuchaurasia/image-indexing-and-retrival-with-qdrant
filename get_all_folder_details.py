import os

def get_subfolder_paths(main_folder):
    subfolder_paths = []
    for item in os.listdir(main_folder):
        if os.path.isdir(os.path.join(main_folder, item)):
            subfolder_paths.append(rf"{main_folder}/{item}")
    return subfolder_paths

# Example usage
main_folder = r"iamges_path"
subfolder_paths = get_subfolder_paths(main_folder)
print(subfolder_paths)