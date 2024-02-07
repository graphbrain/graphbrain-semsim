import os


def replace_in_names(root_directory, old_string, new_string):
    for path, subdirs, files in os.walk(root_directory):
        for name in files:
            if old_string in name:
                new_name = name.replace(old_string, new_string)
                os.rename(os.path.join(path, name), os.path.join(path, new_name))

        for name in subdirs:
            if old_string in name:
                new_name = name.replace(old_string, new_string)
                os.rename(os.path.join(path, name), os.path.join(path, new_name))


# Usage example
root_directory = '/Users/max18768/Documents/MA/graphbrain-semsim/data'  # Replace with your directory path
old_string = '1-1_pred_wildcard'                # Replace with the substring you want to change
new_string = '1-2_pred_wildcard'                # Replace with the new substring

replace_in_names(root_directory, old_string, new_string)
