import os
import shutil

target_dir = 'data/sourcedScriptsByActivity'

name = input('Enter script writer\'s name : ')
source_dir = os.path.join('data/sourcedScripts',name)

for root, dirs, files in os.walk(source_dir):
    for f in files:
        activity = os.path.basename(root)
        source_filepath = os.path.join(root, f)
        target_filepath = os.path.join(target_dir, activity, name+f)
        copy = input(f'Do you want to move {source_filepath} to {target_filepath}? (y/n) ')
        if copy == 'y':
            if not os.path.isdir(os.path.join(target_dir, activity)):
                os.makedirs(os.path.join(target_dir, activity))
            shutil.copy(source_filepath, target_filepath)

