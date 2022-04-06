import shutil
import os
import shutil

name = input('Enter participant\'s name : ')
for root, dirs, files in os.walk(os.path.join('data/sourcedScripts',name)):
    for f in files:
        filepath = os.path.join(root,f)
        targetpath = os.path.join('data/sourcedScriptsByActivity',os.path.basename(root),name+f)
        move = input(f'Copy {filepath} to {targetpath}? (y/n)')
        if move == 'y':
            shutil.copy(filepath, targetpath)