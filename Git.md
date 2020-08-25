# Git

## Check Status

### git status
Example: `git status`

This will show the current:
- untracked files
- tracked & changed & unstaged files
- tracked & changed & staged (ready to commit) files

### git show
Example: `git show`

This will show (by default):
- the HEAD object, including hash of the commit, position of the branch, author, date and the content.


## Branch

### git checkout <the branch name to switch to>
Example: `git checkout feature1-dev` or `git checkout master`

This will:
- switch to another branch.

### git branch -d <branch name to delete>
Example: `git branch -d fix-bug`

This will:
- delete the branch **locally**. *Make sure the branch has been merged to master before doing so.*

### git push <remote> :<branch name to delete>
Example: `git push origin :fix-bug`

This will:
- delte the branch **remotely**. If you want to delete it locally afterwards, you can `git pull` to update to the local.


## Commit

### git pull
Example: `git pull`

This will:
- update the local branch against the remote branch.

### git add <files/changes to stage>
Example: `git add .` (This will stage all the changes.)
         `git add fileabc` (This will stage the `fileabc` only)

This will:
- stage the specified changes.

### git rm --cached <filename>
Example: `git rm --cached fileabc`

This will:
- remove the file `fileabc` from the staged files, but will **not** remove the file from the disk. The changes made to the file will be kept. This is usually used when you wrongly add a file by `git add .`.

### git commit -m 'comment comes here'
Example: `git commit -m 'update the feature abc'`

This will:
- put all the tracked changes to the commit. Usually this comes after `git add <files/changes to stage>`

### git push
Example: `git push`

This will:
- update the remote branch according to the local branch.


## Git ignore

### .gitignore
Example:
```
/directoryabc/*
file123
!file345
!/directoryfgh/*
```

This will (corresponding to the example above):
- ignore all the files under the directory `directoryabc`
- ignore the file `file123`
- **not** ignore the file `file345`
- **not** ignore the files under directory `directoryfgh`
The file you want to ignore should not be staged before you stage the `.gitignore` file. After you stage the `.gitignore` file, it will take effect.

### git check-ignore [-v] <the ignored file>
Example: `git check-ignore fileabc` or `git check-ignore -v fileabc` (to get verbose result)

This will:
- show which gitignore file (the first example) or which rule in which gitignore file (the second example) ignored the file. 
