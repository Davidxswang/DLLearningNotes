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

## Config

### git config \<options>

Example: `git config user.email abc@abc.com`

This will:

- config the git repository or global git configurations

## Remote

### git remote

Example: `git remote` or `git remote -v`

This will show:

- the name of the remote repository
- the list of the remote repositories, include name, url and type

### git remote add \<name> \<url>

Example: `git remote add origin https:\\xxxxx`

This will:

- add a new remote repository

### git remote remove \<name>

Example: `git remote remove origin`

This will:

- remove the remote repository

### git remote set-url \<name> \<url>

Example: `git remote set-url origin https:\\xxxx`

This will:

- change the url of the remote repository specified

## Branch

### git checkout \<branch name to switch to>

Example: `git checkout feature1-dev` or `git checkout master`

This will:

- switch to another branch.

### git branch -d \<branch name to delete>

Example: `git branch -d fix-bug`

This will:

- delete the branch **locally**. **Make sure the branch has been merged to master before doing so.**

### git push \<remote> :\<branch name to delete>

Example: `git push origin :fix-bug`

This will:

- delte the branch **remotely**. If you want to delete it locally afterwards, you can `git pull` to update to the local.

## Commit

### git pull

Example: `git pull`

This will:

- update the local branch against the remote branch.

### git add <files/changes to stage>

Example: `git add .` (This will stage all the changes.) or `git add fileabc` (This will stage the `fileabc` only)

This will:

- stage the specified changes.

### git rm --cached \<filename>

Example: `git rm --cached fileabc`

This will:

- remove the file `fileabc` from the staged files, but will **not** remove the file from the disk. The changes made to the file will be kept. This is usually used when you wrongly add a file by `git add .`.

### git commit -m 'comment comes here'

Example: `git commit -m 'update the feature abc'`

This will:

- put all the tracked changes to the commit. Usually this comes after `git add <files/changes to stage>`

### git commit --amend

Example: `git commit --amend`

This will:

- allow the use to amend the last commit message

### git push \<name> \<branch>

Example: `git push origin main` or `git push`

This will:

- update the remote branch according to the local branch
- the name and branch can be left out, in which case, `origin` and `main` will be used implicitly

### git clone

Example: `git clone https://github.com/abcd/abc.git`

This will:

- clone the repository from github to the local host

## Git ignore

### .gitignore

Example:

```.gitignore
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

The file you want to ignore should not be staged before you stage the `.gitignore` file. After you stage the `.gitignore` file, the rules in `.gitignore` will take effect.

### git check-ignore [-v] \<the ignored file>

Example: `git check-ignore fileabc` or `git check-ignore -v fileabc` (to get verbose result)

This will:

- show which gitignore file (the first example) or which rule in which gitignore file (the second example) ignored the file. 

## SSH Config for Git

In the /home/\<username>/.ssh/config, use this to allow github pull and push through ssh.

```config
# github account username: abc123
Host github.com-abc123
	HostName github.com
	User git
	IdentityFile ~/.ssh/id_rsa
```
