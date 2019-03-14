# Git

**Transfer codes from ==GitLab== to==GitHub==**:

- create an empty repo on GitHub

  ```c++
  git remote add githubhttps://yourLogin@github.com/yourLogin/yourRepoName.git
  git push --mirror github
  ```



**Convert remote's URL from `https` to `ssh`**:

```bash
# To check if remote's URL is ssh or https
git remote -v
# To switch from https to ssh
git remote set-url origin git@github.com:USERNAME/REPOSITORY.git
```



**Untrack files and don't change working directory**:

```bash
git rm --cached [filename]
```



**Reset `git add` operations**:

```bash
# reset staging area, but leave the working directory unchanged
git reset HEAD .
git reset

# Remove <file> from the staging area, but leave the working directory unchanged
git reset <file>

# reset staging area and reset working directory to match last commit
git reset --hard

# move the current brach tip backward to <commit>, reset the staging area to match, but leave the working directory alone
git reset <commit>

# same as precious, but resets both the staging area and working directory
git reset --hard <commit>
```



**Clean files**:

```bash
# Show which files would be removed from working directory
git clean -n

# Execute the clean
git clean -f
```



**Append changes to last commit**:

```bash
git commit --amend
```



**Display `commits` concerned with the specific file**:

```bash
git log -- <file>
```



**Git diff**:

```bash
# show difference between working directory and last commit
git diff HEAD

# show difference between changes and last commit
git diff --cached
```





