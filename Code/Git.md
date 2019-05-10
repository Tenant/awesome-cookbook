## Git

### 1. Initialize Repo

**Transfer codes from `GitLab` to `GitHub`**:

  ```c++
  git push --mirror github
  ```

### 2. Tracking Files

**Undo modifications**

```bash
git checkout @ -- [filename]
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
git clean -df
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


### 3. Remote

**Convert remote's URL from `https` to `ssh`**:

```bash
git remote -v

git remote add github [url]

git remote set-url origin [url]
```



### 4. Branch

```bash
git branch [new-branch-name]

git branch

git checkout [target-branch-name]

git merge [source-branch-name]

git branch --all

git push origin [local-branch]:[remote-branch]

git push origin :[remote-branch]

git branch -d [branch-name]

git remote set-head origin [branch-new]
```

**List all tracked files**

```bash
git ls-tree -r master --name-only
```





