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

**Untrack files but remain files in the repository**

```bash
git update-index --assume-unchanged [filename]
```


**Untrack files and and remove files from the repository**:

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

## Rebase

**Git追加代码更改到之前某次commit**

```bash
git stash # 保存工作空间中的改动
git log --oneline # 查看commit ID
git rebase f774c^ --interactive # 找到需要修改的commit ID，将行首的pick改成edit并退出
git stash pop # 弹出之前保存的改动
git add . # 提交修改到暂存区
git commit --amend # 追加修改到指定的commit ID
git rebase --continue # 移动HEAD到最新commit处。如发生重复，应修复冲突，提交后重新运行此命令
git push -f origin master # 强制提交更改，慎用
```

**Git调整commit之间的顺序**

```bash
git log --oneline # 查看commit ID
git rebase -i b0aa963^ # 设置修改范围从该commit ID开始
# 手动调整commit的位置，然后保存退出即可
```

查看指定文件的修改历史

```bash
git log -p FILE
```

搜集修改符合指定模式的历史

```bash
git log -s'PATTERN`
```

交互式的保存和取消保存变化

```bash
git add -p
```

返回指定非HEAD分支的提交记录。

```bash
git log ..BRANCH
```

返回一个简单版的git status

```bash
git status -s
```

显示你在本地已完成的操作列表

```bash
git reflog
```

显示提交记录的参与者列表。和Github的参与者列表相同。

```bash
git shortlog -sn
```



