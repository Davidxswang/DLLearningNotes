# Python 3

## uninstall Python 3 from Mac

1. Make sure we know the version of Python 3: `python3 --version`
2. Delete the frameworks: `sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.7`, here we need to change the `3.7` into the version we want to uninstall
3. Delete the applications: `sudo rm -rf /Applications/Python\ 3.7/`, here be careful with the space with `Python` and `3.7`, and change the `3.7` into the version we want to uninstall
4. Remove the system links in `/usr/local/bin/` that point to the version that we want to remove
5. Check the shell configuration files to make sure no `python3.7` path is still in use, e.g., `~/.bash_profile`, or `~/.zshrc`

## built-in data types

## itertools

## collections

## heapq

## bisect

## functools

## Reference

1. [How to Uninstall Python 3.7 Cleanly from Mac](https://www.jianshu.com/p/98383a7688a5)