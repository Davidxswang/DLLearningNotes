# Python 3

## uninstall Python 3 from Mac

1. Make sure we know the version of Python 3: `python3 --version`
2. Delete the frameworks: `sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.7`, here we need to change the `3.7` into the version we want to uninstall
3. Delete the applications: `sudo rm -rf /Applications/Python\ 3.7/`, here be careful with the space between `Python` and `3.7`, and change the `3.7` into the version we want to uninstall
4. Remove the system links in `/usr/local/bin/` that point to the version that we want to remove
5. Check the shell configuration files to make sure no `python3.7` path is still in use, e.g., `~/.bash_profile`, or `~/.zshrc`

## interesting built-in functions and topics

### five kinds of parameter

1. positional-or-keyword: this is the default type of parameter, e.g., all the parameters in

```python
def function(arg1, arg2='test')
```

2. positional-only: argument supplied only by position, e.g., `po_only1` and `po_only2` in

```python
def function(po_only1, po_only2, /, arg1, arg2='test')
```

3. keyword-only: argument supplied only by keywork, e.g., `kw_only1` and `kw_only2` in

```python
def function(arg1, arg2, *, kw_only1, kw_only2)
```

4. var-position: arbitrary sequence of positional parameters, e.g., `*arg` in

```python
def function(arg1, arg2, *args)
```

5. var_keyword: arbitrarily many keyword arguments, e.g., `**kwargs` in

```python
def function(arg1, arg2, **kwards)
```

## built-in data types

### list

#### sort

Definition:

```python
sort(*, key=None, reverse=False)
```

Example:

```python
a = [(1, 2, 4), (3, 2, 5), (2, 4, 2)]
a.sort(key=lambda x:x[0], reverse=True)
# a will be [(3, 2, 5), (2, 4, 2), (1, 2, 4)]
```

This method sorts the list in place, using only `<` comparisons between items.

### dict

#### get

Definition:

```python
get(key[, default])
```

Return the value for `key` if `key` is in the dictionary, else `default`. If `default` is not given, it defaults to `None`, so that this method never raises a `KeyError`.

#### setdefault

Definition:

```python
setdefault(key[, default])
```

If key is in the dictionary, return its value. If not, insert `key` with a value of `default`, and return `default`. `default` defaults to `None`.

## itertools

## collections

### collections.namedtuple

Create tuple subclass with named fields, which is indexable and iterable.

Definition:

```python
collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)
```

Examples:

```python
from collections import namedtuple
Person = namedtuple('person', ['name', 'age', 'gender'])
person_a = Person('david', 18, 'male')
person_b = Person._make(['john', 20, 'male'])
person_b_dict = person_b._asdict()
person_a_replaced = person_a._replace(name='steven')
fields = person_a._fields
fields = Person._fields # can be class method or instance method
field_defaults = Person._field_defaults # dictionary
person_a_name = getattr(person, 'name')
# person_a[0] is 'david'
# person_a.age is 18
# name, age, gender = person_b
# person_b_dict is {'name':'john', 'age': 20, 'gender': 'male'}
# fields is ('name', 'age', 'gender')

```

Notes:

- `field_names` can also be `'name age gender'` or `'name, age, gender'`.
- `defaults` are applied to the rightmost parameters.
- No per-instance dictionaries are needed, so it requires no more memory than tuple, i.e. `__slots__ = ()`.

## heapq

## bisect

## functools

### functools.lru_cache()

Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can save time if the function is called multiple times.

Definition:

```python
@lru_cache(user_function)
@lru_cache(maxsize=128, typed=False)
```

Examples:

```python
@lru_cache
def count(input_string):
    return input_string.count('1')
# this will leaves maxsize at its default
count.cache_parameters()
>>> {"maxsize": 128, "typed": False}
count.cache_info()
>>> CacheInfo(hits=3, misses=8, maxsize=32, currsize=8)
count.cache_clear()
# clear the cache
```

Notes:

- LRU: Least Recent Used.
- A dictionary is used to cache the results, so the positional and keyword arguments must be hashable.
- If `maxsize` is `None`, lru function is disabled.
- If `typed` is `True`, f(3) and f(3.0) will be cached as distinct results.

## glob

Unix style pathname pattern expansion.

### glob.glob

Returns a possibly empty list containing file names matching `pathname`.

Definition:

```python
glob.glob(pathname, *, recursive=False)
```

Example:

```python
result = glob.glob("/usr/david/home/test/*.py")
```

Notes:

- If `recursive` is `True`, pattern `**` will match any files, zero or more directories, symbolic links, and subdirectories, recursively.

## os.path

### os.path.exists

Return `True` if `path` is an exsiting path or an open file descriptor, `False` if a broken symbolic link.

Definition:

```python
os.path.exists(path)
```

Example:

```python
result = os.path.exists('./abc/test.py')
```

Notes:

- `path` can be an integer (open file descriptor), a path-like object, or a string.

### os.path.isfile

Return `True` if `path` is an existing regular file. It follows symbolic links, so both `isfile` and `islink` can be true for the same path.

Definition:

```python
os.path.isfile(path)
```

Example:

```python
result = os.path.isfile('./abc/test.py')
```

Notes:

- `path` can be a path-like object or a string.

### os.path.isdir

Return `True` if `path` is an existing directory. It follows symbolic links, so both `isdir` and `islink` can be true for the same path.

Definition:

```python
os.path.isdir(path)
```

Example:

```python
result = os.path.isdir('./abc/testdir')
```

Notes:

- `path` can be a path-like object or a string.

### os.path.islink

Return `True` if `path` refers to an existing directory entry that is a symbolic link. Always `False` if symbolic links are not supported by the Python runtime.

Definition:

```python
os.path.islink(path)
```

Example:

```python
result = os.path.islink('./abc/test')
```

Notes:

- `path` can be a path-like object or a string.

### os.path.join

Join one or more path components intelligently.

Definition:

```python
os.path.join(path, *paths)
```

Example:

```python
os.path.join('./abc', 'project2', 'abc')
>>> './abc/project2/abc'
```

Notes:

- `path` and `paths` can be a path-like object or a string.

### os.path.split

Split the pathname `path` into a pair, `(head, tail)` where `tail` is the last pathname component and `head` is everything leading up to that.

Definition:

```python
os.path.split(path)
```

Example:

```python
os.path.split('./abc/project2/abc.py')
>>> ['./abc/project2', 'abc.py']
```

Notes:

- `path` can be a path-like object or a string.

## csv

### csv.reader

Return a reader object which will iterate over lines in the given csvfile.

Definition:

```python
csv.reader(csvfile, dialect='excel', **fmtparams)
```

Example:

```python
with open('test.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        print(row[0], row[1]) # each row is a list of strings
```

Notes:

- `csvfile` can be any object that supports iterable protocol and returns a string when its `__next__` method is called.

### csv.writer

Return a writer object responsible for converting the user’s data into delimited strings on the given file-like object.

Definition:

```python
csv.writer(csvfile, dialect='excel', **fmtparams)
```

Example:

```python
with open('writer.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([1, 2, 3, 'abc', 'abass'])
```

Notes:

- `csvfile` can be any object with a `write()` method.

## argparse

Parser for command-line options, arguments and sub-commands.

### ArgumentParser

Create a new ArgumentParser object.

Definition:

```python
class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
```

Example:

```python
parser = argparse.ArgumentParser(description="text to display before the argument help", epilog="text to display after the argument help", parents=[parser1, parser2])
```

### add_argument

Define how a single command-line argument should be parsed.

Definition:

```python
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

Example:

```python
parser.add_argument('-f', '--foo')
parser.add_argument('bar')
parser.add_argument('--setnumber', action='store_const', const=42)
parser.add_argument('--trainmode', action="store_true")
parser.add_argument('--addnumber', action="append")
parser.add_argument('--int', dest="types", action="append_const", const=int)
parser.add_argument('--str', dest="types", action="append_const", const=str)
parser.add_argument('--verbose', '-v', action="count", default=0)
parser.add_argument('--version', action="version", version="%(prog)s 2.0")
parser.add_argument('--plot', action="extend", nargs="+", type=str)
```

Notes:

- `name or flags`: optional argument: `'-f', '--foo'`, positional argument: `bar`.
- `action` defaults to `store`, other options are: `store_const`, `store_true`, `store_false`, `append`, `append_const`, `count`, `help` (automatically added to the parser), `version`, `extend`, and any Action subclass.
- `default` defaults to `None`.
- `nargs` have options: `N` (an integer, e.g., 1-item, 2-item list), `'?'` (zero or single item), `'*'` (zero or more items), `'+'` (1 or more items).
- `metavar` can set the name to refer to for the optional arguments.
- `type` can facilitate type conversion.

### parse_args

Convert argument strings to objects and assign them as attributes of the namespace, then return the namespace.

Definition:

```python
ArgumentParser.parse_args(args=None, namespace=None)
```

Example:

```python
parser.parse_args(['testtext', '--foo', '1'])
>>> Namespace(bar='testtext', foo='1')
parser.parse_args("--addnumber 1 --addnumber 2".split())
>>> Namespace(addnumber=["1", "2"])
parser.parse_args("--int --str".split())
>>> Namespace(types=[<class 'int'>, <class 'str'>])
parser.parse_args("-vvv --trainmode".split())
>>> Namespace(verbose=3, trainmode=True)
parser.parse_args(["--version"])
>>> PROG 2.0
parser.parse_args("--plot train_acc --plot train_loss val_acc val_loss".split())
>>> Namespace(plot=["train_acc", "train_loss", "val_acc", "val_loss"])
parser.parse_args("--case=test")
>>> Namespace(case="test")
parser.parse_args("-n1")
>>> Namespace(n="1")
parser.parse_args("-sabcC")
>>> Namespace(s=True, a=True, b=True, c="C")
result.s
>>> True
```

Notes:

- For long options (longer than a single character), `'='` can be used to parse the argument string.
- For short options, the option and its value can be concatenated.
- For short options that don't require a value (or only the last one requires a value), they can be concatenated.

## datetime

### datetime.timedelta

Definition:

```python
class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
```

Example:

```python
t1 = datetime.timedelta(hours=1, seconds=1)
t1.total_seconds()
>>>  3601.0
t1.days, t1.seconds, t1.microseconds
>>> (0, 3601, 3601000)
```

### datetime.date

Definition:

```python
class datetime.date(year, month, day)
```

Example:

```python
datetime.date.today() == datetime.date.fromtimestamp(time.time())
>>> True
d1 = datetime.date.today()
d1.year, d1.month, d1.day
>>> (2020, 12, 10)
d2 = d1.replace(day=d1.day+1)
d2 - d1
>>> timedelta(days=1)
```

### datetime.time

Definition:

```python

```

Example:

```python

```

### datetime.datetime

Definition:

```python
class datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
```

Example:

```python

```

## Reference

1. [How to Uninstall Python 3.7 Cleanly from Mac](https://www.jianshu.com/p/98383a7688a5)
2. [Python Docs](https://docs.python.org)
