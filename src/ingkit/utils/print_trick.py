def tree(obj, prefix="", name=None, is_last=True):
    connector = "└── " if is_last else "├── "
    if name is not None:
        print(f"{prefix}{connector}{name}: {type(obj).__name__}")
        prefix += "    " if is_last else "│   "
    if isinstance(obj, dict):
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            tree(
                value,
                prefix=prefix,
                name=str(key),
                is_last=(i == len(items) - 1),
            )