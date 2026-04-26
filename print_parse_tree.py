from parser import ASTNode


def print_parse_tree(node, prefix="", is_last=True, label=None):
    connector = "`-- " if is_last else "|-- "
    line_prefix = prefix + connector if prefix else ""
    child_prefix = prefix + ("    " if is_last else "|   ")

    def format_scalar(value):
        return repr(value) if isinstance(value, str) else str(value)

    if isinstance(node, ASTNode):
        header = node.token_type

        if not isinstance(node.value, (dict, list)):
            header += f": {format_scalar(node.value)}"

        if label is not None:
            print(f"{line_prefix}{label}: {header}")
        else:
            print(f"{line_prefix}{header}")

        if isinstance(node.value, dict):
            items = list(node.value.items())
            for i, (key, value) in enumerate(items):
                print_parse_tree(value, child_prefix, i == len(items) - 1, key)

        elif isinstance(node.value, list):
            for i, item in enumerate(node.value):
                print_parse_tree(item, child_prefix, i == len(node.value) - 1, f"[{i}]")

    elif isinstance(node, dict):
        if label is not None:
            print(f"{line_prefix}{label}")
        else:
            print(f"{line_prefix}dict")

        items = list(node.items())
        for i, (key, value) in enumerate(items):
            print_parse_tree(value, child_prefix, i == len(items) - 1, key)

    elif isinstance(node, list):
        if label is not None:
            print(f"{line_prefix}{label}")
        else:
            print(f"{line_prefix}list")

        for i, item in enumerate(node):
            print_parse_tree(item, child_prefix, i == len(node) - 1, f"[{i}]")

    else:
        text = format_scalar(node)
        if label is not None:
            print(f"{line_prefix}{label}: {text}")
        else:
            print(f"{line_prefix}{text}")
