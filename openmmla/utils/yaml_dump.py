import re

import yaml


def is_nested_dict(value):
    """If the value is a dict and any of its values is a dict or list, consider it a nested structure."""
    return isinstance(value, dict) and any(isinstance(v, (dict, list)) for v in value.values())


def should_use_inline(value):
    """If the value is a flat list (all elements are scalars), return True to output inline."""
    return isinstance(value, list) and all(not isinstance(i, (dict, list)) for i in value)


def represent_sequence(dumper, data):
    """Use inline style for flat lists; otherwise use the default multi-line style."""
    flow = should_use_inline(data)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=flow)


class FinalSmartDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

    def write_line_break(self, data=None):
        # After writing a line, if add_extra_line_break is set and we're at the start of a line, insert an extra blank line
        super().write_line_break(data)
        if getattr(self, 'add_extra_line_break', False) and self.column == 0:
            super().write_line_break()
            self.add_extra_line_break = False

    def represent_mapping(self, tag, mapping, flow_style=None):
        """Iterate over each key-value pair in the mapping. If the value is nested, we want a blank line before the key.

        Note: Since PyYAML's emitter doesn't provide precise control over inserting blank lines,
        we rely on post-processing the generated YAML string to insert blank lines where needed.
        """
        value_nodes = []
        for key, value in mapping.items():
            node_key = self.represent_data(key)
            node_value = self.represent_data(value)
            value_nodes.append((node_key, node_value))
        return yaml.MappingNode(tag, value_nodes, flow_style=flow_style)


FinalSmartDumper.add_representer(list, represent_sequence)


def format_yaml_output(yaml_str):
    """Insert blank lines before keys whose values are nested structures based on the following rules:

    - For lines like "key:" (ending with a colon, no value on the same line),
    - If the next line has more indentation (indicating a nested structure),
    - And the previous line is not blank,
    Then insert a blank line before this key.
    """
    lines = yaml_str.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        # Check for "key:" format (no inline value, ends with colon)
        if re.match(r'^\s*\S+:\s*$', line):
            # Check if next line exists and is more indented, indicating nested structure
            if i + 1 < len(lines):
                indent_current = len(re.match(r'^(\s*)', line).group(1))
                indent_next = len(re.match(r'^(\s*)', lines[i + 1]).group(1))
                if indent_next > indent_current:
                    # If previous line is not blank, insert an empty line
                    if new_lines and new_lines[-1].strip() != "":
                        new_lines.append("")
        new_lines.append(line)
    return "\n".join(new_lines)


def dump_yaml_pretty(data, path):
    # First dump the YAML as a string using the custom Dumper
    raw_yaml = yaml.dump(
        data,
        Dumper=FinalSmartDumper,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=120
    )
    # Post-process: insert blank lines before keys with nested values
    formatted_yaml = format_yaml_output(raw_yaml)
    with open(path, 'w') as f:
        f.write(formatted_yaml)
