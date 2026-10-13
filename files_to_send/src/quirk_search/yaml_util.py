"""Small YAML reader for the config files in this project.

The configs use maps, lists, numbers, and comments. This parser covers that
subset so the package does not need PyYAML.
"""

from __future__ import annotations


def strip_comment(line: str) -> str:
    out = []
    quote = None
    for ch in line:
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out)


def parse_scalar(raw: str):
    text = raw.strip()
    if text == "" or text.lower() in {"null", "~"}:
        return None
    if (text[0] == text[-1]) and text[0] in {"'", '"'}:
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner == "":
            return []
        return [parse_scalar(part) for part in _split_top_level(inner)]
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _split_top_level(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


class _Parser:
    def __init__(self, text: str):
        self.lines = []
        for raw in text.splitlines():
            cleaned = strip_comment(raw).rstrip()
            if cleaned.strip():
                self.lines.append(cleaned)
        self.i = 0

    def parse(self):
        if not self.lines:
            return {}
        return self._block(self._indent(self.lines[0]))

    def _indent(self, line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _block(self, indent: int):
        if self.i >= len(self.lines):
            return {}
        stripped = self.lines[self.i].lstrip()
        if stripped.startswith("- "):
            return self._list(indent)
        return self._dict(indent)

    def _dict(self, indent: int) -> dict:
        mapping = {}
        while self.i < len(self.lines):
            line = self.lines[self.i]
            current = self._indent(line)
            if current < indent:
                break
            if current > indent:
                raise ValueError(f"Unexpected indent in config line: {line}")
            stripped = line.strip()
            if stripped.startswith("- "):
                break
            key, separator, rest = stripped.partition(":")
            if separator != ":":
                raise ValueError(f"Expected key: value, got: {line}")
            rest = rest.strip()
            self.i += 1
            if rest == "":
                mapping[key.strip()] = self._block(indent + 2)
            else:
                mapping[key.strip()] = parse_scalar(rest)
        return mapping

    def _list(self, indent: int) -> list:
        items = []
        while self.i < len(self.lines):
            line = self.lines[self.i]
            current = self._indent(line)
            if current < indent:
                break
            if current > indent:
                raise ValueError(f"Unexpected indent in config list: {line}")
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            rest = stripped[2:].strip()
            self.i += 1
            if rest == "":
                items.append(self._block(indent + 2))
            elif ":" in rest and not rest.startswith("["):
                key, _, value = rest.partition(":")
                item = {}
                value = value.strip()
                if value == "":
                    item[key.strip()] = self._block(indent + 2)
                else:
                    item[key.strip()] = parse_scalar(value)
                item.update(self._dict(indent + 2))
                items.append(item)
            else:
                items.append(parse_scalar(rest))
        return items


def load_yaml(text: str):
    return _Parser(text).parse()


def load_yaml_file(path) -> dict:
    return load_yaml(path.read_text())
