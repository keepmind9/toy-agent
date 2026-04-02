"""File operation tools: read, write, edit."""

from toy_agent.tools import tool


@tool(description="Read the content of a file")
def read_file(path: str) -> str:
    """path: Absolute or relative file path to read"""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool(description="Write content to a file, creates the file if it does not exist")
def write_file(path: str, content: str) -> str:
    """path: Absolute or relative file path to write
    content: The content to write to the file"""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"OK: wrote {len(content)} chars to {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool(description="Replace an exact string in a file with a new string")
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """path: Absolute or relative file path to edit
    old_string: The exact string to find and replace
    new_string: The string to replace it with"""
    try:
        with open(path) as f:
            content = f.read()

        if old_string not in content:
            return f"Error: old_string not found in {path}"

        if content.count(old_string) > 1:
            return f"Error: old_string is not unique in {path}, found {content.count(old_string)} matches"

        new_content = content.replace(old_string, new_string, 1)

        with open(path, "w") as f:
            f.write(new_content)
        return f"OK: replaced in {path}"
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"
