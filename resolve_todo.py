import re
with open("TODO.md", "r", encoding="utf-8") as f:
    content = f.read()

# Custom regex replacement to resolve any conflict blocks
def replacer(match):
    head_content = match.group(1)
    main_content = match.group(2)
    # Return head_content, but if main_content has unique lines, append them.
    # Actually, head_content usually has all the progress.
    lines = head_content.strip().split("\n")
    for line in main_content.strip().split("\n"):
        if line and line not in lines:
            lines.append(line)
    return "\n".join(lines) + "\n"

resolved = re.sub(r"<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> origin/main-3336522528379146385", replacer, content, flags=re.DOTALL)

with open("TODO.md", "w", encoding="utf-8") as f:
    f.write(resolved)
