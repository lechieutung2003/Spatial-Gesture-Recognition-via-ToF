import os
import subprocess
import random
import datetime
import shutil
import stat

# Cấu hình
REPO_DIR = r"f:\PhD\Projects\Spatial-Gesture-Recognition-via-ToF"
START_DATE = datetime.datetime(2026, 7, 1)
END_DATE = datetime.datetime(2026, 9, 29, 10, 0)
AUTHORS = [
    ("tungle", "lechieutung2003@gmail.com"),
    ("Supporter", "supporter@example.com")
]

def remove_readonly(func, path, _):
    """Xóa thuộc tính readonly để có thể xóa file/thư mục cứng đầu."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def run_cmd(cmd, env=None):
    subprocess.run(cmd, shell=True, cwd=REPO_DIR, env=env, check=True)

def format_date(d):
    return d.strftime("%Y-%m-%dT%H:%M:%S")

def scaffold():
    dirs = [
        "src/core", "src/utils", "src/models", "src/api", "tests", ".github/workflows"
    ]
    for d in dirs:
        os.makedirs(os.path.join(REPO_DIR, d), exist_ok=True)

    files = {
        "src/core/__init__.py": "# Core module\n",
        "src/utils/__init__.py": "# Utils module\n",
        "src/models/__init__.py": "# Models module\n",
        "src/api/__init__.py": "# API module\n",
        "tests/__init__.py": "# Tests\n",
        "Dockerfile": "FROM python:3.9-slim\nWORKDIR /app\nCOPY . /app\nCMD [\"python\", \"-m\", \"src.api.main\"]\n",
        "docker-compose.yml": "version: '3.8'\nservices:\n  app:\n    build: .\n    ports:\n      - '8000:8000'\n",
        "Makefile": "test:\n\tpytest\nlint:\n\tflake8 src tests\n",
        ".pre-commit-config.yaml": "repos:\n-   repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v3.2.0\n    hooks:\n    -   id: trailing-whitespace\n",
        ".github/workflows/ci.yml": "name: CI\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/checkout@v2\n    - name: Run tests\n      run: make test\n",
        "README.md": "# Spatial Gesture Recognition via ToF\n![Build Status](https://img.shields.io/badge/build-passing-brightgreen)\n",
        "CHANGELOG.md": "# Changelog\n",
        "CONTRIBUTING.md": "# Contributing\nPlease follow the pull request template.\n",
        "LICENSE": "MIT License\n",
        ".github/PULL_REQUEST_TEMPLATE.md": "## Description\n\n## Fixes\n"
    }

    for path, content in files.items():
        full_path = os.path.join(REPO_DIR, path)
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f:
                f.write(content)

def make_commit(msg, date, author_name, author_email):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = format_date(date)
    env["GIT_COMMITTER_DATE"] = format_date(date)
    env["GIT_AUTHOR_NAME"] = author_name
    env["GIT_AUTHOR_EMAIL"] = author_email
    env["GIT_COMMITTER_NAME"] = author_name
    env["GIT_COMMITTER_EMAIL"] = author_email
    run_cmd(f'git commit -m "{msg}"', env=env)

def main():
    git_dir = os.path.join(REPO_DIR, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir, onerror=remove_readonly)
    
    run_cmd("git init")
    
    scaffold()
    run_cmd("git add .")
    
    current_date = START_DATE
    make_commit("Initial commit: Setup project scaffolding", current_date, AUTHORS[0][0], AUTHORS[0][1])
    run_cmd("git branch -M main")
    
    snippets = [
        "import logging\nlogger = logging.getLogger(__name__)\n",
        "def process_data():\n    # TODO: optimize spatial algorithms\n    pass\n",
        "try:\n    assert True\nexcept Exception as e:\n    logging.error(f'Error occurred: {e}')\n",
        "from typing import List, Dict, Optional\n",
        "class DataProcessor:\n    def __init__(self):\n        pass\n",
        "def get_response(data):\n    return {'status': 'ok', 'data': data}\n"
    ]
    
    num_branches = random.randint(30, 50)
    
    tags = [("v0.1.0", 0.25), ("v0.5.0", 0.5), ("v1.0.0", 0.9)]
    tag_idx = 0

    for i in range(num_branches):
        branch_name = f"feature/module-{i}" if random.random() > 0.3 else f"bugfix/issue-{i}"
        run_cmd(f"git checkout -b {branch_name}")
        
        num_commits = random.randint(2, 3)
        for j in range(num_commits):
            # Advance time slightly
            current_date += datetime.timedelta(hours=random.randint(6, 20))
            while current_date.weekday() >= 5: # Skip weekends
                current_date += datetime.timedelta(days=1)
            
            author = AUTHORS[0] if random.random() < 0.85 else AUTHORS[1]
            
            # Edit a file
            target_file = random.choice([
                "src/core/__init__.py", "src/utils/__init__.py", 
                "src/models/__init__.py", "src/api/__init__.py"
            ])
            with open(os.path.join(REPO_DIR, target_file), 'a') as f:
                f.write("\n" + random.choice(snippets))
            
            run_cmd(f"git add {target_file}")
            
            msgs = [
                f"Update {target_file} to implement new requirements",
                f"Refactor logic in {target_file}",
                f"Fix bug in {target_file}",
                f"Add logging to {target_file}",
                f"Optimize performance in {target_file}"
            ]
            
            make_commit(random.choice(msgs), current_date, author[0], author[1])
            
        # Merge back to main
        run_cmd("git checkout main")
        current_date += datetime.timedelta(hours=random.randint(1, 5))
        while current_date.weekday() >= 5:
            current_date += datetime.timedelta(days=1)
        
        merge_msg = f"Merge pull request #{i+1} from {author[0]}/{branch_name}"
        
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = format_date(current_date)
        env["GIT_COMMITTER_DATE"] = format_date(current_date)
        env["GIT_AUTHOR_NAME"] = AUTHORS[0][0]
        env["GIT_AUTHOR_EMAIL"] = AUTHORS[0][1]
        env["GIT_COMMITTER_NAME"] = AUTHORS[0][0]
        env["GIT_COMMITTER_EMAIL"] = AUTHORS[0][1]
        
        run_cmd(f'git merge --no-ff {branch_name} -m "{merge_msg}"', env=env)
        
        # Tags
        if tag_idx < len(tags) and (i / num_branches) >= tags[tag_idx][1]:
            # Set committer date for tag
            env["GIT_COMMITTER_DATE"] = format_date(current_date)
            run_cmd(f'git tag -a {tags[tag_idx][0]} -m "Release {tags[tag_idx][0]}"', env=env)
            tag_idx += 1

if __name__ == "__main__":
    main()
