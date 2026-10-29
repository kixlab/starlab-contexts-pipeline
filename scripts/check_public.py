"""Scan tracked files, optionally all reachable history; never print secrets."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SECRET_RULES = {
    "OpenAI key": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}",
    "Google API key": rb"\bAIza[0-9A-Za-z_-]{30,}",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "private key": rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    "machine-specific path": rb"/(?:home/[^/\s\"']+|Users/[^/\s\"']+)/",
}
FORBIDDEN = re.compile(r"(^|/)(?:\.env(?:\.[^/]*)?|\.claude|\.codex|\.agents|\.DS_Store|__pycache__|node_modules|\.next|pilot-study-records|logs|raw_organization_data(?:_old)?|raw_video_data)(?:/|$)|(?:^|/)task_rep/(?:agentic|design_agent)/|(?:^|/)[^/]*cookies[^/]*\.txt$")


def git(*args, **kwargs):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Scan all objects reachable from branches and tags")
    args = parser.parse_args()
    findings = []

    def check_path(path):
        if FORBIDDEN.search(path) and not path.endswith((".example", ".local.example")):
            findings.append((path, "excluded path"))

    def check_content(path, data):
        for label, pattern in SECRET_RULES.items():
            if re.search(pattern, data):
                findings.append((path, label))

    paths = git("ls-files", "-z").decode().split("\0")
    for path in filter(None, paths):
        check_path(path)
        full = ROOT / path
        if full.is_file():
            check_content(path, full.read_bytes())

    if args.history:
        objects = git("rev-list", "--objects", "--all").decode().splitlines()
        names = {}
        for line in objects:
            oid, _, path = line.partition(" ")
            names[oid] = path
            check_path(path)
        for path in git("log", "--all", "--format=", "--name-only", "-z").decode().split("\0"):
            check_path(path.strip("\n"))
        metadata = git("cat-file", "--batch-check=%(objectname) %(objecttype)",
                       input=("\n".join(names) + "\n").encode()).decode()
        proc = subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "--batch"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for line in metadata.splitlines():
                oid, kind = line.split()
                if kind not in {"blob", "commit", "tag"}:
                    continue
                proc.stdin.write((oid + "\n").encode()); proc.stdin.flush()
                size = int(proc.stdout.readline().split()[2])
                data = proc.stdout.read(size); proc.stdout.read(1)
                check_content(names[oid] or oid, data)
        finally:
            proc.stdin.close(); proc.stdout.close(); proc.wait()
    for path, rule in sorted(set(findings)):
        print(f"FAIL {path}: {rule}")
    if findings:
        raise SystemExit(1)
    print("Public-file scan passed" + (" (including reachable Git history)" if args.history else ""))


if __name__ == "__main__":
    main()
