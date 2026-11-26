import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import re

VALID_SET = {
    "cd", "ls", "ll", "pwd", "exit", "cat", "cp", "head", "find", "chmod", "nano", "man",
    "ssh", "ssh-add", "scp", "wget", "curl", "ip", "open", "passwd",
    "nmap", "sqlmap", "msfconsole", "john", "fcrackzip", "python", "python3",
    "clear", "ping", "sh", "nc", "service", "ss", "sftp", "touch", "less", "git",
    "history", "test", "ifconfig", "ssh-copy-id", "msfvenom", "python2", "unzip", "dpkg"
}
TYPO = {
    "mfsconsole": "msfconsole",
    "ipconfig": "ifconfig",
    "ccd": "cd",
    "/ls": "ls",
}
INTENT = {
    "metasploit": "msfconsole",
    "msf5": "msfconsole",
    "Downloads": "cd Downloads",
    "/opt/level-5": "cd /opt/level-5",
    "script.py": "./script.py",
    "sshjohn.py": "./sshjohn.py",
    "msf": "msfconsole",
    "/opt/metasploit-framework/bin/msfconsole": "msfconsole",
}

CORRECTIONS = {**TYPO, **INTENT}

def apply_label(row: pd.Series) -> str:
    if row["first"] in VALID_SET:
        return "valid"
    if row["first"] in TYPO:
        return "typo"
    if row["first"] in INTENT:
        return "intent"
    return "non_command"


def apply_target(row: pd.Series):
    if row["label"] == "valid":
        return row["actual"]
    if row["label"] == "non_command":
        return None
    return row["actual"].replace(row["first"], CORRECTIONS[row["first"]], 1)


def label_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Strips actual, extracts the command name (ignoring a leading sudo), and adds label and target columns."""
    df = df.copy()
    df["actual"] = df["actual"].str.strip()
    df["first"] = df["actual"].str.replace(r"^sudo\s+", "", regex=True).str.split().str[0]
    # TODO: add custom set params to label passed through args=
    df["label"] = df.apply(apply_label, axis=1)
    df["target"] = df.apply(apply_target, axis=1)
    return df


def command_name(s: pd.Series) -> pd.Series:
    return s.str.strip().str.replace(r"^sudo\s+", "", regex=True).str.split().str[0]

def build_prefix(s: pd.Series) -> pd.DataFrame:
    parts = s.str.extract(r"^(?P<sudo>sudo\s+)?(?P<cmd>\S+)\s*(?P<rest>.*)")
    parts["sudo"] = parts["sudo"].fillna("")
    parts["prefix"] = parts["sudo"] + parts["cmd"] + " "
    return parts[["prefix", "rest"]]


def score(pred: pd.Series, target: pd.Series):
    """Score by matching on the full string and on the command name, returning both boolean series"""
    pred = pred.str.strip()
    target = target.str.strip()
    s1 = command_name(pred)
    s2 = command_name(target)
    exact = pred == target
    cmd_match = s1 == s2
    return (exact, cmd_match)


def run_mcnemar(model_right: pd.Series, base_right: pd.Series, name: str):
    b = int((model_right & ~base_right).sum())  
    c = int((~model_right & base_right).sum())  
    a = int((model_right & base_right).sum())
    d = int((~model_right & ~base_right).sum())
    res = mcnemar([[a, b], [c, d]], exact=True)
    print(f"{name}: both right={a}, only model={b}, only baseline={c}, both wrong={d}, p={res.pvalue:.3f}")

