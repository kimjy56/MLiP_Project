"""Diagnose DNS versus Windows sandbox network denial; never change OS/network settings."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import platform
import socket
import time
import tomllib
import urllib.request


def appcontainer_status():
    if platform.system()!="Windows":return None
    from ctypes import wintypes
    adv=ctypes.WinDLL("advapi32",use_last_error=True)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    adv.OpenProcessToken.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE)]
    adv.GetTokenInformation.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    token=wintypes.HANDLE()
    if not adv.OpenProcessToken(kernel.GetCurrentProcess(),8,ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        value=wintypes.DWORD();needed=wintypes.DWORD()
        if not adv.GetTokenInformation(token,29,ctypes.byref(value),ctypes.sizeof(value),ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(value.value)
    finally:kernel.CloseHandle(token)


def diagnose(check_http=True):
    report={"recorded_at_kst":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
            "dns":{},"tcp":{},"system_settings_modified":False}
    for host in ["localhost","huggingface.co","pypi.org","www.microsoft.com"]:
        try:
            addresses=sorted({r[4][0] for r in socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)})
            report["dns"][host]={"ok":True,"addresses":addresses}
        except OSError as exc:
            report["dns"][host]={"ok":False,"errno":exc.errno,"winerror":getattr(exc,"winerror",None),"message":str(exc)}
    for host in ["1.1.1.1","8.8.8.8"]:
        tick=time.monotonic()
        try:
            with socket.create_connection((host,443),timeout=3):pass
            report["tcp"][host]={"ok":True}
        except OSError as exc:
            report["tcp"][host]={"ok":False,"errno":exc.errno,"winerror":getattr(exc,"winerror",None),"message":str(exc)}
        report["tcp"][host]["seconds"]=round(time.monotonic()-tick,3)
    try:report["process_is_appcontainer"]=appcontainer_status()
    except Exception as exc:report["token_check_error"]=type(exc).__name__
    config=Path.home()/".codex/config.toml"
    try:
        content=tomllib.loads(config.read_text(encoding="utf-8-sig"))
        report["configured_workspace_network_access"]=content.get("sandbox_workspace_write",{}).get("network_access")
        report["configured_windows_sandbox"]=content.get("windows",{}).get("sandbox")
    except (OSError,ValueError):report["configured_workspace_network_access"]=None
    blocked=report.get("process_is_appcontainer") and all(
        item.get("winerror")==10013 for item in report["tcp"].values())
    if blocked:
        report["diagnosis"]="sandbox_network_access_denied"
        report["resolution_scope"]="Active execution permissions, not a DNS-server replacement"
    elif not report["dns"]["huggingface.co"]["ok"]:
        report["diagnosis"]="dns_resolution_failed_requires_host_comparison"
    else:
        report["diagnosis"]="dns_available"
    if check_http and report["dns"]["huggingface.co"]["ok"]:
        try:
            url="https://huggingface.co/api/models/LGAI-EXAONE/EXAONE-4.0-1.2B"
            with urllib.request.urlopen(url,timeout=15) as response:
                value=json.load(response)
            report["huggingface_https"]={"ok":True,"model_id":value.get("id"),"revision":value.get("sha")}
        except Exception as exc:
            report["huggingface_https"]={"ok":False,"error_type":type(exc).__name__,"message":str(exc)}
    report["ready_for_model_download"]=bool(report.get("huggingface_https",{}).get("ok"))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report=diagnose()
    if args.output:
        if args.output.exists():raise FileExistsError("Do not replace an earlier network diagnosis")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report["ready_for_model_download"] else (2 if report["diagnosis"]=="sandbox_network_access_denied" else 1))


if __name__=="__main__":
    main()
