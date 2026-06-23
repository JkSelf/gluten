#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import base64
import hashlib
import hmac
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, quote
from pyspark import SparkConf
from pyspark.sql import SparkSession
DAS = "org.apache.hadoop.fs.azurebfs.sas.IbmlhcasSASTokenProvider"
FIXED = "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider"
OAUTH = "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
CASES = ["oauth", "oauth-das", "sharedkey", "sharedkey-das", "sas", "sas-das"]
def load_config():
    # Configuration is intentionally supplied by environment variables or CLI args.
    pass


def e(key, default=""):
    return os.getenv(key, "").strip() or default


def first(*keys):
    return next((e(key) for key in keys if e(key)), "")


def truth(key, default="false"):
    return e(key, default).lower() in {"1", "true", "yes", "y", "on"}


def acct(args):
    return f"{args.account}.dfs.core.windows.net"


def hkey(args, name):
    return f"spark.hadoop.fs.azure.{name}.{acct(args)}"


def sas_time(value):
    value = value.strip()
    value = value[:-1] + "+00:00" if value.endswith("Z") else value
    if "." in value:
        head, tail = value.split(".", 1)
        for sign in ("+", "-"):
            if sign in tail:
                frac, tz = tail.split(sign, 1)
                value = f"{head}.{frac[:6]}{sign}{tz}"
                break
    parsed = datetime.fromisoformat(value)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def valid_sas(token):
    expiry = parse_qs(token.lstrip(
        "?"), keep_blank_values=True).get("se", [""])[0]
    return bool(expiry and sas_time(expiry) > datetime.now(timezone.utc) + timedelta(minutes=10))


def generate_sas(args):
    key = first("ABFS_ACCOUNT_KEY", "AZURE_STORAGE_ACCOUNT_KEY")
    if not key:
        raise ValueError("missing ABFS_ACCOUNT_KEY")
    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=5)
             ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    expiry = (now + timedelta(hours=int(e("ABFS_GENERATED_SAS_EXPIRY_HOURS", "48")))
              ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    perm, services, resources = e("ABFS_SAS_PERMISSIONS", "rwdlacup"), e(
        "ABFS_SAS_SERVICES", "b"), e("ABFS_SAS_RESOURCE_TYPES", "sco")
    protocol, version, ip = e("ABFS_SAS_PROTOCOL", "https"), e(
        "ABFS_SAS_VERSION", "2020-02-10"), e("ABFS_SAS_IP")
    text = "\n".join([args.account, perm, services, resources,
                     start, expiry, ip, protocol, version]) + "\n"
    sig = base64.b64encode(hmac.new(base64.b64decode(
        key), text.encode(), hashlib.sha256).digest()).decode()
    params = [("sv", version), ("ss", services), ("srt", resources),
              ("sp", perm), ("se", expiry), ("st", start)]
    if ip:
        params.append(("sip", ip))
    return "&".join(f"{k}={quote(v, safe='')}" for k, v in params + [("spr", protocol), ("sig", sig)])


def fixed_sas(args):
    token = first("ABFS_SAS_TOKEN", "AZURE_STORAGE_SAS_TOKEN")
    if token and valid_sas(token):
        return token, ""
    if not truth("ABFS_GENERATE_SAS_TOKEN", "true"):
        return "", "ABFS_SAS_TOKEN is expired or missing" if token else "missing ABFS_SAS_TOKEN"
    try:
        print("[sas] generated SAS token from ABFS_ACCOUNT_KEY", flush=True)
        return generate_sas(args), ""
    except Exception as exc:
        return "", f"missing or expired ABFS_SAS_TOKEN and generation failed: {exc}"


def base_conf(args):
    conf = SparkConf().setAll([
        ("spark.hadoop.fs.azure.createRemoteFileSystemDuringInitialization",
         e("AZURE_CREATE_REMOTE_FILESYSTEM_DURING_INITIALIZATION", "false")),
    ])
    return conf


def add_das(conf, args):
    missing = [key for key in (
        "DAS_ENDPOINT", "INSTANCE_ID", "API_KEY") if not e(key)]
    if missing:
        return False, "missing " + ", ".join(missing)
    conf.setAll([
        ("spark.hadoop.wxd.cas.endpoint", e("DAS_ENDPOINT")
         ), ("spark.hadoop.wxd.instanceId", e("INSTANCE_ID")),
        ("spark.hadoop.wxd.apikey", e("API_KEY")), ("spark.hadoop.wxd.cas.ssl.no.verify", e(
            "WXD_CAS_SSL_NO_VERIFY", "true")),
        ("spark.hadoop.wxd.cas.sas.expiry.period", e("WXD_CAS_SAS_EXPIRY_PERIOD",
         "600")), (hkey(args, "sas.token.provider.type"), DAS),
    ])
    return True, ""


def configure(conf, args, case):
    case = case.lower()
    if case in {"oauth", "oauth-das"}:
        cid, tenant, secret = first("AZURE_CLIENT_ID", "ABFS_CLIENT_ID"), first(
            "AZURE_TENANT_ID", "ABFS_TENANT_ID"), first("AZURE_CLIENT_SECRET", "ABFS_CLIENT_SECRET")
        missing = [name for name, value in [("AZURE_CLIENT_ID", cid), (
            "AZURE_TENANT_ID", tenant), ("AZURE_CLIENT_SECRET", secret)] if not value]
        if missing:
            return False, "missing " + ", ".join(missing)
        conf.setAll([(hkey(args, "account.auth.type"), "OAuth"), (hkey(args, "account.oauth.provider.type"), OAUTH),
                     (hkey(args, "account.oauth2.client.id"), cid), (hkey(
                         args, "account.oauth2.client.secret"), secret),
                     (hkey(args, "account.oauth2.client.endpoint"), f"https://login.microsoftonline.com/{tenant}/oauth2/token")])
        if case.endswith("-das"):
            conf.set(hkey(args, "sas.token.provider.type"), DAS)
        return True, ""
    if case in {"sharedkey", "sharedkey-das"}:
        key = first("ABFS_ACCOUNT_KEY", "AZURE_STORAGE_ACCOUNT_KEY")
        if not key:
            return False, "missing ABFS_ACCOUNT_KEY"
        conf.set(hkey(args, "account.auth.type"), "SharedKey").set(
            hkey(args, "account.key"), key)
        if case.endswith("-das"):
            conf.set(hkey(args, "sas.token.provider.type"), DAS)
        return True, ""
    if case == "sas-das":
        conf.set(hkey(args, "account.auth.type"), "SAS")
        return add_das(conf, args)
    if case == "sas":
        token, reason = fixed_sas(args)
        if not token:
            return False, reason
        conf.setAll([(hkey(args, "account.auth.type"), "SAS"), (hkey(
            args, "sas.token.provider.type"), FIXED), (hkey(args, "sas.fixed.token"), token)])
        return True, ""
    return False, f"unknown case '{case}'"


def run_case(args, case):
    active = SparkSession.getActiveSession()
    if active:
        active.stop()
    conf = base_conf(args)
    ok, reason = configure(conf, args, case)
    if not ok:
        return "skip", reason
    spark = SparkSession.builder.master(args.master).appName(
        f"abfs-auth-{case}").config(conf=conf).getOrCreate()
    try:
        spark.sparkContext.setLogLevel(args.log_level)
        path = f"abfss://{args.container}@{args.account}.dfs.core.windows.net/{args.path_prefix}/{args.run_id}/{case}/parquet"
        print(f"[{case}] writing parquet smoke data", flush=True)
        spark.createDataFrame([(1, case), (2, case)], "id INT, auth_case STRING").write.mode(
            "overwrite").parquet(path)
        spark.read.parquet(path).createOrReplaceTempView("abfs_auth_smoke")
        row = spark.sql(
            "SELECT auth_case, COUNT(*) row_count, SUM(id) id_sum FROM abfs_auth_smoke GROUP BY auth_case").collect()
        assert len(
            row) == 1 and row[0]["auth_case"] == case and row[0]["row_count"] == 2 and row[0]["id_sum"] == 3, row
        if args.show_plan:
            print(spark.sql("SELECT * FROM abfs_auth_smoke WHERE id = 1")
                  ._jdf.queryExecution().executedPlan(), flush=True)
        print(f"[{case}] PASS", flush=True)
        return "pass", ""
    finally:
        spark.stop()


def child_command(args, case):
    cmd = [sys.executable, os.path.abspath(__file__), "--cases", case, "--account", args.account, "--container", args.container,
           "--path-prefix", args.path_prefix, "--run-id", args.run_id, "--master", args.master, "--off-heap-size", args.off_heap_size,
           "--log-level", args.log_level, "--child-case"]
    for flag, on in [("--show-plan", args.show_plan), ("--strict", args.strict)]:
        if on:
            cmd.append(flag)
    return cmd


def child_env(args, case):
    child = dict(os.environ, ABFS_AUTH_CASES=case,
                 RUN_ID=args.run_id, ABFS_AUTH_MODES_CHILD="true")
    for key in list(child):
        if key.startswith("PYSPARK_GATEWAY") or key in {"PYSPARK_DRIVER_CONN_INFO_PATH", "_PYSPARK_DRIVER_CONN_INFO_PATH"}:
            child.pop(key, None)
    return child


def run_children(args, cases):
    print(f"Cases: {', '.join(cases)}", flush=True)
    failures, skipped = [], []
    for case in cases:
        print(f"[{case}] START isolated process", flush=True)
        try:
            code = subprocess.run(child_command(args, case), env=child_env(
                args, case), check=False).returncode
        except FileNotFoundError:
            code = 127
        if code == 0:
            print(f"[{case}] isolated process PASS", flush=True)
        elif code == 77:
            skipped.append(case)
            print(f"[{case}] isolated process SKIP", flush=True)
        else:
            failures.append(case)
            print(f"[{case}] isolated process FAIL: exit code {code}",
                  file=sys.stderr, flush=True)
            if not args.continue_on_failure:
                break
    return finish(failures, skipped, False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ABFS auth smoke tests.")
    account = e("ABFS_ACCOUNT_NAME")
    container = e("ABFS_CONTAINER_NAME")
    parser.add_argument("--account", default=account, required=not account)
    parser.add_argument("--container", default=container, required=not container)
    for name, default in [("cases", e("ABFS_AUTH_CASES", "all")), ("path-prefix",
                                                                   e("ABFS_TEST_PATH_PREFIX", "abfs-auth-smoke")),
                          ("run-id", e("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
                           ), ("master", e("SPARK_MASTER", "local[1]")),
                          ("off-heap-size", e("SPARK_OFF_HEAP_SIZE", "2g")), ("log-level", e("SPARK_LOG_LEVEL", "WARN"))]:
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--child-case", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def finish(failures, skipped, child_case):
    if skipped:
        print("Skipped: " + ", ".join(skipped), flush=True)
    if failures:
        print("Failed: " + ", ".join(failures), file=sys.stderr, flush=True)
        return 1
    if skipped and child_case:
        return 77
    print("All runnable ABFS auth cases passed", flush=True)
    return 0


def main():
    load_config()
    args = parse_args()
    cases = CASES if args.cases.strip().lower() == "all" else [
        case.strip() for case in args.cases.split(",") if case.strip()]
    if len(cases) > 1:
        return 2 if args.child_case else run_children(args, cases)
    print(f"Cases: {', '.join(cases)}", flush=True)
    failures, skipped = [], []
    for case in cases:
        try:
            status, reason = run_case(args, case)
        except Exception as exc:
            status, reason = "fail", str(exc)
        if status == "skip":
            print(f"[{case}] SKIP: {reason}", flush=True)
            (failures if args.strict else skipped).append(case)
        elif status == "fail":
            failures.append(case)
            print(f"[{case}] FAIL: {reason}", file=sys.stderr, flush=True)
            if not args.continue_on_failure:
                break
    return finish(failures, skipped, args.child_case)


if __name__ == "__main__":
    sys.exit(main())
