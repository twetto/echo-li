import gzip, json, collections, subprocess, sys

SYM = r"C:\Users\twetto\scoop\apps\llvm\current\bin\llvm-symbolizer.exe"
EXE = r"target\release\echo-li-cli.exe"

p = json.load(gzip.open("profile.json.gz", "rt", encoding="utf-8"))
libs = p["libs"]

# leaf self-time: collect per-rva counts for echo-li-cli.exe, bucket others by module
app_rva = collections.Counter()
mod_other = collections.Counter()
total = 0
for t in p["threads"]:
    sa = t["stringArray"]; fn = t["funcTable"]; rt = t["resourceTable"]
    f_func = t["frameTable"]["func"]; s_frame = t["stackTable"]["frame"]
    cache = {}
    def info(fi):
        if fi in cache: return cache[fi]
        res = fn["resource"][fi]; libname = None
        if res is not None and res >= 0:
            li = rt["lib"][res]
            if li is not None and li >= 0: libname = libs[li]["name"]
        raw = sa[fn["name"][fi]]
        cache[fi] = (libname, raw); return cache[fi]
    for si in t["samples"]["stack"]:
        if si is None: continue
        total += 1
        libname, raw = info(f_func[s_frame[si]])
        if libname == "echo-li-cli.exe" and raw.startswith("0x"):
            app_rva[int(raw, 16)] += 1
        else:
            mod_other[libname or "?"] += 1

# batch-symbolize unique app rvas via llvm-symbolizer
uniq = list(app_rva)
stdin = "\n".join(hex(a) for a in uniq) + "\n"
proc = subprocess.run(
    [SYM, "--obj=" + EXE, "--relative-address", "--no-inlines",
     "--output-style=LLVM", "--functions=short"],
    input=stdin, capture_output=True, text=True)
# parse: per address -> first non-empty line is the function name, then a location line, blank sep
names = []
for block in proc.stdout.split("\n\n"):
    lines = [l for l in block.splitlines() if l.strip()]
    names.append(lines[0] if lines else "??")
if len(names) < len(uniq):
    names += ["??"] * (len(uniq) - len(names))

by_func = collections.Counter()
for a, nm in zip(uniq, names):
    by_func[("app", nm)] += app_rva[a]
for m, c in mod_other.items():
    by_func[("mod", m)] += c

print(f"total samples: {total}   app addrs: {len(uniq)} unique\n=== TOP 40 SELF TIME (llvm-symbolizer) ===")
for (k, nm), c in by_func.most_common(40):
    nm = nm if len(nm) <= 95 else nm[:92] + "..."
    print(f"{100*c/total:6.2f}%  {c:6d}  [{k}] {nm}")
