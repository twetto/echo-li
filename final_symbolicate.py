import gzip, json, collections, bisect, re

# ---- load pdb symbol table ----
procs = []      # (rva, end, name) with real length
starts = []     # (rva, name) all, for nearest-preceding fallback
for line in open("pdb_syms.tsv", encoding="utf-8"):
    rva_s, len_s, name = line.rstrip("\n").split("\t", 2)
    rva = int(rva_s, 16); ln = int(len_s, 16)
    starts.append((rva, name))
    if ln > 0:
        procs.append((rva, rva + ln, name))
procs.sort()
starts.sort()
proc_starts = [p[0] for p in procs]
start_addrs = [s[0] for s in starts]

def name_for(rva):
    i = bisect.bisect_right(proc_starts, rva) - 1
    if 0 <= i < len(procs) and rva < procs[i][1]:
        return procs[i][2]
    j = bisect.bisect_right(start_addrs, rva) - 1
    if 0 <= j < len(starts):
        return starts[j][1] + " (approx)"
    return None

# ---- minimal legacy rust demangler ----
ESC = {"$LT$": "<", "$GT$": ">", "$u20$": " ", "$C$": ",", "$RF$": "&",
       "$BP$": "*", "$LP$": "(", "$RP$": ")", "$u7b$": "{", "$u7d$": "}",
       "$u5b$": "[", "$u5d$": "]", "$u27$": "'", "..": "::"}
def deesc(s):
    for k, v in ESC.items():
        s = s.replace(k, v)
    return s
def demangle(n):
    if n.startswith("_ZN") and n.endswith("E"):
        body = n[3:-1]
        parts, i = [], 0
        while i < len(body):
            m = re.match(r"\d+", body[i:])
            if not m: break
            ln = int(m.group()); i += m.end()
            parts.append(body[i:i+ln]); i += ln
        # drop trailing hash component like 'h0123abcd...'
        if parts and re.fullmatch(r"h[0-9a-f]+", parts[-1]):
            parts.pop()
        return deesc("::".join(parts))
    return n

# ---- aggregate self time over echo-li-cli frames ----
p = json.load(gzip.open("echo-li-profile.json.gz", "rt", encoding="utf-8"))
libs = p["libs"]
self_counts = collections.Counter()
total = 0
for t in p["threads"]:
    sa = t["stringArray"]; fn = t["funcTable"]; rt = t["resourceTable"]
    f_func = t["frameTable"]["func"]; s_frame = t["stackTable"]["frame"]
    cache = {}
    def label(fi):
        if fi in cache: return cache[fi]
        res = fn["resource"][fi]
        libname = None
        if res is not None and res >= 0:
            li = rt["lib"][res]
            if li is not None and li >= 0: libname = libs[li]["name"]
        raw = sa[fn["name"][fi]]
        if libname == "echo-li-cli.exe" and raw.startswith("0x"):
            nm = name_for(int(raw, 16))
            r = ("app", demangle(nm) if nm else raw)
        elif libname in ("ntoskrnl.exe", "ntkrnlmp.exe"):
            r = ("kernel", raw)
        elif libname:
            r = (libname, raw)
        else:
            r = ("?", raw)
        cache[fi] = r; return r
    for si in t["samples"]["stack"]:
        if si is None: continue
        total += 1
        self_counts[label(f_func[s_frame[si]])] += 1

print(f"total samples: {total}\n=== TOP 40 SELF TIME (named) ===")
for (mod, nm), c in self_counts.most_common(40):
    nm = nm if len(nm) <= 95 else nm[:92] + "..."
    print(f"{100*c/total:6.2f}%  {c:6d}  [{mod}] {nm}")
