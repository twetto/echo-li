import gzip, json, collections, bisect

p = json.load(gzip.open("echo-li-profile.json.gz", "rt", encoding="utf-8"))
syms = json.load(open("echo-li-prof2.json.syms.json", encoding="utf-8"))
sym_strings = syms["string_table"]
libs = p["libs"]

# Build per-module sorted (rva -> name) interval tables, keyed by debug_name.
mod_by_debugname = {}
for m in syms["data"]:
    st = m["symbol_table"]
    starts = [e["rva"] for e in st]
    order = sorted(range(len(st)), key=lambda i: st[i]["rva"])
    starts_sorted = [st[i]["rva"] for i in order]
    ends_sorted = [st[i]["rva"] + e["size"] for i, e in ((j, st[j]) for j in order)]
    names_sorted = [sym_strings[st[i]["symbol"]] for i in order]
    mod_by_debugname[m["debug_name"].lower()] = (starts_sorted, ends_sorted, names_sorted)

def lookup(debugname, rva):
    tab = mod_by_debugname.get((debugname or "").lower())
    if not tab:
        return None
    starts, ends, names = tab
    i = bisect.bisect_right(starts, rva) - 1
    if 0 <= i < len(starts) and rva < ends[i]:
        return names[i]
    return None

KERNEL = {"ntoskrnl.exe", "ntkrnlmp.exe"}

def resolve_func(t, fi):
    """Return a readable label for func index fi in thread t."""
    sa = t["stringArray"]; fn = t["funcTable"]; rt = t["resourceTable"]
    res = fn["resource"][fi]
    libname = None; debugname = None
    if res is not None and res >= 0:
        li = rt["lib"][res]
        if li is not None and li >= 0:
            libname = libs[li]["name"]; debugname = libs[li].get("debugName")
    raw = sa[fn["name"][fi]]
    # RVA is the hex func name (matches frame.address)
    rva = None
    if isinstance(raw, str) and raw.startswith("0x"):
        try: rva = int(raw, 16)
        except ValueError: rva = None
    if rva is not None and debugname:
        nm = lookup(debugname, rva)
        if nm:
            return ("echo-li-cli" if libname == "echo-li-cli.exe" else libname or "?", nm)
    # unresolved: bucket by module
    if libname in KERNEL: return ("[kernel]", libname)
    if libname: return (libname, raw)
    return ("[unknown]", raw)

self_counts = collections.Counter()
mod_counts = collections.Counter()
total = 0
for t in p["threads"]:
    samples = t["samples"]; stackTable = t["stackTable"]
    s_frame = stackTable["frame"]; f_func = t["frameTable"]["func"]
    cache = {}
    for si in samples["stack"]:
        if si is None: continue
        total += 1
        fi = f_func[s_frame[si]]
        if fi not in cache:
            cache[fi] = resolve_func(t, fi)
        mod, nm = cache[fi]
        self_counts[(mod, nm)] += 1
        mod_counts[mod] += 1

print(f"total samples: {total}\n")
print("=== SELF TIME BY MODULE ===")
for mod, c in mod_counts.most_common(15):
    print(f"{100*c/total:6.2f}%  {c:7d}  {mod}")

print("\n=== TOP 35 FUNCTIONS BY SELF TIME ===")
for (mod, nm), c in self_counts.most_common(35):
    label = nm if len(nm) <= 90 else nm[:87] + "..."
    print(f"{100*c/total:6.2f}%  {c:6d}  [{mod}] {label}")
