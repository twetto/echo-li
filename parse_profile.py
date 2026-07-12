import gzip, json, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "echo-li-profile.json.gz"
with gzip.open(path, "rt", encoding="utf-8") as f:
    prof = json.load(f)

threads = prof["threads"]

def strings(t):
    # newer profiles: stringArray on thread; older: shared stringTable
    if "stringArray" in t:
        return t["stringArray"]
    if "stringTable" in t:
        st = t["stringTable"]
        return st["_array"] if isinstance(st, dict) else st
    return prof.get("shared", {}).get("stringArray")

self_counts = collections.Counter()   # leaf self-time
incl_counts = collections.Counter()   # inclusive (function anywhere on stack)
total_samples = 0
per_thread = collections.Counter()

for t in threads:
    sarr = strings(t)
    samples = t["samples"]
    stack_col = samples["stack"]
    stackTable = t["stackTable"]
    s_prefix = stackTable["prefix"]
    s_frame = stackTable["frame"]
    frameTable = t["frameTable"]
    f_func = frameTable["func"]
    funcTable = t["funcTable"]
    fn_name = funcTable["name"]

    def func_of_stack(si):
        return f_func[s_frame[si]]

    tname = sarr[t["name"]] if isinstance(t.get("name"), int) else t.get("name", "?")

    for si in stack_col:
        if si is None:
            continue
        total_samples += 1
        per_thread[tname] += 1
        # self: leaf frame's function
        leaf_func = func_of_stack(si)
        self_counts[sarr[fn_name[leaf_func]]] += 1
        # inclusive: walk prefix chain, unique funcs
        seen = set()
        cur = si
        while cur is not None:
            fn = sarr[fn_name[func_of_stack(cur)]]
            if fn not in seen:
                seen.add(fn)
                incl_counts[fn] += 1
            cur = s_prefix[cur]

print(f"total samples: {total_samples}\n")

print("=== TOP 30 BY SELF TIME ===")
for name, c in self_counts.most_common(30):
    print(f"{100*c/total_samples:6.2f}%  {c:6d}  {name}")

print("\n=== TOP 25 BY INCLUSIVE TIME ===")
for name, c in incl_counts.most_common(25):
    print(f"{100*c/total_samples:6.2f}%  {c:6d}  {name}")

print("\n=== SAMPLES PER THREAD ===")
for name, c in per_thread.most_common(20):
    print(f"{100*c/total_samples:6.2f}%  {c:6d}  {name}")
