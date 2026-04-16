import requests
import csv
import io
import time
import sys

BASE = "http://127.0.0.1:8000/api"
PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def upload_csv(content, filename):
    return requests.post(f"{BASE}/upload", files={"file": (filename, content, "text/csv")})


def test_health():
    r = requests.get(f"{BASE}/health")
    report("Health check", r.status_code == 200)


def test_upload_empty():
    r = upload_csv(b"", "empty.csv")
    report("Empty file rejected (400)", r.status_code == 400, f"got {r.status_code}")


def test_upload_non_csv():
    r = requests.post(f"{BASE}/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    report("Non-CSV rejected (400)", r.status_code == 400, f"got {r.status_code}")


def test_upload_normal():
    csv_content = b"id,age,income,target\n1,25,50000,1\n2,35,75000,0\n3,45,90000,1\n4,28,55000,0\n5,32,65000,1\n6,41,85000,0\n7,29,52000,1\n8,38,78000,0\n9,22,49000,1\n10,36,73000,0\n"
    r = upload_csv(csv_content, "normal.csv")
    report("Normal CSV accepted (200)", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
    if r.status_code == 200:
        return r.json().get("filename")
    return None


def test_upload_heavy_missing():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "age", "income", "education", "target"])
    for i in range(100):
        writer.writerow([
            i,
            "" if i % 2 == 0 else str(20 + i % 50),
            "" if i % 3 == 0 else str(30000 + i * 100),
            "" if i % 4 == 0 else ["Bachelor", "Master", "PhD"][i % 3],
            i % 2,
        ])
    r = upload_csv(buf.getvalue().encode(), "heavy_missing.csv")
    report("Heavy missing CSV accepted (200)", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")


def test_upload_all_text():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "desc", "category"])
    for i in range(50):
        writer.writerow([f"item_{i}", f"description text {i}" * 5, f"cat_{i % 3}"])
    r = upload_csv(buf.getvalue().encode(), "all_text.csv")
    report("All-text CSV accepted (200)", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")


def test_upload_large_rejected():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id"] + [f"feat_{j}" for j in range(20)] + ["target"])
    for i in range(50001):
        row = [i] + [float(i * j % 100) / 10 for j in range(20)] + [i % 2]
        writer.writerow(row)
    r = upload_csv(buf.getvalue().encode(), "large.csv")
    report("Large CSV (>50k rows) rejected (413)", r.status_code == 413, f"got {r.status_code}: {r.text[:200]}")


def test_upload_too_few_rows():
    csv_content = b"id,target\n1,0\n"
    r = upload_csv(csv_content, "tiny.csv")
    report("Too few rows rejected (400)", r.status_code == 400, f"got {r.status_code}: {r.text[:200]}")


def test_train_invalid_task():
    r = requests.post(f"{BASE}/train", json={
        "filename": "nonexist.csv",
        "task_type": "invalid_type",
        "target_column": "target",
    })
    report("Invalid task_type rejected (400)", r.status_code == 400, f"got {r.status_code}")


def test_train_nonexistent_target():
    csv_content = b"age,income,target\n25,50000,1\n35,75000,0\n45,90000,1\n28,55000,0\n32,65000,1\n41,85000,0\n29,52000,1\n38,78000,0\n"
    r = upload_csv(csv_content, "target_test.csv")
    if r.status_code != 200:
        report("Train nonexistent target - upload failed", False, r.text[:200])
        return
    filename = r.json()["filename"]
    r2 = requests.post(f"{BASE}/train", json={
        "filename": filename,
        "task_type": "classification",
        "target_column": "nonexistent_column",
        "use_tuning": False,
        "use_ensembling": False,
    })
    report("Nonexistent target training started (200)", r2.status_code == 200, f"got {r2.status_code}")
    if r2.status_code == 200:
        sid = r2.json()["session_id"]
        for _ in range(30):
            time.sleep(1)
            r3 = requests.get(f"{BASE}/train/status/{sid}")
            data = r3.json()
            if data.get("status") in ("completed", "error"):
                break
        is_error = data.get("status") == "error"
        friendly = "不存在" in data.get("error", "") or "检查" in data.get("error", "")
        report("Nonexistent target -> friendly error", is_error and friendly, f"error={data.get('error', '')[:200]}")


def test_train_nonexistent_file():
    r = requests.post(f"{BASE}/train", json={
        "filename": "nonexistent_abc123.csv",
        "task_type": "classification",
        "target_column": "target",
        "selected_models": ["lr"],
        "use_tuning": False,
        "use_ensembling": False,
    })
    report("Nonexistent file training started (200)", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        sid = r.json()["session_id"]
        for _ in range(30):
            time.sleep(1)
            r2 = requests.get(f"{BASE}/train/status/{sid}")
            data = r2.json()
            if data.get("status") in ("completed", "error"):
                break
        is_error = data.get("status") == "error"
        friendly = "不存在" in data.get("error", "") or "重新上传" in data.get("error", "")
        report("Nonexistent file -> friendly error", is_error and friendly, f"error={data.get('error', '')[:200]}")


def test_train_normal_flow():
    csv_content = b"age,income,education,target\n25,50000,Bachelor,1\n35,75000,Master,0\n45,90000,PhD,1\n28,55000,Bachelor,0\n32,65000,Master,1\n41,85000,PhD,0\n29,52000,Bachelor,1\n38,78000,Master,0\n22,49000,Bachelor,1\n36,73000,Master,0\n"
    r = upload_csv(csv_content, "train_test.csv")
    if r.status_code != 200:
        report("Normal train flow - upload failed", False, r.text[:200])
        return
    filename = r.json()["filename"]
    r2 = requests.post(f"{BASE}/train", json={
        "filename": filename,
        "task_type": "classification",
        "target_column": "target",
        "selected_models": ["lr"],
        "use_tuning": False,
        "use_ensembling": False,
        "use_outlier_removal": False,
        "use_advanced_imputation": False,
    })
    report("Normal train started (200)", r2.status_code == 200, f"got {r2.status_code}")
    if r2.status_code == 200:
        sid = r2.json()["session_id"]
        for _ in range(60):
            time.sleep(2)
            r3 = requests.get(f"{BASE}/train/status/{sid}")
            data = r3.json()
            if data.get("status") in ("completed", "error"):
                break
        report("Normal train completed", data.get("status") == "completed", f"status={data.get('status')}, error={data.get('error', '')[:200]}")


def test_train_stop():
    csv_content = b"age,income,education,target\n25,50000,Bachelor,1\n35,75000,Master,0\n45,90000,PhD,1\n28,55000,Bachelor,0\n32,65000,Master,1\n41,85000,PhD,0\n29,52000,Bachelor,1\n38,78000,Master,0\n22,49000,Bachelor,1\n36,73000,Master,0\n"
    r = upload_csv(csv_content, "stop_test.csv")
    if r.status_code != 200:
        report("Train stop - upload failed", False, r.text[:200])
        return
    filename = r.json()["filename"]
    r2 = requests.post(f"{BASE}/train", json={
        "filename": filename,
        "task_type": "classification",
        "target_column": "target",
        "selected_models": ["rf", "gbc", "xgb"],
        "use_tuning": True,
        "use_ensembling": True,
    })
    if r2.status_code != 200:
        report("Train stop - start failed", False, f"got {r2.status_code}")
        return
    sid = r2.json()["session_id"]
    time.sleep(2)
    r3 = requests.post(f"{BASE}/train/stop/{sid}")
    report("Train stop request accepted", r3.status_code == 200, f"got {r3.status_code}")
    time.sleep(2)
    r4 = requests.get(f"{BASE}/train/status/{sid}")
    data = r4.json()
    report("Train stopped status", data.get("status") == "stopped", f"status={data.get('status')}")


if __name__ == "__main__":
    print("=" * 60)
    print("VibeMine Edge Case Pressure Test v2")
    print("=" * 60)

    tests = [
        ("Basic", [test_health]),
        ("Upload Edge Cases", [
            test_upload_empty,
            test_upload_non_csv,
            test_upload_normal,
            test_upload_heavy_missing,
            test_upload_all_text,
            test_upload_large_rejected,
            test_upload_too_few_rows,
        ]),
        ("Training Edge Cases", [
            test_train_invalid_task,
            test_train_nonexistent_target,
            test_train_nonexistent_file,
            test_train_normal_flow,
            test_train_stop,
        ]),
    ]

    for group_name, group_tests in tests:
        print(f"\n--- {group_name} ---")
        for t in group_tests:
            try:
                t()
            except Exception as e:
                FAIL += 1
                print(f"  ❌ {t.__name__} — Exception: {e}")

    print("\n" + "=" * 60)
    print(f"Results: ✅ {PASS} passed, ❌ {FAIL} failed")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
