# tar/.tar.gz 支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `tianyi_importer.py` 和 `lutan_importer.py` 中增加 `.tar`、`.tar.gz`、`.tgz` 归档的原生支持，通过 GDAL `/vsitar/` 虚拟文件系统流式读取，无需解压。

**Architecture:** 在 `_prepare_input_path()` 中增加 TAR 检测标志 `is_tar`，在各路径构建和 XML 解析方法中平行于 `is_zip` 增加 `is_tar` 分支。`common_processing.py` 的 `resolve_manifest_path()` 增加 `storage == "tar"` 处理。

**Tech Stack:** Python `tarfile` 标准库、GDAL `/vsitar/` VFS、现有 ZIP 逻辑作参考。

---

## 文件变更概览

| 文件 | 改动 |
|---|---|
| `scripts/tianyi_importer.py` | 增加 `is_tar` 标志、`_discover_tar_members()`、`parse_xml_root()` 的 TAR 分支、`build_slc_path()/build_member_path()` 的 TAR 分支、manifest 的 `"storage": "tar"` |
| `scripts/lutan_importer.py` | 同上，完全平行 |
| `scripts/common_processing.py` | `resolve_manifest_path()` 增加 `storage == "tar"` → `/vsitar/` 处理 |

---

## Task 1: tianyi_importer.py — 核心改动

**Files:**
- Modify: `scripts/tianyi_importer.py`

### Step 1: 增加 `is_tar` 标志和 `tarfile` import

在 `__init__` 中，`is_zip = False` 后增加 `is_tar = False`。

在文件顶部 import 列表中增加 `import tarfile`。

- [ ] **Step 1: 修改 `__init__` 和 import**

```python
# import zipfile 已有
import tarfile  # 新增

class TianyiImporter:
    def __init__(self, product_path: str):
        self.product_path = Path(product_path)
        self._archive_dir = None
        self.is_zip = False
        self.is_tar = False  # 新增
        self._prepare_input_path()
```

### Step 2: 修改 `_prepare_input_path()` 增加 TAR 检测

在 `is_zip` 检测后增加：

```python
        if self.product_path.is_file() and self.product_path.suffix.lower() == ".zip":
            self.is_zip = True
            return
        # 新增
        suffix = self.product_path.suffix.lower()
        if suffix in (".tar", ".tar.gz", ".tgz"):
            self.is_tar = True
            return
        # 现有的其他归档 fallback 逻辑保持不变
        if self.product_path.is_file():
            tmp_dir = Path(shutil.mkdtemp(prefix="tianyi_import_"))
            shutil.unpack_archive(str(self.product_path), str(tmp_dir))
            ...
```

- [ ] **Step 2: 更新 `_prepare_input_path()`**

### Step 3: 增加 `_discover_tar_members()` 方法

在 `_discover_zip_members()` 后增加，逻辑完全平行：

```python
    def _discover_tar_members(self) -> dict[str, str]:
        members: dict[str, str] = {}
        with tarfile.open(self.product_path) as tf:
            for name in tf.getnames():
                low = name.lower()
                if (
                    low.endswith(".xml")
                    and "/annotation/" in low
                    and "/annotation/calibration/" not in low
                ):
                    members["annotation"] = name
                elif low.endswith(".xml") and "/annotation/calibration/" in low:
                    members["calibration"] = name
                elif low.endswith("manifest.safe"):
                    members["manifest"] = name
                elif low.endswith(".tiff") and "/measurement/" in low:
                    members["tiff"] = name
        return members
```

- [ ] **Step 3: 添加 `_discover_tar_members()` 方法**

### Step 4: 更新 `discover_files()`

```python
    def discover_files(self) -> dict[str, str]:
        if self.is_zip:
            return self._discover_zip_members()
        if self.is_tar:  # 新增
            return self._discover_tar_members()
        return self._discover_dir_members()
```

- [ ] **Step 4: 更新 `discover_files()`**

### Step 5: 更新 `parse_xml_root()` 增加 `is_tar` 分支

```python
    def parse_xml_root(self, annotation_name: str) -> ET.Element:
        if self.is_zip:
            with zipfile.ZipFile(self.product_path) as zf:
                with zf.open(annotation_name) as f:
                    return ET.fromstring(f.read())
        if self.is_tar:  # 新增
            with tarfile.open(self.product_path) as tf:
                with tf.extractfile(annotation_name) as f:
                    return ET.fromstring(f.read())
        return ET.parse(annotation_name).getroot()
```

- [ ] **Step 5: 更新 `parse_xml_root()`**

### Step 6: 更新 `build_slc_path()` 增加 `is_tar` 分支

```python
    def build_slc_path(self, tiff_name: str) -> str:
        if self.is_zip:
            return f"/vsizip/{self.product_path}/{tiff_name}"
        if self.is_tar:  # 新增
            return f"/vsitar/{self.product_path}/{tiff_name}"
        return str(Path(tiff_name).resolve())
```

- [ ] **Step 6: 更新 `build_slc_path()`**

### Step 7: 更新 `build_member_path()` 增加 `is_tar` 分支

```python
    def build_member_path(self, member_name: str) -> str:
        if self.is_zip:
            return f"/vsizip/{self.product_path}/{member_name}"
        if self.is_tar:  # 新增
            return f"/vsitar/{self.product_path}/{member_name}"
        return str(Path(member_name).resolve())
```

- [ ] **Step 7: 更新 `build_member_path()`**

### Step 8: 更新 manifest 引用中的 `storage` 值

在 `_manifest_ref()` 方法中，当 `member is not None` 时返回的 dict 需要区分 `is_tar` 的情况：

```python
        if member is not None:
            if self.is_tar:
                return {
                    "path": self._relative_to_output(output_dir, path_value),
                    "storage": "tar",
                    "member": member,
                }
            if self.is_zip:
                return {
                    "path": self._relative_to_output(output_dir, path_value),
                    "storage": "zip",
                    "member": member,
                }
            return {
                "path": self._relative_to_output(output_dir, path_value),
            }
        return self._relative_to_output(output_dir, path_value)
```

检查所有调用 `_manifest_ref()` 的地方，确认 `is_tar` 时代码路径正确。

- [ ] **Step 8: 更新 `_manifest_ref()` 和相关调用点**

---

## Task 2: lutan_importer.py — 核心改动

**Files:**
- Modify: `scripts/lutan_importer.py`

### Step 1: 增加 `is_tar` 标志和 `tarfile` import

```python
import tarfile  # 新增

class LutanImporter:
    def __init__(self, product_path: str):
        self.product_path = Path(product_path)
        self._archive_dir = None
        self.is_zip = False
        self.is_tar = False  # 新增
        self._prepare_input_path()
```

- [ ] **Step 1: 修改 `__init__` 和 import**

### Step 2: 修改 `_prepare_input_path()` 增加 TAR 检测

```python
        if self.product_path.is_file() and self.product_path.suffix.lower() == ".zip":
            self.is_zip = True
            return
        # 新增
        suffix = self.product_path.suffix.lower()
        if suffix in (".tar", ".tar.gz", ".tgz"):
            self.is_tar = True
            return
        # fallback 解压逻辑保持不变
```

- [ ] **Step 2: 更新 `_prepare_input_path()`**

### Step 3: 增加 `_discover_tar_members()` 方法

在 `_discover_zip_members()` 后增加，逻辑平行于 `_discover_dir_members()`（Lutan 用 `iterdir()` 而非 `rglob`）：

```python
    def _discover_tar_members(self) -> dict[str, str]:
        files = {}
        with tarfile.open(self.product_path) as tf:
            for name in tf.getnames():
                upper = name.upper()
                if name.endswith(".tiff") and "SLC" in upper:
                    files["tiff"] = name
                elif name.endswith(".meta.xml"):
                    files["meta_xml"] = name
                elif name.endswith(".incidence.xml"):
                    files["incidence_xml"] = name
                elif name.endswith(".rpc"):
                    files["rpc"] = name
        return files
```

- [ ] **Step 3: 添加 `_discover_tar_members()` 方法**

### Step 4: 更新 `discover_files()` 增加 `is_tar` 分支

```python
    def discover_files(self) -> dict[str, str]:
        if self.is_zip:
            return self._discover_zip_members()
        if self.is_tar:  # 新增
            return self._discover_tar_members()
        return self._discover_dir_members()
```

- [ ] **Step 4: 更新 `discover_files()`**

### Step 5: 更新 `parse_meta_xml()` 增加 `is_tar` 分支

```python
    def parse_meta_xml(self, path: str | Path) -> ET.Element:
        if self.is_zip:
            with zipfile.ZipFile(self.product_path) as zf:
                with zf.open(str(path)) as f:
                    return ET.fromstring(f.read())
        if self.is_tar:  # 新增
            with tarfile.open(self.product_path) as tf:
                with tf.extractfile(str(path)) as f:
                    return ET.fromstring(f.read())
        tree = ET.parse(Path(path))
        root = tree.getroot()
        return root
```

- [ ] **Step 5: 更新 `parse_meta_xml()`**

### Step 6: 更新 `build_slc_path()` 和 `build_member_path()` 增加 `is_tar` 分支

与 Tianyi 完全平行：

```python
    def build_slc_path(self, tiff_name: str) -> str:
        if self.is_zip:
            return f"/vsizip/{self.product_path}/{tiff_name}"
        if self.is_tar:
            return f"/vsitar/{self.product_path}/{tiff_name}"
        return str(Path(tiff_name).resolve())

    def build_member_path(self, member_name: str) -> str:
        if self.is_zip:
            return f"/vsizip/{self.product_path}/{member_name}"
        if self.is_tar:
            return f"/vsitar/{self.product_path}/{member_name}"
        return str(Path(member_name).resolve())
```

- [ ] **Step 6: 更新两个 build 方法**

### Step 7: 更新 `_manifest_ref()` 增加 `is_tar` 分支

```python
    def _manifest_ref(
        self, output_dir: Path, path_value: str | Path, member: str | None = None
    ):
        if member is not None:
            if self.is_tar:
                return {
                    "path": self._relative_to_output(output_dir, path_value),
                    "storage": "tar",
                    "member": member,
                }
            if self.is_zip:
                return {
                    "path": self._relative_to_output(output_dir, path_value),
                    "storage": "zip",
                    "member": member,
                }
            return {
                "path": self._relative_to_output(output_dir, path_value),
            }
        return self._relative_to_output(output_dir, path_value)
```

- [ ] **Step 7: 更新 `_manifest_ref()`**

---

## Task 3: common_processing.py — 路径解析支持

**Files:**
- Modify: `scripts/common_processing.py:89-98`

### Step 1: 更新 `resolve_manifest_path()` 增加 `storage == "tar"` 处理

在现有 `storage == "zip"` 分支后增加：

```python
        if entry.get("storage") == "tar":
            member = entry.get("member")
            if not member:
                raise ValueError(f"tar manifest entry missing member: {entry}")
            return f"/vsitar/{resolved}/{member}"
        return resolved
```

同时在 `entry.startswith("/vsizip/")` 后增加：

```python
    if entry.startswith("/vsitar/"):
        return entry
```

- [ ] **Step 1: 更新 `resolve_manifest_path()`**

---

## Task 4: 验证

### Step 1: 语法检查

```bash
python3 -m py_compile scripts/tianyi_importer.py && echo "tianyi_importer.py: OK"
python3 -m py_compile scripts/lutan_importer.py && echo "lutan_importer.py: OK"
python3 -m py_compile scripts/common_processing.py && echo "common_processing.py: OK"
```

- [ ] **Step 1: py_compile 检查**

### Step 2: 运行现有测试确保无回归

```bash
python3 -m unittest discover tests/ -v 2>&1 | tail -20
```

- [ ] **Step 2: 回归测试**

---

## Task 5: 新增单元测试（可选，视时间而定）

为以下场景补充测试：
- `TianyiImporter` 用 `.tar` / `.tar.gz` / `.tgz` 路径构造时 `is_tar = True`
- `_discover_tar_members()` 找到正确的归档成员
- `build_slc_path()` 返回正确的 `/vsitar/` 格式路径
- `common_processing.resolve_manifest_path()` 正确处理 `storage == "tar"`
