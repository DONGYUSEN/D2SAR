# tar/.tar.gz 支持设计

## 背景

`tianyi_importer.py` 和 `lutan_importer.py` 目前支持三种输入形态：
- **目录**：直接解压的产品目录
- **.zip**：用 `zipfile` 读取，走 GDAL `/vsizip/` 虚拟文件系统
- **其他归档**（fallback）：用 `shutil.unpack_archive` 解压到临时目录

用户请求增加 `.tar`、`.tar.gz`、`.tgz` 的原生支持，无需解压，直接流式读取。

## 技术基础

GDAL 3.11 原生支持 `/vsitar/` 虚拟文件系统，格式为：
```
/vsitar/archive.tar/path/inside/archive/file.tif
```
支持的格式：`.tar`、`.tar.gz`、`.tgz`。无需预解压，GDAL 流式读取内部成员。

## 变更范围

### 1. `tianyi_importer.py`

#### `_prepare_input_path()`
增加 `.tar` / `.tar.gz` / `.tgz` 检测，设置 `is_tar = True`（与 `is_zip` 平行）：

```python
if suffix in (".tar", ".tar.gz", ".tgz"):
    self.is_tar = True
```

#### `_discover_tar_members()`
用 `tarfile` 遍历归档成员，逻辑与 `_discover_zip_members()` 平行：

```python
def _discover_tar_members(self) -> dict[str, str]:
    members = {}
    with tarfile.open(self.product_path) as tf:
        for name in tf.getnames():
            low = name.lower()
            # annotation, calibration, manifest, tiff 查找逻辑同 ZIP
    return members
```

#### `parse_xml_root()`
增加 `is_tar` 分支：

```python
if self.is_tar:
    with tarfile.open(self.product_path) as tf:
        with tf.extractfile(member_name) as f:
            return ET.fromstring(f.read())
```

#### `build_slc_path()` / `build_member_path()`
增加 `is_tar` 分支：

```python
if self.is_tar:
    return f"/vsitar/{self.product_path}/{member_name}"
```

#### manifest 引用
`"storage": "tar"`（与 `"zip"` 平行）

---

### 2. `lutan_importer.py`

完全平行于 `tianyi_importer.py` 的改动：
- `_prepare_input_path()`：检测 `.tar` / `.tar.gz` / `.tgz`
- `_discover_tar_members()`
- `parse_meta_xml()`：增加 `is_tar` 分支
- `build_slc_path()` / `build_member_path()`：增加 `is_tar` 分支
- manifest 引用 `"storage": "tar"`

---

### 3. `common_processing.py`

#### `resolve_manifest_path()`
在 `storage == "zip"` 分支后增加 `storage == "tar"`：

```python
if entry.get("storage") == "tar":
    member = entry.get("member")
    if not member:
        raise ValueError(f"tar manifest entry missing member: {entry}")
    return f"/vsitar/{resolved}/{member}"
```

同时在 `entry.startswith("/vsizip/")` 后增加：
```python
if entry.startswith("/vsitar/"):
    return entry
```

---

## 输入检测优先级

1. `path.is_dir()` → 目录
2. `suffix == ".zip"` → ZIP
3. `suffix in (".tar", ".tar.gz", ".tgz")` → TAR
4. 其他归档文件（fallback）→ `shutil.unpack_archive` 解压

---

## manifest storage 值

| 后缀 | storage | VFS 路径格式 |
|---|---|---|
| .zip | `"zip"` | `/vsizip/{path}/{member}` |
| .tar | `"tar"` | `/vsitar/{path}/{member}` |
| .tar.gz | `"tar"` | `/vsitar/{path}/{member}` |
| .tgz | `"tar"` | `/vsitar/{path}/{member}` |

---

## 测试计划

- 单元测试覆盖 `.tar`、`.tar.gz`、`.tgz` 三种后缀的输入检测
- `is_tar` 标志正确设置
- `_discover_tar_members()` 能找到正确成员
- `build_slc_path()` 返回正确的 `/vsitar/` 路径
- `common_processing.resolve_manifest_path()` 正确处理 `storage == "tar"`
