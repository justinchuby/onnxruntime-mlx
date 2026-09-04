"""Hatchling build hook for onnxruntime-ep-mlx.

Builds the Rust execution provider (`cargo build --release` in ../rust), then
bundles the resulting `libonnxruntime_mlx_ep.dylib` together with its mlx-c/mlx
runtime dependencies into the wheel package, relinked so they load from
``@loader_path`` (a self-contained wheel). Finally it forces a platform wheel
tag: the package ships no CPython-ABI extension, so a single
``py3-none-macosx_*_arm64`` wheel installs on 3.12, 3.13 and the free-threaded
builds alike.

The onnxruntime dependency is intentionally NOT bundled — it is resolved at
runtime from the host ``onnxruntime`` package (two-level namespace), matching the
EP's design.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PLUGIN_DYLIB = "libonnxruntime_mlx_ep.dylib"
MLX_VERSION = "0.32.2"
MLX_COMMIT = "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
MLXC_COMMIT = "c74db5307cc8ce122f48d97ef951b30578674e7f"


def _dependency_prefix(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} was not initialized by the pinned MLX setup")
    return Path(value)


def _linked_dependency(binary: Path, basename: str) -> str | None:
    out = subprocess.run(
        ["otool", "-L", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines()[1:]:
        dependency = line.strip().split(" ", 1)[0]
        if Path(dependency).name == basename:
            return dependency
    return None


def _run(cmd: list[str], **kw) -> None:
    print("[onnxruntime-ep-mlx build] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def _resolve_ort_include() -> str:
    """Mirror rust/build.rs: ORT_INCLUDE_DIR, else $ORT_HOME/include."""
    inc = os.environ.get("ORT_INCLUDE_DIR")
    if not inc:
        home = os.environ.get("ORT_HOME")
        if home:
            inc = str(Path(home) / "include")
    if inc and (Path(inc) / "onnxruntime_c_api.h").is_file():
        return inc
    raise RuntimeError(
        "Could not locate the ONNX Runtime headers. Set ORT_INCLUDE_DIR to the "
        "ORT C-API include dir, or ORT_HOME to an ONNX Runtime release root "
        "(expects $ORT_HOME/include/onnxruntime_c_api.h)."
    )


def _ensure_mlx(project_root: Path) -> None:
    mlx_value = os.environ.get("MLX_PREFIX")
    mlxc_value = os.environ.get("MLXC_PREFIX")
    if bool(mlx_value) != bool(mlxc_value):
        raise RuntimeError("MLX_PREFIX and MLXC_PREFIX must be set together")

    if not mlx_value:
        script = project_root / "scripts" / "setup_mlx.sh"
        if not script.is_file():
            script = project_root.parent / "rust" / "scripts" / "setup_mlx.sh"
        if not script.is_file():
            raise RuntimeError("Pinned MLX setup script not found in repository checkout or sdist")
        prefix = project_root / ".deps" / "mlx"
        env = {
            **os.environ,
            "MLX_VERSION": MLX_VERSION,
            "MLX_COMMIT": MLX_COMMIT,
            "MLXC_COMMIT": MLXC_COMMIT,
        }
        _run([str(script), str(prefix)], env=env)
        mlx_value = mlxc_value = str(prefix)
        os.environ["MLX_PREFIX"] = mlx_value
        os.environ["MLXC_PREFIX"] = mlxc_value

    mlx_prefix = Path(mlx_value)
    mlxc_prefix = Path(mlxc_value)
    version_header = mlx_prefix / "include" / "mlx" / "version.h"
    ops_header = mlxc_prefix / "include" / "mlx" / "c" / "ops.h"
    fast_header = mlxc_prefix / "include" / "mlx" / "c" / "fast.h"
    version_text = version_header.read_text()
    ops_text = ops_header.read_text()
    fast_text = fast_header.read_text()
    expected = (
        "#define MLX_VERSION_MAJOR 0",
        "#define MLX_VERSION_MINOR 32",
        "#define MLX_VERSION_PATCH 2",
    )
    if not all(line in version_text for line in expected):
        raise RuntimeError(f"MLX_PREFIX must provide MLX {MLX_VERSION}: {mlx_prefix}")
    if "mlx_cumsum_axis(" not in ops_text or "force_fused" not in fast_text:
        raise RuntimeError(f"MLXC_PREFIX does not provide the MLX 0.32.2 C ABI: {mlxc_prefix}")


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("onnxruntime-ep-mlx only builds on macOS (Apple Silicon).")

        project_root = Path(self.root)          # the python/ dir
        # In a repository checkout the crate is ../rust. In an sdist it is
        # included as ./rust so installers can build the wheel from source.
        rust_dir = project_root / "rust"
        if not (rust_dir / "Cargo.toml").is_file():
            rust_dir = project_root.parent / "rust"
        if not (rust_dir / "Cargo.toml").is_file():
            raise RuntimeError(
                "Rust crate not found. Expected ./rust in an sdist or ../rust "
                "in a repository checkout."
            )

        pkg_dir = project_root / "src" / "onnxruntime_ep_mlx"

        _ensure_mlx(project_root)

        # 1) Build the Rust EP dylib.
        env = dict(os.environ)
        env["ORT_INCLUDE_DIR"] = _resolve_ort_include()
        _run(["cargo", "build", "--release"], cwd=str(rust_dir), env=env)

        built = rust_dir / "target" / "release" / PLUGIN_DYLIB
        if not built.is_file():
            raise RuntimeError(f"cargo did not produce {built}")

        # 2) Copy the plugin into the package.
        dest_plugin = pkg_dir / PLUGIN_DYLIB
        shutil.copy2(built, dest_plugin)
        os.chmod(dest_plugin, 0o755)

        # 3) Bundle + relink the mlx runtime next to the plugin (self-contained).
        self._bundle_mlx(pkg_dir, dest_plugin)

        # 4) This is a platform wheel with no Python-ABI extension: force a
        #    py3-none-macosx_*_arm64 tag so ONE wheel serves every interpreter
        #    (CPython 3.10+, free-threaded 3.13t/3.14t, ...).
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        if plat.startswith("macosx_"):
            # setup-python may provide a universal2 interpreter even though MLX and the EP dylib
            # are Apple-Silicon-only. Advertising universal2 would let Intel pip install a wheel
            # that dyld cannot load.
            parts = plat.split("_")
            if len(parts) < 4:
                raise RuntimeError(f"unexpected macOS platform tag: {plat}")
            plat = f"macosx_{parts[1]}_{parts[2]}_arm64"
        # Honour MACOSX_DEPLOYMENT_TARGET for the platform floor: the bundled
        # dylibs (and mlx) target it, so the tag should advertise it rather than
        # whatever floor the running interpreter was built against.
        dep_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if dep_target and plat.startswith("macosx_"):
            major, _, minor = dep_target.partition(".")
            plat = f"macosx_{major}_{minor or '0'}_arm64"
        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{plat}"

    # -- mlx bundling ---------------------------------------------------------
    def _bundle_mlx(self, pkg_dir: Path, plugin: Path) -> None:
        mlxc_pfx = _dependency_prefix("MLXC_PREFIX")
        mlx_pfx = _dependency_prefix("MLX_PREFIX")
        mlxc_src = mlxc_pfx / "lib" / "libmlxc.dylib"
        mlx_src = mlx_pfx / "lib" / "libmlx.dylib"
        metallib_src = mlx_pfx / "lib" / "mlx.metallib"
        for f in (mlxc_src, mlx_src, metallib_src):
            if not f.is_file():
                raise RuntimeError(
                    f"Required MLX artifact missing: {f}. Run the pinned setup script first."
                )

        mlxc_dst = pkg_dir / "libmlxc.dylib"
        mlx_dst = pkg_dir / "libmlx.dylib"
        for src, dst in ((mlxc_src, mlxc_dst), (mlx_src, mlx_dst), (metallib_src, pkg_dir / "mlx.metallib")):
            shutil.copy2(src, dst)
            os.chmod(dst, 0o644)
        os.chmod(mlxc_dst, 0o755)
        os.chmod(mlx_dst, 0o755)

        jaccl_src = mlx_pfx / "lib" / "libjaccl.dylib"
        jaccl_dst = pkg_dir / "libjaccl.dylib"
        if jaccl_src.is_file():
            shutil.copy2(jaccl_src, jaccl_dst)
            os.chmod(jaccl_dst, 0o755)

        def name_tool(*args: str) -> None:
            _run(["install_name_tool", *args])

        def resign(f: Path) -> None:
            subprocess.run(["codesign", "--force", "--sign", "-", str(f)], check=False)

        # Bundled mlx install ids -> @loader_path.
        name_tool("-id", "@loader_path/libmlxc.dylib", str(mlxc_dst))
        name_tool("-id", "@loader_path/libmlx.dylib", str(mlx_dst))
        # Relink MLX dependencies by basename so both Homebrew absolute paths and
        # custom-build @rpath install names work.
        mlxc_mlx = _linked_dependency(mlxc_dst, "libmlx.dylib")
        if mlxc_mlx:
            name_tool("-change", mlxc_mlx, "@loader_path/libmlx.dylib", str(mlxc_dst))

        # Plugin's mlx deps -> colocated copies.
        plugin_mlxc = _linked_dependency(plugin, "libmlxc.dylib")
        plugin_mlx = _linked_dependency(plugin, "libmlx.dylib")
        if plugin_mlxc:
            name_tool("-change", plugin_mlxc, "@loader_path/libmlxc.dylib", str(plugin))
        if plugin_mlx:
            name_tool("-change", plugin_mlx, "@loader_path/libmlx.dylib", str(plugin))

        if jaccl_src.is_file():
            name_tool("-id", "@loader_path/libjaccl.dylib", str(jaccl_dst))
            mlx_jaccl = _linked_dependency(mlx_dst, "libjaccl.dylib")
            if mlx_jaccl:
                name_tool("-change", mlx_jaccl, "@loader_path/libjaccl.dylib", str(mlx_dst))

        # The Rust EP does NOT link libonnxruntime — it reaches ORT purely through
        # the OrtApi function-pointer table handed to CreateEpFactories (see
        # rust/build.rs). So there is no onnxruntime dependency to relink here;
        # ORT dlopen()s the plugin by the absolute path library_path() returns.

        # Re-sign everything we mutated (install_name_tool voids the ad-hoc sig;
        # dyld SIGKILLs unsigned/invalid arm64 images).
        bundled = [mlxc_dst, mlx_dst, plugin]
        if jaccl_src.is_file():
            bundled.append(jaccl_dst)
        for f in bundled:
            resign(f)
