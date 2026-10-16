"""Bounded local Git fixtures; no network, dependency installs or native compile."""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rl_native_manifest", ROOT / "scripts/rl/native_build_manifest.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@local",
         "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f"Fixture git {args}: {result.stderr.decode()}")
    return result.stdout.decode().strip()


def commit(repo, message):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_patch(before, after):
    parts = []
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) != after.get(name):
            parts.append(f"diff --git a/{name} b/{name}\n")
            if name not in before:
                parts.append("new file mode 100644\n")
            elif name not in after:
                parts.append("deleted file mode 100644\n")
            parts.extend(difflib.unified_diff(before.get(name, "").splitlines(keepends=True),
                                             after.get(name, "").splitlines(keepends=True),
                                             fromfile=f"a/{name}" if name in before else "/dev/null",
                                             tofile=f"b/{name}" if name in after else "/dev/null"))
    return "".join(parts).encode()


class SourceFixture:
    def __init__(self, directory):
        self.root = Path(directory) / "project"
        self.source = Path(directory) / "source"
        self.build = self.source / "build-fixture"
        self.source.mkdir()
        git(self.source, "init", "-q")
        deps = {}
        for name in ("json", "pybind11"):
            dep = self.source / name
            dep.mkdir()
            git(dep, "init", "-q")
            (dep / "dependency.h").write_text(f"// {name} upstream\n")
            deps[name] = commit(dep, "dependency upstream")
        self.json_pin = deps["json"]
        (self.source / "pybind11/dependency.h").write_text("// pybind11 pinned release\n")
        self.pybind_pin = commit(self.source / "pybind11", "pinned release")
        git(self.source / "pybind11", "tag", "v2.13.6")
        self.base = {
            "CMakeLists.txt": "cmake_minimum_required(VERSION 3.5)\nproject(fixture)\n",
            "cmake/options.cmake": "set(FIXTURE ON)\n",
            "src/engine.cpp": 'const char *version = "upstream";\n',
            "include/engine.h": "void engine();\n",
            "bindings/api.cpp": "void upstream() {}\n",
            ".gitignore": "build*/\nignored/\n*.log\n",
            ".gitmodules": '[submodule "json"]\n\tpath = json\n\turl = ../json\n'
                           '[submodule "pybind11"]\n\tpath = pybind11\n\turl = ../pybind11\n',
        }
        self.put(self.base)
        git(self.source, "add", ".gitignore", ".gitmodules", "CMakeLists.txt", "cmake", "src", "include", "bindings")
        for name in deps:
            git(self.source, "update-index", "--add", "--cacheinfo", f"160000,{deps[name]},{name}")
        git(self.source, "commit", "-qm", "immutable upstream")
        self.pin = git(self.source, "rev-parse", "HEAD")
        self.core = {**self.base, "src/engine.cpp": 'const char *version = "core015";\n',
                     "bindings/api.cpp": "void upstream() {}\nvoid core() {}\n",
                     "include/core.h": "void core();\n"}
        self.panel = {**self.core,
                      "bindings/api.cpp": "void upstream() {}\nvoid core() {}\nvoid panel() {}\n",
                      "include/core.h": "void core();\nvoid panel();\n"}
        patch_dir = self.root / "patches"
        patch_dir.mkdir(parents=True)
        self.core_patch = patch_dir / "sts_lightspeed_python_api.patch"
        self.overlay_patch = patch_dir / "sts_lightspeed_rl_panel.patch"
        self.core_patch.write_bytes(make_patch(self.base, self.core))
        self.overlay_patch.write_bytes(make_patch(self.core, self.panel))
        self.expected = {"upstream_commit": self.pin, "pybind11_tag": "v2.13.6",
                         "patch_sha256": sha(self.core_patch.read_bytes()), "backend_api": 1}
        self.expected_file = self.root / "src/sts_env/_backend/native_manifest.json"
        self.expected_file.parent.mkdir(parents=True)
        self.expected_file.write_text(json.dumps(self.expected))

    def put(self, contents):
        for name, text in contents.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def install(self, *, panel=True, committed_core=False):
        git(self.source, "apply", str(self.core_patch))
        if committed_core:
            git(self.source, "add", "src", "include", "bindings")
            git(self.source, "commit", "-qm", "versioned environment patch")
        if panel:
            git(self.source, "apply", str(self.overlay_patch))

    def build_outputs(self):
        self.build.mkdir(exist_ok=True)
        self.module = self.build / ("slaythespire" + sysconfig.get_config_var("EXT_SUFFIX"))
        self.module.write_bytes(b"fixture native module; never imported")
        self.old_manifest = b'{"previous": "preserve on failure"}\n'
        (self.build / "build_manifest.json").write_bytes(self.old_manifest)
        tools = self.root / "tools"
        tools.mkdir(exist_ok=True)
        cmake, compiler = tools / "cmake", tools / "c++"
        for path, version in ((cmake, "cmake version fixture-9.8"), (compiler, "Fixture C++ 4.2")):
            path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n")
            path.chmod(0o755)
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={self.source}\nCMAKE_COMMAND:INTERNAL={cmake}\n"
            f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n")
        return cmake, compiler

    def script_runner(self):
        """Run the real shell helper with fake configure/build commands only."""
        self.build_outputs()
        scripts = self.root / "scripts/rl"
        scripts.mkdir(parents=True)
        script = scripts / "build_native_rl.sh"
        shutil.copyfile(ROOT / "scripts/rl/build_native_rl.sh", script)
        # Only substitute the immutable external dependency SHA for this fixture.
        (scripts / "native_build_manifest.py").write_text(
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('helper', {str(ROOT / 'scripts/rl/native_build_manifest.py')!r})\n"
            "helper = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(helper)\n"
            f"helper.PYBIND11_COMMIT = {self.pybind_pin!r}\nhelper.main()\n")
        runtime = self.root / ".venv-native/bin/python"
        runtime.parent.mkdir(parents=True)
        runtime.write_text(
            f"#!{sys.executable}\n"
            "import os, pathlib, sys\n"
            "with open(os.environ['FIXTURE_EVENTS'], 'a') as stream:\n"
            "    stream.write(' '.join(sys.argv[1:]) + '\\n')\n"
            "if '--build' in sys.argv:\n"
            "    build = pathlib.Path(sys.argv[sys.argv.index('--build') + 1])\n"
            f"    (build / {self.module.name!r}).write_bytes(b'fixture native module; never imported')\n"
            "    if os.environ.get('FIXTURE_MUTATE'):\n"
            "        pathlib.Path(os.environ['FIXTURE_MUTATE']).write_text('// changed during build\\n')\n"
            "else:\n"
            "    build = pathlib.Path(sys.argv[sys.argv.index('-B') + 1])\n"
            "    build.mkdir(parents=True, exist_ok=True)\n"
            f"    (build / 'CMakeCache.txt').write_text({(self.build / 'CMakeCache.txt').read_text()!r})\n")
        runtime.chmod(0o755)
        self.events = self.root / "events.log"
        self.script = script

    def run_script(self, mutate=None, *, build=None, env=None):
        build = build if build is not None else self.build
        return subprocess.run(["bash", str(self.script)], capture_output=True, text=True, timeout=30,
                              env={**os.environ, "PYTHON": sys.executable,
                                   "STS_RL_SOURCE": str(self.source), "BUILD_DIR": str(build.relative_to(self.source)),
                                   "STS_BUILD_JOBS": "2", "FIXTURE_EVENTS": str(self.events),
                                   "FIXTURE_MUTATE": str(mutate or ""), **(env or {})})


class RLNativeBuildTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rl-native-test-")
        self.addCleanup(self.directory.cleanup)
        self.f = SourceFixture(self.directory.name)
        pin = patch.object(helper, "PYBIND11_COMMIT", self.f.pybind_pin)
        pin.start()
        self.addCleanup(pin.stop)

    def identity(self):
        return helper.source_identity(self.f.source, self.f.root, self.f.build)

    def assert_preserved(self):
        self.assertEqual((self.f.build / "build_manifest.json").read_bytes(), self.f.old_manifest)
        self.assertEqual(self.f.module.read_bytes(), b"fixture native module; never imported")
        self.assertEqual(list(self.f.build.glob(".build_manifest-*.tmp")), [])

    def test_accepts_exact_core_plus_overlay_and_canonical_digest(self):
        self.f.install()
        before = git(self.f.source, "status", "--porcelain")
        identity = self.identity()
        hashes = {name: sha(text.encode()) for name, text in self.f.panel.items()}
        canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(identity["source_tree_sha256"], sha(canonical))
        self.assertEqual(identity["source_head"], self.f.pin)
        self.assertEqual(identity["source_upstream_commit"], self.f.pin)
        self.assertEqual(identity["rl_panel_patch_sha256"], sha(self.f.overlay_patch.read_bytes()))
        self.assertEqual(git(self.f.source, "status", "--porcelain"), before)
        self.assertEqual(self.identity(), identity)

    def test_accepts_synthetic_committed_core_uncommitted_overlay(self):
        self.f.install(committed_core=True)
        identity = self.identity()
        self.assertEqual(identity["source_head"], git(self.f.source, "rev-parse", "HEAD"))
        self.assertNotEqual(identity["source_head"], self.f.pin)
        self.assertNotIn("panel", git(self.f.source, "show", "HEAD:bindings/api.cpp"))
        self.assertIn("panel", (self.f.source / "bindings/api.cpp").read_text())

    def test_accepts_gitdir_file_and_prior_cmake_outputs(self):
        self.f.install()
        initial = self.identity()
        gitdir = Path(self.directory.name) / "source.git"
        shutil.move(self.f.source / ".git", gitdir)
        (self.f.source / ".git").write_text(f"gitdir: {gitdir}\n")
        previous = self.f.source / "build-previous"
        previous.mkdir()
        (previous / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={self.f.source}\n")
        generated = previous / "CMakeFiles/compilerid.cpp"
        generated.parent.mkdir()
        generated.write_text("// generated compiler probe\n")
        self.assertEqual(self.identity(), initial)
        self.assertEqual(generated.read_text(), "// generated compiler probe\n")

    def test_archive_attributes_cannot_hide_versioned_inputs(self):
        self.f.install()
        initial = self.identity()
        (self.f.source / ".git/info/attributes").write_text("src/engine.cpp export-ignore\n")
        self.assertEqual(self.identity(), initial)
        (self.f.source / "src/engine.cpp").write_text("// hidden mutation\n")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.identity()

    def test_build_directory_cannot_hide_versioned_sources(self):
        self.f.install()
        for name in ("src", "src/new-directory", "json/build-hidden"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "Build directory contains versioned source"):
                    helper.source_identity(self.f.source, self.f.root, self.f.source / name)

    def test_rejects_previous_core_version_even_with_committed_core(self):
        self.f.install(committed_core=True)
        (self.f.source / "src/engine.cpp").write_text('const char *version = "core014";\n')
        git(self.f.source, "add", "src/engine.cpp")
        git(self.f.source, "commit", "-qm", "stale core")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.identity()

    def test_rejects_missing_overlay_on_every_rerun(self):
        self.f.install(committed_core=True)
        self.f.build_outputs()
        checked = self.identity()
        # Preserve a real precheck, then remove the overlay from the fixture.
        self.f.put(self.f.core)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
            self.assert_preserved()

    def test_rejects_partial_overlay(self):
        self.f.install()
        (self.f.source / "include/core.h").write_text(self.f.core["include/core.h"])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.identity()

    def test_rejects_patch_sha_mismatch_and_unapplicable_overlay(self):
        self.f.install()
        with self.f.core_patch.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(RuntimeError, "Core patch SHA"):
            self.identity()
        self.f.core_patch.write_bytes(make_patch(self.f.base, self.f.core))
        self.f.overlay_patch.write_bytes(make_patch(self.f.base, self.f.panel))
        with self.assertRaisesRegex(RuntimeError, r"git --no-replace-objects -C failed:.*patch failed:"):
            self.identity()

    def test_rejects_modified_cmake_inputs_sources_and_missing_files(self):
        self.f.install()
        for name in ("CMakeLists.txt", "cmake/options.cmake", "src/engine.cpp", "bindings/api.cpp", "include/core.h"):
            with self.subTest(name=name):
                path = self.f.source / name
                saved = path.read_bytes()
                path.write_bytes(saved + b"// mutation\n")
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    self.identity()
                path.write_bytes(saved)
        (self.f.source / "include/engine.h").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or non-regular"):
            self.identity()

    def test_rejects_new_tracked_untracked_and_ignored_source_files(self):
        self.f.install()
        for name in ("src/extra.cpp", "ignored/extra.hpp", "extra.cmake", "src/ignored.log"):
            with self.subTest(name=name):
                path = self.f.source / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("// new input\n")
                with self.assertRaisesRegex(RuntimeError, "Unexpected unversioned source"):
                    self.identity()
                path.unlink()
        extra = self.f.source / "extra.cpp"
        extra.write_text("// staged source\n")
        git(self.f.source, "add", "extra.cpp")
        with self.assertRaisesRegex(RuntimeError, "Unexpected tracked"):
            self.identity()

    def test_rejects_wrong_pin_even_if_working_bytes_match(self):
        self.f.install()
        unrelated = Path(self.directory.name) / "unrelated"
        unrelated.mkdir()
        git(unrelated, "init", "-q")
        (unrelated / "file").write_text("unrelated")
        other = commit(unrelated, "wrong origin")
        git(self.f.source, "fetch", str(unrelated), "HEAD")
        git(self.f.source, "update-ref", "HEAD", other)
        with patch.object(helper, "_git", wraps=helper._git) as calls:
            with self.assertRaisesRegex(RuntimeError, "git --no-replace-objects -C failed:"):
                self.identity()
        self.assertEqual(calls.call_args.args[1:], ("merge-base", "--is-ancestor", self.f.pin, other))

    def test_rejects_dependency_head_mismatch_and_moved_tag(self):
        self.f.install()
        for name in ("json", "pybind11"):
            with self.subTest(name=name):
                dep = self.f.source / name
                old = git(dep, "rev-parse", "HEAD")
                (dep / "dependency.h").write_text("// wrong dependency\n")
                commit(dep, "wrong pin")
                if name == "pybind11":
                    git(dep, "tag", "-f", "v2.13.6")
                with self.assertRaisesRegex(RuntimeError, f"Dependency {name} pin mismatch"):
                    self.identity()
                # Fixture-only recovery: keep HEAD at the pin and restore its bytes.
                git(dep, "update-ref", "HEAD", old)
                (dep / "dependency.h").write_text(git(dep, "show", "HEAD:dependency.h") + "\n")
                git(dep, "add", "dependency.h")

    def test_rejects_dirty_or_unpopulated_dependencies(self):
        self.f.install()
        for name in ("json", "pybind11"):
            dep = self.f.source / name
            for tracked in (True, False):
                with self.subTest(name=name, tracked=tracked):
                    path = dep / ("dependency.h" if tracked else "new.h")
                    old = path.read_bytes() if path.exists() else None
                    path.write_text("// dirty\n")
                    with self.assertRaisesRegex(RuntimeError, f"Dependency {name} has unversioned"):
                        self.identity()
                    if old is None:
                        path.unlink()
                    else:
                        path.write_bytes(old)
        shutil.rmtree(self.f.source / "json/.git")
        with self.assertRaisesRegex(RuntimeError, "Missing Git checkout"):
            self.identity()

    def test_postcheck_rejects_source_mutation_and_preserves_outputs(self):
        self.f.install()
        self.f.build_outputs()
        checked = self.identity()
        (self.f.source / "src/engine.cpp").write_text("// changed after precheck\n")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
        self.assert_preserved()

    def test_postcheck_rejects_valid_identity_changed_since_precheck(self):
        self.f.install()
        self.f.build_outputs()
        checked = self.identity()
        git(self.f.source, "add", "src", "include", "bindings")
        git(self.f.source, "commit", "-qm", "same verified bytes, different head")
        with self.assertRaisesRegex(RuntimeError, "changed since"):
            helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
        self.assert_preserved()

    def test_manifest_propagates_verified_actual_identities_and_tool_versions(self):
        self.f.install(committed_core=True)
        cmake, compiler = self.f.build_outputs()
        checked = self.identity()
        result = helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
        self.assertEqual(result, json.loads((self.f.build / "build_manifest.json").read_text()))
        for key, value in {**self.f.expected, **checked}.items():
            self.assertEqual(result[key], value, key)
        self.assertEqual(result["json_commit"], self.f.json_pin)
        self.assertEqual(result["pybind11_commit"], self.f.pybind_pin)
        self.assertEqual(result["cmake_version"], "cmake version fixture-9.8")
        self.assertEqual(result["compiler_version"], "Fixture C++ 4.2")
        self.assertEqual(result["cmake_executable"], str(cmake))
        self.assertEqual(result["compiler_executable"], str(compiler))
        self.assertEqual(result["module_sha256"], sha(self.f.module.read_bytes()))
        self.assertEqual(result["module_file"], self.f.module.name)
        self.assertEqual(result["python_cache_tag"], sys.implementation.cache_tag)
        self.assertEqual(result["python_soabi"], sysconfig.get_config_var("SOABI"))
        self.assertEqual(result["platform"], sysconfig.get_platform())
        self.assertEqual(result["variant"], "sts_rl_panel")

    def test_atomic_publication_failure_preserves_existing_manifest(self):
        self.f.install()
        self.f.build_outputs()
        checked = self.identity()
        with patch.object(helper.os, "replace", side_effect=OSError("fixture publish failure")):
            with self.assertRaisesRegex(OSError, "publish failure"):
                helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
        self.assert_preserved()

    def test_tool_identity_failure_preserves_existing_manifest(self):
        self.f.install()
        cmake, _ = self.f.build_outputs()
        checked = self.identity()
        cmake.write_text("#!/bin/sh\nexit 3\n")
        with self.assertRaisesRegex(RuntimeError, r"cmake --version failed:"):
            helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
        self.assert_preserved()

    def test_mutation_during_tool_query_prevents_manifest_write(self):
        self.f.install()
        _, compiler = self.f.build_outputs()
        checked = self.identity()
        original_run = helper._run

        def mutate(*args):
            result = original_run(*args)
            if args[0] == str(compiler):
                (self.f.source / "src/engine.cpp").write_text("// tool-time mutation\n")
            return result

        with patch.object(helper, "_run", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
        self.assert_preserved()

    def test_write_api_requires_explicit_nonempty_precheck(self):
        self.f.install()
        self.f.build_outputs()
        with patch.object(helper, "_run") as commands:
            with self.assertRaisesRegex(TypeError, "checked"):
                helper.write_manifest(self.f.source, self.f.root, self.f.build)
            for checked in (None, {}):
                with self.subTest(checked=checked):
                    with self.assertRaisesRegex(RuntimeError, "pre-build check is required"):
                        helper.write_manifest(self.f.source, self.f.root, self.f.build, checked)
                    self.assert_preserved()
            commands.assert_not_called()

    def test_write_cli_requires_checked_source_before_accessing_outputs(self):
        self.f.install()
        self.f.build_outputs()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/rl/native_build_manifest.py"), "write",
             str(self.f.source), str(self.f.root), str(self.f.build)],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("the following arguments are required: --checked-source", result.stderr)
        self.assert_preserved()

    def test_shell_rejects_nonempty_outputs_before_cmake_preserving_all_files(self):
        self.f.install()
        self.f.script_runner()
        stale = self.f.build / "CMakeFiles/stale.o"
        stale.parent.mkdir()
        stale.write_bytes(b"stale object with misleading newer timestamp\x00")
        future = (self.f.source / "src/engine.cpp").stat().st_mtime + 3600
        os.utime(stale, (future, future))
        (self.f.build / ".hidden").write_bytes(b"preserve hidden output")
        hidden_only = self.f.source / "build-hidden-only"
        hidden_only.mkdir()
        (hidden_only / ".sentinel").write_bytes(b"hidden files make an output nonempty")
        directory_only = self.f.source / "build-directory-only"
        (directory_only / "CMakeFiles").mkdir(parents=True)
        for build in (self.f.build, hidden_only, directory_only):
            with self.subTest(build=build.name):
                paths = sorted(path.relative_to(build) for path in build.rglob("*"))
                before = {path: (build / path).read_bytes() for path in paths if (build / path).is_file()}
                result = self.f.run_script(build=build)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("choose fresh BUILD_DIR", result.stderr)
                self.assertFalse(self.f.events.exists(), result.stdout + result.stderr)
                self.assertEqual(sorted(path.relative_to(build) for path in build.rglob("*")), paths)
                self.assertEqual({path: (build / path).read_bytes() for path in before}, before)
                self.assert_preserved()
        self.assertEqual(stale.stat().st_mtime, future)

    def test_shell_missing_runtime_rejected_before_clone_or_source_work(self):
        self.f.install()
        self.f.script_runner()
        runtime = self.f.root / ".venv-native/bin/python"
        runtime.rename(runtime.with_name("python.unavailable"))
        tools = self.f.root / "early-command-traps"
        tools.mkdir()
        for name in ("git", "python"):
            command = tools / name
            command.write_text('#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$FIXTURE_EVENTS"\nexit 97\n')
            command.chmod(0o755)
        absent = self.f.source.parent / "not-cloned"
        for source in (self.f.source, absent):
            with self.subTest(source=source.name):
                result = self.f.run_script(build=self.f.source / "build-no-runtime", env={
                    "STS_RL_SOURCE": str(source), "PYTHON": str(tools / "python"),
                    "PATH": str(tools) + os.pathsep + os.environ["PATH"]})
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("Missing .venv-native build-tools runtime", result.stderr)
                self.assertFalse(self.f.events.exists(), result.stdout + result.stderr)
                self.assertFalse(absent.exists())
                self.assertFalse((source / "build-no-runtime").exists())
                self.assert_preserved()

    def test_shell_accepts_empty_output_with_matching_source_checks(self):
        self.f.install()
        self.f.script_runner()
        empty = self.f.source / "build-empty"
        empty.mkdir()
        checked = self.identity()
        before = git(self.f.source, "status", "--porcelain")
        result = self.f.run_script(build=empty)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata = json.loads((empty / "build_manifest.json").read_text())
        for key, value in checked.items():
            self.assertEqual(metadata[key], value, key)
        self.assertEqual(len(self.f.events.read_text().splitlines()), 2)
        self.assertEqual(git(self.f.source, "status", "--porcelain"), before)
        self.assertEqual(self.identity(), checked)
        self.assert_preserved()

    def test_shell_rejects_stale_missing_and_modified_source_before_cmake(self):
        self.f.install()
        self.f.script_runner()
        cases = {"src/engine.cpp": 'const char *version = "core014";\n',
                 "bindings/api.cpp": self.f.core["bindings/api.cpp"],
                 "CMakeLists.txt": "project(mutated)\n"}
        for name, changed in cases.items():
            with self.subTest(name=name):
                path = self.f.source / name
                old = path.read_text()
                path.write_text(changed)
                fresh = self.f.source / "build-rejected-source"
                result = self.f.run_script(build=fresh)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("does not match", result.stderr)
                self.assertFalse(self.f.events.exists(), result.stdout + result.stderr)
                self.assertFalse(fresh.exists())
                self.assertEqual(path.read_text(), changed)
                self.assert_preserved()
                path.write_text(old)

    def test_shell_success_and_postbuild_mutation_failure(self):
        self.f.install(committed_core=True)
        self.f.script_runner()
        fresh = self.f.source / "build-success"
        result = self.f.run_script(build=fresh)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = self.f.events.read_text().splitlines()
        self.assertEqual(len(events), 2)
        self.assertIn("--build", events[1])
        self.assertIn("-j 2", events[1])
        manifest = (fresh / "build_manifest.json").read_bytes()
        self.assertEqual(json.loads(manifest)["source_head"], git(self.f.source, "rev-parse", "HEAD"))
        self.assertEqual(json.loads(manifest)["source_tree_sha256"], self.identity()["source_tree_sha256"])
        failed = self.f.source / "build-mutated"
        result = self.f.run_script(build=failed, mutate=self.f.source / "src/engine.cpp")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)
        self.assertEqual(len(self.f.events.read_text().splitlines()), 4)
        self.assertTrue((failed / self.f.module.name).exists())
        self.assertFalse((failed / "build_manifest.json").exists())
        self.assertEqual(list(failed.glob(".build_manifest-*.tmp")), [])
        self.assertEqual((fresh / "build_manifest.json").read_bytes(), manifest)
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main()
