# Celik Kubbe Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the existing Celik Kubbe runtime by making threat range handling, WTA battery assignment, and core tests match the current codebase.

**Architecture:** Add small pure-Python helpers for coordinate/range conversion and battery profiles, then wire `main.py` to those helpers without restructuring the GUI. Replace the stale integration test script with tests that exercise modules that actually exist in this project.

**Tech Stack:** Python, PySide6, NumPy, SciPy, Ultralytics, stdlib `unittest`.

---

### Task 1: Core Behavior Tests

**Files:**
- Create: `tests/test_core_behaviors.py`

- [ ] **Step 1: Write failing tests**

Add tests for `src.coordinate_utils` and battery profiles before those helpers exist.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_core_behaviors -v`
Expected: failure caused by missing helpers.

### Task 2: Range and Battery Helpers

**Files:**
- Create: `src/coordinate_utils.py`
- Create: `src/battery_profiles.py`
- Modify: `main.py`

- [ ] **Step 1: Implement range helpers**

Create helpers that convert physical kilometers to the normalized radar display radius and back.

- [ ] **Step 2: Implement battery profiles**

Map `PIL-ALFA`, `PIL-BETA`, `PIL-GAMMA`, and `PIL-DELTA` to explicit range and kill-probability values.

- [ ] **Step 3: Wire helpers into `main.py`**

Use physical range fields for table/logging and normalized fields only for radar display coordinates.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_core_behaviors -v`
Expected: all tests pass.

### Task 3: Stale Test Entrypoint

**Files:**
- Modify: `src/test_celikkubbe_pipeline.py`

- [ ] **Step 1: Replace imports for missing legacy modules**

Make the script test current modules: WTA, radar params, sensor fusion scoring, config, and model paths.

- [ ] **Step 2: Run script**

Run: `python src/test_celikkubbe_pipeline.py`
Expected: exit code 0.

### Task 4: Final Verification

**Files:**
- No additional files.

- [ ] **Step 1: Parse all Python files**

Run AST parse for `main.py`, `src/**/*.py`, and `tests/**/*.py`.

- [ ] **Step 2: Run unit tests**

Run: `python -m unittest discover -s tests -v`
Expected: all tests pass.

- [ ] **Step 3: Report caveats**

Report any dependency, GUI, camera, or hardware checks that were not run.
