#!/usr/bin/env python3
"""
Comprehensive Code Checker

Runs all validation checks without requiring dependencies installed.
Perfect for pre-commit validation.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class ColorOutput:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def print_header(text: str):
        print(f"\n{ColorOutput.HEADER}{ColorOutput.BOLD}{text}{ColorOutput.ENDC}")

    @staticmethod
    def print_success(text: str):
        print(f"{ColorOutput.OKGREEN}✓ {text}{ColorOutput.ENDC}")

    @staticmethod
    def print_error(text: str):
        print(f"{ColorOutput.FAIL}✗ {text}{ColorOutput.ENDC}")

    @staticmethod
    def print_warning(text: str):
        print(f"{ColorOutput.WARNING}⚠ {text}{ColorOutput.ENDC}")


def run_check(name: str, command: List[str]) -> Tuple[bool, str]:
    """Run a check command"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_syntax():
    """Check Python syntax"""
    ColorOutput.print_header("1. Checking Python Syntax...")
    success, output = run_check(
        "Syntax Check",
        ["python3", "validate_code.py"]
    )

    if success:
        ColorOutput.print_success("All Python files have valid syntax")
    else:
        ColorOutput.print_error("Syntax errors found")
        print(output)

    return success


def check_imports():
    """Check import structure"""
    ColorOutput.print_header("2. Checking Import Structure...")

    root = Path(__file__).parent
    issues = []

    # Check that __init__.py files exist
    src_dirs = [
        root / "src",
        root / "src" / "models",
        root / "src" / "data",
        root / "src" / "training",
        root / "src" / "analysis",
        root / "src" / "utils",
        root / "tests",
    ]

    for dir_path in src_dirs:
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            issues.append(f"Missing __init__.py in {dir_path}")

    if not issues:
        ColorOutput.print_success("All packages have __init__.py files")
        return True
    else:
        for issue in issues:
            ColorOutput.print_error(issue)
        return False


def check_file_structure():
    """Check project structure"""
    ColorOutput.print_header("3. Checking Project Structure...")

    root = Path(__file__).parent
    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "src/__init__.py",
        "tests/__init__.py",
        "configs/medium_model.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/analyze.py",
    ]

    missing = []
    for file_path in required_files:
        if not (root / file_path).exists():
            missing.append(file_path)

    if not missing:
        ColorOutput.print_success(f"All {len(required_files)} required files present")
        return True
    else:
        ColorOutput.print_error(f"Missing {len(missing)} required files:")
        for file_path in missing:
            print(f"  - {file_path}")
        return False


def check_code_stats():
    """Show code statistics"""
    ColorOutput.print_header("4. Code Statistics...")

    root = Path(__file__).parent

    # Count Python files
    py_files = [f for f in root.rglob("*.py")
                if '__pycache__' not in str(f) and '.git' not in str(f)]

    src_files = [f for f in py_files if str(f).startswith(str(root / "src"))]
    test_files = [f for f in py_files if str(f).startswith(str(root / "tests"))]
    script_files = [f for f in py_files if str(f).startswith(str(root / "scripts"))]

    # Count lines
    total_lines = 0
    for file in py_files:
        try:
            with open(file, 'r') as f:
                total_lines += len(f.readlines())
        except Exception:
            pass

    print(f"  Python files: {len(py_files)}")
    print(f"    - Source: {len(src_files)}")
    print(f"    - Tests: {len(test_files)}")
    print(f"    - Scripts: {len(script_files)}")
    print(f"  Total lines of code: {total_lines:,}")

    ColorOutput.print_success("Statistics generated")
    return True


def check_yaml_configs():
    """Check YAML configuration files"""
    ColorOutput.print_header("5. Checking Configuration Files...")

    root = Path(__file__).parent
    config_dir = root / "configs"

    if not config_dir.exists():
        ColorOutput.print_error("configs/ directory not found")
        return False

    yaml_files = list(config_dir.glob("*.yaml"))

    if len(yaml_files) < 3:
        ColorOutput.print_warning(f"Only {len(yaml_files)} config files found (expected 3)")

    issues = []
    for yaml_file in yaml_files:
        try:
            # Try to read as text (YAML parsing requires PyYAML)
            with open(yaml_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    issues.append(f"{yaml_file.name} is empty")
        except Exception as e:
            issues.append(f"{yaml_file.name}: {str(e)}")

    if not issues:
        ColorOutput.print_success(f"{len(yaml_files)} YAML config files are readable")
        return True
    else:
        for issue in issues:
            ColorOutput.print_error(issue)
        return False


def main():
    """Run all checks"""
    print("="*70)
    print("COMPREHENSIVE CODE VALIDATION")
    print("="*70)

    checks = [
        ("Syntax Validation", check_syntax),
        ("Import Structure", check_imports),
        ("File Structure", check_file_structure),
        ("Code Statistics", check_code_stats),
        ("YAML Configs", check_yaml_configs),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            ColorOutput.print_error(f"Error running {name}: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        color = ColorOutput.OKGREEN if result else ColorOutput.FAIL
        print(f"{color}{status}{ColorOutput.ENDC} - {name}")

    print("="*70)

    if all_passed:
        ColorOutput.print_success("\n✓ All checks passed! Code is ready.")
        return 0
    else:
        ColorOutput.print_error("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
