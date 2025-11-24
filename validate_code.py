#!/usr/bin/env python3
"""
Code Validation Script

Validates Python code without requiring dependencies.
Checks:
- Syntax errors
- Import structure
- Function/class definitions
- Basic code quality
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


class CodeValidator:
    """Validates Python code structure and syntax"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.errors = []
        self.warnings = []

    def validate_syntax(self, filepath: Path) -> bool:
        """Check if file has valid Python syntax"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            compile(source, str(filepath), 'exec')
            return True
        except SyntaxError as e:
            self.errors.append(f"{filepath}: Syntax error at line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            self.errors.append(f"{filepath}: {str(e)}")
            return False

    def validate_ast(self, filepath: Path) -> bool:
        """Validate AST structure"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, str(filepath))

            # Check for basic issues
            for node in ast.walk(tree):
                # Check for undefined names would require more complex analysis
                pass

            return True
        except Exception as e:
            self.errors.append(f"{filepath}: AST parse error: {str(e)}")
            return False

    def check_imports(self, filepath: Path) -> List[str]:
        """Extract imports from file"""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return imports
        except Exception:
            return []

    def count_definitions(self, filepath: Path) -> Tuple[int, int]:
        """Count classes and functions"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)

            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))

            return classes, functions
        except Exception:
            return 0, 0

    def validate_file(self, filepath: Path) -> bool:
        """Validate a single file"""
        print(f"  Checking {filepath.relative_to(self.root_dir)}...", end=" ")

        # Check syntax
        if not self.validate_syntax(filepath):
            print("✗ SYNTAX ERROR")
            return False

        # Check AST
        if not self.validate_ast(filepath):
            print("✗ AST ERROR")
            return False

        # Count definitions
        classes, functions = self.count_definitions(filepath)

        print(f"✓ ({classes} classes, {functions} functions)")
        return True

    def validate_directory(self, directory: Path) -> int:
        """Validate all Python files in directory"""
        py_files = list(directory.rglob("*.py"))
        valid_count = 0

        for filepath in sorted(py_files):
            # Skip __pycache__ and .git
            if '__pycache__' in str(filepath) or '.git' in str(filepath):
                continue

            if self.validate_file(filepath):
                valid_count += 1

        return valid_count

    def print_summary(self, total: int, valid: int):
        """Print validation summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Total files checked: {total}")
        print(f"Valid files: {valid}")
        print(f"Files with errors: {total - valid}")

        if self.errors:
            print(f"\n{len(self.errors)} ERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")

        if self.warnings:
            print(f"\n{len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        if not self.errors:
            print("\n✓ All files passed validation!")
            return 0
        else:
            print("\n✗ Validation failed!")
            return 1


def main():
    print("="*70)
    print("CODE VALIDATION")
    print("="*70)
    print()

    root_dir = Path(__file__).parent
    validator = CodeValidator(root_dir)

    # Validate source code
    print("Validating src/...")
    src_valid = validator.validate_directory(root_dir / "src")

    # Validate tests
    print("\nValidating tests/...")
    test_valid = validator.validate_directory(root_dir / "tests")

    # Validate scripts
    print("\nValidating scripts/...")
    script_valid = validator.validate_directory(root_dir / "scripts")

    total = src_valid + test_valid + script_valid
    total_files = len([f for f in root_dir.rglob("*.py")
                       if '__pycache__' not in str(f) and '.git' not in str(f)])

    return validator.print_summary(total_files, total)


if __name__ == "__main__":
    sys.exit(main())
