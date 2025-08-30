#!/usr/bin/env python3
"""
Dependency checker for the crypto portfolio tracker.
This script verifies that all required dependencies are installed and compatible.
"""

import sys
import pkg_resources
import subprocess
import json
from pathlib import Path


def find_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml.
    
    Returns:
        Path: The project root directory
    """
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent
    # If we can't find pyproject.toml, return the current file's directory
    return Path(__file__).parent.parent.parent


def check_dependencies():
    """
    Check if all dependencies are installed and compatible.
    
    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """
    print("Checking dependencies...")
    
    # Try to read dependencies from pyproject.toml
    project_root = find_project_root()
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    # Parse dependencies from pyproject.toml
    dependencies = []
    extras = {}
    
    try:
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        # Simple parsing of dependencies (this is a basic implementation)
        # In a real implementation, you might want to use tomli or similar
        lines = content.split('\n')
        in_dependencies = False
        in_dev_dependencies = False
        
        for line in lines:
            line = line.strip()
            
            # Check for main dependencies section
            if line.startswith('dependencies = ['):
                in_dependencies = True
                continue
            elif line.startswith('[project.optional-dependencies]'):
                in_dependencies = False
                continue
            elif line.startswith('dev = [') or line.startswith('"dev" = ['):
                in_dev_dependencies = True
                continue
            elif line.startswith(']') and (in_dependencies or in_dev_dependencies):
                in_dependencies = False
                in_dev_dependencies = False
                continue
            
            # Extract dependencies
            if in_dependencies or in_dev_dependencies:
                if line.startswith('"') and line.endswith('",'):
                    dep = line[1:-2]  # Remove quotes and comma
                    if in_dev_dependencies:
                        extras.setdefault('dev', []).append(dep)
                    else:
                        dependencies.append(dep)
                elif line.startswith('"') and line.endswith('"'):
                    dep = line[1:-1]  # Remove quotes
                    if in_dev_dependencies:
                        extras.setdefault('dev', []).append(dep)
                    else:
                        dependencies.append(dep)
        
        print(f"Found {len(dependencies)} main dependencies and {sum(len(deps) for deps in extras.values())} extra dependencies")
        
    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return False
    
    # Check main dependencies
    all_dependencies_satisfied = True
    
    print("\nChecking main dependencies:")
    for requirement in dependencies:
        try:
            pkg_resources.require(requirement)
            print(f"  ✅ {requirement}")
        except pkg_resources.DistributionNotFound:
            print(f"  ❌ {requirement} (not found)")
            all_dependencies_satisfied = False
        except pkg_resources.VersionConflict as e:
            print(f"  ❌ {requirement} (version conflict: {e})")
            all_dependencies_satisfied = False
        except Exception as e:
            print(f"  ❌ {requirement} (error: {e})")
            all_dependencies_satisfied = False
    
    # Check dev dependencies if requested
    if extras.get('dev'):
        print("\nChecking development dependencies:")
        for requirement in extras['dev']:
            try:
                pkg_resources.require(requirement)
                print(f"  ✅ {requirement}")
            except pkg_resources.DistributionNotFound:
                print(f"  ❌ {requirement} (not found)")
                all_dependencies_satisfied = False
            except pkg_resources.VersionConflict as e:
                print(f"  ❌ {requirement} (version conflict: {e})")
                all_dependencies_satisfied = False
            except Exception as e:
                print(f"  ❌ {requirement} (error: {e})")
                all_dependencies_satisfied = False
    
    return all_dependencies_satisfied


def check_python_version():
    """
    Check if the Python version is compatible.
    
    Returns:
        bool: True if Python version is compatible, False otherwise
    """
    print("Checking Python version...")
    
    # Check minimum version (3.10+)
    if sys.version_info < (3, 10):
        print(f"❌ Python version {sys.version_info} is too old. Minimum required is 3.10")
        return False
    else:
        print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
        return True


def check_uv():
    """
    Check if uv is installed and working.
    
    Returns:
        bool: True if uv is available, False otherwise
    """
    print("Checking uv...")
    
    try:
        result = subprocess.run(['uv', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ uv is available: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv is not available or not working properly")
            return False
    except FileNotFoundError:
        print("❌ uv is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking uv: {e}")
        return False


def main():
    """Main function to run all checks."""
    print("Crypto Portfolio Tracker - Dependency Checker")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("uv Tool", check_uv),
        ("Dependencies", check_dependencies),
    ]
    
    results = []
    
    for check_name, check_function in checks:
        print(f"\n{check_name}:")
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    all_passed = True
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Your environment is ready.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())