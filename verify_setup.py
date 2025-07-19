#!/usr/bin/env python3
"""
Setup Verification Script for AutoSocialMedia Pipeline
This script checks all dependencies, configurations, and environment variables
"""

import os
import sys
import subprocess
import importlib
from colorama import init, Fore, Style

# Initialize colorama for Windows
init()

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

def print_success(text):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_warning(text):
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    print_header("Checking FFmpeg Installation")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print_success(f"FFmpeg installed: {version_line}")
            return True
    except FileNotFoundError:
        pass
    
    print_error("FFmpeg not found in PATH")
    print_info("Install FFmpeg from: https://ffmpeg.org/download.html")
    print_info("Make sure to add it to your system PATH")
    return False

def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print_success(f"{package_name} is installed")
        return True
    except ImportError:
        print_error(f"{package_name} is NOT installed")
        return False

def check_python_dependencies():
    """Check all required Python packages"""
    print_header("Checking Python Dependencies")
    
    packages = [
        ("playwright", "playwright"),
        ("google-generativeai", "google.generativeai"),
        ("requests", "requests"),
        ("Pillow", "PIL"),
        ("moviepy", "moviepy"),
        ("python-dotenv", "dotenv"),
        ("configparser", "configparser"),
        ("whisper-timestamped", "whisper_timestamped"),
        ("beautifulsoup4", "bs4"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("httpx", "httpx"),
        ("colorama", "colorama"),
    ]
    
    missing_packages = []
    for package, import_name in packages:
        if not check_python_package(package, import_name):
            missing_packages.append(package)
    
    # Check optional packages
    print_info("\nChecking optional packages:")
    if not check_python_package("google-cloud-texttospeech", "google.cloud.texttospeech"):
        print_warning("Google Cloud TTS not installed (required only if using Google TTS)")
    
    if missing_packages:
        print_warning(f"\nTo install missing packages, run:")
        print_info(f"pip install -r requirements.txt")
        return False
    
    return True

def check_playwright_browsers():
    """Check if Playwright browsers are installed"""
    print_header("Checking Playwright Browsers")
    try:
        result = subprocess.run(['playwright', 'install', '--dry-run'], capture_output=True, text=True)
        if "chromium" in result.stdout.lower() and "already installed" in result.stdout.lower():
            print_success("Chromium browser is installed")
            return True
        else:
            print_warning("Chromium browser may not be installed")
            print_info("Run: playwright install chromium")
            return False
    except:
        print_error("Could not check Playwright browsers")
        print_info("Run: playwright install chromium")
        return False

def check_environment_variables():
    """Check if required environment variables are set"""
    print_header("Checking Environment Variables")
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        "DEEPSEEK_API_KEY": "Required for AI script generation",
        "GEMINI_API_KEY": "Required for webpage analysis and visual validation"
    }
    
    recommended_vars = {
        "PEXELS_API_KEY": "Recommended for stock photos/videos",
        "UNSPLASH_ACCESS_KEY": "Recommended for stock photos"
    }
    
    optional_vars = {
        "REPLICATE_API_TOKEN": "Optional for AI image generation",
        "GOOGLE_CSE_API_KEY": "Optional for Google Images search",
        "GOOGLE_CSE_ID": "Optional for Google Images search",
        "SEARXNG_URL": "Optional for SearXNG image search",
        "TENOR_API_KEY": "Optional for Tenor reaction GIFs",
        "COVERR_API_KEY": "Optional for Coverr stock videos",
        "OPENVERSE_API_KEY": "Optional for Openverse CC-licensed images"
    }
    
    all_good = True
    
    # Check required variables
    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print_success(f"{var} is set ({desc})")
        else:
            print_error(f"{var} is NOT set or has placeholder value ({desc})")
            all_good = False
    
    # Check recommended variables
    print_info("\nChecking recommended API keys:")
    has_visual_api = False
    for var, desc in recommended_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print_success(f"{var} is set ({desc})")
            has_visual_api = True
        else:
            print_warning(f"{var} is not set ({desc})")
    
    if not has_visual_api:
        print_error("At least one visual API key (PEXELS or UNSPLASH) is recommended!")
        all_good = False
    
    # Check optional variables
    print_info("\nChecking optional API keys:")
    for var, desc in optional_vars.items():
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print_success(f"{var} is set ({desc})")
        else:
            print_info(f"{var} is not set ({desc})")
    
    return all_good

def check_configuration_files():
    """Check if configuration files exist and are valid"""
    print_header("Checking Configuration Files")
    
    files_ok = True
    
    # Check config.ini
    if os.path.exists("config.ini"):
        print_success("config.ini exists")
        # Try to parse it
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read('config.ini')
            tts_provider = config.get('audio', 'tts_provider', fallback='kokoro')
            print_info(f"  TTS Provider: {tts_provider}")
        except Exception as e:
            print_error(f"  Error reading config.ini: {e}")
            files_ok = False
    else:
        print_error("config.ini not found")
        files_ok = False
    
    # Check .env file
    if os.path.exists(".env"):
        print_success(".env file exists")
    else:
        print_error(".env file not found")
        print_info("Copy .env.example to .env and add your API keys")
        files_ok = False
    
    return files_ok

def check_directories():
    """Check and create necessary directories"""
    print_header("Checking Directories")
    
    directories = [
        "runs",
        "runs/current",
        "cache",
        "cache/visual_assets",
        "assets",
        "assets/luts"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print_success(f"Directory exists: {directory}")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                print_success(f"Created directory: {directory}")
            except Exception as e:
                print_error(f"Could not create directory {directory}: {e}")
                return False
    
    return True

def main():
    """Run all verification checks"""
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}AutoSocialMedia Pipeline - Setup Verification{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    checks = [
        ("Python Version", check_python_version),
        ("FFmpeg", check_ffmpeg),
        ("Python Dependencies", check_python_dependencies),
        ("Playwright Browsers", check_playwright_browsers),
        ("Environment Variables", check_environment_variables),
        ("Configuration Files", check_configuration_files),
        ("Directories", check_directories)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("Verification Summary")
    
    all_passed = all(results.values())
    critical_passed = results.get("Python Version", False) and \
                     results.get("Environment Variables", False) and \
                     results.get("Configuration Files", False)
    
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
    
    print()
    if all_passed:
        print_success("All checks passed! You're ready to run the pipeline.")
        print_info("\nRun the pipeline with: python main.py")
    elif critical_passed:
        print_warning("Some optional checks failed, but critical components are ready.")
        print_info("You can run the pipeline, but some features may not work.")
        print_info("\nRun the pipeline with: python main.py")
    else:
        print_error("Critical checks failed. Please fix the issues above before running the pipeline.")
        print_info("\nRefer to the README.md for setup instructions.")

if __name__ == "__main__":
    main()
