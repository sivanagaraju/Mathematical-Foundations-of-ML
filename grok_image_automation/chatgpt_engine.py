"""
ChatGPT Images 2.5 Automation Engine
Automates image generation in ChatGPT Images (https://chatgpt.com/images)
using Playwright with a persistent Chrome profile.

Features:
- Headful Google Chrome with persistent profile in .chatgpt_profile
- Automated login workflow:
    1. Enters email (sivanagarajupachipulusu@gmail.com)
    2. Clicks "Continue"
    3. Handles email-verification prompt by clicking "Continue with password"
    4. Enters password (Pulk@_ta!nt_01!)
    5. Gracefully waits for 2FA / email passcode entry and verification
- Persistent session storage across runs (login once, reuse forever)
- Prompt input and submission via circular send button
- Automated generation monitoring and image downloading
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHATGPT_PROFILE_DIR = SCRIPT_DIR / ".chatgpt_profile"

CHATGPT_BASE_URL = "https://chatgpt.com"
CHATGPT_IMAGES_URL = "https://chatgpt.com/images"
CHATGPT_LOGIN_URL = "https://chatgpt.com/auth/login"

DEFAULT_EMAIL = "sivanagarajupachipulusu@gmail.com"
DEFAULT_PASSWORD = "Pulk@_ta!nt_01!"

def launch_chatgpt_browser(p, profile_dir: Path = CHATGPT_PROFILE_DIR, headless: bool = False):
    """Launch Google Chrome persistent context for ChatGPT."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    context = p.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        channel="chrome",
        headless=headless,
        viewport={"width": 1440, "height": 900},
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox"
        ]
    )
    return context

def is_chatgpt_logged_in(page) -> bool:
    """Check if the user is already authenticated in ChatGPT."""
    try:
        current_url = page.url.lower()
        if "auth/login" in current_url or "auth.openai.com" in current_url:
            return False
        
        # Check 1: If explicit "Log in" or "Sign up" buttons are visible, definitely not logged in
        login_btns = page.locator("button:has-text('Log in'), a:has-text('Log in'), [data-testid='login-button']").all()
        for b in login_btns:
            if b.is_visible():
                return False

        # Check 2: Account or user profile menu is visible (only when authenticated)
        user_indicators = page.locator("[data-testid='user-menu-button'], [data-testid='profile-button'], button[aria-label*='User' i], [aria-label*='Account' i]").all()
        for u in user_indicators:
            if u.is_visible():
                return True

        # Check 3: If on chatgpt.com and no login button is visible at all
        if len(login_btns) == 0 and page.locator("#prompt-textarea, [contenteditable='true']").count() > 0:
            return True
            
    except Exception:
        pass
    return False

def chatgpt_login(
    context,
    login_timeout_seconds: int = 60,
    **kwargs
) -> bool:
    """
    Open Chrome with the persistent profile so the USER can log in manually.
    Waits 60 seconds for the user to complete login in the open Chrome window.
    Saves and preserves all session cookies, tokens, and storage state in .chatgpt_profile.
    """
    page = context.pages[0] if context.pages else context.new_page()
    print("=" * 70, flush=True)
    print("MANUAL CHATGPT LOGIN FLOW", flush=True)
    print(f"Target URL:  {CHATGPT_IMAGES_URL}", flush=True)
    print(f"Profile Dir: {CHATGPT_PROFILE_DIR}", flush=True)
    print(f"Window Time: {login_timeout_seconds} seconds", flush=True)
    print("=" * 70, flush=True)

    print("Opening ChatGPT Images in Chrome...", flush=True)
    try:
        page.goto(CHATGPT_IMAGES_URL, timeout=45000)
    except Exception as e:
        print(f"  Note during initial navigation: {e}", flush=True)
    time.sleep(3)

    if is_chatgpt_logged_in(page):
        print("\n[OK] Already logged into ChatGPT! Session is active.", flush=True)
        return True

    # Navigate to login page if login button exists or not authenticated
    print("Navigating to login page...", flush=True)
    try:
        login_btn = page.locator("button:has-text('Log in'), a:has-text('Log in'), [data-testid='login-button']").first
        if login_btn.count() > 0 and login_btn.is_visible():
            login_btn.click()
        else:
            page.goto(CHATGPT_LOGIN_URL, timeout=30000)
    except Exception:
        page.goto(CHATGPT_LOGIN_URL, timeout=30000)

    print("\n" + "=" * 70, flush=True)
    print("👉 PLEASE COMPLETE LOGIN MANUALLY IN THE OPEN CHROME WINDOW:", flush=True)
    print("  1. Enter your email and password.")
    print("  2. Enter the passcode / verification code or approve on your device.")
    print("  3. Complete any Cloudflare verification if prompted.")
    print(f"Waiting {login_timeout_seconds} seconds for you to log in in Chrome...", flush=True)
    print("=" * 70, flush=True)

    start_wait = time.time()
    logged_in = False

    while True:
        elapsed = int(time.time() - start_wait)
        remaining = max(0, login_timeout_seconds - elapsed)

        # Check if login completed
        try:
            current_url = page.url.lower()
            if ("chatgpt.com" in current_url and "auth" not in current_url) and is_chatgpt_logged_in(page):
                print("\n[OK] Successful login detected!", flush=True)
                logged_in = True
                break
        except Exception:
            # If browser was closed by user
            print("\nBrowser window was closed by user.", flush=True)
            break

        if elapsed >= login_timeout_seconds:
            # Final check on /images
            try:
                page.goto(CHATGPT_IMAGES_URL, timeout=30000)
                time.sleep(3)
                if is_chatgpt_logged_in(page):
                    logged_in = True
                    break
            except Exception:
                pass
            print(f"\nTime limit ({login_timeout_seconds}s) reached.", flush=True)
            break

        if remaining % 10 == 0 or remaining <= 5:
            print(f"  Waiting for manual login in Chrome... ({remaining}s remaining)", flush=True)
        time.sleep(2)

    # Let Chrome persist all cookies and tokens to disk
    print("\nPersisting session tokens and cookies to profile directory...", flush=True)
    time.sleep(4)

    if logged_in or is_chatgpt_logged_in(page):
        print("=" * 70, flush=True)
        print("[SUCCESS] ChatGPT profile saved successfully to:", flush=True)
        print(f"  {CHATGPT_PROFILE_DIR}", flush=True)
        print("This profile will now be automatically reused by:")
        print("  - grok_mathsterms_runner.py --engine chatgpt")
        print("  - grok_imagine_runner.py --engine chatgpt")
        print("=" * 70, flush=True)
        return True
    else:
        print("\n⚠️ Notice: Login not fully confirmed yet. You can re-run with `--login` anytime.", flush=True)
        return False

def ensure_chatgpt_images_page(page, reset: bool = False):
    """Ensure page is on https://chatgpt.com/images and the input box is ready."""
    page.bring_to_front()
    if reset or "/images" not in page.url:
        print(f"Navigating to {CHATGPT_IMAGES_URL}...", flush=True)
        page.goto(CHATGPT_IMAGES_URL, timeout=45000)
        time.sleep(2)

    try:
        page.wait_for_selector("div#prompt-textarea, div.ProseMirror, [contenteditable='true']:visible", timeout=20000)
    except Exception:
        img_nav = page.locator("a[href*='/images'], button:has-text('Images')").first
        if img_nav.count() > 0:
            img_nav.click()
            time.sleep(2)

def is_chatgpt_generating(page) -> bool:
    """Check if ChatGPT is actively generating an image."""
    try:
        stop_btn = page.locator("button[data-testid='stop-button'], button[aria-label*='Stop' i]")
        if stop_btn.count() > 0 and stop_btn.first.is_visible():
            return True
        if page.locator("text='Creating image', text='Thinking', [aria-busy='true']").count() > 0:
            return True
    except Exception:
        pass
    return False

def submit_chatgpt_prompt(page, prompt_text: str, max_retries: int = 2) -> bool:
    """
    Enter prompt into ChatGPT Images prompt textarea and click the circular submit arrow button.
    """
    ensure_chatgpt_images_page(page)

    editor = page.locator("div#prompt-textarea, div.ProseMirror, [contenteditable='true']:visible").first
    editor.wait_for(state="visible", timeout=15000)
    editor.click()
    time.sleep(0.3)

    # Clear previous text
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    time.sleep(0.2)

    # Insert prompt text
    print(f"Entering prompt into ChatGPT Images ({len(prompt_text)} chars)...", flush=True)
    page.keyboard.insert_text(prompt_text)
    time.sleep(0.8)

    # Click the circular Submit / Send button
    for attempt in range(1, max_retries + 1):
        if is_chatgpt_generating(page):
            print("  [OK] Generation already underway!", flush=True)
            return True

        send_btn = page.locator("button[data-testid='send-button'], button[aria-label*='Send' i]").first
        if send_btn.count() > 0 and send_btn.is_visible() and not send_btn.is_disabled():
            print(f"  Attempt {attempt}: Clicking circular Send button...", flush=True)
            send_btn.click()
        else:
            alt_send = page.locator("button.rounded-full").filter(has=page.locator("svg"))
            if alt_send.count() > 0 and alt_send.last.is_visible() and not alt_send.last.is_disabled():
                print(f"  Attempt {attempt}: Clicking alternate Send button...", flush=True)
                alt_send.last.click()
            else:
                print(f"  Attempt {attempt}: Pressing Enter to send...", flush=True)
                page.keyboard.press("Enter")

        for _ in range(8):
            time.sleep(0.5)
            if is_chatgpt_generating(page):
                print("  [OK] ChatGPT generation started successfully!", flush=True)
                return True

        print(f"  Notice: Generation not detected after attempt {attempt}.", flush=True)

    return is_chatgpt_generating(page)

def extract_image_elements(page) -> list:
    """Extract rendered generated image URLs or elements from ChatGPT conversation."""
    image_urls = []
    try:
        imgs = page.locator("article img, main img, img[alt*='Generated' i], img[src*='oaiusercontent.com'], img[src*='blob:']").all()
        for img in imgs:
            src = img.get_attribute("src") or ""
            if src and (src.startswith("http") or src.startswith("blob:")):
                try:
                    width = img.evaluate("e => e.naturalWidth || e.width || 0")
                    if width > 100 or "oaiusercontent" in src or "blob:" in src:
                        if src not in image_urls:
                            image_urls.append(src)
                except Exception:
                    if src not in image_urls:
                        image_urls.append(src)
    except Exception:
        pass
    return image_urls

def wait_and_download_chatgpt_images(
    context,
    page,
    output_dir: Path,
    file_prefix: str,
    pre_existing_urls: set,
    expected_count: int = 1,
    max_wait: int = 180,
    initial_wait: int = 25
) -> list:
    """
    Wait for ChatGPT image generation to finish and download the newly created image(s).
    Includes generous max_wait (up to 180s) and periodic progress heartbeat.
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    if initial_wait > 0:
        print(f"Waiting initial {initial_wait}s for ChatGPT to generate...", flush=True)
        time.sleep(initial_wait)

    new_urls = []
    last_heartbeat = time.time()
    while time.time() - start_time < max_wait:
        generating = is_chatgpt_generating(page)
        current_urls = extract_image_elements(page)
        fresh_urls = [u for u in current_urls if u not in pre_existing_urls]

        if not generating and len(fresh_urls) >= 1:
            new_urls = fresh_urls
            print(f"Generation completed! Found {len(new_urls)} new generated image(s).", flush=True)
            break
        elif len(fresh_urls) >= expected_count:
            new_urls = fresh_urls
            print(f"Reached expected count ({len(new_urls)} images).", flush=True)
            break

        # Progress heartbeat every 15 seconds
        if time.time() - last_heartbeat >= 15:
            elapsed = int(time.time() - start_time)
            remaining = max(0, max_wait - elapsed)
            print(f"  [Generating] Waiting for ChatGPT... ({elapsed}s elapsed, {remaining}s max remaining)", flush=True)
            last_heartbeat = time.time()

        time.sleep(3)

    if not new_urls:
        all_urls = extract_image_elements(page)
        fresh_urls = [u for u in all_urls if u not in pre_existing_urls]
        if fresh_urls:
            new_urls = fresh_urls

    if new_urls:
        print(f"Downloading {len(new_urls)} image(s) for {file_prefix}...", flush=True)
        for idx, img_url in enumerate(new_urls[:expected_count], start=1):
            save_path = output_dir / f"{file_prefix}_img{idx}.jpg"
            try:
                if img_url.startswith("http"):
                    resp = context.request.get(img_url)
                    if resp.status == 200 and len(resp.body()) > 10000:
                        save_path.write_bytes(resp.body())
                        print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size:,} bytes)", flush=True)
                        downloaded_files.append(save_path)
                        continue
                
                print(f"  Attempting canvas/blob extraction for image {idx}...", flush=True)
                img_loc = page.locator(f"img[src='{img_url}']").first
                if img_loc.is_visible():
                    img_bytes = img_loc.screenshot()
                    if len(img_bytes) > 10000:
                        save_path.write_bytes(img_bytes)
                        print(f"  [SAVED via render] {save_path.name} ({save_path.stat().st_size:,} bytes)", flush=True)
                        downloaded_files.append(save_path)
            except Exception as e:
                print(f"  Error saving image {idx}: {e}", flush=True)
    else:
        print(f"⚠️ Warning: No new image URLs detected for {file_prefix}.", flush=True)

    return downloaded_files

def main():
    parser = argparse.ArgumentParser(description="ChatGPT Images 2.5 Automation Engine")
    parser.add_argument("--login", action="store_true", help="Log into ChatGPT and save session into .chatgpt_profile")
    parser.add_argument("--test", action="store_true", help="Test ChatGPT Images session and prompt generation")
    parser.add_argument("--wait", type=int, default=60, help="Seconds to wait for manual login (default: 60)")
    parser.add_argument("--email", type=str, default=DEFAULT_EMAIL, help="ChatGPT account email")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="ChatGPT account password")
    parser.add_argument("--profile-dir", type=str, default=str(CHATGPT_PROFILE_DIR), help="Path to profile directory")

    args = parser.parse_args()
    profile_path = Path(args.profile_dir)

    with sync_playwright() as p:
        context = launch_chatgpt_browser(p, profile_path, headless=False)
        try:
            if args.login:
                chatgpt_login(context, login_timeout_seconds=args.wait)
            elif args.test:
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(CHATGPT_IMAGES_URL, timeout=45000)
                time.sleep(3)
                if not is_chatgpt_logged_in(page):
                    print("Not logged in. Running login first...", flush=True)
                    chatgpt_login(context, login_timeout_seconds=args.wait)
                
                print("Testing prompt generation on ChatGPT Images...", flush=True)
                test_prompt = "A high-contrast mathematical diagram of Jensen's inequality with secant lines and convex curve."
                pre_urls = set(extract_image_elements(page))
                submit_chatgpt_prompt(page, test_prompt)
                test_out = SCRIPT_DIR / "test_chatgpt_output"
                downloaded = wait_and_download_chatgpt_images(context, page, test_out, "test_topic", pre_urls, expected_count=1)
                print(f"Test completed! Downloaded {len(downloaded)} images.", flush=True)
            else:
                parser.print_help()
        finally:
            context.close()

if __name__ == "__main__":
    main()
