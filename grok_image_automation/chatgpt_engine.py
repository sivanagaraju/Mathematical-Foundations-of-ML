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
import base64
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

def get_chatgpt_profile_dir(account: str = "1", custom_dir: str = "") -> Path:
    """
    Resolve profile directory for a given ChatGPT account / session.
    - account="1" or "default" or "main" -> SCRIPT_DIR / ".chatgpt_profile"
    - account="2" -> SCRIPT_DIR / ".chatgpt_profile_2"
    - account="<name>" -> SCRIPT_DIR / f".chatgpt_profile_{name}"
    - If custom_dir is provided, it takes precedence.
    """
    if custom_dir:
        return Path(custom_dir).resolve()
    acc = str(account).strip()
    if acc in ["1", "default", "", "main"]:
        return CHATGPT_PROFILE_DIR
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', acc)
    return SCRIPT_DIR / f".chatgpt_profile_{safe_name}"

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
        accept_downloads=True,
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
    profile_dir: Path = CHATGPT_PROFILE_DIR,
    login_timeout_seconds: int = 60,
    **kwargs
) -> bool:
    """
    Open Chrome with the persistent profile so the USER can log in manually.
    Waits 60 seconds for the user to complete login in the open Chrome window.
    Saves and preserves all session cookies, tokens, and storage state in profile_dir.
    """
    page = context.pages[0] if context.pages else context.new_page()
    print("=" * 70, flush=True)
    print("MANUAL CHATGPT LOGIN FLOW", flush=True)
    print(f"Target URL:  {CHATGPT_IMAGES_URL}", flush=True)
    print(f"Profile Dir: {profile_dir}", flush=True)
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
            try:
                dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
            except Exception:
                pass
        time.sleep(2)

    # Let Chrome persist all cookies and tokens to disk
    print("\nPersisting session tokens and cookies to profile directory...", flush=True)
    time.sleep(4)

    if logged_in or is_chatgpt_logged_in(page):
        print("=" * 70, flush=True)
        print("[SUCCESS] ChatGPT profile saved successfully to:", flush=True)
        print(f"  {profile_dir}", flush=True)
        print("This profile will now be automatically reused by:")
        print("  - grok_mathsterms_runner.py --engine chatgpt")
        print("  - grok_imagine_runner.py --engine chatgpt")
        print("=" * 70, flush=True)
        return True
    else:
        print("\n⚠️ Notice: Login not fully confirmed yet. You can re-run with `--login` anytime.", flush=True)
        return False

def dismiss_chatgpt_modals(page, cooldown_if_rate_limited: bool = False, cooldown_seconds: int = 45) -> bool:
    """
    Detect and dismiss ChatGPT popups/modals, specifically:
    - 'Too many requests' dialog with 'Got it' button
    - Any modal/dialog with 'Got it', 'Got It', 'Dismiss', 'Close', or 'OK' button
    If 'Too many requests' is detected and cooldown_if_rate_limited=True,
    waits cooldown_seconds with countdown logs so ChatGPT rate limit can recover.
    Returns True if a modal was dismissed, False otherwise.
    """
    dismissed = False
    rate_limited = False

    try:
        # Check for Rate Limit markers in text or headings
        rate_limit_indicators = [
            "text='Too many requests'",
            "text='making requests too quickly'",
            "text='temporarily limited access'",
            "text='wait a few minutes'"
        ]
        for ind in rate_limit_indicators:
            if page.locator(ind).count() > 0:
                rate_limited = True
                break
    except Exception:
        pass

    # Selectors for 'Got it' and other dismiss buttons
    button_selectors = [
        "button:has-text('Got it')",
        "button:has-text('Got It')",
        "button:has-text('got it')",
        "div[role='dialog'] button:has-text('Got it')",
        "div[role='dialog'] button:has-text('Got It')",
        "div[role='alertdialog'] button:has-text('Got it')",
        "div[role='alertdialog'] button:has-text('Got It')",
        "div[role='dialog'] button",
        "div[role='alertdialog'] button",
        "button:has-text('Dismiss')",
        "button:has-text('Close')",
        "button:has-text('OK')"
    ]

    for sel in button_selectors:
        try:
            btns = page.locator(sel).all()
            for btn in btns:
                if btn.is_visible():
                    btn_text = btn.inner_text().strip().lower()
                    if any(w in btn_text for w in ["got it", "dismiss", "close", "ok"]):
                        print(f"  [MODAL DETECTED] Clicking '{btn.inner_text().strip()}' button...", flush=True)
                        btn.click(timeout=3000, force=True)
                        dismissed = True
                        time.sleep(1)
                        break
            if dismissed:
                break
        except Exception:
            continue

    # Fallback via JavaScript evaluation in case of layout overlays
    if not dismissed:
        try:
            js_res = page.evaluate("""() => {
                const buttons = Array.from(document.querySelectorAll('button'));
                for (const b of buttons) {
                    const txt = (b.innerText || '').trim().toLowerCase();
                    if (txt === 'got it' || txt === 'got it.' || txt === 'dismiss' || txt === 'close' || txt === 'ok') {
                        b.click();
                        return b.innerText.trim();
                    }
                }
                return null;
            }""")
            if js_res:
                print(f"  [MODAL DETECTED] Clicked '{js_res}' button via JavaScript evaluation.", flush=True)
                dismissed = True
                time.sleep(1)
        except Exception:
            pass

    # Extra fallback: press Escape if dialog still active
    if not dismissed and rate_limited:
        try:
            page.keyboard.press("Escape")
            time.sleep(0.5)
        except Exception:
            pass

    if rate_limited:
        print("  ⚠️ [RATE LIMIT] ChatGPT 'Too many requests' dialog was encountered (clicked 'Got it').", flush=True)
        if cooldown_if_rate_limited:
            print(f"  Cooling down for {cooldown_seconds}s before retrying...", flush=True)
            for rem in range(cooldown_seconds, 0, -10):
                print(f"    Rate limit cooldown: {rem}s remaining...", flush=True)
                time.sleep(10)
            print("  Rate limit cooldown completed! Resuming...", flush=True)

    return dismissed or rate_limited

def ensure_chatgpt_images_page(page, reset: bool = False):
    """
    Ensure page is ready for prompt submission.
    If reset=True, navigates to CHATGPT_IMAGES_URL to start a fresh topic conversation.
    If reset=False and already in an active chat (/c/...) or /images, preserves the current conversation!
    """
    page.bring_to_front()
    dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
    curr_url = page.url or ""
    # Only navigate if reset is True, or not on chatgpt.com at all, or not in a chat/images view
    if reset or ("chatgpt.com" not in curr_url) or ("/c/" not in curr_url and "/images" not in curr_url and "/g/" not in curr_url):
        print(f"Navigating to {CHATGPT_IMAGES_URL}...", flush=True)
        page.goto(CHATGPT_IMAGES_URL, timeout=45000)
        time.sleep(2)
        dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)

    try:
        page.wait_for_selector("div#prompt-textarea, div.ProseMirror, [contenteditable='true']:visible", timeout=20000)
    except Exception:
        dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
        img_nav = page.locator("a[href*='/images'], button:has-text('Images')").first
        if img_nav.count() > 0:
            img_nav.click()
            time.sleep(2)
        dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)

def is_chatgpt_generating(page) -> bool:
    """Check if ChatGPT is actively generating an image or streaming a response."""
    try:
        # Check 1: Stop button (standard indicator in ChatGPT during generation/streaming)
        stop_btn = page.locator(
            "button[data-testid='stop-button'], button[aria-label*='Stop' i], button:has(svg rect), button:has(rect)"
        )
        if stop_btn.count() > 0 and stop_btn.first.is_visible():
            return True

        # Check 2: Generating text or busy states
        if page.locator(
            "text='Creating image', text='Thinking', text='Generating', text='Drawing', text='Working on that', [aria-busy='true'], div[data-testid*='loading']"
        ).count() > 0:
            return True

        # Check 3: Active shimmering / pulse placeholders
        if page.locator("div[class*='animate-pulse'], [data-testid*='image-loading']").count() > 0:
            return True

        # Check 4: Progress percentage pill (e.g. 35%, 68%)
        pills = page.locator("div:has-text('%'), span:has-text('%')")
        for i in range(min(5, pills.count())):
            txt = pills.nth(i).inner_text().strip()
            if re.search(r'\b\d{1,3}%\b', txt):
                return True
    except Exception:
        pass
    return False

def submit_chatgpt_prompt(page, prompt_text: str, max_retries: int = 3, reset_conversation: bool = False) -> bool:
    """
    Enter prompt into ChatGPT prompt textarea and submit.
    If reset_conversation is True, navigates to CHATGPT_IMAGES_URL for a new topic.
    If reset_conversation is False (e.g. for follow-ups in the active conversation), stays directly in the current chat (/c/...)!
    """
    try:
        if reset_conversation:
            ensure_chatgpt_images_page(page, reset=True)
        else:
            dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
            curr_url = page.url or ""
            if "chatgpt.com" not in curr_url:
                ensure_chatgpt_images_page(page, reset=True)

        editor = page.locator("div#prompt-textarea, div.ProseMirror, [contenteditable='true']:visible").first
        editor.wait_for(state="visible", timeout=15000)
        editor.click()
        time.sleep(0.3)

        # Clear previous text
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        time.sleep(0.2)

        # Insert prompt text
        print(f"Entering prompt into ChatGPT ({len(prompt_text)} chars)...", flush=True)
        page.keyboard.insert_text(prompt_text)
        time.sleep(1.0)

        # Click the circular Submit / Send button or press Enter
        for attempt in range(1, max_retries + 1):
            dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
            if is_chatgpt_generating(page):
                print("  [OK] Generation already underway!", flush=True)
                return True

            clicked = False
            send_btn = page.locator("button[data-testid='send-button'], button[aria-label*='Send' i], button[aria-label*='Send prompt' i]").first
            if send_btn.count() > 0 and send_btn.is_visible() and not send_btn.is_disabled():
                try:
                    print(f"  Attempt {attempt}: Clicking circular Send button...", flush=True)
                    send_btn.click(timeout=4000, force=True)
                    clicked = True
                except Exception as e:
                    print(f"  Send button click failed ({e}), checking modals & falling back to Enter...", flush=True)
                    dismiss_chatgpt_modals(page, cooldown_if_rate_limited=True, cooldown_seconds=45)

            if not clicked:
                print(f"  Attempt {attempt}: Pressing Enter to send prompt...", flush=True)
                try:
                    editor.click(timeout=3000, force=True)
                except Exception:
                    pass
                time.sleep(0.2)
                page.keyboard.press("Enter")

            # Wait up to 12s for generation to register
            for check_step in range(24):
                time.sleep(0.5)
                if is_chatgpt_generating(page):
                    print("  [OK] ChatGPT generation started successfully!", flush=True)
                    return True
                if check_step % 6 == 0:
                    was_modal = dismiss_chatgpt_modals(page, cooldown_if_rate_limited=True, cooldown_seconds=45)
                    if was_modal:
                        print("  Modal dismissed during generation wait. Retrying prompt submission...", flush=True)
                        break

            print(f"  Notice: Generation not detected after attempt {attempt}.", flush=True)

        return is_chatgpt_generating(page)
    except Exception as e:
        print(f"⚠️ Error submitting prompt to ChatGPT: {e}", flush=True)
        dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
        return False

def extract_image_elements(page) -> list:
    """Extract rendered generated image URLs or elements from ChatGPT conversation."""
    image_urls = []
    try:
        locs = [
            "[data-message-author-role='assistant'] img",
            "article img",
            "main img",
            "img[alt*='Generated' i]",
            "img[src*='oaiusercontent.com']",
            "img[src*='blob:']",
            "button:has(img) img",
            "div[data-testid*='image'] img"
        ]
        for sel in locs:
            imgs = page.locator(sel).all()
            for img in imgs:
                try:
                    src = img.get_attribute("src") or ""
                    if not src or src in image_urls:
                        continue
                    if any(icon in src.lower() for icon in ["avatar", "icon", "favicon", "profile", "user-"]):
                        continue
                    try:
                        w = img.evaluate("e => e.naturalWidth || e.width || 0")
                    except Exception:
                        w = 200
                    if w > 80 or "oaiusercontent" in src or "blob:" in src:
                        image_urls.append(src)
                except Exception:
                    continue
    except Exception:
        pass
    return image_urls

CHATGPT_FOLLOWUP_IMAGE_PROMPT = "generate the image for the details"

def check_chatgpt_text_message(page) -> str:
    """
    Check if ChatGPT responded with a text message instead of generating an image.
    Inspects the latest assistant conversation turn.
    Returns the message text if detected and no image was created, otherwise empty string.
    """
    try:
        turns = page.locator("[data-message-author-role='assistant'], article:has([data-message-author-role='assistant']), div.markdown").all()
        if turns:
            last_turn = turns[-1]
            turn_imgs = last_turn.locator("img").all()
            has_generated_img = False
            for img in turn_imgs:
                src = img.get_attribute("src") or ""
                if "oaiusercontent.com" in src or "blob:" in src or (src.startswith("http") and not any(icon in src for icon in ["avatar", "icon", "favicon", "profile"])):
                    has_generated_img = True
                    break
            if not has_generated_img:
                txt = last_turn.inner_text().strip()
                if txt:
                    return txt
    except Exception:
        pass
    return ""

def download_chatgpt_highres_image(
    page,
    img_loc,
    output_dir: Path,
    file_prefix: str,
    idx: int = 1,
    img_url: str = "",
    context = None
) -> Path:
    """
    Download the master high-resolution generated image from ChatGPT.
    1. Clicks the generated image in chat to open the high-res lightbox modal.
    2. Clicks the Download button in the top-right toolbar with page.expect_download().
    3. Falls back to in-page blob/URL extraction if download event is not caught.
    4. Closes the lightbox modal to restore clean chat state.
    """
    saved_file = None
    output_dir.mkdir(parents=True, exist_ok=True)
    default_save_path = output_dir / f"{file_prefix}_img{idx}.jpg"

    # Step 1: Locate image in active chat if not already set
    if img_loc is None or (hasattr(img_loc, "count") and img_loc.count() == 0) or (hasattr(img_loc, "is_visible") and not img_loc.is_visible()):
        chat_imgs = page.locator("[data-message-author-role='assistant'] img, article img, main img, div[data-testid*='image'] img").all()
        valid = [i for i in chat_imgs if not any(icon in (i.get_attribute("src") or "").lower() for icon in ["avatar", "icon", "favicon", "profile", "user-"])]
        if valid:
            img_loc = valid[-1]

    # Open lightbox by clicking the image in chat
    opened_lightbox = False
    try:
        if img_loc and img_loc.count() > 0 and img_loc.is_visible():
            print("  Opening ChatGPT full-resolution image viewer...", flush=True)
            img_loc.scroll_into_view_if_needed()
            time.sleep(0.5)

            # Check if parent is a clickable button/container
            parent_btn = img_loc.locator("xpath=ancestor::button | xpath=ancestor::div[@role='button']").first
            if parent_btn.count() > 0 and parent_btn.is_visible():
                parent_btn.click(timeout=4000, force=True)
            else:
                img_loc.click(timeout=4000, force=True)

            # Wait for lightbox / modal to appear
            for _ in range(12):
                time.sleep(0.5)
                for check_sel in [
                    "div[role='dialog']",
                    "div[aria-modal='true']",
                    "button[aria-label*='Close' i]",
                    "button[aria-label*='Download' i]",
                    "button:has-text('Share')"
                ]:
                    if page.locator(check_sel).count() > 0:
                        opened_lightbox = True
                        break
                if opened_lightbox:
                    break

            if not opened_lightbox:
                # Direct JS click on the image or its parent
                page.evaluate("""() => {
                    const imgs = Array.from(document.querySelectorAll("[data-message-author-role='assistant'] img, article img, main img"));
                    const valid = imgs.filter(i => !i.src.includes('avatar') && !i.src.includes('icon') && !i.src.includes('favicon'));
                    if (valid.length > 0) {
                        const target = valid[valid.length - 1];
                        const btn = target.closest('button') || target.closest('[role=\"button\"]') || target;
                        btn.click();
                    }
                }""")
                for _ in range(8):
                    time.sleep(0.5)
                    for check_sel in ["div[role='dialog']", "div[aria-modal='true']", "button[aria-label*='Download' i]", "button:has-text('Share')"]:
                        if page.locator(check_sel).count() > 0:
                            opened_lightbox = True
                            break
                    if opened_lightbox:
                        break
    except Exception as e:
        print(f"  Note while opening image viewer: {e}", flush=True)

    # Step 2: In lightbox, locate the Download button in top-right and trigger high-res download
    if opened_lightbox:
        print("  Full-resolution viewer active. Locating Download button...", flush=True)
        download_btn = None

        # Strategy A: Standard accessible attributes
        for sel in [
            "button[aria-label*='Download' i]",
            "a[aria-label*='Download' i]",
            "button[title*='Download' i]",
            "a[download]",
            "[data-testid*='download' i]",
            "div[role='dialog'] button[aria-label*='Download' i]",
        ]:
            loc = page.locator(sel)
            for i in range(loc.count()):
                btn = loc.nth(i)
                if btn.is_visible():
                    download_btn = btn
                    break
            if download_btn:
                break

        # Strategy B: Proximity - Download button is immediately before 'Share' in top-right toolbar
        if not download_btn:
            try:
                js_btn = page.evaluate_handle("""() => {
                    const btns = Array.from(document.querySelectorAll("div[role='dialog'] button, header button, div[aria-modal='true'] button, button"));
                    // 1. Check aria-label or title
                    for (const b of btns) {
                        const label = (b.getAttribute('aria-label') || '').toLowerCase();
                        const title = (b.getAttribute('title') || '').toLowerCase();
                        if (label.includes('download') || title.includes('download')) {
                            return b;
                        }
                    }
                    // 2. Button immediately preceding 'Share'
                    for (let i = 0; i < btns.length; i++) {
                        const txt = (btns[i].innerText || '').trim().toLowerCase();
                        const label = (btns[i].getAttribute('aria-label') || '').toLowerCase();
                        if ((txt === 'share' || label.includes('share')) && i > 0) {
                            return btns[i - 1];
                        }
                    }
                    return null;
                }""")
                if js_btn and js_btn.as_element():
                    download_btn = js_btn.as_element()
            except Exception:
                pass

        if download_btn:
            try:
                print("  Clicking Lightbox Download button (top right) to download master image...", flush=True)
                with page.expect_download(timeout=15000) as download_info:
                    download_btn.click(timeout=4000, force=True)
                download = download_info.value
                suggested = download.suggested_filename or ""
                ext = Path(suggested).suffix.lower() if suggested else ".jpg"
                if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                    ext = ".jpg"

                target_path = output_dir / f"{file_prefix}_img{idx}{ext}"
                download.save_as(str(target_path))
                if target_path.exists() and target_path.stat().st_size > 25000:
                    print(f"  [SAVED FULL-RES via Download button] {target_path.name} ({target_path.stat().st_size:,} bytes)", flush=True)
                    saved_file = target_path
                    if ext != ".jpg":
                        try:
                            default_save_path.write_bytes(target_path.read_bytes())
                        except Exception:
                            pass
            except Exception as e:
                print(f"  Direct download button note ({e}), trying Share modal flow...", flush=True)

        # Strategy C: The Share Modal -> Download Button Flow (User-verified golden path)
        # In ChatGPT's image viewer, clicking 'Share' opens "Share a public link to this image"
        # which has a clear, unambiguous [Download] button at the bottom right.
        if not saved_file:
            try:
                print("  Attempting download via Share Modal flow...", flush=True)
                share_modal_visible = False
                for chk in [
                    "text='Share a public link'",
                    "text='public link to this image'",
                    "div[role='dialog'] button:has-text('Download')",
                    "div[role='dialog'] [role='button']:has-text('Download')"
                ]:
                    if page.locator(chk).count() > 0:
                        share_modal_visible = True
                        break

                if not share_modal_visible:
                    # Locate and click the Share button in the image viewer
                    share_btn = None
                    for s_sel in [
                        "button:has-text('Share')",
                        "div[role='dialog'] button:has-text('Share')",
                        "div[aria-modal='true'] button:has-text('Share')",
                        "button[aria-label*='Share' i]",
                        "[data-testid*='share' i]"
                    ]:
                        loc = page.locator(s_sel)
                        for i in range(loc.count()):
                            b = loc.nth(i)
                            if b.is_visible():
                                share_btn = b
                                break
                        if share_btn:
                            break

                    if share_btn:
                        print("  Clicking 'Share' button in viewer to open share options...", flush=True)
                        share_btn.click(timeout=4000, force=True)
                        time.sleep(1.2)
                        for _ in range(8):
                            if (page.locator("text='Share a public link'").count() > 0 or 
                                page.locator("button:has-text('Download')").count() > 0):
                                share_modal_visible = True
                                break
                            time.sleep(0.4)

                # Locate the Download button inside the Share modal dialog
                download_in_share = None
                for d_sel in [
                    "div[role='dialog'] button:has-text('Download')",
                    "div[role='dialog'] [role='button']:has-text('Download')",
                    "div[aria-modal='true'] button:has-text('Download')",
                    "button:has-text('Download')",
                    "[role='button']:has-text('Download')",
                ]:
                    loc = page.locator(d_sel)
                    for i in range(loc.count()):
                        b = loc.nth(i)
                        if b.is_visible() and "download" in (b.inner_text() or "").strip().lower():
                            download_in_share = b
                            break
                    if download_in_share:
                        break

                # JS element search fallback inside dialogs
                if not download_in_share:
                    try:
                        js_dl = page.evaluate_handle("""() => {
                            const dialogs = document.querySelectorAll("div[role='dialog'], div[aria-modal='true']");
                            for (const d of dialogs) {
                                const candidates = Array.from(d.querySelectorAll("button, [role='button'], div, a, span"));
                                for (const el of candidates) {
                                    const txt = (el.innerText || '').trim().toLowerCase();
                                    if (txt === 'download') {
                                        return el.closest('button') || el.closest('[role=\"button\"]') || el;
                                    }
                                }
                            }
                            return null;
                        }""")
                        if js_dl and js_dl.as_element():
                            download_in_share = js_dl.as_element()
                    except Exception:
                        pass

                if download_in_share:
                    print("  Clicking 'Download' button in Share modal...", flush=True)
                    with page.expect_download(timeout=15000) as download_info:
                        download_in_share.click(timeout=4000, force=True)
                    download = download_info.value
                    suggested = download.suggested_filename or ""
                    ext = Path(suggested).suffix.lower() if suggested else ".jpg"
                    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                        ext = ".jpg"

                    target_path = output_dir / f"{file_prefix}_img{idx}{ext}"
                    download.save_as(str(target_path))
                    if target_path.exists() and target_path.stat().st_size > 25000:
                        print(f"  [SAVED FULL-RES via Share Modal] {target_path.name} ({target_path.stat().st_size:,} bytes)", flush=True)
                        saved_file = target_path
                        if ext != ".jpg":
                            try:
                                default_save_path.write_bytes(target_path.read_bytes())
                            except Exception:
                                pass

                    # Dismiss the Share modal
                    try:
                        page.keyboard.press("Escape")
                        time.sleep(0.5)
                        close_btns = page.locator("button[aria-label*='Close' i], div[role='dialog'] button:has-text('✕')").all()
                        for cb in close_btns:
                            if cb.is_visible():
                                cb.click(timeout=1500)
                                break
                    except Exception:
                        pass
                    time.sleep(0.5)
            except Exception as e:
                print(f"  Share modal flow note: {e}", flush=True)

        # Fallback in lightbox: extract full-resolution image data from lightbox img
        if not saved_file:
            try:
                blob_b64 = page.evaluate("""async () => {
                    try {
                        const img = document.querySelector("div[role='dialog'] img, div[aria-modal='true'] img, img[alt*='Generated']");
                        if (!img || !img.src) return null;
                        const r = await fetch(img.src);
                        const b = await r.blob();
                        return new Promise((resolve) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result);
                            reader.readAsDataURL(b);
                        });
                    } catch (e) {
                        return null;
                    }
                }""")
                if blob_b64 and "," in blob_b64:
                    raw_bytes = base64.b64decode(blob_b64.split(",", 1)[1])
                    if len(raw_bytes) > 25000:
                        default_save_path.write_bytes(raw_bytes)
                        print(f"  [SAVED FULL-RES via viewer blob] {default_save_path.name} ({len(raw_bytes):,} bytes)", flush=True)
                        saved_file = default_save_path
            except Exception as e:
                print(f"  Viewer extraction note: {e}", flush=True)

        # Close the lightbox modal
        try:
            page.keyboard.press("Escape")
            time.sleep(0.5)
            close_btn = page.locator("button[aria-label*='Close' i], div[role='dialog'] button:has-text('✕')").first
            if close_btn.count() > 0 and close_btn.is_visible():
                close_btn.click(timeout=2000)
        except Exception:
            pass
        time.sleep(0.5)

    # Step 3: Fallback if lightbox could not be opened: fetch binary blob directly in-page
    if not saved_file and img_url and img_url.startswith("blob:"):
        try:
            print("  Fetching blob binary directly from page context...", flush=True)
            blob_b64 = page.evaluate("""async (u) => {
                try {
                    const r = await fetch(u);
                    const b = await r.blob();
                    return new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result);
                        reader.readAsDataURL(b);
                    });
                } catch (e) {
                    return null;
                }
            }""", img_url)
            if blob_b64 and "," in blob_b64:
                raw_bytes = base64.b64decode(blob_b64.split(",", 1)[1])
                if len(raw_bytes) > 20000:
                    default_save_path.write_bytes(raw_bytes)
                    print(f"  [SAVED via page blob fetch] {default_save_path.name} ({len(raw_bytes):,} bytes)", flush=True)
                    saved_file = default_save_path
        except Exception as e:
            print(f"  Blob fetch error: {e}", flush=True)

    # Step 4: Fallback for http URLs
    if not saved_file and img_url and img_url.startswith("http") and context:
        try:
            resp = context.request.get(img_url)
            if resp.status == 200 and len(resp.body()) > 20000:
                default_save_path.write_bytes(resp.body())
                print(f"  [SAVED via HTTP request] {default_save_path.name} ({len(resp.body()):,} bytes)", flush=True)
                saved_file = default_save_path
        except Exception as e:
            print(f"  HTTP request error: {e}", flush=True)

    # Step 5: Last-resort fallback: element screenshot
    if not saved_file and img_loc and img_loc.is_visible():
        try:
            print(f"  Notice: Falling back to element screenshot for {default_save_path.name}...", flush=True)
            img_bytes = img_loc.screenshot()
            if len(img_bytes) > 10000:
                default_save_path.write_bytes(img_bytes)
                print(f"  [SAVED via render fallback] {default_save_path.name} ({len(img_bytes):,} bytes)", flush=True)
                saved_file = default_save_path
        except Exception:
            pass

    return saved_file

def download_from_chatgpt_chat_url(
    context,
    page,
    chat_url: str,
    output_dir: Path,
    file_prefix: str,
    idx: int = 1,
    max_wait: int = 60
) -> Path:
    """
    Navigate directly to an existing ChatGPT chat URL (e.g. https://chatgpt.com/c/6ab96387-104c-83ee-8bfa-e5b8a76d463c),
    find the generated image in that conversation, and download it at full resolution.
    """
    if chat_url and page.url.strip() != chat_url.strip():
        print(f"Navigating to existing chat conversation: {chat_url}...", flush=True)
        page.goto(chat_url, timeout=60000)
        time.sleep(3)
    dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)

    # Check if a dialog or share modal is already open
    if page.locator("div[role='dialog'], div[aria-modal='true']").count() > 0:
        print("  Detected active dialog/modal already open in chat. Proceeding directly to download...", flush=True)
        return download_chatgpt_highres_image(
            page=page,
            img_loc=None,
            output_dir=output_dir,
            file_prefix=file_prefix,
            idx=idx,
            context=context
        )

    img_loc = None
    start = time.time()
    while time.time() - start < max_wait:
        dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)
        chat_imgs = page.locator("[data-message-author-role='assistant'] img, article img, main img, div[data-testid*='image'] img").all()
        valid = [i for i in chat_imgs if not any(icon in (i.get_attribute("src") or "").lower() for icon in ["avatar", "icon", "favicon", "profile", "user-"])]
        if valid:
            img_loc = valid[-1]
            if img_loc.is_visible():
                break
        time.sleep(2)

    if not img_loc:
        print(f"⚠️ Could not locate generated image in chat {chat_url}", flush=True)
        return None

    return download_chatgpt_highres_image(
        page=page,
        img_loc=img_loc,
        output_dir=output_dir,
        file_prefix=file_prefix,
        idx=idx,
        context=context
    )

def wait_and_download_chatgpt_images(
    context,
    page,
    output_dir: Path,
    file_prefix: str,
    pre_existing_urls: set,
    expected_count: int = 1,
    max_wait: int = 360,
    initial_wait: int = 20,
    follow_up_message: str = CHATGPT_FOLLOWUP_IMAGE_PROMPT
) -> list:
    """
    Wait for ChatGPT image generation to finish and download the newly created image(s).
    Includes generous max_wait (default 360s / 6 minutes), continuous wait while actively generating,
    periodic modal dismissal, and progress heartbeat.
    If ChatGPT responds with a text message rather than generating an image, automatically sends
    the follow-up message: 'generate the image for the details' staying in the active chat.
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    follow_up_sent = False

    if initial_wait > 0:
        print(f"Waiting initial {initial_wait}s for ChatGPT to process...", flush=True)
        for step in range(max(1, initial_wait // 3)):
            time.sleep(3)
            dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)

            # Check if ChatGPT prematurely stopped or responded with text instead of generating
            if step >= 1:
                gen_now = is_chatgpt_generating(page)
                curr_urls = extract_image_elements(page)
                fr_urls = [u for u in curr_urls if u not in pre_existing_urls]

                if not gen_now and len(fr_urls) >= 1:
                    break

                if not gen_now and len(fr_urls) == 0:
                    text_resp = check_chatgpt_text_message(page)
                    if text_resp:
                        preview = (text_resp[:120] + "...") if len(text_resp) > 120 else text_resp
                        print(f"  [TEXT RESPONSE DETECTED] ChatGPT replied with message rather than image: '{preview}'", flush=True)
                        print(f"  [FOLLOW-UP] Staying in current chat and sending: '{follow_up_message}'...", flush=True)
                        submit_chatgpt_prompt(page, follow_up_message, reset_conversation=False)
                        follow_up_sent = True
                        start_time = time.time()
                        time.sleep(4)
                        break

    new_urls = []
    last_heartbeat = time.time()
    while (time.time() - start_time < max_wait) or (is_chatgpt_generating(page) and (time.time() - start_time < max_wait + 120)):
        # Check and dismiss any 'Got it' or 'Too many requests' modal that popped up
        dismiss_chatgpt_modals(page, cooldown_if_rate_limited=False)

        generating = is_chatgpt_generating(page)
        current_urls = extract_image_elements(page)
        fresh_urls = [u for u in current_urls if u not in pre_existing_urls]

        if not generating and len(fresh_urls) >= 1:
            new_urls = fresh_urls
            print(f"Generation completed! Found {len(new_urls)} new generated image(s) after {int(time.time() - start_time)}s.", flush=True)
            break
        elif len(fresh_urls) >= expected_count:
            new_urls = fresh_urls
            print(f"Reached expected count ({len(new_urls)} images).", flush=True)
            break

        # Check if assistant turn has an image even if URL diff was tricky
        if not generating and len(fresh_urls) == 0:
            chat_imgs = page.locator("[data-message-author-role='assistant'] img, article:has([data-message-author-role='assistant']) img").all()
            valid_chat = []
            for ci in chat_imgs:
                s = ci.get_attribute("src") or ""
                if s and not any(icon in s.lower() for icon in ["avatar", "icon", "favicon", "profile"]):
                    valid_chat.append(s)
            candidate_fresh = [u for u in valid_chat if u not in pre_existing_urls]
            if candidate_fresh:
                new_urls = candidate_fresh
                print(f"Generation completed! Found assistant image(s) after {int(time.time() - start_time)}s.", flush=True)
                break

        # If not generating and no fresh images, check if ChatGPT responded with a text message
        if not generating and len(fresh_urls) == 0 and not follow_up_sent:
            text_resp = check_chatgpt_text_message(page)
            elapsed = time.time() - start_time
            if text_resp or elapsed >= 20:
                preview = (text_resp[:120] + "...") if len(text_resp) > 120 else (text_resp or "No image generated, editor idle")
                print(f"  [TEXT RESPONSE DETECTED] ChatGPT replied with message rather than image: '{preview}'", flush=True)
                print(f"  [FOLLOW-UP] Staying in current chat and sending: '{follow_up_message}'...", flush=True)
                submit_chatgpt_prompt(page, follow_up_message, reset_conversation=False)
                follow_up_sent = True
                start_time = time.time()
                last_heartbeat = time.time()
                time.sleep(4)
                continue

        # Progress heartbeat every 15 seconds
        if time.time() - last_heartbeat >= 15:
            elapsed = int(time.time() - start_time)
            remaining = max(0, max_wait - elapsed)
            gen_label = "Generating image..." if generating else "Waiting for render..."
            print(f"  [{gen_label}] ChatGPT active... ({elapsed}s elapsed, {remaining}s remaining)", flush=True)
            last_heartbeat = time.time()

        time.sleep(3)

    if not new_urls:
        all_urls = extract_image_elements(page)
        fresh_urls = [u for u in all_urls if u not in pre_existing_urls]
        if fresh_urls:
            new_urls = fresh_urls

    if new_urls:
        print(f"Downloading {len(new_urls)} full-resolution image(s) for {file_prefix}...", flush=True)
        for idx, img_url in enumerate(new_urls[:expected_count], start=1):
            img_loc = page.locator(f"img[src='{img_url}']").first
            if img_loc.count() == 0 or not img_loc.is_visible():
                all_imgs = page.locator("[data-message-author-role='assistant'] img, article img, main img, img[alt*='Generated' i], img[src*='oaiusercontent.com'], img[src*='blob:']").all()
                valid_imgs = [i for i in all_imgs if not any(icon in (i.get_attribute("src") or "").lower() for icon in ["avatar", "icon", "favicon", "profile"])]
                if valid_imgs:
                    img_loc = valid_imgs[-1]

            saved_path = download_chatgpt_highres_image(
                page=page,
                img_loc=img_loc,
                output_dir=output_dir,
                file_prefix=file_prefix,
                idx=idx,
                img_url=img_url,
                context=context
            )
            if saved_path and saved_path.exists():
                downloaded_files.append(saved_path)
    else:
        # Final fallback: check if any assistant image is visible right now in the chat
        all_imgs = page.locator("[data-message-author-role='assistant'] img, article img, main img").all()
        valid_imgs = [i for i in all_imgs if not any(icon in (i.get_attribute("src") or "").lower() for icon in ["avatar", "icon", "favicon", "profile"])]
        if valid_imgs:
            print(f"  Attempting download from visible assistant image in chat for {file_prefix}...", flush=True)
            saved_path = download_chatgpt_highres_image(
                page=page,
                img_loc=valid_imgs[-1],
                output_dir=output_dir,
                file_prefix=file_prefix,
                idx=1,
                context=context
            )
            if saved_path and saved_path.exists():
                downloaded_files.append(saved_path)
        else:
            print(f"⚠️ Warning: No new image detected for {file_prefix}.", flush=True)

    return downloaded_files

def main():
    parser = argparse.ArgumentParser(description="ChatGPT Images 2.5 Automation Engine")
    parser.add_argument("--account", type=str, default="1", help="ChatGPT account / session identifier (e.g. 1, 2, 'alt', default: '1')")
    parser.add_argument("--login", action="store_true", help="Log into ChatGPT and save session into profile")
    parser.add_argument("--test", action="store_true", help="Test ChatGPT Images session and prompt generation")
    parser.add_argument("--wait", type=int, default=60, help="Seconds to wait for manual login (default: 60)")
    parser.add_argument("--email", type=str, default=DEFAULT_EMAIL, help="ChatGPT account email")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="ChatGPT account password")
    parser.add_argument("--profile-dir", type=str, default="", help="Path to profile directory (overrides --account)")

    args = parser.parse_args()
    profile_path = get_chatgpt_profile_dir(args.account, custom_dir=args.profile_dir)

    print(f"ChatGPT Profile: {profile_path} (Account: {args.account})", flush=True)

    with sync_playwright() as p:
        context = launch_chatgpt_browser(p, profile_path, headless=False)
        try:
            if args.login:
                chatgpt_login(context, profile_dir=profile_path, login_timeout_seconds=args.wait)
            elif args.test:
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(CHATGPT_IMAGES_URL, timeout=45000)
                time.sleep(3)
                if not is_chatgpt_logged_in(page):
                    print("Not logged in. Running login first...", flush=True)
                    chatgpt_login(context, profile_dir=profile_path, login_timeout_seconds=args.wait)
                
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
