"""
Grok Imagine Batch Automation
Automates image generation in Grok Imagine using Playwright with persistent session storage.

Features:
- Headful Google Chrome with persistent profile (cookies & tokens saved across runs)
- Interactive login workflow (gives you time to log in via Google, X, or email)
- Imagine configuration: Quality 2.0, 2:3 aspect ratio, 2 images (x2)
- Explicit clicking of the blue circle Submit button (button[aria-label='Submit'])
- Pre-existing image ID filtering (never downloads stale/old images)
- In-browser authenticated download via context.request.get (bypasses 403 Forbidden)
- Full UTF-8 support on Windows terminal
"""

import os
import re
import sys
import time
import json
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

# Force UTF-8 stdout on Windows to prevent charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Locate script and project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROFILE_DIR = SCRIPT_DIR / ".grok_profile"

GROK_URL = "https://grok.com"
GROK_IMAGINE_URL = "https://grok.com/imagine"
GROK_LIBRARY_URL = "https://grok.com/library"

def get_topic_titles(target_dir: Path):
    """Load topic titles from metadata.json if available."""
    if not target_dir:
        return []
    metadata_file = target_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("topics", [])
        except Exception:
            pass
    return []

def clean_transcript(text: str) -> str:
    """Remove timestamp prefixes like [00:01] or **[00:01]** and clean excess whitespace."""
    cleaned = re.sub(r'\*?\*?\[\d{2}:\d{2}\]\*?\*?', '', text)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

def format_prompt(topic_idx: int, topic_title: str, text: str, mode: str = "cleaned") -> str:
    """Format prompt with comprehensive explanatory infographic directive followed by full topic content."""
    content = text.strip() if mode == "raw" else clean_transcript(text)
    title_suffix = f" of '{topic_title}'" if topic_title else ""
    prompt = (
        f"Comprehensive educational technical infographic clearly explaining the core mechanics{title_suffix}. Critical is for given context need to work and explain that."
        f"Step-by-step visual explanation with annotated mathematical formulas, labeled architecture block diagrams, directional data flow arrows, and structured explanation. "
        f"Clear technical intuition, readable typography, professional university lecture poster layout. "
        f"Topic content:\n{content}"
    )
    return prompt

def launch_browser(p, profile_dir: Path, headless: bool = False):
    """Launch persistent Chrome browser context."""
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

def interactive_login(profile_dir: Path):
    """Open Chrome, navigate to Grok, and let the user log in interactively."""
    print("=" * 70, flush=True)
    print("GROK INTERACTIVE LOGIN", flush=True)
    print("=" * 70, flush=True)
    print(f"Profile directory: {profile_dir}", flush=True)
    print("Please log into your account in the opened browser window.", flush=True)
    print("Once you are fully logged in and see Grok, return to this terminal.", flush=True)
    print("=" * 70, flush=True)

    with sync_playwright() as p:
        context = launch_browser(p, profile_dir, headless=False)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(GROK_URL)
            
            try:
                input("\n👉 Press [ENTER] in this terminal once you have finished logging into Grok...")
            except KeyboardInterrupt:
                print("\nLogin cancelled.", flush=True)
                return False

            print("\nSaving session state...", flush=True)
            current_url = page.url
            print(f"Current page URL: {current_url}", flush=True)
            time.sleep(2)
            print(f"[OK] Session successfully saved to: {profile_dir}", flush=True)
            return True
        finally:
            context.close()

def configure_imagine_settings(page, quality: str = "2.0", aspect_ratio: str = "2:3", num_images: int = 2):
    """Ensure Grok Imagine settings (Quality 2.0, 2:3 ratio, x2 count) are selected."""
    page.bring_to_front()
    try:
        page.wait_for_selector(".tiptap, .ProseMirror", timeout=15000)
        page.wait_for_selector("button[aria-label='Aspect Ratio']", timeout=10000)
    except Exception:
        pass

    print(f"Configuring Imagine settings (Quality: {quality}, Ratio: {aspect_ratio}, Count: x{num_images})...", flush=True)
    
    # 1. Quality 2.0
    try:
        quality_btn = page.locator("button, [role='button']").filter(has_text="Quality 2.0")
        if quality_btn.count() > 0:
            quality_btn.first.click()
            print("  Selected Quality 2.0", flush=True)
            time.sleep(0.5)
    except Exception as e:
        print(f"  Note on Quality selector: {e}", flush=True)

    # 2. Aspect Ratio (2:3)
    try:
        ar_btn = page.locator("button[aria-label='Aspect Ratio']")
        if ar_btn.count() > 0:
            if aspect_ratio not in ar_btn.first.inner_text():
                ar_btn.first.click()
                time.sleep(0.5)
                opt = page.locator("[role='menuitemradio'], [role='option']").filter(has_text=aspect_ratio)
                if opt.count() > 0:
                    opt.first.click()
                print(f"  Selected Aspect Ratio: {aspect_ratio}", flush=True)
                time.sleep(0.5)
            else:
                print(f"  Verified Aspect Ratio: {aspect_ratio}", flush=True)
        else:
            print("  Note: Aspect Ratio button not visible", flush=True)
    except Exception as e:
        print(f"  Note on Aspect Ratio selector: {e}", flush=True)

    # 3. Image Count (x2)
    try:
        cnt_btn = page.locator("button[aria-label='Image Count']")
        if cnt_btn.count() > 0:
            cnt_text = cnt_btn.first.inner_text().strip()
            target_str = str(num_images)
            if target_str not in cnt_text:
                cnt_btn.first.click()
                time.sleep(0.5)
                opt_cnt = page.locator("[role='menuitemradio']").filter(has_text=re.compile(rf"^{target_str}$|^x{target_str}$", re.I))
                if opt_cnt.count() > 0:
                    opt_cnt.first.click()
                    print(f"  Selected Image Count: {num_images} images (x{num_images})", flush=True)
                time.sleep(0.5)
            else:
                print(f"  Verified Image Count: x{num_images}", flush=True)
        else:
            print("  Note: Image Count button not visible", flush=True)
    except Exception as e:
        print(f"  Note on Image Count selector: {e}", flush=True)

def is_generation_underway(page, initial_canvases: int, pre_ids: set) -> bool:
    """Check if Grok Imagine has actively started generating images."""
    try:
        # Check 1: Canvas placeholders for the dot matrix animation (seen in generating cards)
        current_canvases = page.locator("canvas").count()
        if current_canvases > initial_canvases:
            return True

        # Check 2: New post IDs appearing in DOM
        current_ids = set(extract_post_ids(page))
        if len(current_ids - pre_ids) > 0:
            return True

        # Check 3: Submit button disabled while prompt is in editor
        submit_btn = page.locator("button[aria-label='Submit']")
        if submit_btn.count() > 0 and submit_btn.first.is_visible():
            if submit_btn.first.is_disabled():
                return True

        # Check 4: Generating pulse or busy states
        if page.locator("[class*='animate-pulse'], [aria-busy='true']").count() > 0:
            return True
    except Exception:
        pass
    return False

def trim_content_at_end(content: str, max_chars: int = 3800) -> str:
    """Trim transcript content at the end for rare scenarios where context limit is hit."""
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    # Try to cleanly break at paragraph or sentence boundary
    last_para = truncated.rfind('\n\n')
    if last_para > int(max_chars * 0.7):
        truncated = truncated[:last_para]
    else:
        last_dot = truncated.rfind('. ')
        if last_dot > int(max_chars * 0.7):
            truncated = truncated[:last_dot + 1]
    return truncated.strip() + "\n\n[...context trimmed at end due to length limit...]"

def trigger_submit_safely(page, initial_canvases: int, pre_ids: set, max_attempts: int = 2) -> bool:
    """
    Attempt to trigger generation. If not generating after an attempt, safely re-click button.
    CRITICAL: Only click if generation is NOT already underway.
    """
    for attempt in range(1, max_attempts + 1):
        if is_generation_underway(page, initial_canvases, pre_ids):
            print(f"  [OK] Generation already underway. Button will NOT be re-pressed.", flush=True)
            return True

        submit_btn = page.locator("button[aria-label='Submit']")
        if submit_btn.count() > 0 and submit_btn.first.is_visible():
            if not submit_btn.first.is_disabled():
                print(f"  Submit attempt {attempt}/{max_attempts}: clicking submit button...", flush=True)
                submit_btn.first.click()
            else:
                print(f"  Submit attempt {attempt}/{max_attempts}: button disabled, checking generation status...", flush=True)
        else:
            alt_btn = page.locator("button.rounded-full, button.bg-primary").filter(has=page.locator("svg"))
            if alt_btn.count() > 0 and alt_btn.last.is_visible():
                print(f"  Submit attempt {attempt}/{max_attempts}: clicking alternate submit button...", flush=True)
                alt_btn.last.click()
            else:
                print(f"  Submit attempt {attempt}/{max_attempts}: pressing Enter...", flush=True)
                page.keyboard.press("Enter")

        # Poll for 3-4 seconds to confirm if generation started
        for _ in range(7):
            time.sleep(0.5)
            if is_generation_underway(page, initial_canvases, pre_ids):
                print("  [OK] Generation started successfully!", flush=True)
                return True

        print(f"  Notice: Generation not detected after attempt {attempt}.", flush=True)

    return is_generation_underway(page, initial_canvases, pre_ids)

def enter_prompt_and_generate(
    page,
    prompt_text: str,
    raw_content: str = "",
    topic_idx: int = 1,
    topic_title: str = "",
    prompt_mode: str = "cleaned",
    pre_ids: set = None
):
    """
    Find prompt input, clear, type prompt, and safely trigger submit.
    1. Re-clicks submit only if image generation has NOT started (never clicks if already generating).
    2. Rare fallback: if context limit is hit causing failure, trims context at the end and re-submits.
    """
    page.bring_to_front()
    if pre_ids is None:
        pre_ids = set(extract_post_ids(page))
    initial_canvases = page.locator("canvas").count()

    print(f"Entering prompt ({len(prompt_text)} chars)...", flush=True)
    
    # Locate TipTap / ProseMirror rich text editor
    editor = page.locator(".tiptap, .ProseMirror, [contenteditable='true']").first
    try:
        editor.wait_for(state="visible", timeout=15000)
    except Exception:
        print("  Editor not visible, navigating to Grok Imagine...", flush=True)
        page.goto(GROK_IMAGINE_URL, timeout=30000)
        time.sleep(2)
        editor.wait_for(state="visible", timeout=15000)

    editor.click()
    time.sleep(0.3)

    # Clear existing content
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    time.sleep(0.2)

    # Insert full prompt text
    page.keyboard.insert_text(prompt_text)
    time.sleep(0.8)

    # Step 1: Safely trigger submit (re-clicks only if image is NOT generating)
    started = trigger_submit_safely(page, initial_canvases, pre_ids, max_attempts=2)

    # Step 2: Context length hit - fallback only in rare failure scenario
    if not started and (len(prompt_text) > 3500 or raw_content):
        print("\n⚠️ Prompt generation did not start. Rare context limit likely hit.", flush=True)
        print("  Applying rare fallback: Trimming context at the end...", flush=True)

        base_content = raw_content if raw_content else prompt_text
        trimmed_content = trim_content_at_end(base_content, max_chars=3800)
        trimmed_prompt = format_prompt(topic_idx, topic_title, trimmed_content, mode=prompt_mode)
        print(f"  Trimmed prompt from {len(prompt_text)} to {len(trimmed_prompt)} chars.", flush=True)

        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        time.sleep(0.2)
        page.keyboard.insert_text(trimmed_prompt)
        time.sleep(0.8)

        # Trigger submit safely with trimmed prompt
        started = trigger_submit_safely(page, initial_canvases, pre_ids, max_attempts=2)

    if started:
        print("Generation triggered! Waiting for images to render...", flush=True)
    else:
        print("⚠️ Warning: Could not confirm generation start. Proceeding to image wait...", flush=True)

def extract_post_ids(page) -> list:
    """Extract all unique post IDs currently found in links on the page."""
    post_ids = []
    try:
        links = page.locator("a[href*='/imagine/post/']").all()
        for lnk in links:
            href = lnk.get_attribute("href") or ""
            m = re.search(r'/imagine/post/([a-zA-Z0-9\-]{36})', href)
            if m:
                pid = m.group(1)
                if pid not in post_ids:
                    post_ids.append(pid)
    except Exception:
        pass
    return post_ids

def get_dynamic_user_id(page) -> str:
    """Extract user UUID dynamically from page content or asset links."""
    try:
        content = page.content()
        matches = re.findall(r'assets\.grok\.com/users/([a-f0-9\-]{36})/', content)
        if matches:
            return matches[0]
    except Exception:
        pass
    return "783b837f-0a80-4964-b078-df408b4d4293"

def wait_and_download_images(ctx, page, output_dir: Path, topic_prefix: str, pre_existing_ids: set, expected_count: int = 2, max_wait: int = 65, initial_wait: int = 38) -> list:
    """
    Wait for Grok image generation to finish,
    identify the newly created post IDs on the Imagine screen,
    and download the full-resolution images.
    Never downloads old/cached images.
    """
    start_time = time.time()
    downloaded_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if initial_wait > 0:
        print(f"Waiting {initial_wait}s for image generation to complete...", flush=True)
        time.sleep(initial_wait)
    
    # Poll for the newly generated post IDs on the Imagine screen
    new_ids = []
    while time.time() - start_time < max_wait:
        current_ids = extract_post_ids(page)
        fresh = [pid for pid in current_ids if pid not in pre_existing_ids]
        
        if len(fresh) >= expected_count:
            new_ids = fresh[:expected_count]
            print(f"Found {len(new_ids)} newly generated image post IDs: {new_ids}", flush=True)
            break
        elif len(fresh) > 0 and (time.time() - start_time) >= (initial_wait + 10):
            new_ids = fresh
            print(f"Found {len(new_ids)} newly generated image post IDs: {new_ids}", flush=True)
            break
        time.sleep(2)

    user_id = get_dynamic_user_id(page)

    # Download each newly generated image
    if new_ids:
        print(f"Downloading {len(new_ids)} newly generated image(s) for {topic_prefix}...", flush=True)
        for idx, pid in enumerate(new_ids, start=1):
            save_path = output_dir / f"{topic_prefix}_img{idx}.jpg"
            img_url = f"https://assets.grok.com/users/{user_id}/generated/{pid}/image.jpg"
            try:
                resp = ctx.request.get(img_url)
                if resp.status == 200 and len(resp.body()) > 10000:
                    save_path.write_bytes(resp.body())
                    print(f"  [SAVED] {save_path.name} ({save_path.stat().st_size:,} bytes)", flush=True)
                    downloaded_files.append(save_path)
                else:
                    print(f"  HTTP status {resp.status} for image {pid}", flush=True)
            except Exception as e:
                print(f"  Download error for {pid}: {e}", flush=True)
    else:
        print(f"⚠️ Warning: No newly generated post IDs detected for {topic_prefix}.", flush=True)

    return downloaded_files

def resolve_paths(dir_arg: str, transcripts_arg: str = "", output_arg: str = ""):
    """Resolve transcripts_dir, output_dir, and target_dir robustly."""
    transcripts_dir = None
    target_dir = None

    if transcripts_arg:
        p = Path(transcripts_arg).resolve()
        if p.exists() and p.is_dir():
            transcripts_dir = p
            if p.parent.name == "raw":
                target_dir = p.parent.parent
            else:
                target_dir = p.parent

    if not transcripts_dir and dir_arg:
        p = Path(dir_arg).resolve()
        if not p.exists():
            p = (PROJECT_ROOT / dir_arg).resolve()
        
        if p.exists():
            if p.name == "transcript-by-topic":
                transcripts_dir = p
                target_dir = p.parent.parent if p.parent.name == "raw" else p.parent
            elif (p / "raw" / "transcript-by-topic").exists():
                transcripts_dir = p / "raw" / "transcript-by-topic"
                target_dir = p
            elif (p / "transcript-by-topic").exists():
                transcripts_dir = p / "transcript-by-topic"
                target_dir = p

    # Default fallback: 13-Tutorial11-f-Divergence-Examples
    if not transcripts_dir:
        default_t26 = PROJECT_ROOT / "Mathematical-Foundation-for-GenerativeAI" / "13-Tutorial11-f-Divergence-Examples"
        if (default_t26 / "raw" / "transcript-by-topic").exists():
            transcripts_dir = default_t26 / "raw" / "transcript-by-topic"
            target_dir = default_t26
        else:
            cwd = Path.cwd().resolve()
            if (cwd / "raw" / "transcript-by-topic").exists():
                transcripts_dir = cwd / "raw" / "transcript-by-topic"
                target_dir = cwd

    if output_arg:
        output_dir = Path(output_arg).resolve()
    elif target_dir:
        output_dir = target_dir / "grok_images"
    elif transcripts_dir:
        output_dir = transcripts_dir.parent / "grok_images"
    else:
        output_dir = SCRIPT_DIR / "grok_images"

    return transcripts_dir, output_dir, target_dir

def is_topic_complete(output_dir: Path, t_num: int, expected_count: int = 2) -> bool:
    """Check if all expected images exist and are valid (>10KB) for a topic."""
    for idx in range(1, expected_count + 1):
        jpg_f = output_dir / f"topic-{t_num:02d}_img{idx}.jpg"
        png_f = output_dir / f"topic-{t_num:02d}_img{idx}.png"
        valid_jpg = jpg_f.exists() and jpg_f.stat().st_size > 10000
        valid_png = png_f.exists() and png_f.stat().st_size > 10000
        if not (valid_jpg or valid_png):
            return False
    return True

def run_topic_batch(context, chunk, topic_titles, prompt_mode, output_dir):
    """Submit prompts for a chunk of topics across tabs and download generated images."""
    while len(context.pages) < len(chunk):
        context.new_page()
    tabs = context.pages[:len(chunk)]

    batch_info = []
    for tab_idx, (t_num, f) in enumerate(chunk):
        tab = tabs[tab_idx]
        t_title = topic_titles[t_num - 1] if 0 <= (t_num - 1) < len(topic_titles) else f"Topic {t_num}"
        print(f"  [Tab {tab_idx+1}] Setting up Topic {t_num:02d} ('{t_title}')...", flush=True)

        with open(f, "r", encoding="utf-8") as tf:
            raw_content = tf.read()
        prompt = format_prompt(t_num, t_title, raw_content, mode=prompt_mode)

        tab.bring_to_front()
        tab.goto(GROK_IMAGINE_URL, timeout=30000)
        try:
            tab.wait_for_selector(".tiptap, .ProseMirror", timeout=20000)
        except Exception:
            time.sleep(2)

        configure_imagine_settings(tab, quality="2.0", aspect_ratio="2:3", num_images=2)
        pre_ids = set(extract_post_ids(tab))
        enter_prompt_and_generate(
            tab,
            prompt,
            raw_content=raw_content,
            topic_idx=t_num,
            topic_title=t_title,
            prompt_mode=prompt_mode,
            pre_ids=pre_ids
        )
        batch_info.append((tab, t_num, pre_ids))
        time.sleep(1.5)

    print(f"\nAll {len(chunk)} prompts in batch submitted! Waiting 40s for parallel generations...", flush=True)
    time.sleep(38)

    results = []
    for tab_idx, (tab, t_num, pre_ids) in enumerate(batch_info):
        tab.bring_to_front()
        prefix = f"topic-{t_num:02d}"
        print(f"  [Tab {tab_idx+1}] Downloading images for {prefix}...", flush=True)
        downloaded = wait_and_download_images(context, tab, output_dir, prefix, pre_ids, expected_count=2, max_wait=30, initial_wait=2)
        print(f"  [Tab {tab_idx+1}] Topic {t_num:02d} complete: {len(downloaded)} images.", flush=True)
        results.append((t_num, downloaded))

    return results

def process_topics(
    profile_dir: Path,
    transcripts_dir: Path,
    output_dir: Path,
    target_dir: Path,
    topic_filter: list = None,
    prompt_mode: str = "cleaned",
    delay: int = 6,
    parallel: int = 1,
    skip_existing: bool = False,
    max_retries: int = 2
):
    """Iterate through topic files, generate images in Grok, and download them with automatic retries."""
    if not transcripts_dir or not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}", flush=True)
        return

    topic_titles = get_topic_titles(target_dir) if target_dir else []
    all_files = sorted(list(transcripts_dir.glob("topic-*.txt")))
    if not all_files:
        all_files = sorted(list(transcripts_dir.glob("topic-*.md")))

    # Filter files
    topic_files = []
    for f in all_files:
        m = re.search(r'topic-(\d+)', f.stem)
        if not m:
            continue
        t_num = int(m.group(1))
        if topic_filter and t_num not in topic_filter:
            continue
        topic_files.append((t_num, f))

    if not topic_files:
        print(f"Error: No matching topic files found in {transcripts_dir}", flush=True)
        return

    # Check whether to skip fully generated topics or regenerate all
    if skip_existing:
        valid_topic_files = []
        for t_num, f in topic_files:
            if is_topic_complete(output_dir, t_num, expected_count=2):
                print(f"  [SKIP] Topic {t_num:02d} already fully generated in {output_dir.name}", flush=True)
            else:
                valid_topic_files.append((t_num, f))
        topic_files = valid_topic_files
    else:
        print(f"  [REGENERATE] Processing all {len(topic_files)} topic(s) in {output_dir.name} (full regeneration)", flush=True)

    if not topic_files:
        print(f"All topics already completed for {output_dir.parent.name}!", flush=True)
        return

    parallel = max(1, min(parallel, 3))
    print("=" * 70, flush=True)
    print(f"PROCESSING {len(topic_files)} TOPIC(S) IN GROK IMAGINE (Parallel Concurrency: {parallel})", flush=True)
    print(f"Transcripts: {transcripts_dir}", flush=True)
    print(f"Profile:     {profile_dir}", flush=True)
    print(f"Output:      {output_dir}", flush=True)
    print("=" * 70, flush=True)

    with sync_playwright() as p:
        context = launch_browser(p, profile_dir, headless=False)
        try:
            main_page = context.pages[0] if context.pages else context.new_page()
            print(f"Navigating to Grok Imagine: {GROK_IMAGINE_URL}...", flush=True)
            main_page.goto(GROK_IMAGINE_URL, timeout=60000)
            time.sleep(3)

            if "login" in main_page.url.lower() or "signin" in main_page.url.lower():
                print("⚠️ Session expired or not logged in. Please run `python grok_imagine_runner.py --login` first!", flush=True)
                return

            # Process in batches of size `parallel`
            for i in range(0, len(topic_files), parallel):
                chunk = topic_files[i : i + parallel]
                chunk_nums = [t_num for t_num, _ in chunk]
                print(f"\n>>> Starting Parallel Batch: Topics {chunk_nums} ({len(chunk)} tabs)", flush=True)

                run_topic_batch(context, chunk, topic_titles, prompt_mode, output_dir)

                # Retry any topic in this chunk that is not fully generated (missing 2 valid images)
                retry_queue = [item for item in chunk if not is_topic_complete(output_dir, item[0], expected_count=2)]
                retry_count = 1
                while retry_queue and retry_count <= max_retries:
                    missing_nums = [t[0] for t in retry_queue]
                    print(f"\n⚠️ Topic(s) {missing_nums} not fully generated (missing 2 valid images). Retrying attempt {retry_count}/{max_retries}...", flush=True)
                    time.sleep(3)
                    run_topic_batch(context, retry_queue, topic_titles, prompt_mode, output_dir)
                    retry_queue = [item for item in chunk if not is_topic_complete(output_dir, item[0], expected_count=2)]
                    retry_count += 1

                # Cooldown before next batch
                if i + parallel < len(topic_files) and delay > 0:
                    print(f"\nWaiting {delay}s cooldown before next parallel batch...", flush=True)
                    time.sleep(delay)

            # Final verification pass across all topics in this run
            incomplete_topics = [item for item in topic_files if not is_topic_complete(output_dir, item[0], expected_count=2)]
            sweep_count = 1
            while incomplete_topics and sweep_count <= max_retries:
                inc_nums = [t[0] for t in incomplete_topics]
                print(f"\n⚠️ Final Verification: Topic(s) {inc_nums} still not fully generated. Running sweep retry {sweep_count}/{max_retries}...", flush=True)
                for j in range(0, len(incomplete_topics), parallel):
                    sweep_chunk = incomplete_topics[j : j + parallel]
                    run_topic_batch(context, sweep_chunk, topic_titles, prompt_mode, output_dir)
                    time.sleep(3)
                incomplete_topics = [item for item in topic_files if not is_topic_complete(output_dir, item[0], expected_count=2)]
                sweep_count += 1

            # Close extra tabs to save resources
            while len(context.pages) > 1:
                try:
                    context.pages[-1].close()
                except Exception:
                    break

            all_complete = all(is_topic_complete(output_dir, t[0], 2) for t in topic_files)
            print("\n" + "=" * 70, flush=True)
            if all_complete:
                print("ALL REQUESTED TOPICS COMPLETED AND FULLY GENERATED (2 IMAGES EACH)!", flush=True)
            else:
                missing = [t[0] for t in topic_files if not is_topic_complete(output_dir, t[0], 2)]
                print(f"Completed with some topics still missing images: {missing}", flush=True)
            print(f"Images are saved in: {output_dir}", flush=True)
            print("=" * 70, flush=True)
        finally:
            context.close()

def run_range(
    profile_path: Path,
    start_num: int,
    end_num: int,
    skip_existing: bool = False,
    parallel: int = 3,
    prompt_mode: str = "cleaned",
    delay: int = 6,
    max_retries: int = 2
):
    """Process all tutorial/lecture folders within start_num and end_num range."""
    lectures_root = PROJECT_ROOT / "Mathematical-Foundation-for-GenerativeAI"
    if not lectures_root.exists():
        print(f"Error: Lectures root not found: {lectures_root}", flush=True)
        return

    matching_dirs = []
    for d in sorted(lectures_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r'^(\d+)-', d.name)
        if m:
            num = int(m.group(1))
            if start_num <= num <= end_num:
                matching_dirs.append((num, d))

    print("=" * 70, flush=True)
    print(f"BATCH RANGE RUN: Folders {start_num} to {end_num} ({len(matching_dirs)} folders found)", flush=True)
    print("=" * 70, flush=True)

    for num, folder in matching_dirs:
        trans_dir = folder / "raw" / "transcript-by-topic"
        if not trans_dir.exists():
            trans_dir = folder / "transcript-by-topic"
        
        if not trans_dir.exists():
            print(f"\n[SKIP] {folder.name}: No transcript-by-topic directory found.", flush=True)
            continue

        topic_files = list(trans_dir.glob("topic-*.txt"))
        if not topic_files:
            topic_files = list(trans_dir.glob("topic-*.md"))
        expected_imgs = len(topic_files) * 2

        out_dir = folder / "grok_images"
        if skip_existing and out_dir.exists():
            existing_imgs = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
            if len(existing_imgs) >= expected_imgs and expected_imgs > 0:
                # Verify each individual topic has 2 valid images
                all_done = True
                for tf in topic_files:
                    m = re.search(r'topic-(\d+)', tf.stem)
                    if m and not is_topic_complete(out_dir, int(m.group(1)), 2):
                        all_done = False
                        break
                if all_done:
                    print(f"\n[SKIP] {folder.name}: Already complete ({len(existing_imgs)}/{expected_imgs} images).", flush=True)
                    continue
                else:
                    print(f"\n[PARTIAL] {folder.name}: Found topics not fully generated. Running completion...", flush=True)

        print(f"\n" + "#" * 70)
        print(f"PROCESSING FOLDER {num:02d}: {folder.name}")
        print(f"Topics: {len(topic_files)} (Expected images: {expected_imgs})")
        print("#" * 70)

        process_topics(
            profile_dir=profile_path,
            transcripts_dir=trans_dir,
            output_dir=out_dir,
            target_dir=folder,
            topic_filter=None,
            prompt_mode=prompt_mode,
            delay=delay,
            parallel=parallel,
            skip_existing=skip_existing,
            max_retries=max_retries
        )

def main():
    parser = argparse.ArgumentParser(description="Grok Imagine Batch Automation")
    parser.add_argument("--login", action="store_true", help="Launch browser to log in interactively and save session")
    parser.add_argument("--run", action="store_true", help="Run batch image generation for topics")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="Run across folder number range, e.g. --range 14 33")
    parser.add_argument("--regenerate", "--force", dest="regenerate", action="store_true", help="Force regenerate all topics even if images already exist")
    parser.add_argument("--skip-existing", action="store_true", help="Skip topics that already have 2 complete images")
    parser.add_argument("--max-retries", type=int, default=2, help="Number of retries for topics not fully generated (default: 2)")
    parser.add_argument("--transcripts-dir", type=str, default="", help="Direct path to transcript-by-topic directory")
    parser.add_argument("--dir", type=str, default="", help="Target lecture/tutorial directory (default: 13-Tutorial11-f-Divergence-Examples)")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory to save images")
    parser.add_argument("--topic", type=int, nargs="+", help="Specific topic number(s) to run (e.g. --topic 1 2)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel generation calls (max: 3, e.g. --parallel 3)")
    parser.add_argument("--raw", action="store_true", help="Paste raw transcript text without cleaning timestamps")
    parser.add_argument("--delay", type=int, default=6, help="Delay (in seconds) between generations/batches")
    parser.add_argument("--profile-dir", type=str, default=str(PROFILE_DIR), help="Path to browser profile directory")

    args = parser.parse_args()
    profile_path = Path(args.profile_dir)
    prompt_mode = "raw" if args.raw else "cleaned"

    if args.login:
        interactive_login(profile_path)
    elif args.range:
        skip_existing = args.skip_existing
        start_num, end_num = args.range
        parallel_val = args.parallel if args.parallel > 1 else 3
        run_range(
            profile_path=profile_path,
            start_num=start_num,
            end_num=end_num,
            skip_existing=skip_existing,
            parallel=parallel_val,
            prompt_mode=prompt_mode,
            delay=args.delay,
            max_retries=args.max_retries
        )
    elif args.run:
        transcripts_dir, output_dir, target_dir = resolve_paths(args.dir, args.transcripts_dir, args.output_dir)
        # When targeting a specific folder, default to regenerate unless --skip-existing is explicitly passed
        skip_existing = args.skip_existing and not args.regenerate
        parallel_val = args.parallel if args.parallel > 1 else 3
        process_topics(
            profile_path,
            transcripts_dir,
            output_dir,
            target_dir,
            topic_filter=args.topic,
            prompt_mode=prompt_mode,
            delay=args.delay,
            parallel=parallel_val,
            skip_existing=skip_existing,
            max_retries=args.max_retries
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
