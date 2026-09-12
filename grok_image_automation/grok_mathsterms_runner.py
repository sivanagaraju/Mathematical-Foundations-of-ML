r"""
Grok Imagine Batch Automation for MathsTerms
Parses monolithic topic markdown files into individual section topics,
generates educational technical infographics in Grok Imagine,
and saves them organized in segregated folders with topic-matching filenames.

Folder structure:
  MathsTerms/<Category>/grok_images/<MarkdownFileStem>/
    ├── topic-01_<TopicSlug>_img1.jpg
    ├── topic-01_<TopicSlug>_img2.jpg
    ├── topic-02_<TopicSlug>_img1.jpg
    └── ...

RUN COMMANDS (Copy & Paste):
  # 1. Run all files in Category 01 with ChatGPT:
  python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --category "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\MathsTerms\01-Primal-Analysis-and-Foundations" --skip-existing

  # 2. Run all topics for a specific file with ChatGPT:
  python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --skip-existing

  # 3. Run all files in an entire category with Grok (parallel 2 tabs):
  python grok_image_automation/grok_mathsterms_runner.py --engine grok --category "01-Primal-Analysis-and-Foundations" --parallel 2 --skip-existing

  # 4. Run specific topic(s) of a file (e.g. Topics 1, 2, 4):
  python grok_image_automation/grok_mathsterms_runner.py --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --topic 1 2 4

  # 5. Run across all 46 files in MathsTerms (skipping completed topics):
  python grok_image_automation/grok_mathsterms_runner.py --all --parallel 3 --skip-existing
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

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MATHS_TERMS_DIR = PROJECT_ROOT / "MathsTerms"
PROFILE_DIR = SCRIPT_DIR / ".grok_profile"

GROK_URL = "https://grok.com"
GROK_IMAGINE_URL = "https://grok.com/imagine"

# Import ChatGPT Engine
try:
    from chatgpt_engine import (
        CHATGPT_PROFILE_DIR,
        CHATGPT_IMAGES_URL,
        DEFAULT_EMAIL,
        DEFAULT_PASSWORD,
        launch_chatgpt_browser,
        chatgpt_login,
        is_chatgpt_logged_in,
        ensure_chatgpt_images_page,
        submit_chatgpt_prompt,
        wait_and_download_chatgpt_images,
        extract_image_elements
    )
except ImportError:
    pass

def clean_title(raw_title: str) -> str:
    """Remove emoji and special characters from heading title."""
    cleaned = re.sub(r'[^\w\s\-\(\)\.,&/]', '', raw_title)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def slugify_title(title: str, max_len: int = 35) -> str:
    """Generate filesystem-safe short slug from section title."""
    cleaned = clean_title(title)
    slug = re.sub(r'[\s\-\(\)\.,&/]+', '_', cleaned).strip('_')
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip('_')
    return slug if slug else "topic"

def parse_markdown_topics(file_path: Path) -> tuple:
    """
    Parse a MathsTerms markdown file into its document title and numbered section topics.
    Returns: (doc_title, list_of_topics)
    """
    text = file_path.read_text(encoding="utf-8")
    
    # Extract Document Title from top # heading
    doc_title_match = re.search(r'(?m)^#\s+(.+)$', text)
    doc_title = doc_title_match.group(1).strip() if doc_title_match else file_path.stem
    doc_title = clean_title(doc_title)
    
    # Match numbered sections: ### 1. ... or ## 1. ...
    pattern = r'(?m)^###\s+(\d+)\.\s*(.+)$'
    matches = list(re.finditer(pattern, text))
    if not matches:
        pattern = r'(?m)^##\s+(\d+)\.\s*(.+)$'
        matches = list(re.finditer(pattern, text))
        
    topics = []
    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        raw_title = m.group(2).strip()
        title = clean_title(raw_title)
        slug = slugify_title(title)
        
        start_pos = m.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_pos:end_pos].strip()
        # Clean trailing horizontal rule --- if present
        content = re.sub(r'\n---\s*$', '', content).strip()
        
        topics.append({
            "num": sec_num,
            "raw_title": raw_title,
            "title": title,
            "slug": slug,
            "content": content,
            "char_count": len(content)
        })
        
    return doc_title, topics

def format_mathsterms_prompt(topic_num: int, topic_title: str, doc_title: str, content: str) -> str:
    """Format prompt combining educational infographic directive, topic context, and section content."""
    prompt = (
        f"Comprehensive educational technical infographic clearly explaining the core mechanics of '{topic_title}' in '{doc_title}'. "
        f"Critical is for given context need to work and explain that. "
        f"Step-by-step visual explanation with annotated mathematical formulas, labeled architecture block diagrams, directional data flow arrows, and structured explanation. "
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

def configure_imagine_settings(page, quality: str = "2.0", aspect_ratio: str = "2:3", num_images: int = 2):
    """Ensure Grok Imagine settings (Quality 2.0, 2:3 ratio, x2 count) are selected."""
    page.bring_to_front()
    try:
        page.wait_for_selector(".tiptap, .ProseMirror", timeout=15000)
        page.wait_for_selector("button[aria-label='Aspect Ratio']", timeout=10000)
    except Exception:
        pass

    # 1. Quality 2.0
    try:
        quality_btn = page.locator("button, [role='button']").filter(has_text="Quality 2.0")
        if quality_btn.count() > 0:
            quality_btn.first.click()
            time.sleep(0.5)
    except Exception:
        pass

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
                time.sleep(0.5)
    except Exception:
        pass

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
                time.sleep(0.5)
    except Exception:
        pass

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

def is_generation_underway(page, initial_canvases: int, pre_ids: set) -> bool:
    """
    Check if Grok Imagine has actively started generating images.
    CRITICAL: Only true generation indicators (canvases, fresh post IDs, pulsing cards)
    are checked. A disabled submit button alone does NOT mean generation is underway,
    because Grok disables the submit button when prompt length exceeds limits!
    """
    try:
        # Check 1: Canvas placeholders for the dot matrix animation (most reliable indicator)
        current_canvases = page.locator("canvas").count()
        if current_canvases > initial_canvases:
            return True

        # Check 2: New post IDs appearing in DOM links
        current_ids = set(extract_post_ids(page))
        if len(current_ids - pre_ids) > 0:
            return True

        # Check 3: Shimmering or active pulsing placeholder elements
        if page.locator("a[href*='/imagine/post/'] [class*='animate-pulse'], [aria-busy='true']").count() > 0:
            return True
    except Exception:
        pass
    return False

def trim_content_at_end(content: str, max_chars: int = 3200) -> str:
    """Trim section content at the end for rare scenarios where context limit is hit."""
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
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
            print(f"    [OK] Generation already underway. Button will NOT be re-pressed.", flush=True)
            return True

        submit_btn = page.locator("button[aria-label='Submit']")
        if submit_btn.count() > 0 and submit_btn.first.is_visible():
            if not submit_btn.first.is_disabled():
                print(f"    Submit attempt {attempt}/{max_attempts}: clicking submit button...", flush=True)
                submit_btn.first.click()
            else:
                print(f"    Submit attempt {attempt}/{max_attempts}: button disabled, checking generation status...", flush=True)
        else:
            alt_btn = page.locator("button.rounded-full, button.bg-primary").filter(has=page.locator("svg"))
            if alt_btn.count() > 0 and alt_btn.last.is_visible():
                print(f"    Submit attempt {attempt}/{max_attempts}: clicking alternate submit button...", flush=True)
                alt_btn.last.click()
            else:
                print(f"    Submit attempt {attempt}/{max_attempts}: pressing Enter...", flush=True)
                page.keyboard.press("Enter")

        # Poll for 3-4 seconds to confirm if generation started
        for _ in range(7):
            time.sleep(0.5)
            if is_generation_underway(page, initial_canvases, pre_ids):
                print("    [OK] Generation started successfully!", flush=True)
                return True

        print(f"    Notice: Generation not detected after attempt {attempt}.", flush=True)

    return is_generation_underway(page, initial_canvases, pre_ids)

def enter_prompt_and_generate(
    page,
    prompt_text: str,
    raw_content: str = "",
    topic_num: int = 1,
    topic_title: str = "",
    doc_title: str = "",
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

    print(f"    Entering prompt ({len(prompt_text)} chars)...", flush=True)
    editor = page.locator(".tiptap, .ProseMirror, [contenteditable='true']").first
    try:
        editor.wait_for(state="visible", timeout=15000)
    except Exception:
        page.goto(GROK_IMAGINE_URL, timeout=30000)
        time.sleep(2)
        editor.wait_for(state="visible", timeout=15000)

    editor.click()
    time.sleep(0.3)

    # Clear existing content
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    time.sleep(0.2)

    # Insert full prompt
    page.keyboard.insert_text(prompt_text)
    time.sleep(0.8)

    # Step 1: Safely trigger submit (re-clicks only if image is NOT generating)
    started = trigger_submit_safely(page, initial_canvases, pre_ids, max_attempts=2)

    # Step 2: Context length hit - fallback only in rare failure scenario
    if not started and (len(prompt_text) > 3500 or raw_content):
        print("\n    ⚠️ Prompt generation did not start. Context limit likely hit.", flush=True)
        print("    Applying rare fallback: Trimming context at the end...", flush=True)

        base_content = raw_content if raw_content else prompt_text
        trimmed_content = trim_content_at_end(base_content, max_chars=3800)
        trimmed_prompt = format_mathsterms_prompt(topic_num, topic_title, doc_title, trimmed_content)
        print(f"    Trimmed prompt from {len(prompt_text)} to {len(trimmed_prompt)} chars.", flush=True)

        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        time.sleep(0.2)
        page.keyboard.insert_text(trimmed_prompt)
        time.sleep(0.8)

        started = trigger_submit_safely(page, initial_canvases, pre_ids, max_attempts=2)

    if started:
        print("    Generation triggered! Waiting for images to render...", flush=True)
    else:
        print("    ⚠️ Warning: Could not confirm generation start. Proceeding to image wait...", flush=True)

def wait_and_download_images(
    ctx,
    page,
    output_dir: Path,
    file_prefix: str,
    pre_existing_ids: set,
    expected_count: int = 2,
    max_wait: int = 65,
    initial_wait: int = 38
) -> list:
    """Wait for Grok image generation to complete and download fresh images."""
    start_time = time.time()
    downloaded_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if initial_wait > 0:
        print(f"    Waiting {initial_wait}s for image generation to complete...", flush=True)
        time.sleep(initial_wait)
    
    new_ids = []
    while time.time() - start_time < max_wait:
        current_ids = extract_post_ids(page)
        fresh = [pid for pid in current_ids if pid not in pre_existing_ids]
        
        if len(fresh) >= expected_count:
            new_ids = fresh[:expected_count]
            print(f"    Found {len(new_ids)} newly generated post IDs: {new_ids}", flush=True)
            break
        elif len(fresh) > 0 and (time.time() - start_time) >= (initial_wait + 10):
            new_ids = fresh
            print(f"    Found {len(new_ids)} newly generated post IDs: {new_ids}", flush=True)
            break
        time.sleep(2)

    user_id = get_dynamic_user_id(page)

    if new_ids:
        print(f"    Downloading {len(new_ids)} image(s) for {file_prefix}...", flush=True)
        for idx, pid in enumerate(new_ids, start=1):
            save_path = output_dir / f"{file_prefix}_img{idx}.jpg"
            img_url = f"https://assets.grok.com/users/{user_id}/generated/{pid}/image.jpg"
            try:
                resp = ctx.request.get(img_url)
                if resp.status == 200 and len(resp.body()) > 10000:
                    save_path.write_bytes(resp.body())
                    print(f"      [SAVED] {save_path.name} ({save_path.stat().st_size:,} bytes)", flush=True)
                    downloaded_files.append(save_path)
                else:
                    print(f"      HTTP status {resp.status} for image {pid}", flush=True)
            except Exception as e:
                print(f"      Download error for {pid}: {e}", flush=True)
    else:
        print(f"    ⚠️ Warning: No newly generated post IDs detected for {file_prefix}.", flush=True)

    return downloaded_files

def is_topic_complete(output_dir: Path, file_prefix: str, expected_count: int = 2) -> bool:
    """Check if all expected images exist and are valid (>10KB) for a topic prefix."""
    for idx in range(1, expected_count + 1):
        jpg_f = output_dir / f"{file_prefix}_img{idx}.jpg"
        png_f = output_dir / f"{file_prefix}_img{idx}.png"
        valid_jpg = jpg_f.exists() and jpg_f.stat().st_size > 10000
        valid_png = png_f.exists() and png_f.stat().st_size > 10000
        if not (valid_jpg or valid_png):
            return False
    return True

def run_topic_batch(context, chunk, doc_title: str, output_dir: Path, engine: str = "grok"):
    """Submit prompts for a chunk of topics across tabs and download generated images."""
    if engine == "chatgpt":
        while len(context.pages) < len(chunk):
            context.new_page()
        tabs = context.pages[:len(chunk)]

        batch_info = []
        for tab_idx, t in enumerate(chunk):
            tab = tabs[tab_idx]
            t_num = t["num"]
            t_title = t["title"]
            slug = t["slug"]
            file_prefix = f"topic-{t_num:02d}_{slug}"
            print(f"  [Tab {tab_idx+1}] Setting up ChatGPT Images for Topic {t_num:02d} ('{t_title}')...", flush=True)

            tab.bring_to_front()
            ensure_chatgpt_images_page(tab, reset=(tab_idx == 0))

            prompt = format_mathsterms_prompt(t_num, t_title, doc_title, t["content"])
            pre_urls = set(extract_image_elements(tab))

            submit_chatgpt_prompt(tab, prompt)
            batch_info.append((tab, t, file_prefix, pre_urls))
            time.sleep(2)

        if len(chunk) > 1:
            print(f"\nAll {len(chunk)} ChatGPT prompts in batch submitted! Waiting 25s for parallel generations...", flush=True)
            time.sleep(25)

        results = []
        for tab_idx, (tab, t, file_prefix, pre_urls) in enumerate(batch_info):
            tab.bring_to_front()
            print(f"  [Tab {tab_idx+1}] Downloading ChatGPT image for {file_prefix}...", flush=True)
            downloaded = wait_and_download_chatgpt_images(
                context,
                tab,
                output_dir,
                file_prefix,
                pre_urls,
                expected_count=1,
                max_wait=180,
                initial_wait=5 if len(chunk) > 1 else 20
            )
            print(f"  [Tab {tab_idx+1}] Topic {t['num']:02d} complete: {len(downloaded)} image(s).", flush=True)
            results.append((t, downloaded))

        # Close extra tabs beyond tab 0
        while len(context.pages) > 1:
            try:
                context.pages[-1].close()
            except Exception:
                break

        return results

    while len(context.pages) < len(chunk):
        context.new_page()
    tabs = context.pages[:len(chunk)]

    batch_info = []
    for tab_idx, t in enumerate(chunk):
        tab = tabs[tab_idx]
        t_num = t["num"]
        t_title = t["title"]
        slug = t["slug"]
        file_prefix = f"topic-{t_num:02d}_{slug}"
        print(f"  [Tab {tab_idx+1}] Setting up Topic {t_num:02d} ('{t_title}')...", flush=True)

        prompt = format_mathsterms_prompt(t_num, t_title, doc_title, t["content"])

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
            raw_content=t["content"],
            topic_num=t_num,
            topic_title=t_title,
            doc_title=doc_title,
            pre_ids=pre_ids
        )
        batch_info.append((tab, t, file_prefix, pre_ids))
        time.sleep(1.5)

    print(f"\nAll {len(chunk)} prompts in batch submitted! Waiting 40s for parallel generations...", flush=True)
    time.sleep(38)

    results = []
    for tab_idx, (tab, t, file_prefix, pre_ids) in enumerate(batch_info):
        tab.bring_to_front()
        print(f"  [Tab {tab_idx+1}] Downloading images for {file_prefix}...", flush=True)
        downloaded = wait_and_download_images(
            context,
            tab,
            output_dir,
            file_prefix,
            pre_ids,
            expected_count=2,
            max_wait=30,
            initial_wait=2
        )
        print(f"  [Tab {tab_idx+1}] Topic {t['num']:02d} complete: {len(downloaded)} images.", flush=True)
        results.append((t, downloaded))

    return results

def process_single_mathsterms_file(
    context,
    md_file: Path,
    topic_filter: list = None,
    delay: int = 6,
    parallel: int = 2,
    skip_existing: bool = False,
    max_retries: int = 2,
    engine: str = "grok"
):
    """Process a single MathsTerms markdown file, creating its segregated images directory."""
    doc_title, all_topics = parse_markdown_topics(md_file)
    if not all_topics:
        print(f"⚠️ No topics extracted from {md_file.name}. Skipping.", flush=True)
        return

    category_dir = md_file.parent
    file_slug = md_file.stem
    # Segregated output folder: MathsTerms/<Category>/<engine>_images/<MarkdownFileStem>/
    img_folder = "chatgpt_images" if engine == "chatgpt" else "grok_images"
    output_dir = category_dir / img_folder / file_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_imgs = 1 if engine == "chatgpt" else 2

    # Filter topics if requested
    topics_to_process = []
    for t in all_topics:
        if topic_filter and t["num"] not in topic_filter:
            continue
        file_prefix = f"topic-{t['num']:02d}_{t['slug']}"
        if skip_existing and is_topic_complete(output_dir, file_prefix, expected_count=expected_imgs):
            print(f"  [SKIP] Topic {t['num']:02d} ({t['title'][:30]}) already complete in {output_dir.name}", flush=True)
        else:
            topics_to_process.append(t)

    if not topics_to_process:
        print(f"All topics already completed for {md_file.name}!", flush=True)
        return

    print("=" * 75, flush=True)
    print(f"ENGINE:  {engine.upper()}")
    print(f"FILE:    {md_file.name}")
    print(f"TITLE:   {doc_title}")
    print(f"TOPICS:  {len(topics_to_process)} / {len(all_topics)} topic(s) to generate (Expected: {len(topics_to_process) * expected_imgs} images)")
    print(f"OUTPUT:  {output_dir}")
    print("=" * 75, flush=True)

    parallel = max(1, min(parallel, 3))

    for i in range(0, len(topics_to_process), parallel):
        chunk = topics_to_process[i : i + parallel]
        chunk_nums = [t["num"] for t in chunk]
        print(f"\n>>> Starting Batch: Topics {chunk_nums} ({len(chunk)} tab(s))", flush=True)

        run_topic_batch(context, chunk, doc_title, output_dir, engine=engine)

        # Retry loop for any topic in this chunk that is incomplete
        retry_queue = [t for t in chunk if not is_topic_complete(output_dir, f"topic-{t['num']:02d}_{t['slug']}", expected_count=expected_imgs)]
        retry_count = 1
        while retry_queue and retry_count <= max_retries:
            missing_nums = [t["num"] for t in retry_queue]
            print(f"\n⚠️ Topic(s) {missing_nums} not fully generated (missing {expected_imgs} valid images). Retrying attempt {retry_count}/{max_retries}...", flush=True)
            time.sleep(3)
            run_topic_batch(context, retry_queue, doc_title, output_dir, engine=engine)
            retry_queue = [t for t in chunk if not is_topic_complete(output_dir, f"topic-{t['num']:02d}_{t['slug']}", expected_count=expected_imgs)]
            retry_count += 1

        if i + parallel < len(topics_to_process) and delay > 0:
            print(f"\nWaiting {delay}s cooldown before next batch...", flush=True)
            time.sleep(delay)

    # Final sweep verification pass
    incomplete = [t for t in topics_to_process if not is_topic_complete(output_dir, f"topic-{t['num']:02d}_{t['slug']}", expected_count=expected_imgs)]
    sweep_count = 1
    while incomplete and sweep_count <= max_retries:
        inc_nums = [t["num"] for t in incomplete]
        print(f"\n⚠️ Final Sweep: Topic(s) {inc_nums} still missing images. Running retry sweep {sweep_count}/{max_retries}...", flush=True)
        for j in range(0, len(incomplete), parallel):
            sweep_chunk = incomplete[j : j + parallel]
            run_topic_batch(context, sweep_chunk, doc_title, output_dir, engine=engine)
            time.sleep(3)
        incomplete = [t for t in topics_to_process if not is_topic_complete(output_dir, f"topic-{t['num']:02d}_{t['slug']}", expected_count=expected_imgs)]
        sweep_count += 1

    all_done = all(is_topic_complete(output_dir, f"topic-{t['num']:02d}_{t['slug']}", expected_count=expected_imgs) for t in topics_to_process)
    if all_done:
        print(f"\n[DONE] All {len(topics_to_process)} topics completed in: {output_dir}\n", flush=True)
    else:
        missing = [t["num"] for t in topics_to_process if not is_topic_complete(output_dir, f"topic-{t['num']:02d}_{t['slug']}", expected_count=expected_imgs)]
        print(f"\n[PARTIAL] Topics still missing images: {missing}\n", flush=True)

def find_target_files(file_arg: str = "", category_arg: str = "", all_flag: bool = False) -> list:
    """Resolve target markdown files based on arguments."""
    target_files = []
    
    if file_arg:
        p = Path(file_arg).resolve()
        if not p.exists():
            p = (PROJECT_ROOT / file_arg).resolve()
        if not p.exists():
            p = (MATHS_TERMS_DIR / file_arg).resolve()
        if p.exists() and p.is_file():
            target_files.append(p)
        else:
            print(f"Error: File not found: {file_arg}", flush=True)
            return []

    elif category_arg:
        cat_p = Path(category_arg).resolve()
        if not (cat_p.exists() and cat_p.is_dir()):
            cat_p = (MATHS_TERMS_DIR / category_arg).resolve()
        if not (cat_p.exists() and cat_p.is_dir()):
            # Try partial match on category folder name
            for d in MATHS_TERMS_DIR.iterdir():
                if d.is_dir() and category_arg.lower() in d.name.lower():
                    cat_p = d
                    break
        if cat_p.exists() and cat_p.is_dir():
            md_files = sorted(list(cat_p.glob("*.md")))
            target_files = [f for f in md_files if f.name not in ["README.md", "CONCEPT_MAP.md"]]
        else:
            print(f"Error: Category folder not found: {category_arg}", flush=True)
            return []

    elif all_flag:
        all_md = sorted(list(MATHS_TERMS_DIR.glob("**/*.md")))
        target_files = [f for f in all_md if f.name not in ["README.md", "CONCEPT_MAP.md"]]

    else:
        # Default fallback: 01-Probability_Basics_and_Axioms.md
        default_file = MATHS_TERMS_DIR / "01-Primal-Analysis-and-Foundations" / "01-Probability_Basics_and_Axioms.md"
        if default_file.exists():
            target_files.append(default_file)

    return target_files

def main():
    parser = argparse.ArgumentParser(description="Image Automation for MathsTerms (Grok Imagine & ChatGPT Images 2.5)")
    parser.add_argument("--engine", choices=["grok", "chatgpt"], default="grok", help="Image generation engine (grok or chatgpt, default: grok)")
    parser.add_argument("--login", action="store_true", help="Launch browser to log in and save session into profile")
    parser.add_argument("--wait", type=int, default=60, help="Seconds to wait for manual login (default: 60)")
    parser.add_argument("--email", type=str, default=DEFAULT_EMAIL, help="Email for ChatGPT login (default: sivanagarajupachipulusu@gmail.com)")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="Password for ChatGPT login")
    parser.add_argument("--file", type=str, default="", help="Path to specific MathsTerms markdown file")
    parser.add_argument("--category", type=str, default="", help="Category folder name or number (e.g. 01-Primal-Analysis-and-Foundations)")
    parser.add_argument("--all", action="store_true", help="Process all markdown files across all categories in MathsTerms")
    parser.add_argument("--topic", type=int, nargs="+", help="Specific topic number(s) to run (e.g. --topic 1 2 4)")
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel generation tabs (1-3, default: 2)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip topics that already have complete images")
    parser.add_argument("--regenerate", "--force", dest="regenerate", action="store_true", help="Force regenerate all topics even if images exist")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries for incomplete topics (default: 2)")
    parser.add_argument("--delay", type=int, default=6, help="Cooldown delay (seconds) between batches (default: 6)")
    parser.add_argument("--profile-dir", type=str, default="", help="Custom Chrome profile directory")

    args = parser.parse_args()

    # Determine profile directory based on engine
    if args.profile_dir:
        profile_path = Path(args.profile_dir)
    elif args.engine == "chatgpt":
        profile_path = CHATGPT_PROFILE_DIR
    else:
        profile_path = PROFILE_DIR

    # Handle interactive / automated login
    if args.login:
        with sync_playwright() as p:
            if args.engine == "chatgpt":
                context = launch_chatgpt_browser(p, profile_path, headless=False)
                try:
                    chatgpt_login(context, login_timeout_seconds=args.wait)
                finally:
                    context.close()
            else:
                from grok_imagine_runner import interactive_login
                interactive_login(profile_path)
        return

    target_files = find_target_files(args.file, args.category, args.all)
    if not target_files:
        print("No target files found to process. Use --file, --category, or --all.", flush=True)
        return

    skip_existing = args.skip_existing and not args.regenerate
    parallel_val = max(1, min(args.parallel, 3))

    print("=" * 75, flush=True)
    print(f"MATHSTERMS IMAGE AUTOMATION RUNNER")
    print(f"Engine:           {args.engine.upper()}")
    print(f"Target files:     {len(target_files)}")
    print(f"Parallel tabs:    {parallel_val}")
    print(f"Skip existing:    {skip_existing}")
    print(f"Max retries:      {args.max_retries}")
    print(f"Profile:          {profile_path}")
    print("=" * 75, flush=True)

    with sync_playwright() as p:
        context = launch_chatgpt_browser(p, profile_path, headless=False) if args.engine == "chatgpt" else launch_browser(p, profile_path, headless=False)
        try:
            main_page = context.pages[0] if context.pages else context.new_page()
            if args.engine == "chatgpt":
                print(f"Navigating to ChatGPT Images: {CHATGPT_IMAGES_URL}...", flush=True)
                main_page.goto(CHATGPT_IMAGES_URL, timeout=60000)
                time.sleep(3)
                if not is_chatgpt_logged_in(main_page):
                    print("Session not active. Running ChatGPT login...", flush=True)
                    chatgpt_login(context, login_timeout_seconds=args.wait)
            else:
                print(f"Navigating to Grok Imagine: {GROK_IMAGINE_URL}...", flush=True)
                main_page.goto(GROK_IMAGINE_URL, timeout=60000)
                time.sleep(3)

                if "login" in main_page.url.lower() or "signin" in main_page.url.lower():
                    print("⚠️ Session expired or not logged in. Please log in via python grok_mathsterms_runner.py --engine grok --login first!", flush=True)
                    return

            for idx, md_file in enumerate(target_files, start=1):
                print(f"\n[{idx}/{len(target_files)}] Processing file: {md_file.name}...")
                process_single_mathsterms_file(
                    context=context,
                    md_file=md_file,
                    topic_filter=args.topic,
                    delay=args.delay,
                    parallel=parallel_val,
                    skip_existing=skip_existing,
                    max_retries=args.max_retries,
                    engine=args.engine
                )

            while len(context.pages) > 1:
                try:
                    context.pages[-1].close()
                except Exception:
                    break

            print("\n" + "=" * 75, flush=True)
            print("ALL MATHSTERMS FILES PROCESSED SUCCESSFULLY!", flush=True)
            print("=" * 75, flush=True)
        finally:
            context.close()

if __name__ == "__main__":
    main()
