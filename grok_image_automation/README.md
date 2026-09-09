# Grok Imagine Automation (`grok_image_automation`)

Automate AI image generation in Grok Imagine using Playwright with persistent session storage.

---

## Directory Structure

```
Mathematical-Foundations-of-ML/
│
└── grok_image_automation/
    ├── grok_imagine_runner.py    # Main automation script
    ├── README.md                 # Documentation & usage guide
    └── .grok_profile/            # Persistent Chrome profile (stores your login session)
```

---

## How It Works

1. **Persistent Session (`.grok_profile`)**:
   - Uses real Google Chrome with a persistent user data directory (`.grok_profile/`).
   - All cookies, auth tokens, and session context are saved directly to disk.
   - You only need to log into Grok once. Future runs reuse this session automatically.

2. **Automated Settings**:
   - **Aspect Ratio**: Automatically selects **`2:3`** (portrait).
   - **Quality**: Selects **`Quality 2.0`**.
   - **Image Count**: Selects **`2 images`** per generation.

3. **Batch Processing**:
   - Reads every topic file (`topic-01.txt` through `topic-10.txt`) from the transcript folder.
   - Cleans timestamp markers `[00:01]` and crafts an educational infographic diagram prompt.
   - Submits prompt to Grok Imagine by clicking the blue circle Submit button.
   - Waits 40 seconds for generation and tracks the exact new post IDs generated on the Imagine screen (ensuring old/cached images are never downloaded).
   - Automatically downloads the full-resolution (1152x1728) images to `grok_images/` as `topic-01_img1.jpg`, `topic-01_img2.jpg`, etc.

---

## Quick Start Guide

Open a PowerShell terminal in `grok_image_automation`:

```powershell
cd c:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\grok_image_automation
```

### 1. First Time: Log in and Save Session (Only needed once)
```powershell
python grok_imagine_runner.py --login
```
- A visible Chrome window opens to `https://grok.com`.
- Log in with your Google, X, or email credentials.
- Once you see your dashboard, return to the terminal and press **`[Enter]`**.
- Your login is now permanently saved in `.grok_profile/`.

---

### 2. Run for Tutorial 11 Transcripts

#### Option A: Direct Transcripts Path
```powershell
python grok_imagine_runner.py --run --transcripts-dir "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\Mathematical-Foundation-for-GenerativeAI\26-Tutorial11-f-Divergence-Examples\raw\transcript-by-topic"
```

#### Option B: Test with Topic 1 Only (Recommended first check)
```powershell
python grok_imagine_runner.py --run --transcripts-dir "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\Mathematical-Foundation-for-GenerativeAI\26-Tutorial11-f-Divergence-Examples\raw\transcript-by-topic" --topic 1
#### Run all topics in a tutorial with 3 parallel tabs:
```powershell
python grok_imagine_runner.py --run --parallel 3
```

#### Run a specific tutorial directory:
```powershell
python grok_imagine_runner.py --run --dir "../Mathematical-Foundation-for-GenerativeAI/28-Lec05-Generative-Adversarial-Networks" --parallel 3
```

#### Run a range of folders automatically (e.g. 14 to 33):
```powershell
python grok_imagine_runner.py --range 14 33 --parallel 3
```
*Note: Automatically skips folders that already have complete images.*

---

## CLI Options & Flags

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `--run` | *(flag)* | Start batch generation process for a single tutorial |
| `--range` | `START END` | Batch process all folders in number range (e.g. `--range 14 33`) |
| `--skip-existing` | *(flag)* | Skip folders that already have all images complete (default: True) |
| `--force` | *(flag)* | Force regenerate even if images already exist |
| `--parallel` | `1-3` | Number of parallel generations in concurrent tabs (max: `3`, e.g. `--parallel 3`) |
| `--login` | *(flag)* | Launch Chrome interactively to log in / refresh session |
| `--transcripts-dir` | `<path>` | Direct path to `raw/transcript-by-topic` folder |
| `--dir` | `<path>` | Path to a lecture/tutorial root folder |
| `--topic` | `1 2 3...` | Run specific topic numbers (e.g. `--topic 1` or `--topic 6 7 8`) |
| `--raw` | *(flag)* | Send verbatim raw transcript text without cleaning timestamps |
| `--delay` | `N` | Cooldown pause in seconds between batches (default: `6` seconds) |
| `--output-dir` | `<path>` | Custom directory to save downloaded images |
| `--profile-dir` | `<path>` | Custom path to Chrome profile (default: `.grok_profile`) |

---

## Reusing for Other Tutorials / Lectures

You can point the runner to any other tutorial in the repository without re-authenticating:

```powershell
# Example: Run for Lecture 03
python grok_imagine_runner.py --run --dir "../Mathematical-Foundation-for-GenerativeAI/25-Lec03-f-Divergence-Examples"
```

Images will automatically download into that lecture's `grok_images/` folder!
