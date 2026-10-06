# Multi-Engine Image Automation (`grok_image_automation`)

Automate educational technical infographic generation using **Grok Imagine** and **ChatGPT Images 2.5** (`https://chatgpt.com/images`) with persistent session storage.

---

## Directory & Profile Architecture

```
Mathematical-Foundations-of-ML/
│
└── grok_image_automation/
    ├── chatgpt_engine.py         # ChatGPT Images 2.5 automation engine
    ├── grok_imagine_runner.py    # Lecture notes runner (supports --engine grok | chatgpt)
    ├── grok_mathsterms_runner.py # MathsTerms markdown runner (supports --engine grok | chatgpt)
    ├── README.md                 # Complete documentation & copy-paste commands
    ├── .grok_profile/            # Persistent Chrome profile for Grok
    └── .chatgpt_profile/         # Persistent Chrome profile for ChatGPT
```

---

## Supported Engines


| Engine                                      | URL                          | Storage Profile     | Default Output Directory |
| :-------------------------------------------- | :----------------------------- | :-------------------- | :------------------------- |
| **Grok Imagine** (`--engine grok`)          | `https://grok.com/imagine`   | `.grok_profile/`    | `.../grok_images/`       |
| **ChatGPT Images 2.5** (`--engine chatgpt`) | `https://chatgpt.com/images` | `.chatgpt_profile/` | `.../chatgpt_images/`    |

---

## 1. Authentication & Multi-Account Persistent Sessions (Run Once)

You can maintain multiple independent ChatGPT accounts and sessions side-by-side using `--account 1`, `--account 2`, etc.

### ChatGPT Account 1 Login (Default):

```powershell
python grok_image_automation/chatgpt_engine.py --login --account 1
```

*(Or: `python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --login --account 1`)*

- Chrome opens with `.chatgpt_profile/`.
- Log in manually (enter email, password, passcode/2FA, verify).
- Saved permanently into `.chatgpt_profile/`.

### ChatGPT Account 2 Login (New / Second Gmail Account):

```powershell
python grok_image_automation/chatgpt_engine.py --login --account 2
```

*(Or: `python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --login --account 2`)*

- Chrome opens with an isolated, clean directory: `.chatgpt_profile_2/`.
- Log in with your **second Gmail account**.
- Saved permanently into `.chatgpt_profile_2/`. Both accounts exist side-by-side without interference!
- Optional: Use `--wait 90` if you need more time.

### Grok Login:

```powershell
python grok_image_automation/grok_imagine_runner.py --engine grok --login
```

- Opens Chrome to `https://grok.com`. Log in and press `[Enter]` in the terminal.

---

## 2. MathsTerms Runner (`grok_mathsterms_runner.py`)

Dynamically parses topic sections (`### 1.`, `### 2.`, etc.) from monolithic markdown files in `MathsTerms/` and downloads generated images into segregated topic folders.

### Using ChatGPT Images 2.5 (`--engine chatgpt`):

```powershell
# 1. Run across ALL 6 categories/folders in MathsTerms with ChatGPT:
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --all --parallel 2 --skip-existing

# 2. Run all files in Category 01 in parallel (2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --category "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\MathsTerms\05-Convexity-Duality-and-Metric-Analysis" --parallel 2 --skip-existing

# 3. Or using category folder name shorthand (3 parallel tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --category "05-Convexity-Duality-and-Metric-Analysis" --parallel 3 --skip-existing

# 4. Generate images for Topic 1 of a specific file:
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md" --topic 1

# 5. Generate multiple topics (e.g. Topics 1, 2, 4) in parallel:
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md" --topic 1 2 4 --parallel 3

# 6. Process all topics in a single file (parallel 2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md" --parallel 2 --skip-existing

# 7. Run using your SECOND ChatGPT account (--account 2):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --account 2 --all --parallel 2 --skip-existing
```

> [!NOTE]
> **Automatic "Too many requests" / "Got it" Modal Handling**:
> If ChatGPT shows the "Too many requests" popup dialog at any point (during tab navigation, prompt submission, or generation polling), the engine automatically detects it, clicks the **"Got it"** button, runs a safe cooldown countdown, and retries prompt generation smoothly without crashing or stalling.

### Using Grok Imagine (`--engine grok`):

```powershell
# 1. Run Category 01 with Grok (parallel 2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine grok --category "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\MathsTerms\05-Convexity-Duality-and-Metric-Analysis" --parallel 2 --skip-existing

# 2. Run all topics for a file in parallel (2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine grok --file "MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md" --parallel 2

# 3. Force regenerate a specific topic:
python grok_image_automation/grok_mathsterms_runner.py --engine grok --file "MathsTerms/03-Probability-and-Statistical-Estimation/00-Probability_Basics_and_Axioms.md" --topic 4 --regenerate
```

### Output Folder Structure:

```
MathsTerms/<Category>/chatgpt_images/<MarkdownFileStem>/
  ├── topic-01_<TopicSlug>_img1.jpg
  ├── topic-01_<TopicSlug>_img2.jpg
  └── ...
```

---

## 3. Lecture Transcripts Runner (`grok_imagine_runner.py`)

Processes topic transcript files (`topic-*.txt` or `topic-*.md`) across lecture notes (`01-*` through `73-*`).

### A. Run a Folder Range with Parallel Tabs (Recommended)

Specify both the **parent directory** (`--dir`) and the **lecture number range** (`--range START END`):

```powershell
# 1. Run Folders 08 to 14 in 'Mathematical-foundation-ml' using ChatGPT (2 parallel tabs):
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --dir "Mathematical-foundation-ml" --range 8 14 --parallel 2

# 2. Run Folders 08 to 14 in 'Mathematical-foundation-ml' using Grok (3 parallel tabs):
python grok_image_automation/grok_imagine_runner.py --engine grok --dir "Mathematical-foundation-ml" --range 8 14 --parallel 3

# 3. Run Folders 14 to 33 in 'Mathematical-Foundation-for-GenerativeAI' using ChatGPT (2 parallel tabs):
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --dir "Mathematical-Foundation-for-GenerativeAI" --range 14 33 --parallel 2

# 4. Force REGENERATE all topics across Folders 08 to 14 (overwrites existing images):
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --dir "Mathematical-foundation-ml" --range 8 14 --parallel 2 --regenerate
```

---

### B. Run a Single Lecture Folder

```powershell
# 1. Run all topics in Lecture 08 in parallel (2 tabs, ChatGPT):
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --run --dir "Mathematical-foundation-ml/08-Lec07-IID-Assumption" --parallel 2

# 2. Run a specific topic (e.g. Topic 1) in a lecture folder:
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --run --dir "Mathematical-foundation-ml/08-Lec07-IID-Assumption" --topic 1

# 3. Force regenerate a single topic:
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --run --dir "Mathematical-foundation-ml/08-Lec07-IID-Assumption" --topic 1 --regenerate
```

---

### C. Direct Download from Existing ChatGPT Chat URL

If an image already generated in your ChatGPT chat history, you can download the master image directly without re-generating:

```powershell
python grok_image_automation/grok_imagine_runner.py --dir "Mathematical-foundation-ml/08-Lec07-IID-Assumption" --topic 1 --chat-url "https://chatgpt.com/c/6ab96387-104c-83ee-8bfa-e5b8a76d463c"
```

---

### D. Preview Topics Without Launching Browser (`--dry-run`)

```powershell
# Check which topics are DONE vs PENDING in a lecture folder:
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --dir "Mathematical-foundation-ml/08-Lec07-IID-Assumption" --dry-run
```

---

## CLI Options & Flags


| Flag                      | Argument            | Description                                                                                   |
| :-------------------------- | :-------------------- | :---------------------------------------------------------------------------------------------- |
| `--engine`                | `grok` \| `chatgpt` | Engine to use for image generation (default:`grok`)                                           |
| `--dir`                   | `<path>`            | Target lecture directory or parent directory containing numbered lecture folders              |
| `--range`                 | `START END`         | Process a range of numbered lecture folders (e.g.`--range 8 14`)                              |
| `--parallel`              | `1-3`               | Number of parallel generation tabs/browsers (1 to 3, default:`1` for lectures, `2` for maths) |
| `--run`                   | *(flag)*            | Run batch image generation for topics in a single target directory                            |
| `--all`                   | *(flag)*            | Process all lecture/tutorial folders in the target directory                                  |
| `--topic`                 | `1 2 3...` or `1,2` | Specific topic number(s) to run                                                               |
| `--skip-existing`         | *(flag)*            | Automatically skip topics that already have complete images (**Default: TRUE**)               |
| `--regenerate`, `--force` | *(flag)*            | Force re-generate topics even if images already exist on disk                                 |
| `--no-skip-existing`      | *(flag)*            | Disable skipping existing images                                                              |
| `--timeout`, `--max-wait` | `SECONDS`           | Maximum wait time in seconds for image generation (default:`360` / 6 min)                     |
| `--chat-url`              | `<url>`             | Download generated image directly from an existing ChatGPT conversation URL                   |
| `--dry-run`               | *(flag)*            | Print discovered topics and status without launching the browser                              |
| `--account`               | `<id>`              | Account / session identifier (`1`, `2`, `'alt'`, etc., default: `1`)                          |
| `--login`                 | *(flag)*            | Launch browser to log in and save session to engine's profile                                 |
| `--email`                 | `<email>`           | Account email for ChatGPT (default:`sivanagarajupachipulusu@gmail.com`)                       |
| `--password`              | `<password>`        | Password for ChatGPT                                                                          |
| `--file`                  | `<path>`            | Path to a MathsTerms markdown file                                                            |
| `--category`              | `<name>`            | Category name or absolute path in MathsTerms                                                  |
| `--max-retries`           | `N`                 | Number of retries for incomplete topics (default:`2`)                                         |
| `--include-all-sections`  | *(flag)*            | Include non-image sections (References & Beginner Comprehension, skipped by default)          |
| `--clean-ignored`         | *(flag)*            | Clean/delete obsolete generated image files for ignored sections (e.g. topic-13, topic-14)    |
| `--delay`                 | `N`                 | Cooldown seconds between batches (default:`6`)                                                |
| `--profile-dir`           | `<path>`            | Custom Chrome profile directory                                                               |

> [!TIP]
> **Automatic Non-Image Section Skipping**:
> By default, `grok_mathsterms_runner.py` automatically skips generating images for non-visual sections:
>
> - **Section 13:** Beginner Comprehension Confidence Audit (rubrics / checklists)
> - **Section 14:** Curated External Learning References & Further Study (link portfolios / tables)
>
> Pass `--include-all-sections` if you wish to generate images for these sections as well.
> Pass `--clean-ignored` to automatically delete any existing image files for these sections from output folders.
