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

## 1. Authentication & Persistent Profiles (Run Once)

### Grok Login:

```powershell
python grok_image_automation/grok_imagine_runner.py --engine grok --login
```

- Opens Chrome to `https://grok.com`. Log in and press `[Enter]` in the terminal.

### ChatGPT Images Login (Manual 60-Second Window):

```powershell
python grok_image_automation/chatgpt_engine.py --login
```

*(Or via either runner: `python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --login`)*

- Chrome opens directly with the persistent `.chatgpt_profile/` directory.
- You have **60 seconds** to log in manually (enter email, password, passcode/2FA, verify).
- Once completed, the browser cleanly saves all session cookies, tokens, and storage state into `.chatgpt_profile/`.
- That profile is then **permanently reused** by both `grok_mathsterms_runner.py` and `grok_imagine_runner.py`!
- Optional: Use `--wait <seconds>` (e.g. `--wait 90`) if you need more time.

---

## 2. MathsTerms Runner (`grok_mathsterms_runner.py`)

Dynamically parses topic sections (`### 1.`, `### 2.`, etc.) from monolithic markdown files in `MathsTerms/` and downloads generated images into segregated topic folders.

### Using ChatGPT Images 2.5 (`--engine chatgpt`):

```powershell
# 1. Run all files in Category 01 in parallel (2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --category "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\MathsTerms\01-Primal-Analysis-and-Foundations" --parallel 2 --skip-existing

# 2. Or using category folder name shorthand (3 parallel tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --category "01-Primal-Analysis-and-Foundations" --parallel 3 --skip-existing

# 3. Generate images for Topic 1 of a specific file:
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --topic 1

# 4. Generate multiple topics (e.g. Topics 1, 2, 4) in parallel:
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --topic 1 2 4 --parallel 3

# 5. Process all topics in a single file (parallel 2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine chatgpt --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --parallel 2 --skip-existing
```

### Using Grok Imagine (`--engine grok`):

```powershell
# 1. Run Category 01 with Grok (parallel 2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine grok --category "C:\Users\sivan\Learning\Code\GenerativeAI\Mathematical-Foundations-of-ML\MathsTerms\01-Primal-Analysis-and-Foundations" --parallel 2 --skip-existing

# 2. Run all topics for a file in parallel (2 tabs):
python grok_image_automation/grok_mathsterms_runner.py --engine grok --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --parallel 2

# 3. Force regenerate a specific topic:
python grok_image_automation/grok_mathsterms_runner.py --engine grok --file "MathsTerms/01-Primal-Analysis-and-Foundations/01-Probability_Basics_and_Axioms.md" --topic 4 --regenerate
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

Processes topic transcript files (`topic-*.txt` or `topic-*.md`) across lecture notes (`01-*` through `34-*`).

### Using ChatGPT Images 2.5:

```powershell
# Run Topic 1 for Lecture 13:
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --run --dir "Mathematical-Foundation-for-GenerativeAI/13-Tutorial11-f-Divergence-Examples" --topic 1

# Run across all topics in a lecture folder in parallel (2 tabs):
python grok_image_automation/grok_imagine_runner.py --engine chatgpt --run --dir "Mathematical-Foundation-for-GenerativeAI/13-Tutorial11-f-Divergence-Examples" --parallel 2 --skip-existing
```

### Using Grok Imagine:

```powershell
# Run 3 parallel tabs:
python grok_image_automation/grok_imagine_runner.py --engine grok --run --parallel 3

# Run across a range of lecture folders (e.g. 14 to 33):
python grok_image_automation/grok_imagine_runner.py --engine grok --range 14 33 --parallel 3
```

---

## CLI Options & Flags


| Flag                      | Argument            | Description                                                             |
| :-------------------------- | :-------------------- | :------------------------------------------------------------------------ |
| `--engine`                | `grok` \| `chatgpt` | Engine to use for image generation (default:`grok`)                     |
| `--login`                 | *(flag)*            | Launch browser to log in and save session to engine's profile           |
| `--email`                 | `<email>`           | Account email for ChatGPT (default:`sivanagarajupachipulusu@gmail.com`) |
| `--password`              | `<password>`        | Password for ChatGPT (default:`Pulk@_ta!nt_01!`)                        |
| `--file`                  | `<path>`            | Path to a MathsTerms markdown file                                      |
| `--category`              | `<name>`            | Category name or absolute path in MathsTerms                            |
| `--all`                   | *(flag)*            | Process all files across all categories in MathsTerms                   |
| `--topic`                 | `1 2 3...`          | Specific topic numbers to run                                           |
| `--parallel`              | `1-3`               | Number of parallel generation tabs (default: 2)                         |
| `--skip-existing`         | *(flag)*            | Skip topics that already have valid downloaded images                   |
| `--regenerate`, `--force` | *(flag)*            | Force regenerate topics even if images already exist                    |
| `--max-retries`           | `N`                 | Number of retries for incomplete topics (default:`2`)                   |
| `--delay`                 | `N`                 | Cooldown seconds between batches (default:`6`)                          |
| `--profile-dir`           | `<path>`            | Custom Chrome profile directory                                         |
