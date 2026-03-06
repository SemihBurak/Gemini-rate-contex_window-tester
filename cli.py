# ============================================================
# WORK@HOME 2 - CLI Version
# Testing and Detecting Google Gemini Rate Limits & Context Window
# ============================================================
# Run: python cli.py

import os
import re
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── Configuration ────────────────────────────────────────────
AVAILABLE_MODELS = [
    "gemma-3-1b-it",
    "gemini-2.5-flash",
]

WARNING_THRESHOLD = 0.60
MAX_RETRIES = 3
BASE_DELAY = 2

# ── Example prompts for testing ──────────────────────────────
# Context Window (per minute) TPM limits (free tier): gemma = 15,000 tokens/min, gemini = 250,000 tokens/min
# From testing: short sentence ~13 tokens/rep, long sentence ~25 tokens/rep

EXAMPLE_2K = ("The following is a very long document for testing context window limits. " * 153
    + "\n\nNow summarize everything above in one sentence.")

EXAMPLE_5K = ("The following is a very long document for testing context window limits. " * 382
    + "\n\nNow summarize everything above in one sentence.")

EXAMPLE_10K = ("The following is a very long document for testing context window limits. " * 764
    + "\n\nNow summarize everything above in one sentence.")

EXAMPLE_15K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 600
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_25K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 1000
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_50K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 2000
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_100K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 4000
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_125K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 5000
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_150K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 6000
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_250K = ("The following is a comprehensive analysis of global technology trends, artificial intelligence, machine learning, distributed computing, and software engineering practices. " * 10000
    + "\n\nSummarize the key points from this entire document.")

EXAMPLE_PROMPTS = {
    "~2K tokens": EXAMPLE_2K,
    "~5K tokens": EXAMPLE_5K,
    "~10K tokens": EXAMPLE_10K,
    "~15K tokens": EXAMPLE_15K,
    "~25K tokens": EXAMPLE_25K,
    "~50K tokens": EXAMPLE_50K,
    "~100K tokens": EXAMPLE_100K,
    "~125K tokens": EXAMPLE_125K,
    "~150K tokens": EXAMPLE_150K,
    "~250K tokens": EXAMPLE_250K,
}

client = None
MODEL_NAME = None
CONTEXT_LIMIT = None


# ── Helper: Fetch model info ────────────────────────────────
def fetch_model_info(model_name):
    model = client.models.get(model=model_name)
    input_limit = model.input_token_limit or 0
    output_limit = model.output_token_limit or 0

    print("\n" + "=" * 50)
    print(f"  Model:         {model.display_name}")
    print(f"  Name:          {model.name}")
    print("-" * 50)
    print(f"  Input Limit:   {input_limit:,}")
    print(f"  Output Limit:  {output_limit:,}")
    print("=" * 50)

    return input_limit


def get_context_limit(model_name):
    if "gemma" in model_name:
        return 15_000
    else:
        return 250_000


def select_model():
    global MODEL_NAME, CONTEXT_LIMIT

    print("\n  Available Models:")
    for i, name in enumerate(AVAILABLE_MODELS, 1):
        print(f"    {i}) {name}")

    while True:
        choice = input(f"\n  Select model [1-{len(AVAILABLE_MODELS)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_MODELS):
                MODEL_NAME = AVAILABLE_MODELS[idx]
                break
        except ValueError:
            pass
        print("  Invalid choice, try again.")

    print(f"\n  Fetching info for {MODEL_NAME}...")
    fetch_model_info(MODEL_NAME)
    CONTEXT_LIMIT = get_context_limit(MODEL_NAME)
    print(f"  Context Window Limit: {CONTEXT_LIMIT:,} tokens")
    print(f"  Warning at: {WARNING_THRESHOLD*100:.0f}% ({int(CONTEXT_LIMIT * WARNING_THRESHOLD):,} tokens)")


# ── Helper: Check context window usage ──────────────────────
def check_context_window(used_tokens):
    ratio = used_tokens / CONTEXT_LIMIT
    pct = ratio * 100
    if ratio >= WARNING_THRESHOLD:
        print(
            f"\n  [WARNING] Context Window: {used_tokens:,} / {CONTEXT_LIMIT:,} tokens "
            f"({pct:.1f}%) — APPROACHING THE LIMIT!"
        )
    else:
        print(
            f"\n  [INFO] Context Window: {used_tokens:,} / {CONTEXT_LIMIT:,} tokens "
            f"({pct:.1f}%)"
        )


# ── Helper: Parse rate limit error ──────────────────────────
def show_rate_limit_error(error):
    err_str = str(error)

    if "PerDay" in err_str and "input_token" in err_str.lower():
        limit_type = "Daily Input Token Limit (TPD)"
        unit = "tokens/day"
    elif "PerDay" in err_str:
        limit_type = "Daily Request Limit (RPD)"
        unit = "requests/day"
    elif "PerMinute" in err_str and "input_token" in err_str.lower():
        limit_type = "Input Tokens Per Minute Limit (TPM)"
        unit = "tokens/min"
    elif "PerMinute" in err_str and "request" in err_str.lower():
        limit_type = "Requests Per Minute Limit (RPM)"
        unit = "requests/min"
    else:
        limit_type = "Rate Limit"
        unit = ""

    quota_match = re.search(r"quotaValue.*?(\d+)", err_str)
    quota_value = quota_match.group(1) if quota_match else "unknown"

    delay_match = re.search(r"retryDelay.*?(\d+)", err_str)
    retry_delay = delay_match.group(1) + "s" if delay_match else "unknown"

    model_match = re.search(r"'model':\s*'([\w.-]+)'", err_str)
    model_name = model_match.group(1) if model_match else "unknown"

    print(f"\n  [ERROR] {limit_type} Exceeded")
    print(f"  Model: {model_name} | Limit: {quota_value} {unit} | Retry after: {retry_delay}")
    print(f"\n  Full error: {err_str}")


# ── Retry wrapper ─────────────────────────────────────────────
def call_with_retry(api_func):
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return api_func()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err:
                if attempt < MAX_RETRIES:
                    print(
                        f"  [RETRY] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"  [ERROR] Rate limit exceeded after {MAX_RETRIES} retries.")
                    raise
            else:
                raise


# ── Text Generation ──────────────────────────────────────────
def generate_text(prompt):
    def _call():
        return client.models.generate_content(
            model=MODEL_NAME, contents=prompt
        )

    response = call_with_retry(_call)

    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count or 0
    response_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count or 0

    print(f"  Prompt: {prompt_tokens}  |  Response: {response_tokens}  |  Total: {total_tokens}")
    check_context_window(total_tokens)

    return response.text


# ── Chat Mode ────────────────────────────────────────────────
class GeminiChat:
    def __init__(self):
        self.session = client.chats.create(model=MODEL_NAME)
        self.total_tokens = 0

    def send_message(self, message):
        def _call():
            return self.session.send_message(message)

        response = call_with_retry(_call)

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count or 0
        response_tokens = usage.candidates_token_count or 0
        self.total_tokens += usage.total_token_count or 0

        print(f"  Prompt: {prompt_tokens}  |  Response: {response_tokens}  |  Cumulative: {self.total_tokens}")
        check_context_window(self.total_tokens)

        return response.text

    def reset(self):
        self.session = client.chats.create(model=MODEL_NAME)
        self.total_tokens = 0
        print("  Chat history cleared.\n")


# ── List all models ──────────────────────────────────────────
def list_all_models():
    print("\n  Fetching all models...\n")
    for model in client.models.list():
        print(f"    {model.name}  in:{model.input_token_limit:,}  out:{model.output_token_limit:,}")
    print()


# ── Example prompt picker ────────────────────────────────────
def pick_example_prompt():
    print("\n  Example Prompts:")
    keys = list(EXAMPLE_PROMPTS.keys())
    for i, k in enumerate(keys, 1):
        print(f"    {i}) {k}")
    print(f"    0) Custom prompt")

    while True:
        choice = input(f"\n  Select [0-{len(keys)}]: ").strip()
        try:
            idx = int(choice)
            if idx == 0:
                return input("\n  Enter your prompt: ").strip()
            if 1 <= idx <= len(keys):
                print(f"  Loaded: {keys[idx - 1]}")
                return EXAMPLE_PROMPTS[keys[idx - 1]]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


# ── Mode: Text Generation ───────────────────────────────────
def text_generation_mode():
    print("\n--- Text Generation Mode ---")
    print("Type 'back' to return, 'example' to pick an example prompt.\n")

    while True:
        prompt = input("Prompt> ").strip()
        if prompt.lower() == "back":
            break
        if prompt.lower() == "example":
            prompt = pick_example_prompt()
            if not prompt:
                continue
        if not prompt:
            print("  Empty prompt, try again.")
            continue

        try:
            result = generate_text(prompt)
            print(f"\nGemini:\n{result}\n")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                show_rate_limit_error(e)
            else:
                print(f"  [ERROR] {e}\n")


# ── Mode: Chat ──────────────────────────────────────────────
def chat_mode():
    print("\n--- Chat Mode ---")
    print("Commands: 'reset' = clear history, 'example' = load example, 'back' = return\n")

    chat = GeminiChat()

    while True:
        user_input = input("You> ").strip()
        if user_input.lower() == "back":
            break
        if user_input.lower() == "reset":
            chat.reset()
            continue
        if user_input.lower() == "example":
            user_input = pick_example_prompt()
            if not user_input:
                continue
        if not user_input:
            continue

        try:
            reply = chat.send_message(user_input)
            print(f"\nGemini: {reply}\n")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                show_rate_limit_error(e)
            else:
                print(f"  [ERROR] {e}\n")


# ── Mode: Rate Limit Stress Test ────────────────────────────
def rate_limit_test():
    print("\n--- Rate Limit Stress Test ---")

    try:
        count = int(input("  How many rapid requests? [5]: ").strip() or "5")
    except ValueError:
        count = 5

    prompt = input("  Prompt for each request ['Say hello in one sentence.']: ").strip() or "Say hello in one sentence."
    print(f"\n  Sending {count} rapid requests...\n")

    successes, failures = 0, 0
    for i in range(1, count + 1):
        print(f"--- Request {i}/{count} ---")
        try:
            result = generate_text(prompt)
            print(f"  Response: {result.strip()[:100]}\n")
            successes += 1
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                show_rate_limit_error(e)
            else:
                print(f"  [FAILED] {e}\n")
            failures += 1

    print(f"\n  Results: {successes} succeeded, {failures} failed\n")


# ── Menu ─────────────────────────────────────────────────────
def print_menu():
    print("\n" + "=" * 50)
    print(f"  GEMINI API TESTER (CLI)")
    print("=" * 50)
    print("  1) Choose/Switch Model")
    print("  2) List All Models")
    print("  3) Show Model Info")
    print("  4) Text Generation")
    print("  5) Chat Mode")
    print("  6) Rate Limit Stress Test")
    print("  7) Exit")
    print("=" * 50)


# ── Main ─────────────────────────────────────────────────────
def main():
    global client

    print("=" * 50)
    print("  WORK@HOME 2 — Gemini Rate Limit & Context Tester (CLI)")
    print("=" * 50)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  No GEMINI_API_KEY found in .env file. Exiting.")
        return

    client = genai.Client(api_key=api_key)
    print("  Client initialized.\n")

    while True:
        print_menu()
        choice = input("  Select [1-7]: ").strip()

        if choice in ("3", "4", "5", "6") and MODEL_NAME is None:
            print("  Please select a model first (option 1).")
            continue

        if choice == "1":
            select_model()
        elif choice == "2":
            list_all_models()
        elif choice == "3":
            fetch_model_info(MODEL_NAME)
        elif choice == "4":
            text_generation_mode()
        elif choice == "5":
            chat_mode()
        elif choice == "6":
            rate_limit_test()
        elif choice == "7":
            print("  Goodbye!")
            break
        else:
            print("  Invalid choice, try again.")


if __name__ == "__main__":
    main()
