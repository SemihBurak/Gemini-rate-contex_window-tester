# ============================================================
# WORK@HOME 2 - Streamlit UI
# Testing and Detecting Google Gemini Rate Limits & Context Window
# ============================================================
# Run: streamlit run app.py

import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── Configuration ────────────────────────────────────────────
AVAILABLE_MODELS = [
    "gemma-3-1b-it",
    "gemini-2.5-flash-lite",
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


# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Gemini API Tester", layout="centered")
st.title("Gemini API Tester")
st.caption("Work@Home 2 — Rate Limits & Context Window Detection")


# ── Init client ──────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


client = get_client()
if client is None:
    st.error("No GEMINI_API_KEY found in .env file.")
    st.stop()


# ── Fetch model info ─────────────────────────────────────────
def get_model_info(model_name):
    model = client.models.get(model = model_name)
    return {
        "display_name": model.display_name,
        "name": model.name,
        "input_token_limit": model.input_token_limit or 0,
        "output_token_limit": model.output_token_limit or 0,
    }


def get_all_models():
    models = []
    for model in client.models.list():
        models.append({
            "name": model.name,
            "input_limit": model.input_token_limit or 0,
            "output_limit": model.output_token_limit or 0,
        })
    return models


# ── Error display helper ──────────────────────────────────────
def show_rate_limit_error(error):
    """Parse rate limit error and show a user-friendly message with expandable details."""
    err_str = str(error)

    # Detect limit type and set appropriate unit
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

    # Extract quota value
    quota_match = re.search(r"quotaValue.*?(\d+)", err_str)
    quota_value = quota_match.group(1) if quota_match else "unknown"

    # Extract retry delay
    delay_match = re.search(r"retryDelay.*?(\d+)", err_str)
    retry_delay = delay_match.group(1) + "s" if delay_match else "unknown"

    # Extract model
    model_match = re.search(r"'model':\s*'([\w.-]+)'", err_str)
    model_name = model_match.group(1) if model_match else "unknown"

    st.error(
        f"**{limit_type} Exceeded**  \n"
        f"Model: `{model_name}` | Limit: **{quota_value} {unit}** | Retry after: **{retry_delay}**"
    )

    with st.expander("Show full error details"):
        st.code(err_str, language="text")


# ── Retry wrapper ─────────────────────────────────────────────
def call_with_retry(api_func, log_container):
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return api_func()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err:
                if attempt < MAX_RETRIES:
                    log_container.warning(
                        f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    log_container.error(
                        f"Rate limit exceeded after {MAX_RETRIES} retries."
                    )
                    raise
            else:
                raise


# ── Context window display ────────────────────────────────────
def show_context_status(used, limit):
    ratio = used / limit
    pct = ratio * 100
    if ratio >= WARNING_THRESHOLD:
        st.warning(
            f"**Context Window Warning!** {used:,} / {limit:,} tokens "
            f"({pct:.1f}%) — approaching the limit!"
        )
    else:
        st.progress(ratio, text=f"Token usage: {used:,} / {limit:,} ({pct:.1f}%)")


# ── Sidebar: Model selection ─────────────────────────────────
with st.sidebar:
    st.header("Choose Model")
    selected_model = st.selectbox("Model", AVAILABLE_MODELS)

    # Fetch and display model info
    info = get_model_info(selected_model)
    context_limit = info["input_token_limit"]

    st.markdown("---")
    st.subheader("Model Info")
    st.text(f"Name:          {info['name']}")
    st.text(f"Display Name:  {info['display_name']}")
    st.text(f"Input Limit:   {info['input_token_limit']:,}")
    st.text(f"Output Limit:  {info['output_token_limit']:,}")

    if "gemma" in selected_model:
        context_limit = 15_000
    else:
        context_limit = 250_000

    st.markdown("---")
    st.subheader("Context Window Limit (Actually it is TPM)")
    WARNING_THRESHOLD = st.slider("Warning Threshold?", 0.0, 1.0, 0.6)
    st.text(f"Limit:         {context_limit:,} tokens")
    st.text(f"Warning at:    {WARNING_THRESHOLD*100:.0f}% ({int(context_limit * WARNING_THRESHOLD):,} tokens)")
    

    st.markdown("---")
    if st.button("List All Models"):
        all_models = get_all_models()
        for m in all_models:
            st.text(f"{m['name']}  in:{m['input_limit']:,}  out:{m['output_limit']:,}")


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab_text, tab_chat, tab_ratelimit = st.tabs(
    ["Text Generation", "Chat Mode", "Rate Limit Stress Test"]
)

# ── TAB 1: Text Generation ───────────────────────────────────
with tab_text:
    st.subheader("Text Generation + Context Window Monitor")

    # Example prompt selector
    st.markdown("**Example Prompts** (pick one to auto-fill, or write your own)")
    example_choice = st.selectbox(
        "Load example prompt",
        ["-- Custom --"] + list(EXAMPLE_PROMPTS.keys()),
        key="example_select"
    )

    if example_choice != "-- Custom --":
        default_prompt = EXAMPLE_PROMPTS[example_choice]
    else:
        default_prompt = ""

    prompt = st.text_area("Enter your prompt", value=default_prompt, height=200)

    if st.button("Generate", type="primary", key="gen_btn"):
        if not prompt.strip():
            st.warning("Prompt is empty.")
        else:
            log = st.empty()
            try:
                def _call():
                    return client.models.generate_content(
                        model=selected_model, contents=prompt
                    )

                with st.spinner("Generating..."):
                    response = call_with_retry(_call, log)

                usage = response.usage_metadata
                prompt_tokens = usage.prompt_token_count or 0
                response_tokens = usage.candidates_token_count or 0
                total_tokens = usage.total_token_count or 0
                # total_tokens = prompt_tokens + response_tokens
                log.empty()

                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", prompt_tokens)
                col2.metric("Response Tokens", response_tokens)
                col3.metric("Total Tokens", total_tokens)

                show_context_status(total_tokens, context_limit)

                st.markdown("### Response")
                st.write(response.text)

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    show_rate_limit_error(e)
                else:
                    st.error(f"Error: {e}")


# ── TAB 2: Chat Mode ─────────────────────────────────────────
with tab_chat:
    st.subheader("Multi-turn Chat + Context Window Monitor")

    # Init session state — recreate chat if model changed
    if "chat_model" not in st.session_state or st.session_state.chat_model != selected_model:
        st.session_state.chat_session = client.chats.create(model=selected_model)
        st.session_state.chat_model = selected_model
        st.session_state.chat_display = []
        st.session_state.chat_tokens = 0
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "chat_tokens" not in st.session_state:
        st.session_state.chat_tokens = 0

    if st.button("Clear Conversation", key="clear_chat"):
        st.session_state.chat_session = client.chats.create(model=selected_model)
        st.session_state.chat_display = []
        st.session_state.chat_tokens = 0
        st.rerun()

    # Example prompts for chat
    st.markdown("**Example Prompts**")
    chat_examples = {
        "-- Custom --": None,
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
    chat_example_choice = st.selectbox("Load example into chat", list(chat_examples.keys()), key="chat_example")

    # Display chat history
    for role, text in st.session_state.chat_display:
        with st.chat_message(role):
            st.write(text)

    # Token bar
    if st.session_state.chat_tokens > 0:
        show_context_status(st.session_state.chat_tokens, context_limit)

    # Helper to send a message via the chat session
    def send_chat_message(message, display_text=None):
        """Send message via client.chats, update tokens and display."""
        display_text = display_text or message
        st.session_state.chat_display.append(("user", display_text))

        log = st.empty()
        try:
            def _call():
                return st.session_state.chat_session.send_message(message)

            with st.spinner("Processing..."):
                response = call_with_retry(_call, log)

            reply = response.text
            usage = response.usage_metadata
            st.session_state.chat_tokens += usage.total_token_count or 0
            st.session_state.chat_display.append(("assistant", reply))

            # # Debug: print chat history from API
            # print("\n--- Chat History from API ---")
            # for msg in st.session_state.chat_session.get_history():
            #     print(f"  {msg.role}: {msg.parts[0].text[:100]}...")
            # print(f"  Total tokens so far: {st.session_state.chat_tokens}")
            # print("---\n")

            log.empty()
            st.rerun()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                show_rate_limit_error(e)
            else:
                st.error(f"Error: {e}")

    # Send example prompt button
    if chat_example_choice != "-- Custom --":
        if st.button("Send Example Prompt", key="send_chat_example"):
            send_chat_message(
                chat_examples[chat_example_choice],
                display_text=f"[Example: {chat_example_choice}]"
            )

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        send_chat_message(user_input)


# ── TAB 3: Rate Limit Stress Test ────────────────────────────
with tab_ratelimit:
    st.subheader("Rate Limit Stress Test")
    st.caption("Sends N rapid requests to trigger rate limiting with exponential backoff.")

    num_requests = st.slider("Number of rapid requests", 2, 40, 5)
    stress_prompt = st.text_input("Prompt for each request", value="Say hello in one sentence.")

    if st.button("Start Stress Test", type="primary", key="stress_btn"):
        results_container = st.container()
        progress = st.progress(0, text="Starting...")

        successes, failures = 0, 0

        for i in range(1, num_requests + 1):
            progress.progress(i / num_requests, text=f"Request {i}/{num_requests}...")
            log_slot = results_container.empty()

            try:
                def _stress_call():
                    return client.models.generate_content(
                        model=selected_model, contents=stress_prompt
                    )

                response = call_with_retry(_stress_call, log_slot)
                usage = response.usage_metadata
                total = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

                with results_container:
                    st.success(
                        f"Request {i} — OK | {total} tokens | "
                        f"{response.text.strip()[:80]}"
                    )
                successes += 1

            except Exception as e:
                with results_container:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        show_rate_limit_error(e)
                    else:
                        st.error(f"Request {i} — FAILED: {e}")
                failures += 1

        progress.empty()
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Succeeded", successes)
        col2.metric("Failed", failures)
