import re

from prompts import EXEC_TOKEN
from utils import clean_text


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
ROLE_PATTERN = re.compile(r"(?i)\b(?:client|therapist|counselor|seeker|supporter)\s*:")
PLACEHOLDER_PATTERN = re.compile(r"\[[^]]+\]|\{[^}]+\}")
PERSONAL_DISCLOSURE_PATTERNS = [
    re.compile(r"(?i)\bI\s+(?:have|had|was|went|used to|grew up|live|lived|work|worked|own|also have)\b"),
    re.compile(
        r"(?i)\bmy\s+(?:child|children|kid|kids|son|daughter|husband|wife|partner|mother|father|parents|family|job|school)\b"
    ),
    re.compile(r"(?i)\b(?:personally|if I were you|I personally|I feel (?:more|less|better|worse|relaxed|great))\b"),
    re.compile(
        r"(?i)\bI can (?:help you )?(?:fill out|submit|send|contact|call|apply|"
        r"obtain|get|provide|arrange|book)\b"
    ),
]
CLOSING_PATTERN = re.compile(
    r"(?i)\b(?:wish you all the best|future endeavors?|make an appointment|always be here for you|"
    r"have a wonderful life|take care|goodbye|see you next time|let me know|ask me anytime|"
    r"happy (?:to|I could|to be able to) help|glad I (?:could|can) (?:be of )?help|"
    r"you(?:'re| are) welcome|call me|contact me|if you need (?:any )?help|"
    r"happy to accompany you|after our conversation|chatting with me today|"
    r"next week|happy holidays?|feedback form|before we end|tell me later|"
    r"tell me (?:about it )?later|tell me later how it went|call it a day|"
    r"see you (?:at some )?future time|come to me anytime|stop here today|"
    r"stop (?:here )?for today|end here today)\b"
)
DIAGNOSTIC_PATTERN = re.compile(
    r"(?i)\b(?:you have|you suffer from|this is)\s+(?:depression|anxiety disorder|ptsd|bipolar|ocd)\b"
)
UNSUPPORTED_CERTAINTY_PATTERN = re.compile(
    r"(?i)\b(?:you will definitely|you(?:'re| are) guaranteed to|definitely succeed|"
    r"everything will (?:be|turn out)|will certainly succeed|"
    r"I (?:believe|know|am sure) you will|you have the ability to)\b"
)


GENERATOR_SYSTEM_PROMPT = """You are a counseling response generator.
- Return only the therapist's next response. Never continue or simulate the client's next turn.
- Write exactly 2 complete sentences and keep the whole response between {min_words} and {max_words} words.
- Ground every factual statement in the dialogue. Do not invent personal experiences, people, events, causes, schedules, diagnoses, or details.
- Be empathetic, calm, non-judgmental, natural, and focused on the client's expressed concern.
- Follow the selected counseling strategy without naming it.
- Sentence 1 must briefly reflect and empathize with only what the client has expressed.
- Sentence 2 must carry out the selected strategy with a grounded response, optional suggestion, or at most one question.
- Ask at most one question.
- Even if the client says thanks, continue with useful support rather than ending the session.
- Do not include role labels, placeholders, medical advice, or session-closing language."""


def word_count(text):
    return len(WORD_PATTERN.findall(str(text or "")))


def target_is_safe(text):
    value = str(text or "")
    if ROLE_PATTERN.search(value) or PLACEHOLDER_PATTERN.search(value):
        return False
    if any(pattern.search(value) for pattern in PERSONAL_DISCLOSURE_PATTERNS):
        return False
    return (
        not DIAGNOSTIC_PATTERN.search(value)
        and not CLOSING_PATTERN.search(value)
        and not UNSUPPORTED_CERTAINTY_PATTERN.search(value)
    )


def build_generator_prompt(
    context_text,
    strategy,
    min_words=20,
    max_words=30,
    include_execution=True,
):
    system_prompt = GENERATOR_SYSTEM_PROMPT.format(
        min_words=min_words,
        max_words=max_words,
    )
    execution_line = f"{EXEC_TOKEN}<|im_end|>\n" if include_execution else "<|im_end|>\n"
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"[CONTEXT]\n{clean_text(context_text)}\n\n"
        f"[THERAPIST_STRATEGY]\n{clean_text(strategy)}\n\n"
        f"[RESPONSE_LENGTH]\n{min_words}-{max_words} words\n\n"
        f"{execution_line}"
        "<|im_start|>assistant\n"
    )


