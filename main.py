from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("🚀 Starting AI Meeting Assistant")

    try:
        chunks = process_input(source)

        transcript = transcribe_all(chunks, language)
        print(f"Transcript preview: {transcript[:300]}")

        title = generate_title(transcript)
        summary = summarize(transcript)

        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        rag_chain = build_rag_chain(transcript)

        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return {}


def chat_loop(rag_chain):
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        if not q:
            continue

        answer = ask_question(rag_chain, q)
        print(f"\n🤖 {answer}\n")


if __name__ == "__main__":
    source = input("Enter YouTube URL or file: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    result = run_pipeline(source, language)

    if not result:
        exit()

    print("\n" + "=" * 60)
    print(f"📌 {result['title']}")
    print("=" * 60)

    print(result["summary"])
    print(result["action_items"])
    print(result["key_decisions"])
    print(result["open_questions"])

    chat_loop(result["rag_chain"])