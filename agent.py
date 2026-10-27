"""
Day 39: AI Help Desk Agent
Uses OpenRouter AI and the Day 38 FastAPI Ticket API.
"""

import os
import logging

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)
TICKET_API_URL = os.getenv(
    "TICKET_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

# Agent safety limit
MAX_TICKETS_PER_RUN = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY missing. Check your .env file."
    )

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def get_tickets():
    """Fetch tickets from the Day 38 FastAPI server."""

    try:
        response = requests.get(
            f"{TICKET_API_URL}/tickets",
            timeout=15
        )
        response.raise_for_status()

        data = response.json()
        tickets = data.get("tickets", [])

        logger.info("Total tickets fetched: %s", len(tickets))

        return tickets

    except requests.RequestException as exc:
        logger.error("Unable to fetch tickets: %s", exc)
        return None


def is_unassigned(ticket):
    """
    Identify unassigned tickets.

    The Day 38 API does not currently contain an assignee field.
    Therefore, Open tickets are treated as unprocessed tickets
    for this demo.
    """

    status = ticket.get("status", "Open")
    return status.lower() == "open"


def generate_ai_solution(ticket):
    """Generate a troubleshooting response using OpenRouter."""

    prompt = f"""
You are an AI Help Desk Assistant.

Analyze the following IT support ticket and suggest
safe, practical troubleshooting steps.

Ticket title: {ticket.get('title', '')}
Ticket description: {ticket.get('description', '')}
Priority: {ticket.get('priority', 'Medium')}

Instructions:
1. Provide a clear and professional response.
2. Give simple troubleshooting steps.
3. Never ask for passwords or secret credentials.
4. Do not claim that you performed actions you did not perform.
5. If the issue needs administrator access or physical repair,
   recommend contacting the IT support team.
6. Keep the response concise.
"""

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful IT help desk assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=400
        )

        solution = response.choices[0].message.content

        if not solution or not solution.strip():
            logger.warning("AI returned an empty response.")
            return None

        return solution.strip()

    except Exception as exc:
        logger.error("AI generation failed: %s", exc)
        return None


def post_solution(ticket, solution):
    """
    Save the AI response and update ticket status.

    The original Day 38 API supports status updates only.
    This demo uses an in-memory local result to display the
    generated answer; it does not save the answer in the ticket.
    """

    ticket_id = ticket.get("id")

    logger.info(
        "Generated response for ticket %s:\n%s",
        ticket_id,
        solution
    )

    try:
        response = requests.patch(
            f"{TICKET_API_URL}/tickets/{ticket_id}",
            params={"status": "Resolved"},
            timeout=15
        )
        response.raise_for_status()

        logger.info(
            "Ticket %s status updated to Resolved.",
            ticket_id
        )

        return True

    except requests.RequestException as exc:
        logger.error(
            "Unable to update ticket %s: %s",
            ticket_id,
            exc
        )
        return False


def run_agent():
    """Process a limited number of open tickets."""

    logger.info("Day 39 AI Help Desk Agent started.")

    tickets = get_tickets()

    if tickets is None:
        logger.error(
            "Stopping agent because the ticket API is unavailable."
        )
        return

    if not tickets:
        logger.info("No tickets found.")
        return

    processed_count = 0

    for ticket in tickets:
        if processed_count >= MAX_TICKETS_PER_RUN:
            logger.warning(
                "Safety limit reached: maximum %s tickets per run.",
                MAX_TICKETS_PER_RUN
            )
            break

        if not is_unassigned(ticket):
            continue

        logger.info(
            "Processing ticket %s: %s",
            ticket.get("id"),
            ticket.get("title")
        )

        solution = generate_ai_solution(ticket)

        if not solution:
            logger.warning(
                "Skipping ticket %s because no AI response was generated.",
                ticket.get("id")
            )
            continue

        print("\n" + "=" * 60)
        print("TICKET ID:", ticket.get("id"))
        print("TITLE:", ticket.get("title"))
        print("AI SOLUTION:\n", solution)
        print("=" * 60)

        if post_solution(ticket, solution):
            processed_count += 1

    logger.info(
        "Agent run completed. Successfully processed: %s",
        processed_count
    )


if __name__ == "__main__":
    run_agent()