"""
LangGraph native tools using official ToolNode pattern
"""

import base64
from email.message import EmailMessage
from typing import Optional, Any

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import pypdf

from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from datetime import datetime, timedelta, timezone
import glob
from pathlib import Path

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from zoneinfo import ZoneInfo

from file_server import get_output_dir, get_download_url

# Global cache to avoid recreating expensive resources

_vectorstore_cache = None
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]


def _load_google_credentials(scopes: list[str] = GOOGLE_SCOPES) -> Credentials:
    cred_dir = Path(os.getenv("GOOGLE_CREDENTIALS_DIR", "./google_credentials")).resolve()
    token_path = cred_dir / "token.json"
    if not token_path.exists():
        raise FileNotFoundError(
            "Google token not found. Run google_credentials/bootstrap_oauth.py to generate it."
        )

    creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _build_google_service(api_name: str, api_version: str) -> Any:
    creds = _load_google_credentials()
    return build(api_name, api_version, credentials=creds, cache_discovery=False)

def create_vectorstore(archivio_path: str) -> InMemoryVectorStore:
    """
    Create or return the vectorstore for document search (singleton pattern).

    If a pre-generated index exists at the configured path, loads it directly.
    Otherwise, loads PDF documents from the specified directory, splits them,
    generates embeddings, saves the index to disk, and returns the vectorstore.

    Args:
        archivio_path: Path to the directory containing PDFs.

    Returns:
        InMemoryVectorStore: Indexed vectorstore for semantic search.
    """
    global _vectorstore_cache
    
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    embeddings = OpenAIEmbeddings()
    base_dir = Path(__file__).parent.resolve()
    index_path = os.getenv('VECTORSTORE_INDEX_PATH', str(base_dir / "data" / "vectorstore.json"))

    # Try loading pre-generated index from disk
    if os.path.exists(index_path):
        print(f"Loading pre-generated vectorstore from {index_path}...")
        try:
            vectorstore = InMemoryVectorStore.load(index_path, embeddings)
            _vectorstore_cache = vectorstore
            print("Vectorstore loaded successfully from disk.")
            return _vectorstore_cache
        except Exception as e:
            print(f"Failed to load vectorstore from {index_path}: {e}. Rebuilding...")

    # Otherwise, rebuild the vector store
    documents = []
    
    # Load PDF documents
    pdf_files = glob.glob(os.path.join(archivio_path, "**", "*.pdf"), recursive=True)
    for pdf_file in pdf_files:     
        try:
            reader = pypdf.PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                text_content = page.extract_text() or ""
                documents.append(
                    Document(
                        page_content=text_content,
                        metadata={"source": pdf_file, "page": i + 1}
                    )
                )
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")

    print(f"Loaded {len(documents)} document pages from {archivio_path}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create the vectorstore
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents=splits)
    
    # Save the vector store to disk
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vectorstore.dump(index_path)
        print(f"Vectorstore saved to {index_path}")
    except Exception as e:
        print(f"Failed to save vectorstore to {index_path}: {e}")

    _vectorstore_cache = vectorstore
    print("Vectorstore created successfully")

    return _vectorstore_cache


@tool
def search_documentation(query: str) -> str:
    """
    Search in the company's insurance portal documentation.
    Useful for finding information about policies, procedures, rules, and FAQs.

    Args:
        query: Question or search terms to find relevant documents

    Returns:
        Most relevant content found in the documentation
    """
    vectorstore = create_vectorstore(os.getenv('DOCS_PATH', './docs'))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    
    if not docs:
        return "No document found for the specified query."
    
    # Format results
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        results.append(f"--- Document {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}")
    
    return "\n\n".join(results)


@tool
def export_to_excel(
    data: dict, 
    filename: Optional[str] = None
) -> str:
    """
    Export data to an Excel file (.xlsx) and return a download link.

    Args:
        data: Data to export - must be a dictionary
        filename: Optional file name (without extension)

    Returns:
        Success message with download URL
        
    Example:
        export_to_excel({"name": "Mario", "age": 30}, "single_user")
        export_to_excel({
            "name": ["Mario", "Luigi"],
            "age": [30, 28]
        }, "users_table")
    """
    try:
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}"
        
        # Remove .xlsx extension if provided
        filename = filename.replace('.xlsx', '')
        
        # Validate data
        if not data or not isinstance(data, dict):
            return "❌ No data to export or invalid format (dict required)"

        # Convert data to DataFrame
        if all(isinstance(v, (list, tuple)) for v in data.values()):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Export to Excel
        full_filename = f"{filename}.xlsx"
        filepath = get_output_dir() / full_filename
        df.to_excel(filepath, index=False, engine='openpyxl')
        
        return (
            f"✅ Excel file exported successfully!\n\n"
            f"📥 **Download:** {get_download_url(full_filename)}"
        )
        
    except Exception as e:
        return f"❌ Export error: {str(e)}"

@tool
def send_email(
    to: list[str], 
    subject: str, 
    body: str,
    cc: list[str] | None = None
) -> str:
    """
    Send an email to a recipient.    
    
    Args:
        to: Email recipient addresses
        subject: Email subject line
        body: Email body content
        cc: Optional list of CC email addresses
                
    Returns:
        Status message about email sending
        
    Example:
        send_email("user@example.com", "Monthly Report", "Here is the report...")
    """
    if not to or not subject or not body:
        return "❌ Missing parameters: to, subject, and body are required"

    recipients = to[:]
    message = EmailMessage()
    message["To"] = ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
        recipients.extend(cc)

    sender = os.getenv("GOOGLE_EMAIL_SENDER")
    if sender:
        message["From"] = sender
    message["Subject"] = subject
    message.set_content(body)

    try:
        service = _build_google_service("gmail", "v1")
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": encoded_message}).execute()
        return f"📧 Email sent successfully to {', '.join(recipients)}"
    except FileNotFoundError as exc:
        return f"❌ {exc}"
    except HttpError as err:
        return f"❌ Gmail API error: {err}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error while sending: {exc}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # Formato ISO: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check Google Calendar availability for the provided attendees."""

    if not attendees:
        return ["❌ No participant specified"]
    if duration_minutes <= 0:
        return ["❌ Invalid duration"]

    # Resolve the calendar timezone first: the requested day must span midnight-to-midnight
    # in the calendar's own timezone, not in UTC, or every returned slot is shifted by the
    # UTC offset (1h in Rome winter, 2h in summer).
    calendar_tz_name = os.getenv("GOOGLE_CALENDAR_TIMEZONE", "UTC")
    try:
        calendar_tz = ZoneInfo(calendar_tz_name)
    except Exception:  # noqa: BLE001
        calendar_tz = timezone.utc
        calendar_tz_name = "UTC"

    try:
        day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=calendar_tz)
    except ValueError:
        return ["❌ Invalid date. Use format YYYY-MM-DD"]

    start_dt = day
    end_dt = start_dt + timedelta(days=1)

    body = {
        "timeMin": start_dt.isoformat().replace("+00:00", "Z"),
        "timeMax": end_dt.isoformat().replace("+00:00", "Z"),
        "timeZone": calendar_tz_name,
        "items": [{"id": email} for email in attendees],
    }

    try:
        service = _build_google_service("calendar", "v3")
        response = service.freebusy().query(body=body).execute()
    except FileNotFoundError as exc:
        return [f"❌ {exc}"]
    except HttpError as err:
        return [f"❌ Calendar API error: {err}"]
    except Exception as exc:  # noqa: BLE001
        return [f"❌ Unexpected error while checking availability: {exc}"]

    busy_intervals: list[tuple[datetime, datetime]] = []
    for calendar_info in response.get("calendars", {}).values():
        for slot in calendar_info.get("busy", []):
            busy_intervals.append(
                (
                    datetime.fromisoformat(slot["start"].replace("Z", "+00:00")),
                    datetime.fromisoformat(slot["end"].replace("Z", "+00:00")),
                )
            )

    busy_intervals.sort(key=lambda item: item[0])
    slot_delta = timedelta(minutes=duration_minutes)
    slots: list[str] = []
    cursor = start_dt

    for busy_start, busy_end in busy_intervals:
        if busy_end <= cursor:
            continue
        if busy_start > cursor:
            free_end = min(busy_start, end_dt)
            current = cursor
            while current + slot_delta <= free_end:
                slot_end = current + slot_delta
                slot_start_local = current.astimezone(calendar_tz)
                slot_end_local = slot_end.astimezone(calendar_tz)
                slots.append(
                    f"{slot_start_local.strftime('%H:%M')} - {slot_end_local.strftime('%H:%M')} {calendar_tz_name}"
                )
                current = slot_end
        cursor = max(cursor, busy_end)
        if cursor >= end_dt:
            break

    if cursor < end_dt:
        current = cursor
        while current + slot_delta <= end_dt:
            slot_end = current + slot_delta
            slot_start_local = current.astimezone(calendar_tz)
            slot_end_local = slot_end.astimezone(calendar_tz)
            slots.append(
                f"{slot_start_local.strftime('%H:%M')} - {slot_end_local.strftime('%H:%M')} {calendar_tz_name}"
            )
            current = slot_end

    if not slots:
        return ["ℹ️ No slots available for the requested duration"]

    return slots

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # Email addresses
) -> str:
    """Create an event on the configured Google Calendar."""

    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    calendar_tz_name = os.getenv("GOOGLE_CALENDAR_TIMEZONE", "UTC")

    event_body = {
        "summary": title,
        "start": {"dateTime": start_time, "timeZone": calendar_tz_name},
        "end": {"dateTime": end_time, "timeZone": calendar_tz_name},
    }
    if attendees:
        event_body["attendees"] = [{"email": email} for email in attendees]

    try:
        service = _build_google_service("calendar", "v3")
        event = (
            service.events()
            .insert(calendarId=calendar_id, body=event_body, sendUpdates="all")
            .execute()
        )
        link = event.get("htmlLink")
        link_info = f" → {link}" if link else ""
        return f"✅ Event created: {title}{link_info}"
    except FileNotFoundError as exc:
        return f"❌ {exc}"
    except HttpError as err:
        return f"❌ Calendar API error: {err}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error while creating event: {exc}"






