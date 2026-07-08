"""
Unit tests for tools in tools.py

Tests each LangChain tool (export_to_excel, send_email, calendar tools,
search_documentation) with proper mocking of external services.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from tools import (
    export_to_excel,
    send_email,
    get_available_time_slots,
    create_calendar_event,
    search_documentation,
)


class TestExportToExcel:
    """Test export_to_excel tool."""

    def test_export_dict_with_lists(self, temp_output_dir):
        """Test exporting tabular data (dict of lists)."""
        with patch('tools.get_output_dir', return_value=temp_output_dir), \
             patch('tools.get_download_url', return_value='http://localhost:8765/test.xlsx'):
            data = {"name": ["Alice", "Bob"], "age": [25, 30]}
            result = export_to_excel.invoke({"data": data, "filename": "test"})
            assert "successfully" in result.lower()
            assert (temp_output_dir / "test.xlsx").exists()

    def test_export_single_row_dict(self, temp_output_dir):
        """Test exporting a flat dict (single-row table)."""
        with patch('tools.get_output_dir', return_value=temp_output_dir), \
             patch('tools.get_download_url', return_value='http://localhost:8765/single.xlsx'):
            data = {"name": "Alice", "age": 25}
            result = export_to_excel.invoke({"data": data, "filename": "single"})
            assert "successfully" in result.lower()
            assert (temp_output_dir / "single.xlsx").exists()

    def test_export_empty_dict_returns_error(self):
        """Test that exporting an empty dict returns an error message."""
        result = export_to_excel.invoke({"data": {}})
        assert "no data" in result.lower() or "invalid" in result.lower()

    def test_export_auto_generates_filename(self, temp_output_dir):
        """Test that a timestamped filename is generated when none is provided."""
        with patch('tools.get_output_dir', return_value=temp_output_dir), \
             patch('tools.get_download_url', return_value='http://localhost:8765/export.xlsx'):
            data = {"col1": [1, 2, 3]}
            result = export_to_excel.invoke({"data": data})
            assert "successfully" in result.lower()

    def test_export_includes_download_url(self, temp_output_dir):
        """Test that the result includes the download URL."""
        url = "http://localhost:8765/report.xlsx"
        with patch('tools.get_output_dir', return_value=temp_output_dir), \
             patch('tools.get_download_url', return_value=url):
            data = {"col": [1]}
            result = export_to_excel.invoke({"data": data, "filename": "report"})
            assert url in result

    def test_export_strips_xlsx_extension_from_filename(self, temp_output_dir):
        """Test that .xlsx in the filename param doesn't cause double extension."""
        with patch('tools.get_output_dir', return_value=temp_output_dir), \
             patch('tools.get_download_url', return_value='http://localhost:8765/test.xlsx'):
            data = {"col": [1]}
            result = export_to_excel.invoke({"data": data, "filename": "test.xlsx"})
            assert "successfully" in result.lower()
            # Should create test.xlsx, not test.xlsx.xlsx
            assert (temp_output_dir / "test.xlsx").exists()
            assert not (temp_output_dir / "test.xlsx.xlsx").exists()


class TestSendEmail:
    """Test send_email tool."""

    @patch('tools._build_google_service')
    def test_send_email_success(self, mock_build_service):
        """Test successful email sending."""
        mock_service = MagicMock()
        mock_build_service.return_value = mock_service
        mock_service.users().messages().send.return_value.execute.return_value = {}

        result = send_email.invoke({
            "to": ["recipient@example.com"],
            "subject": "Test Subject",
            "body": "Test body content"
        })
        assert "successfully" in result.lower()

    @patch('tools._build_google_service')
    def test_send_email_with_cc(self, mock_build_service):
        """Test email sending with CC recipients."""
        mock_service = MagicMock()
        mock_build_service.return_value = mock_service
        mock_service.users().messages().send.return_value.execute.return_value = {}

        result = send_email.invoke({
            "to": ["to@example.com"],
            "subject": "Test",
            "body": "Body",
            "cc": ["cc@example.com"]
        })
        assert "successfully" in result.lower()
        assert "cc@example.com" in result

    def test_send_email_missing_required_fields(self):
        """Test that missing required params (subject, body) raises validation error."""
        with pytest.raises(Exception):
            send_email.invoke({"to": ["test@example.com"]})

    def test_send_email_empty_recipients_returns_error(self):
        """Test that an empty 'to' list returns an error."""
        result = send_email.invoke({
            "to": [],
            "subject": "Test",
            "body": "Body"
        })
        assert "❌" in result

    @patch('tools._build_google_service')
    def test_send_email_google_token_missing(self, mock_build_service):
        """Test graceful handling when Google token is not found."""
        mock_build_service.side_effect = FileNotFoundError("Google token not found")

        result = send_email.invoke({
            "to": ["test@example.com"],
            "subject": "Test",
            "body": "Body"
        })
        assert "❌" in result


class TestCalendarTools:
    """Test calendar-related tools."""

    @patch('tools._build_google_service')
    def test_get_available_slots_all_free(self, mock_build_service):
        """Test getting slots when the entire day is free."""
        mock_service = MagicMock()
        mock_build_service.return_value = mock_service
        mock_service.freebusy().query.return_value.execute.return_value = {
            "calendars": {"test@example.com": {"busy": []}}
        }

        result = get_available_time_slots.invoke({
            "attendees": ["test@example.com"],
            "date": "2024-01-15",
            "duration_minutes": 60
        })
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_slots_invalid_date_format(self):
        """Test that an invalid date format returns an error."""
        result = get_available_time_slots.invoke({
            "attendees": ["test@example.com"],
            "date": "not-a-date",
            "duration_minutes": 60
        })
        assert any("❌" in str(s) for s in result)

    def test_get_slots_empty_attendees(self):
        """Test that an empty attendees list returns an error."""
        result = get_available_time_slots.invoke({
            "attendees": [],
            "date": "2024-01-15",
            "duration_minutes": 60
        })
        assert any("❌" in str(s) for s in result)

    def test_get_slots_zero_duration(self):
        """Test that zero or negative duration returns an error."""
        result = get_available_time_slots.invoke({
            "attendees": ["test@example.com"],
            "date": "2024-01-15",
            "duration_minutes": 0
        })
        assert any("❌" in str(s) for s in result)

    @patch('tools._build_google_service')
    def test_create_calendar_event_success(self, mock_build_service):
        """Test successfully creating a calendar event."""
        mock_service = MagicMock()
        mock_build_service.return_value = mock_service
        mock_service.events().insert.return_value.execute.return_value = {
            "htmlLink": "https://calendar.google.com/event/abc123"
        }

        result = create_calendar_event.invoke({
            "title": "Policy Review Meeting",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T11:00:00",
            "attendees": ["test@example.com"]
        })
        assert "✅" in result or "created" in result.lower()

    @patch('tools._build_google_service')
    def test_create_event_includes_calendar_link(self, mock_build_service):
        """Test that the result includes the Google Calendar link."""
        mock_service = MagicMock()
        mock_build_service.return_value = mock_service
        mock_service.events().insert.return_value.execute.return_value = {
            "htmlLink": "https://calendar.google.com/event/abc123"
        }

        result = create_calendar_event.invoke({
            "title": "Test",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T11:00:00",
            "attendees": ["test@example.com"]
        })
        assert "calendar.google.com" in result

    @patch('tools._build_google_service')
    def test_create_event_token_missing(self, mock_build_service):
        """Test graceful handling when Google token is missing."""
        mock_build_service.side_effect = FileNotFoundError("Google token not found")

        result = create_calendar_event.invoke({
            "title": "Test",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T11:00:00",
            "attendees": ["test@example.com"]
        })
        assert "❌" in result


class TestSearchDocumentation:
    """Test search_documentation tool."""

    @patch('tools.create_vectorstore')
    def test_search_returns_formatted_results(self, mock_create_vs):
        """Test that search results include content and source metadata."""
        from langchain_core.documents import Document

        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            Document(
                page_content="To register a customer, go to Master Data menu.",
                metadata={"source": "operations_manual.pdf", "page": 23}
            )
        ]
        mock_create_vs.return_value = mock_vs

        result = search_documentation.invoke({"query": "how to register a customer"})
        assert "register a customer" in result.lower()
        assert "operations_manual.pdf" in result

    @patch('tools.create_vectorstore')
    def test_search_no_results(self, mock_create_vs):
        """Test that empty results return a 'no document found' message."""
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []
        mock_create_vs.return_value = mock_vs

        result = search_documentation.invoke({"query": "nonexistent topic xyz"})
        assert "no document" in result.lower()

    @patch('tools.create_vectorstore')
    def test_search_multiple_results(self, mock_create_vs):
        """Test that multiple matching documents are returned."""
        from langchain_core.documents import Document

        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [
            Document(page_content="Content A", metadata={"source": "faq.pdf", "page": 1}),
            Document(page_content="Content B", metadata={"source": "manual.pdf", "page": 5}),
        ]
        mock_create_vs.return_value = mock_vs

        result = search_documentation.invoke({"query": "insurance policy"})
        assert "Content A" in result
        assert "Content B" in result
        assert "Document 1" in result
        assert "Document 2" in result