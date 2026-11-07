import io
import unittest
import urllib.error
from unittest.mock import patch

import sync_letterboxd


class FetchFeedTests(unittest.TestCase):
    @patch("sync_letterboxd.time.sleep")
    @patch("sync_letterboxd.urllib.request.urlopen")
    def test_retries_timeout_then_succeeds(self, urlopen, sleep):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"feed"
        urlopen.side_effect = [TimeoutError("timed out"), response]
        self.assertEqual(sync_letterboxd.fetch_feed("https://example.com"), b"feed")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("sync_letterboxd.time.sleep")
    @patch("sync_letterboxd.urllib.request.urlopen", side_effect=urllib.error.URLError("offline"))
    def test_gives_up_after_three_attempts(self, urlopen, sleep):
        with self.assertRaises(urllib.error.URLError):
            sync_letterboxd.fetch_feed("https://example.com")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    @patch("sync_letterboxd.time.sleep")
    @patch("sync_letterboxd.urllib.request.urlopen")
    def test_does_not_retry_client_errors(self, urlopen, sleep):
        urlopen.side_effect = urllib.error.HTTPError("https://example.com", 404, "missing", {}, io.BytesIO())
        with self.assertRaises(urllib.error.HTTPError):
            sync_letterboxd.fetch_feed("https://example.com")
        urlopen.assert_called_once()
        sleep.assert_not_called()

    @patch("sync_letterboxd.time.sleep")
    @patch("sync_letterboxd.urllib.request.urlopen")
    def test_retries_rate_limit_and_server_errors(self, urlopen, sleep):
        urlopen.side_effect = [
            urllib.error.HTTPError("https://example.com", code, "error", {}, io.BytesIO())
            for code in (429, 503)
        ] + [unittest.mock.MagicMock()]
        sync_letterboxd.fetch_feed("https://example.com")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])


if __name__ == "__main__":
    unittest.main()
