"""CNN downloads recover from interrupted TLS without weakening verification."""

import hashlib
import io
from pathlib import Path
import ssl
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import download_cnn_update as downloader


class CNNDownloadTransportTests(unittest.TestCase):
    def setUp(self):
        self.payload = b"verified CNN source\n"
        self.checksum = hashlib.sha256(self.payload).hexdigest()
        sleep = patch.object(downloader.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        stdout = patch("sys.stdout", new_callable=io.StringIO)
        stdout.start()
        self.addCleanup(stdout.stop)

    def test_ssl_eof_retries_and_returns_verified_bytes(self):
        interrupted = ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING")
        with patch.object(downloader.urllib.request, "urlopen", side_effect=[
            interrupted, io.BytesIO(self.payload),
        ]) as request:
            self.assertEqual(downloader._fetch("source.py", self.checksum), self.payload)
        self.assertEqual(request.call_count, 2)
        self.sleep.assert_called_once_with(1)

    def test_ssl_eof_stops_after_the_retry_limit(self):
        interrupted = ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING")
        with patch.object(downloader.urllib.request, "urlopen", side_effect=interrupted) as request:
            with self.assertRaisesRegex(OSError, "Download failed after 3 attempts: source.py") as caught:
                downloader._fetch("source.py", self.checksum)
        self.assertIs(caught.exception.__cause__, interrupted)
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2])

    def test_checksum_failure_is_not_retried(self):
        with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"tampered")) as request:
            with self.assertRaisesRegex(ValueError, "checksum mismatch: source.py"):
                downloader._fetch("source.py", self.checksum)
        request.assert_called_once()
        self.sleep.assert_not_called()

    def test_http_404_fails_immediately(self):
        missing = urllib.error.HTTPError("https://example.invalid/source.py", 404, "Not Found", {}, None)
        with patch.object(downloader.urllib.request, "urlopen", side_effect=missing) as request:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                downloader._fetch("source.py", self.checksum)
        self.assertIs(caught.exception, missing)
        request.assert_called_once()
        self.sleep.assert_not_called()

    def test_powershell_receives_literal_url_and_quoted_path_through_environment(self):
        url = "https://example.invalid/a'b?x=$(Write-Host injected);y=\"quoted\""
        executable = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        make_temporary = tempfile.TemporaryDirectory
        with make_temporary(prefix="cnn space 'quote' ") as parent:
            paths = []

            def run(command, **options):
                self.assertEqual(command[0], executable)
                self.assertEqual(command[1:-1], ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command"])
                script = command[-1]
                self.assertEqual(script, downloader.POWERSHELL_DOWNLOAD)
                self.assertIn("-UseBasicParsing", script)
                self.assertIn("$env:CNN_CODE_FETCH_URL", script)
                self.assertIn("$env:CNN_CODE_FETCH_PATH", script)
                self.assertFalse(options.get("shell", False))
                self.assertGreater(options["timeout"], 0)
                self.assertEqual(options["env"]["CNN_CODE_FETCH_URL"], url)
                path = Path(options["env"]["CNN_CODE_FETCH_PATH"])
                self.assertIn(" ", str(path))
                self.assertIn("'", str(path))
                self.assertNotIn(url, script)
                self.assertNotIn(str(path), script)
                paths.append(path)
                path.write_bytes(self.payload)
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(downloader.shutil, "which", return_value=executable), \
                    patch.object(downloader.tempfile, "TemporaryDirectory", side_effect=lambda **options:
                                 make_temporary(dir=parent, **options)), \
                    patch.object(downloader.subprocess, "run", side_effect=run) as process:
                self.assertEqual(downloader._powershell_download(url), self.payload)
            process.assert_called_once()
            self.assertFalse(paths[0].exists())

    def test_powershell_failure_and_timeout_end_with_clear_download_errors(self):
        cases = [
            (SimpleNamespace(returncode=1, stderr="remote closed the connection"),
             "PowerShell HTTPS request failed: remote closed the connection"),
            (subprocess.TimeoutExpired("powershell.exe", 60), "timed out"),
        ]
        for result, expected_message in cases:
            with self.subTest(failure=expected_message):
                self.sleep.reset_mock()
                options = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                with patch.object(downloader.shutil, "which", return_value="powershell.exe"), \
                        patch.object(downloader.subprocess, "run", **options) as process:
                    with self.assertRaises(OSError) as caught:
                        downloader._fetch("source.py", self.checksum, transport="powershell")
                self.assertIn("Download failed after 3 attempts: source.py", str(caught.exception))
                self.assertIn(expected_message, str(caught.exception))
                self.assertEqual(process.call_count, 3)
                self.assertEqual(self.sleep.call_count, 2)

    def test_no_output_is_written_until_every_source_is_verified(self):
        payloads = {"cnn/source_{}.py".format(index): self.payload + str(index).encode("ascii")
                    for index in range(11)}
        manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
        installer = ("PAYLOAD_SHA256 = " + repr(manifest) + "\n").encode("utf-8")
        sources = {downloader.INSTALLER: installer, **payloads}
        last_source = list(payloads)[-1]

        for corrupt_last_source in (False, True):
            with self.subTest(corrupt_last_source=corrupt_last_source), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "downloaded CNN"

                def request(url, **options):
                    self.assertFalse(output.exists(), "Destination was written before all downloads passed")
                    prefix = downloader.SOURCE_URL + "/"
                    self.assertTrue(url.startswith(prefix))
                    relative = url[len(prefix):]
                    data = b"tampered last source" if corrupt_last_source and relative == last_source else sources[relative]
                    return io.BytesIO(data)

                with patch.object(downloader, "INSTALLER_SHA256", hashlib.sha256(installer).hexdigest()), \
                        patch.object(downloader.urllib.request, "urlopen", side_effect=request) as fetch:
                    if corrupt_last_source:
                        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                            downloader.download_update(output)
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(downloader.download_update(output), output.resolve())
                        actual = {path.relative_to(output).as_posix(): path.read_bytes()
                                  for path in output.rglob("*") if path.is_file()}
                        self.assertEqual(actual, sources)
                self.assertEqual(fetch.call_count, 12)


if __name__ == "__main__":
    unittest.main()
